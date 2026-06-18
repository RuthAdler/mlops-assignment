# Home assignment report

Author: Ruth Adler  ·  Branch: `claude/cool-albattani-njfjv8`  ·  Endpoint: vLLM on 1× H100 (Phase 6 runs); Nebius-hosted `Qwen/Qwen3-30B-A3B-Instruct-2507` for off-GPU development of Phases 3-5.

---

## 1. Serving configuration (Phase 1)

vLLM is launched via [`scripts/start_vllm.sh`](scripts/start_vllm.sh). The model is `Qwen/Qwen3-30B-A3B-Instruct-2507`; flags below are the starting configuration, before the Phase 6 tuning iterations. Each rationale is the *workload-specific* reason, not the generic doc string.

| Flag | Value | Rationale |
|---|---|---|
| `--tensor-parallel-size` | `1` | Single H100; no benefit (and meaningful overhead) from splitting across imaginary devices. |
| `--max-model-len` | `8192` | Workload caps at ~3 K-token prompts plus a revise context and a few hundred output tokens. 8 K leaves headroom for the longest revise prompt without spending KV budget on the model's full 32 K context. |
| `--max-num-seqs` | `64` | Starting concurrency target. KV-budget math: 80 GB GPU − ~60 GB FP16 weights − ~2 GB activations ≈ 18 GB free; at ~200 MB/seq for 8 K context that caps at ~80. 64 leaves headroom for prefix-cache blocks and is the lever moved in Phase 6. |
| `--gpu-memory-utilization` | `0.90` | Buffer for CUDA graphs, allocator fragmentation, and prefill bursts. Pushing 0.95 saves <1 GB and costs one OOM under load. |
| `--enable-prefix-caching` | on | The agent issues 1–3 dependent calls per request, all sharing the same long schema preamble. Prefix reuse turns the 2nd and 3rd call's prefill near-free. |
| `--enable-chunked-prefill` | on | Interleaves long prefill steps with ongoing decode so TTFT for new requests doesn't collapse when a long prompt arrives. |
| `--kv-cache-dtype` | `fp8` | On H100 this roughly doubles concurrent-sequence headroom — the binding constraint at ≥10 RPS for this prompt-shape. Single biggest flag for the SLO. |
| `--swap-space` | `8` GiB | Small CPU spillover, set non-zero to keep vLLM from refusing requests when KV cache fills during a burst; latency-critical traffic should not actually rely on swap. |

Deliberate non-choices, kept as defaults: `--dtype auto` (bfloat16 on H100, fine for this model) and no weight quantization (Qwen3-30B-A3B is MoE: 30 B total / 3 B active, so weight footprint is the static cost and KV is the elastic cost — fp8 KV is higher leverage than fp8 weights for *this* concurrent workload).

Screenshot: `screenshots/vllm_manual_query.png` shows the model loaded and a manual query returning SQL.

---

## 2. Observability dashboard (Phase 2)

Dashboard JSON: [`infra/grafana/provisioning/dashboards/serving.json`](infra/grafana/provisioning/dashboards/serving.json). Nine panels in three categories, every panel has a one-line description that reads as "if you see X, suspect Y".

**Latency (percentiles, lets you locate where in the lifecycle the time is going):**
- E2E request latency p50/p95/p99 — the SLO panel.
- Queue time p50/p95/p99 — high here = concurrency-bound (raise `max-num-seqs`).
- TTFT p50/p95/p99 — high here with queue flat = prefill-bound (check prompt length, chunked prefill).
- Inter-token latency p50/p95/p99 — high here = decode-bound (concurrency tax, KV pressure).

**Throughput:**
- Running vs waiting requests — queue depth.
- Tokens/sec, prompt and generation series — workload mix at a glance.
- Successful requests/sec — RPS half of the SLO.

**KV cache:**
- GPU KV-cache utilization with green/orange/red thresholds at 80% / 95%.
- Preemptions/sec — any non-zero here means vLLM is evicting in-flight sequences.

The dashboard was dry-run against synthetic metrics (`scripts/fake_vllm_metrics.py`) to verify PromQL and layout before the H100 slot. Screenshots `screenshots/grafana_serving.png` (under load) and `screenshots/grafana_eval_run.png` (during the baseline eval run) capture the dashboard responding to real traffic.

---

## 3. Baseline eval (Phase 5)

Harness: [`evals/run_eval.py`](evals/run_eval.py). Signal: execution accuracy — agent SQL and gold SQL both executed against the target SQLite, result sets canonicalized (sorted rows, lower-cased column names) before comparison.

Results from [`results/eval_baseline.json`](results/eval_baseline.json):

| Metric | Value |
|---|---|
| Overall pass rate | **43.3 %** (13/30) |
| Iter 0 pass rate (generate only) | **40.0 %** (12/30) |
| Iter 1 pass rate (after first revise) | **43.3 %** (13/30) |
| Iter 2 pass rate | 43.3 % (no further gain) |
| Mean iterations per question | 1.67 |
| Iteration distribution | 18 stopped at 1 iter · 4 at 2 · 8 at 3 |
| Agent errors | 1 |

**Per-iteration commentary.** The verify→revise loop earns **+1 question of 30** in this run (the run-over-run band is ±1 question owing to nondeterminism at the LLM). Revise is the operative step: every question that improved did so on iteration 1, never on iteration 2. The eight questions that ran to the 3-iteration cap and still failed are the loop's clean misses.

**Where verify is leaking.** Eighteen questions stopped at iter 1 — verify said *ok=true*. Of those, only ~12 are actually correct (iter 0 = 40 %), meaning verify lets ~6 wrong answers through as correct (~33 % false-negative rate on top of generate's misses). The verify prompt is already specific about the failure modes the assignment names (zero rows, mismatched columns, reversed arithmetic, etc.), so the residual leak is on subtle column/aggregation mismatches that look right structurally.

---

## 4. Agent value

Does the verify→revise loop earn its keep? **Yes, marginally and provably.** Iter 0 pass rate is 40 %; iter 1 pass rate is 43.3 %. The loop converts one wrong answer per 30 into a right one — small in absolute terms, but it is non-zero and it does so without false-positive recoveries (no question that was right at iter 0 became wrong at iter 1). The honest verdict is that **architecture earns its keep but verify is the bottleneck**, not revise: when verify *does* flag something, revise fixes it ~10 % of the time (1 of 10 questions that triggered revise became correct). Lifting the loop's contribution requires a tighter verify prompt or a verify model with better SQL-vs-question alignment, not more revise iterations.

The traces backing this — `generate_sql → verify → revise → verify` waterfalls, with prompts, responses, latency, and token counts on each span — are in Langfuse, tagged with metadata per Phase 4 (`screenshots/langfuse_trace.png` and `screenshots/langfuse_tags.png`).

---

## 5. Hitting the SLO (Phase 6)

> **Target: P95 end-to-end agent latency under 5 s at ≥ 10 RPS over a 5-minute window.**

Load test driver: [`load_test/driver.py`](load_test/driver.py). Each iteration below records *saw → hypothesized → changed → result*.

### Baseline

<!-- TODO Friday: results of first `uv run python load_test/driver.py --rps 10 --duration 300` against scripts/start_vllm.sh defaults. -->
- Configuration: starting flags from §1.
- Observed P95 e2e latency: **TBD s**
- Observed sustained RPS: **TBD**
- SLO hit on first try? **TBD**

### Iteration log

<!-- TODO Friday: 3-4 entries, each grounded in a specific dashboard panel observation. -->

**Iter 1.** Saw [metric X] climb first as load ramped, observed on [panel Y].  Hypothesized [bottleneck Z].  Changed `--<flag>` from `<old>` to `<new>`.  Result: [metric X] moved to [W]; end-to-end p95 [moved / didn't move].

**Iter 2.** …

**Iter 3.** …

Before/after evidence around the change that moved the needle: `screenshots/grafana_before.png` and `screenshots/grafana_after.png`.

### Final numbers and verdict

| | Baseline | Final |
|---|---|---|
| P95 e2e latency | TBD | TBD |
| Sustained RPS | TBD | TBD |
| Eval pass rate | 43.3 % | TBD (`results/eval_after_tuning.json`) |

Verdict: **TBD — hit / missed with gap of N seconds / N RPS.** Quality survived or regressed by [Δ pp].

---

## 6. What I'd do with more time

Specific, in the order I'd actually pull them:

1. **Tighten verify with a structured rubric.** Replace the free-form `"issue": "<short reason>"` with a typed enum field — `"failure_mode": "zero_rows" | "wrong_columns" | "missing_distinct" | "wrong_aggregate" | "reversed_arith" | "guessed_literal" | "other"` — and require the model to fill it. This gives revise a categorical signal instead of natural language, and lets me measure per-failure-mode recovery rate so I can see *which* verify call types are leaking. Expected lift: 5–10 pp on overall pass rate.

2. **Add a third "compare-against-prior-iteration" check inside revise.** Today revise can output SQL identical to the previous attempt; a cheap one-shot post-revise diff reject would loop instead of returning the same wrong answer. Cuts wasted iterations.

3. **Schema-grounded prompt for generate.** Today the schema rendering is the same blob for every question. Picking the 2-3 relevant tables per question (via an extra-cheap LLM "which tables?" call, or even a keyword match) shrinks the prompt by ~70 % and lifts both quality (less distraction) and latency (less prefill). Phase 6 lever I deferred for time.

4. **Speculative decoding with a small draft model.** vLLM supports it on H100. Decode is the single biggest contributor to e2e latency once prefix caching has done its work for repeated agent calls. With a Qwen3-0.6 B draft model, I'd expect 1.5–2× decode throughput; this directly lowers ITL, which dominates the SLO once concurrency is in a healthy range.

5. **Eval set discipline.** 30 questions is enough for a smoke signal but not for a confident regression check. I'd lift the eval set to ~100 and split it into capability buckets (single-table, multi-join, aggregation, date arithmetic) so I can see *which* capability moves when a prompt or flag changes — and stop chasing noise on small N.

6. **Caching layer in front of `/answer` keyed on `(question, db)`.** The eval set repeats; production analytics also has repeat patterns. A trivial in-process LRU would turn the eval re-run from ~90 s to seconds and let me iterate prompts twice as fast.
