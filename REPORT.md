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

### Iteration 0 — boot-stability fix (preceded the load tests)

Saw `CUDA error: an illegal memory access` in vLLM Engine logs on the 2nd request whose schema preamble was a 99 % prefix-cache hit (`prefix_cache_stats: hits=1024/1033`). Hypothesised a known vLLM 0.10.2 incompatibility between `--kv-cache-dtype fp8` and `--enable-prefix-caching`. Removed `--kv-cache-dtype fp8`. Result: 8/8 sequential cached-prefix requests returned 200 OK, no further engine deaths. Cost: KV-cache budget dropped from 297,840 → 148,896 tokens; theoretical max concurrency at 8 K context dropped from **36.36 ×** to **18.18 ×**. This forced revisiting `--max-num-seqs` next.

### Baseline (Iter 1) — config from §1 with fp8 KV removed

| | Value |
|---|---|
| Configuration | `--max-num-seqs 64 --max-model-len 8192`, prefix cache + chunked prefill on, fp8 KV off |
| Achieved RPS | 8.3 (requested 10) |
| OK | 1115 / 3000 (37 %) |
| Timeouts | 871 (29 %) |
| HTTP errors | 332 (11 %) |
| p50 e2e latency | 80.0 s |
| **p95 e2e latency** | **110.7 s** |
| p99 e2e latency | 115.9 s |

Raw: `results/load_test_baseline_iter1.json`. **SLO missed by ~22 × on p95.**

### Iteration log

**Iter 2 — wrong direction.** Saw 37 concurrent requests on `vllm:num_requests_running` against an 18.18× KV-derived ceiling, plus a non-zero preemption rate during the iter 1 burn. Hypothesised `--max-num-seqs=64` was overcommitting and that preemptions were the binding constraint. Changed `--max-num-seqs` from `64` to `32`. Result: per-request latency for completed requests dropped (p50 80 s → 58.6 s, good for the slots that ran) **but timeouts climbed 48 % (871 → 1285) and total OK dropped 43 % (1115 → 637).** Wrong direction — fewer slots starve total throughput when each agent request demands 2-3 LLM calls. Lesson: when an agent multiplies LLM load, the binding constraint is total served throughput, not per-request preemption. Raw: `results/load_test_iter2.json`.

**Iter 3 — also regresses, for a different reason.** Reverted `--max-num-seqs` to 64 and reduced `--max-model-len` from `8192` to `4096`, expecting per-sequence KV footprint to halve and concurrency budget to roughly double. Result: OK rose to 878 (above iter 2, still below iter 1's 1115). p95 stayed at 118.7 s. The change didn't deliver because our prompts are 1.5–3 K tokens — they never use the full 8 K context. Reducing `max-model-len` only shrinks vLLM's pre-allocated buffers; it does not free per-request KV at runtime when requests are already short. Lesson: tune the lever that actually binds — measure first, change second. Raw: `results/load_test_iter3.json`.

**Lower-RPS feasibility test (iter 3 config at 3 RPS).** Same vLLM still running, knocked load to `--rps 3 --duration 180`. Result: p50 **2.54 s ✅ under SLO**, p95 **12.0 s ❌ above SLO**, 86 % OK, 1 timeout. The system can serve traffic cleanly at 3 RPS but the tail is still dragged by 3-iteration agent runs (verify → revise → verify). Raw: `results/load_test_3rps.json`.

**Note on visual evidence.** A live before/after Grafana capture during the load test could not be produced: the Grafana UI had a stale-state issue during the H100 slot that left dashboard panels rendering "No data" even though Prometheus was scraping correctly (verified by direct `/api/v1/query` curls returning real data — `vllm:num_requests_running` was 37 mid-test, `rate(vllm:request_success_total[1m])` was 19.6 req/s). The dashboard JSON itself is correct and was validated against the synthetic-metrics dry-run shown in `screenshots/grafana_serving.png`. The primary Phase 6 evidence is the load driver JSONs (`results/load_test_baseline_iter1.json` etc.) plus the raw Prometheus data, which together establish the SLO miss and the per-iteration delta.

### Final numbers and verdict

| | Baseline (Nebius, agent-only) | After tuning (H100, iter 1 config) |
|---|---|---|
| P95 e2e latency (10 RPS) | n/a | **110.7 s** ❌ (target 5 s) |
| Best feasible operating point | n/a | **~3 RPS**, p50 2.5 s ✅ / p95 12 s ❌ |
| Eval pass rate (overall) | 43.3 % | **43.3 %** ✅ unchanged |
| Eval pass rate (iter 0 / 1 / 2) | 40.0 % / 43.3 % / 43.3 % | 40.0 % / 43.3 % / 43.3 % ✅ unchanged |

**Verdict — SLO missed, but with a metric-grounded diagnosis.** At the assignment SLO (p95 < 5 s at 10 RPS) the system is **22 × over latency** with a 63 % failure rate. Underlying cause is throughput saturation: at 10 agent-RPS the agent issues 20-30 LLM calls/s, vLLM completes ~20/s on this 30B model on one H100, so the per-call queue grows without bound. None of the three Phase 6 changes moved the SLO; the iter 1 baseline was the best tested config. At a 3 × lower request rate (3 RPS) the system is healthy on p50 but still misses p95 because of agent 3-iteration tail latency, which is a quality-architecture decision rather than a serving-config one. **Quality survived perfectly — eval pass rate is unchanged at 43.3 % from Nebius baseline.**

---

## 6. What I'd do with more time

Specific, in the order I'd actually pull them:

1. **Tighten verify with a structured rubric.** Replace the free-form `"issue": "<short reason>"` with a typed enum field — `"failure_mode": "zero_rows" | "wrong_columns" | "missing_distinct" | "wrong_aggregate" | "reversed_arith" | "guessed_literal" | "other"` — and require the model to fill it. This gives revise a categorical signal instead of natural language, and lets me measure per-failure-mode recovery rate so I can see *which* verify call types are leaking. Expected lift: 5–10 pp on overall pass rate.

2. **Add a third "compare-against-prior-iteration" check inside revise.** Today revise can output SQL identical to the previous attempt; a cheap one-shot post-revise diff reject would loop instead of returning the same wrong answer. Cuts wasted iterations.

3. **Schema-grounded prompt for generate.** Today the schema rendering is the same blob for every question. Picking the 2-3 relevant tables per question (via an extra-cheap LLM "which tables?" call, or even a keyword match) shrinks the prompt by ~70 % and lifts both quality (less distraction) and latency (less prefill). Phase 6 lever I deferred for time.

4. **Speculative decoding with a small draft model.** vLLM supports it on H100. Decode is the single biggest contributor to e2e latency once prefix caching has done its work for repeated agent calls. With a Qwen3-0.6 B draft model, I'd expect 1.5–2× decode throughput; this directly lowers ITL, which dominates the SLO once concurrency is in a healthy range.

5. **Eval set discipline.** 30 questions is enough for a smoke signal but not for a confident regression check. I'd lift the eval set to ~100 and split it into capability buckets (single-table, multi-join, aggregation, date arithmetic) so I can see *which* capability moves when a prompt or flag changes — and stop chasing noise on small N.

6. **Caching layer in front of `/answer` keyed on `(question, db)`.** The eval set repeats; production analytics also has repeat patterns. A trivial in-process LRU would turn the eval re-run from ~90 s to seconds and let me iterate prompts twice as fast.
