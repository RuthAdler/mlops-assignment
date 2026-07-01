# Report — LLM inference + o11y

Text-to-SQL PoC serving Qwen3-30B-A3B-Instruct-2507 on 1× H100 80 GB, with a
verify → revise LangGraph agent on top and a Prometheus / Grafana / Langfuse
stack watching both layers.

---

## 1. Serving configuration (Phase 1)

Workload the config below is tuned for: prompts of 1.5–3 k tokens (schema +
question + few-shot), short structured output (a SQL query, tens of tokens),
and 2–3 dependent LLM calls per user request. SLO: end-to-end agent p95 < 5 s
at ≥ 10 rps.

Launch script: `scripts/start_vllm.sh`.

| Flag | Value | Why |
|---|---|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Instruct (non-thinking) variant — we don't want per-call thinking tokens on a latency SLO. |
| `--dtype` | `bfloat16` | Native precision for H100 tensor cores; matches training precision. |
| `--tensor-parallel-size` | `1` | Single GPU. |
| `--max-model-len` | `8192` | Fits 3 k prompt + short output with slack. Smaller ctx = more KV blocks per request = higher concurrency ceiling. |
| `--max-num-batched-tokens` | `8192` | With chunked prefill, this sizes each prefill chunk. Big enough to prefill one 3 k prompt in a single chunk while still leaving room to co-schedule decodes. |
| `--max-num-seqs` | `128` | Concurrency cap. High enough to saturate the GPU at 10 rps × 2–3 calls; low enough to prevent KV thrash. |
| `--gpu-memory-utilization` | `0.90` | ~72 GB budget. 30B bf16 weights ≈ 60 GB → ~10–12 GB for KV + activations. Tight; fp8 KV below makes it workable. |
| `--kv-cache-dtype` | `fp8` | Roughly halves KV memory → ~2× concurrent-request headroom, quality loss negligible for structured SQL output. |
| `--enable-prefix-caching` | on | Schema + system prompt are identical across generate / verify / revise within one run, and repeated across runs on the same DB. Direct win on a prefill-dominated workload. |
| `--enable-chunked-prefill` | on | Interleaves long prefills with in-flight decodes so a 3 k-token schema prompt doesn't stall other requests' decode phase and blow tail latency. |
| `--disable-log-requests` | on | Removes per-request stdout from the hot path. |
| `--trust-remote-code` | on | Qwen3 tokenizer needs it. |

Screenshot: `screenshots/vllm_manual_query.png` — vLLM serving on port 8000
and a manual `curl` to `/v1/chat/completions` returning a SQL query for one of
the eval questions.

---

## 2. Baseline eval results (Phase 5)

30 curated BIRD questions, execution-accuracy comparison (canonicalized row
sets). Output at `results/eval_baseline.json`.

| Metric | Value |
|---|---|
| Pass rate if we stopped after iter 0 | `<fill>` |
| Pass rate if we stopped after iter 1 | `<fill>` |
| Pass rate if we stopped after iter 2 (final) | `<fill>` |
| Mean iterations per question | `<fill>` |
| Fraction of questions that triggered ≥ 1 revise | `<fill>` |

**Commentary.** `<one paragraph: does iter 2 > iter 0 by a meaningful margin?
If yes, the loop is doing real work. If not, is verify too permissive, or is
the model already right on the first shot and revise mostly no-ops?>`

Screenshot: `screenshots/grafana_eval_run.png` — Grafana while the eval was
running.

---

## 3. Hitting the SLO (Phase 6)

Target: **p95 end-to-end agent latency < 5 s at ≥ 10 rps over a 5-minute
window.** Load driver: `uv run python load_test/driver.py --rps <n> --duration 300`.

### Baseline run (Phase 1 config, no tuning)

| Metric | Value |
|---|---|
| Target RPS | 10 |
| Achieved RPS | `<fill>` |
| End-to-end p50 / p95 / p99 | `<fill>` / `<fill>` / `<fill>` |
| vLLM p95 e2e latency | `<fill>` |
| Peak KV cache usage | `<fill>` |
| Preemptions during run | `<fill>` |
| SLO hit? | `<yes / no>` |

Screenshot: `screenshots/grafana_before.png`.

### Iteration log

Format per iteration: *saw X → hypothesized Y → changed Z → result was W*.

1. **Iter 1.** `<saw: e.g. TTFT p95 climbs to 3 s under load while TPOT stays flat, num_requests_waiting > 0. hypothesized: prefill queueing — one 3k prompt blocks decodes. changed: raised --max-num-batched-tokens from 8192 to 16384 to admit two prompts per prefill chunk. result: TTFT p95 dropped to <fill>, e2e p95 to <fill>. SLO status: <fill>.>`
2. **Iter 2.** `<...>`
3. **Iter 3.** `<...>`

### Final run

| Metric | Value |
|---|---|
| End-to-end p50 / p95 / p99 | `<fill>` / `<fill>` / `<fill>` |
| Achieved RPS | `<fill>` |
| SLO hit? | `<yes / no>` |

Screenshot: `screenshots/grafana_after.png`.

### Quality check after tuning

Re-ran the eval on the final config. Output at `results/eval_after_tuning.json`.

| Metric | Baseline | After tuning |
|---|---|---|
| Overall pass rate | `<fill>` | `<fill>` |
| Iter 0 pass rate | `<fill>` | `<fill>` |
| Iter 2 pass rate | `<fill>` | `<fill>` |

**Did quality survive?** `<yes / no + one sentence — e.g. fp8 KV had no
measurable pass-rate impact; or lowering max-model-len dropped one long-context
question.>`

---

## 4. Agent value

`<One paragraph. Cite the per-iteration pass rate from § 2. If iter 2 beats
iter 0 by, say, 5+ points, the verify → revise loop is earning its keep and
here's the class of failure it catches (SQL errors, empty result sets when the
question implies rows, wrong-column returns). If not, the loop is mostly a
tax and here's what would have to change (stricter verifier, different
revision prompt, more diverse retry temperature) before it starts helping.>`

---

## 5. What I'd do with more time

- **`<Specific lever 1>`**: e.g. AWQ-int4 quantize the weights (frees ~30 GB, unlocks 3–4× more KV → higher sustainable concurrency at same p95). Would rerun the load test at 20 rps to find the new ceiling.
- **`<Specific lever 2>`**: e.g. add a retrieval step that trims the schema to only tables mentioned in the top-3 nearest gold questions, cutting the prompt from ~3 k to ~1 k tokens — direct win on prefill-bound TTFT.
- **`<Specific lever 3>`**: e.g. cache verify outcomes keyed on (SQL, result-hash) so retries of the same question don't repay verify cost.
- **`<Specific lever 4>`**: e.g. teach the verifier to emit a structured `issue_type` enum, then route to different revise prompts per class (syntax fix vs. semantic rewrite) instead of one generic revise.
