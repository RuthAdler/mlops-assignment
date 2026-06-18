#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

ARGS=(
    --model "$MODEL"
    --host 0.0.0.0
    --port 8000
    # One H100 80GB, so keep tensor parallelism local to a single GPU.
    --tensor-parallel-size 1
    # 3K-token prompts plus revise context/output headroom without using the full model context.
    --max-model-len 8192
    # Starting concurrency target; Phase 6 should tune this against queue time and KV pressure.
    --max-num-seqs 64
    # Leave an OOM buffer for bursts, CUDA graphs, and allocator fragmentation.
    --gpu-memory-utilization 0.90
    # Agent calls share long schema/prompt prefixes, so cache reuse should reduce prefill cost.
    --enable-prefix-caching
    # Interleave long prefills with decode work to protect TTFT under load.
    --enable-chunked-prefill
    # FP8 KV roughly doubles cache headroom, the key constraint for 10+ concurrent agent calls.
    --kv-cache-dtype fp8
    # Small CPU spillover buffer; real SLO tuning should avoid relying on swap for latency.
    --swap-space 8
)

exec uv run python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
