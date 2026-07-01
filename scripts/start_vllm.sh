#!/usr/bin/env bash
#
# Phase 1: vLLM launcher for Qwen3-30B-A3B on 1x H100 80 GB.
#
# Workload profile the flags below are tuned for:
#   - Prompt-heavy: 1.5-3k input tokens (schema + question + few-shot),
#     short structured output (a SQL query, tens of tokens).
#   - Multi-call agent: 2-3 dependent LLM calls per user request.
#   - SLO target: end-to-end agent p95 < 5s at >=10 rps.
#
# Per-flag rationale lives in REPORT.md.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --disable-log-requests \
    --trust-remote-code
