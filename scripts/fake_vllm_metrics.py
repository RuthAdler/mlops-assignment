"""Synthetic vLLM /metrics exposer for dry-running the Grafana dashboard.

Publishes vllm:* counters, gauges, and histograms in Prometheus text format
on a configurable port. Each scrape advances the state so panels animate.
Defaults to port 8000 so Prometheus scrapes it without any infra/prometheus.yml
change - run this when real vLLM is NOT running.

Run:
    uv run python scripts/fake_vllm_metrics.py
    # then open Grafana at http://localhost:3000 -> "vLLM serving"

Numbers are synthetic; this is only for verifying dashboard PromQL + layout
BEFORE the real H100 slot. Do NOT use these numbers in REPORT.md.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf"),
)
MODEL_LABEL = 'model_name="Qwen/Qwen3-0.6B-fake"'
START = time.time()


class Histogram:
    """Cumulative-bucket histogram emitter in Prometheus text format."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.bucket_counts: list[int] = [0] * len(BUCKETS)
        self.sum_value: float = 0.0
        self.count: int = 0

    def observe(self, value: float) -> None:
        self.sum_value += value
        self.count += 1
        for i, upper in enumerate(BUCKETS):
            if value <= upper:
                self.bucket_counts[i] += 1

    def expose(self) -> str:
        lines: list[str] = []
        for upper, count in zip(BUCKETS, self.bucket_counts):
            le = "+Inf" if math.isinf(upper) else f"{upper}"
            lines.append(f'{self.name}_bucket{{le="{le}",{MODEL_LABEL}}} {count}')
        lines.append(f"{self.name}_sum{{{MODEL_LABEL}}} {self.sum_value:.4f}")
        lines.append(f"{self.name}_count{{{MODEL_LABEL}}} {self.count}")
        return "\n".join(lines)


# Module-level state - tiny script, no need to wrap in a class.
e2e = Histogram("vllm:e2e_request_latency_seconds")
queue = Histogram("vllm:request_queue_time_seconds")
ttft = Histogram("vllm:time_to_first_token_seconds")
itl = Histogram("vllm:time_per_output_token_seconds")

prompt_tokens_total: float = 0.0
generation_tokens_total: float = 0.0
request_success_total: int = 0
num_preemptions_total: int = 0


def load_factor(t: float) -> float:
    """Synthetic load curve: ramps over 10 min, then oscillates 0.3-1.0."""
    ramp = min(1.0, t / 600.0)
    return 0.3 + 0.7 * ramp * (0.6 + 0.4 * math.sin(t / 60.0))


def advance() -> None:
    """Generate ~one scrape-interval's worth of synthetic requests."""
    global prompt_tokens_total, generation_tokens_total
    global request_success_total, num_preemptions_total

    load = load_factor(time.time() - START)
    n_reqs = max(1, int(10 * load))
    for _ in range(n_reqs):
        prompt_len = random.randint(1500, 3000)
        output_len = random.randint(20, 200)
        q = random.expovariate(1 / (0.1 + 0.5 * load))
        prefill = max(0.05, random.gauss(0.4 + 0.8 * load, 0.1))
        decode_per_tok = max(0.005, random.gauss(0.015 + 0.03 * load, 0.005))

        queue.observe(q)
        ttft.observe(q + prefill)
        itl.observe(decode_per_tok)
        e2e.observe(q + prefill + decode_per_tok * output_len)

        prompt_tokens_total += prompt_len
        generation_tokens_total += output_len
        request_success_total += 1

    if load > 0.7 and random.random() < 0.1:
        num_preemptions_total += 1


def render_gauges() -> list[str]:
    """Render the gauge metrics that snapshot at scrape time."""
    load = load_factor(time.time() - START)
    running = max(0, int(8 * load + random.gauss(0, 1)))
    waiting = max(0, int(15 * (load - 0.6)) + random.randint(0, 3))
    kv = max(0.0, min(0.99, 0.2 + 0.7 * load + random.gauss(0, 0.03)))
    return [
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{{{MODEL_LABEL}}} {running}",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{{{MODEL_LABEL}}} {waiting}",
        "# TYPE vllm:gpu_cache_usage_perc gauge",
        f"vllm:gpu_cache_usage_perc{{{MODEL_LABEL}}} {kv:.4f}",
    ]


def render() -> str:
    """Produce the full /metrics text body."""
    advance()
    parts: list[str] = render_gauges()
    parts += [
        "# TYPE vllm:prompt_tokens_total counter",
        f"vllm:prompt_tokens_total{{{MODEL_LABEL}}} {prompt_tokens_total:.0f}",
        "# TYPE vllm:generation_tokens_total counter",
        f"vllm:generation_tokens_total{{{MODEL_LABEL}}} {generation_tokens_total:.0f}",
        "# TYPE vllm:request_success_total counter",
        f"vllm:request_success_total{{{MODEL_LABEL}}} {request_success_total}",
        "# TYPE vllm:num_preemptions_total counter",
        f"vllm:num_preemptions_total{{{MODEL_LABEL}}} {num_preemptions_total}",
        "# TYPE vllm:e2e_request_latency_seconds histogram",
        e2e.expose(),
        "# TYPE vllm:request_queue_time_seconds histogram",
        queue.expose(),
        "# TYPE vllm:time_to_first_token_seconds histogram",
        ttft.expose(),
        "# TYPE vllm:time_per_output_token_seconds histogram",
        itl.expose(),
    ]
    return "\n".join(parts) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return  # quiet scrape spam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=8000,
        help="port to listen on (default 8000, matches infra/prometheus.yml)",
    )
    args = parser.parse_args()
    print(f"Fake vLLM /metrics on http://0.0.0.0:{args.port}/metrics")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
