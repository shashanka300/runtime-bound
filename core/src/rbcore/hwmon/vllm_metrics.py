"""Scrape vLLM's Prometheus ``/metrics`` endpoint into long-format rows.

Kept: every ``vllm:*`` gauge and counter (e.g. KV-cache usage, running/waiting
requests, prefix-cache queries/hits, prompt/generation tokens) plus ``_sum`` and
``_count`` of histograms (TTFT, inter-token latency, e2e latency), from which means
over any window are exact: (sum[t1]-sum[t0]) / (count[t1]-count[t0]).
Buckets are dropped to keep volume small; the per-request benchmark output holds the
full latency distributions.
"""

from __future__ import annotations

import httpx
from prometheus_client.parser import text_string_to_metric_families

from rbcore.hwmon.base import Row, Sampler


class VllmMetricsSampler(Sampler):
    name = "vllm"

    def __init__(self, base_url: str, timeout_s: float = 0.5) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def sample(self) -> list[Row]:
        text = self._client.get("/metrics").raise_for_status().text
        return parse_metrics(text)

    def close(self) -> None:
        self._client.close()


def parse_metrics(text: str) -> list[Row]:
    rows: list[Row] = []
    for fam in text_string_to_metric_families(text):
        if not fam.name.startswith("vllm:"):
            continue
        for s in fam.samples:
            if fam.type == "histogram" and not s.name.endswith(("_sum", "_count")):
                continue
            labels = {k: v for k, v in s.labels.items() if k not in ("model_name", "engine")}
            rows.append(
                {
                    "metric": s.name,
                    "labels": ",".join(f"{k}={v}" for k, v in sorted(labels.items())),
                    "value": float(s.value),
                }
            )
    return rows
