from pathlib import Path

import polars as pl

from rbcore.hwmon.base import Monitor, Row, Sampler
from rbcore.hwmon.vllm_metrics import parse_metrics


class Counter(Sampler):
    name = "fake"

    def __init__(self) -> None:
        self.n = 0

    def sample(self) -> list[Row]:
        self.n += 1
        return [{"n": self.n}]


class Broken(Sampler):
    name = "broken"
    calls = 0

    def sample(self) -> list[Row]:
        self.calls += 1
        raise RuntimeError("no device")


def test_monitor_writes_parts_and_disables_broken_sampler(tmp_path: Path) -> None:
    broken = Broken()
    mon = Monitor(tmp_path, [Counter(), broken], interval_s=0.01, flush_every_s=0.02)
    for _ in range(10):
        mon.sample_once()
    mon.flush()
    mon.sample_once()
    mon.stop()  # final flush of the 11th sample
    df = _read(tmp_path / "fake")
    assert df["n"].to_list() == list(range(1, 12))
    assert len(list((tmp_path / "fake").glob("part-*.parquet"))) == 2
    assert broken.calls == Monitor.MAX_CONSECUTIVE_ERRORS
    assert not (tmp_path / "broken").exists()
    assert df["t_ns"].is_sorted()


def test_monitor_thread_runs(tmp_path: Path) -> None:
    mon = Monitor(tmp_path, [Counter()], interval_s=0.01, flush_every_s=10).start()
    import time

    time.sleep(0.2)
    mon.stop()
    assert _read(tmp_path / "fake").height >= 5


def _read(d: Path) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in sorted(d.glob("*.parquet"))])


PROM = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
# HELP vllm:prefix_cache_hits_total Prefix cache hits.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 1200.0
# HELP vllm:time_to_first_token_seconds TTFT.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="m"} 3.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 4.0
vllm:time_to_first_token_seconds_count{model_name="m"} 4.0
vllm:time_to_first_token_seconds_sum{model_name="m"} 0.9
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 5.0
"""


def test_parse_vllm_metrics_keeps_gauges_counters_and_hist_sums() -> None:
    rows = {r["metric"]: r["value"] for r in parse_metrics(PROM)}
    assert rows == {
        "vllm:kv_cache_usage_perc": 0.25,
        "vllm:prefix_cache_hits_total": 1200.0,
        "vllm:time_to_first_token_seconds_count": 4.0,
        "vllm:time_to_first_token_seconds_sum": 0.9,
    }
