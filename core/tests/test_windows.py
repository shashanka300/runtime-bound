import polars as pl
import pytest

from rbcore.windows import counter_delta, gpu_energy_j, per_window

S = 1_000_000_000  # ns per second


def _windows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cell": ["a", "b", "empty"],
            "t_start_ns": [0, 10 * S, 100 * S],
            "t_end_ns": [5 * S, 15 * S, 101 * S],
        }
    )


def test_gpu_energy_sums_gpus_and_uses_counter_delta() -> None:
    t = [i * S for i in range(16)]
    gpu = pl.DataFrame(
        {
            "t_ns": t * 2,
            "gpu": [0] * 16 + [1] * 16,
            "energy_mj": [1000.0 * i for i in range(16)] + [500.0 * i for i in range(16)],
        }
    )
    e = gpu_energy_j(gpu, _windows())
    got = dict(zip(e["cell"], e["gpu_energy_j"], strict=True))
    assert got["a"] == pytest.approx(5 * 1.0 + 5 * 0.5)  # 5 s at 1 W + 0.5 W
    assert got["b"] == pytest.approx(7.5)
    assert got["empty"] == 0.0


def test_counter_delta_and_gauge_mean() -> None:
    t = [i * S for i in range(16)]
    vllm = pl.DataFrame(
        {
            "t_ns": t,
            "metric": "vllm:prefix_cache_hits_total",
            "labels": "",
            "value": [float(10 * i) for i in range(16)],
        }
    )
    d = counter_delta(vllm, _windows(), "vllm:prefix_cache_hits_total")
    assert d.filter(pl.col("cell") == "b")["vllm:prefix_cache_hits_total"].item() == 50.0

    m = per_window(vllm, _windows(), [pl.col("value").mean().alias("mean")])
    assert m.filter(pl.col("cell") == "a")["mean"].item() == pytest.approx(25.0)
    assert m.filter(pl.col("cell") == "empty")["mean"].item() is None
