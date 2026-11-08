"""End-to-end RunContext test with every external service disabled (runs anywhere)."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from rbcore.config import ExperimentConfig, MonitorConfig, TrackingConfig
from rbcore.run import RunContext, phase_windows


class P(BaseModel):
    x: int = 1


def _cfg(**monitor: bool) -> ExperimentConfig[P]:
    off = dict(nvml=False, host=False, docker=False, vllm=False, codecarbon=False)
    return ExperimentConfig[P](
        experiment="T0",
        params=P(),
        monitor=MonitorConfig(interval_s=0.05, **(off | monitor)),
        tracking=TrackingConfig(mlflow_uri=None, phoenix_endpoint=None),
    )


def test_run_writes_complete_record(tmp_path: Path) -> None:
    with RunContext(_cfg(host=True), exp_dir=tmp_path) as run:
        with run.phase("work", concurrency=4):
            run.log_metrics({"tok_s": 10.0}, step=0)
        with run.phase("work", concurrency=8):
            pass
        assert run.hw("host").height >= 1

    d = run.dir
    for name in (
        "config.yaml",
        "manifest.json",
        "summary.json",
        "logs.jsonl",
        "events.jsonl",
        "metrics.jsonl",
        "spans.jsonl",
    ):
        assert (d / name).exists(), name
    summary = json.loads((d / "summary.json").read_text())
    assert summary["status"] == "FINISHED"
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["config_hash"] == run.cfg.hash()
    assert manifest["sandbox"]["matches_eval"] is False

    windows = phase_windows(run.events())
    assert windows["concurrency"].to_list() == [4, 8]
    assert (windows["t_end_ns"] >= windows["t_start_ns"]).all()
    assert windows["status"].to_list() == ["ok", "ok"]

    spans = [json.loads(line) for line in (d / "spans.jsonl").read_text().splitlines()]
    assert [s["name"] for s in spans] == ["work", "work"]
    logs = [json.loads(line) for line in (d / "logs.jsonl").read_text().splitlines()]
    assert all(rec.get("run_id") == run.run_id for rec in logs)


def test_failed_run_is_marked(tmp_path: Path) -> None:
    run = RunContext(_cfg(), exp_dir=tmp_path)
    with pytest.raises(RuntimeError), run, run.phase("boom"):
        raise RuntimeError("x")
    summary = json.loads((run.dir / "summary.json").read_text())
    assert summary["status"] == "FAILED"
    assert phase_windows(run.events())["status"].to_list() == ["error"]


def test_overlapping_phases_pair_by_id(tmp_path: Path) -> None:
    # Same-name phases that overlap (as concurrent tasks do) must keep their own windows.
    run = RunContext(_cfg(), exp_dir=tmp_path)
    with run, run.phase("task", task="outer"), run.phase("task", task="inner"):
        pass
    w = phase_windows(run.events()).sort("t_start_ns")
    assert w["task"].to_list() == ["outer", "inner"]
    outer, inner = w.row(0, named=True), w.row(1, named=True)
    assert outer["t_start_ns"] <= inner["t_start_ns"] <= inner["t_end_ns"] <= outer["t_end_ns"]
