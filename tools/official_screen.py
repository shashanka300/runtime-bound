"""Evaluate a submission bundle with the OFFICIAL harness (wheelhouse V31: swegemma 0.2.11,
adk_submission 0.2.13, google-adk 1.36.1, Docker sandbox) against the local vLLM 0.19.1 server,
configured like the host's getting-started notebook. Budgets come from the bundle's
eval_config.yaml (as on the scorer); max_time_minutes is scaled by --time-scale for the slower
4xL4 scorer.

    infra/official/.venv/bin/python tools/official_screen.py <bundle_dir> <out_dir> \
        --ids a,b,c [--time-scale 0.625]
Writes <out_dir>/results.jsonl (one row per task) plus the harness's own results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import yaml

# swegemma caches its unpacked wheel pool as <tempdir>/swegemma_sp_cache_v8/sp_base.tar, keyed only by
# that name, not by the wheels dir: a rebench run (1450-wheel pool) poisoned the public screens
# (pytest-flake8 broke pytest; LoRA screen 11-08). One temp dir per wheel pool, set before swegemma
# is imported.
_COMP = Path(os.environ.get("COMP_DIR", Path(__file__).resolve().parents[1] / "data" / "competition"))
_TMP = Path("/tmp") / ("swegemma_" + __import__("hashlib").sha1(str(_COMP.resolve()).encode()).hexdigest()[:10])
_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_TMP)
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.cli import _build_cli_model_registry
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

ROOT = Path(__file__).resolve().parents[1]
# COMP_DIR: a competition-shaped root (tasks/snapshots/wheels/graphs), e.g. the SWE-rebench
# tasks of tools/ingest_rebench.py
COMP = Path(os.environ.get("COMP_DIR", ROOT / "data" / "competition"))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--tasks", type=Path, default=COMP / "tasks.jsonl",
                    help="tasks file (e.g. data/lora/tasks_hint1.jsonl with hints_text)")
    args = ap.parse_args()
    ev = yaml.safe_load((args.bundle / "eval_config.yaml").read_text())
    ev = ev.get("evaluation", ev)
    limits, gen_constraints = build_submission_limits()
    args.out.mkdir(parents=True, exist_ok=True)
    config = EvalConfig(
        tasks_path=args.tasks,
        snapshots_dir=COMP / "snapshots",
        results_dir=args.out / "harness",
        submission_dir=args.bundle,
        models=_build_cli_model_registry(submission_dir=args.bundle),
        sandbox="docker",
        wheels_dir=COMP / "wheels",
        graph_dir=str(COMP / "graphs"),
        embeddings_dir=str(COMP / "embeddings"),
        timeout_seconds=ev.get("timeout_seconds"),
        max_time_minutes=float(ev["max_time_minutes"]) * args.time_scale
        if ev.get("max_time_minutes")
        else None,
        max_tool_calls=ev.get("max_tool_calls"),
        max_turns=ev.get("max_turns"),
        limits=limits,
        generation_constraints=gen_constraints,
        # as in the host's getting-started notebook
        context_cache_config=ContextCacheConfig(
            min_tokens=2048, ttl_seconds=1800, cache_intervals=10
        ),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5
        ),
        display_mode="quiet",
    )
    evaluator = Evaluator(config)
    wanted = args.ids.split(",")
    known = {t.instance_id for t in load_tasks(args.tasks)}
    assert set(wanted) <= known, set(wanted) - known
    out_file = args.out / "results.jsonl"
    done = (
        {json.loads(ln)["instance_id"] for ln in out_file.read_text().splitlines() if ln.strip()}
        if out_file.exists()
        else set()
    )
    with out_file.open("a") as fh:
        for iid in wanted:
            if iid in done:  # resumable
                continue
            t0 = time.monotonic()
            # Evaluator.run builds the per-task SwegemmaContext; run it one task at a time so
            # rows stream to results.jsonl and a crash loses at most one task.
            evaluator.config.task_ids = [iid]
            evaluator.config.results_dir = args.out / "harness" / iid
            try:
                (r,) = (await evaluator.run()).task_results
                row = {
                    "instance_id": iid,
                    "resolved": bool(r.resolved),
                    "status": r.status,
                    "exit": r.test_exit_code,
                    "error": (getattr(r, "error_message", None) or "")[:300],
                    "patch_chars": len(r.agent_patch or ""),
                    "tool_calls": r.tool_calls,
                    "llm_calls": r.total_llm_calls,
                    "tokens": r.total_tokens,
                    "duration_s": round(r.duration_seconds, 1),
                }
            except Exception as exc:  # keep screening
                row = {"instance_id": iid, "resolved": False, "error": f"harness: {exc!r}"[:300]}
            row["wall_s"] = round(time.monotonic() - t0, 1)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(iid, row["resolved"], row.get("tool_calls"), row["wall_s"], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
