"""Load swebox results directories into DataFrames (shared by E1, E2, E5, ...)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def load_results(*dirs: Path) -> pl.DataFrame:
    """One row per (task, seed, tag) from ``task_results.jsonl`` files."""
    rows = []
    for d in dirs:
        f = d / "task_results.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                if line:
                    r = json.loads(line)
                    r["tool_counts"] = json.dumps(r.get("tool_counts") or {})
                    rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


# Gold-patch failures whose cause lies in the official evaluation rules, not our harness.
INFEASIBLE_SIGNATURES = {
    "airgap_network": "Network is unreachable",  # tests expect a real TCP timeout
    "async_plugin_disabled": "async def functions are not natively supported",  # -p no:anyio
}


def classify_selfcheck(rows: list[dict[str, Any]], out: Path) -> pl.DataFrame:
    """Per task: ``gradable`` (gold passes, empty fails), ``passes_without_fix`` (empty
    passes: not discriminative), ``infeasible:<reason>`` (gold fails for a reason inherent
    to the official sandbox/command) or ``env_mismatch`` (gold fails otherwise: a harness
    environment defect to investigate)."""
    by_task: dict[str, dict[str, Any]] = {}
    for r in rows:
        by_task.setdefault(r["instance_id"], {"instance_id": r["instance_id"], "repo": r["repo"]})[
            r["which"]
        ] = r["resolved"]
    out_rows = []
    for tid, rec in by_task.items():
        gold, empty = rec.get("gold"), rec.get("empty")
        if gold and not empty:
            status = "gradable"
        elif gold and empty:
            status = "passes_without_fix"
        else:
            log = out / "selfcheck_gold" / f"{tid}.log"
            text = log.read_text(errors="replace") if log.exists() else ""
            status = next(
                (f"infeasible:{k}" for k, sig in INFEASIBLE_SIGNATURES.items() if sig in text),
                "env_mismatch",
            )
        out_rows.append({**rec, "status": status})
    return pl.DataFrame(out_rows).sort("instance_id")


def load_steps(*dirs: Path) -> pl.DataFrame:
    """Every recorded step (model_request / model_response / tool / compaction)."""
    frames = []
    for d in dirs:
        for f in sorted((d / "traces").glob("*.jsonl")):
            if f.stat().st_size:
                frames.append(pl.read_ndjson(f, infer_schema_length=None))
    if not frames:
        return pl.DataFrame()
    steps = pl.concat(frames, how="diagonal_relaxed")
    if "chars" in steps.columns:  # request breakdown struct -> flat columns
        steps = steps.unnest("chars")
    return steps
