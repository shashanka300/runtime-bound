"""Summarise tools/official_gold_sweep.py: gradability per repo under the official offline
environment, the missing modules behind each failure, and (optionally) a run's resolved rate
restricted to the tasks the official environment can grade.

    uv run python tools/official_gold_report.py [--run <swebox results dir> ...]
Writes leaderboard/official_gradability.csv (one row per task).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import polars as pl

from swebox.analysis import load_results

ROOT = Path(__file__).resolve().parents[1]
MISSING = re.compile(r"No module named '([^'.]+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=ROOT / "data/probes/official_gold.jsonl")
    ap.add_argument("--run", type=Path, action="append", default=[])
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.sweep.read_text().splitlines() if line]
    for r in rows:
        mods = sorted(set(MISSING.findall(r.get("patch_tail") or "")))
        r["missing_modules"] = ",".join(mods)
        if r["gradable"]:
            r["reason"] = "gradable"
        elif mods:
            r["reason"] = "missing module"
        elif r.get("base_exit") == r.get("patch_exit"):
            r["reason"] = f"same exit {r.get('base_exit')}"
        else:
            r["reason"] = "other"
    df = pl.DataFrame(
        [
            {
                k: r.get(k)
                for k in (
                    "instance_id",
                    "repo",
                    "gradable",
                    "base_exit",
                    "patch_exit",
                    "reason",
                    "missing_modules",
                    "seconds",
                )
            }
            for r in rows
        ]
    )
    out = ROOT / "leaderboard" / "official_gradability.csv"
    df.write_csv(out)
    print(f"{df['gradable'].sum()}/{df.height} gradable under the official offline environment")
    print(df.group_by("repo").agg(pl.len().alias("n"), pl.col("gradable").sum()).sort("repo"))
    print(df.group_by("reason").len().sort("len", descending=True))
    print(
        "missing modules:", Counter(m for r in rows for m in r["missing_modules"].split(",") if m)
    )
    gradable = set(df.filter(pl.col("gradable"))["instance_id"])
    for run in args.run:
        res = load_results(run)
        sub = res.filter(pl.col("instance_id").is_in(list(gradable)))
        print(
            f"{run}: all {res['resolved'].sum()}/{res.height} = {res['resolved'].mean():.3f}; "
            f"officially gradable only {sub['resolved'].sum()}/{sub.height}; "
            f"as a share of all tasks {sub['resolved'].sum() / max(res.height, 1):.3f}"
        )
    print(out)


if __name__ == "__main__":
    main()
