"""Early-stop check for a paired comparison run: exit 0 = keep going, 1 = stop.

    uv run python tools/early_stop.py <results_dir> <baseline_dir> --min-gain 0.05
Compares resolved rates on the (instance_id, seed) pairs both runs finished.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from swebox.analysis import load_results

ap = argparse.ArgumentParser()
ap.add_argument("results", type=Path)
ap.add_argument("baseline", type=Path)
ap.add_argument("--min-gain", type=float, default=0.05)
args = ap.parse_args()
keys = ["instance_id", "seed"]
a = load_results(args.results).select(*keys, pl.col("resolved").alias("new"))
b = load_results(args.baseline).select(*keys, pl.col("resolved").alias("base"))
j = a.join(b, on=keys)
gain = (j["new"].cast(int).sum() - j["base"].cast(int).sum()) / max(j.height, 1)
print(f"paired n={j.height} new={j['new'].sum()} base={j['base'].sum()} gain={gain:+.3f}")
sys.exit(0 if gain >= args.min_gain else 1)
