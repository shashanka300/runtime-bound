"""Fingerprint the downloaded data so runs can record exactly which data they used.

    python -m rbcore.datainfo $DATA_ROOT > $DATA_ROOT/DATA_MANIFEST.json

Hashes small metadata files fully (tasks.jsonl) and large trees by (relative path,
size) listing, which is cheap and still detects any added/removed/resized file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def fingerprint(root: Path) -> dict[str, object]:
    out: dict[str, object] = {"root": str(root)}
    comp = root / "competition"
    for tasks in sorted(comp.rglob("tasks.jsonl")):
        out[str(tasks.relative_to(root))] = hashlib.sha256(tasks.read_bytes()).hexdigest()
    for sub in ("snapshots", "graphs", "embeddings", "wheels"):
        dirs = sorted(comp.rglob(sub))
        for d in (x for x in dirs if x.is_dir()):
            listing = sorted(
                f"{p.relative_to(d)}:{p.stat().st_size}" for p in d.rglob("*") if p.is_file()
            )
            out[str(d.relative_to(root))] = {
                "files": len(listing),
                "listing_sha256": hashlib.sha256("\n".join(listing).encode()).hexdigest(),
            }
    hf = root / "hf" / "hub"
    if hf.is_dir():
        out["hf_models"] = {
            m.name: sorted(r.name for r in (m / "snapshots").iterdir())
            for m in hf.glob("models--*")
            if (m / "snapshots").is_dir()
        }
    return out


if __name__ == "__main__":
    print(json.dumps(fingerprint(Path(sys.argv[1]).resolve()), indent=2))
