"""Ingest SWE-rebench tasks (nebius/SWE-rebench, filtered split, CC-BY-4.0) into a competition-
shaped data root, for LoRA data generation on repositories outside the four public ones
(docs/design_changes.md, 2026-11-06). Only the tasks and their environments are used; the
trajectories are Gemma's own.

    <root>/competition/tasks.jsonl     our Task schema (problem_statement, patch, test_patch, ...)
    <root>/competition/snapshots/      <id>.tgz: shallow clone at base_commit (depth 1)
    <root>/competition/wheels/         public wheels + py3.13 wheels for the new tasks
    <root>/competition/{graphs,embeddings}/   empty (the bundle never uses those tools)

Steps (each resumable): select -> snapshots -> wheels -> selfcheck (gold, empty) -> keep
(tasks where the gold patch resolves and an empty patch fails).

    python3 tools/ingest_rebench.py select --n 320
    python3 tools/ingest_rebench.py snapshots
    python3 tools/ingest_rebench.py wheels
    uv run python tools/ingest_rebench.py selfcheck --which gold   (then --which empty)
    python3 tools/ingest_rebench.py keep
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/external/swe_rebench/filtered.parquet"
OUT = ROOT / "data/external/rebench_root/competition"
OURS = {"fastapi/fastapi", "Textualize/rich", "psf/requests", "encode/httpx"}
HEAVY = ("torch", "tensorflow", "jax", "cupy", "pyspark", "opencv", "vtk", "pyqt", "pyside")
BUILD = ["setuptools-scm", "hatch-vcs", "hatch-fancy-pypi-readme", "versioneer", "setuptools",
         "wheel", "hatchling", "flit-core", "poetry-core", "pdm-backend", "pytest", "pytest-timeout",
         "pytest-asyncio", "pytest-mock", "pytest-cov", "pytest-xdist", "freezegun", "hypothesis"]


def short_id(row: dict, used: set[str]) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", row["repo"].split("/")[1].lower()).strip("-")
    num = row["instance_id"].rsplit("-", 1)[1]
    iid = f"{name}_{num}"
    if iid in used:
        owner = re.sub(r"[^a-z0-9]+", "-", row["repo"].split("/")[0].lower())
        iid = f"{name}-{owner}_{num}"
    used.add(iid)
    return iid


def select(n: int) -> None:
    import pandas as pd

    d = pd.read_parquet(SRC)
    d["py"] = d.install_config.map(lambda c: c.get("python"))
    d["heavy"] = d.requirements.fillna("").str.lower().map(lambda r: any(h in r for h in HEAVY))
    d["nf"] = d.meta.map(lambda m: m.get("num_modified_files") or 99)
    d["diff"] = d.meta.map(lambda m: (m.get("llm_score") or {}).get("difficulty_score", 9))
    d["bad"] = d.PASS_TO_FAIL.map(len) + d.FAIL_TO_FAIL.map(len)
    d["pyonly"] = d.patch.map(lambda p: all(f.endswith(".py") for f in re.findall(r"(?m)^\+\+\+ b/(\S+)", p)))
    m = d[(d.created_at >= "2023") & d.py.isin(["3.9", "3.10", "3.11", "3.12"]) & ~d.heavy
          & (d.nf <= 3) & (d.FAIL_TO_PASS.map(len) > 0) & (d.bad == 0) & d.pyonly
          & (d["diff"] <= 2) & ~d.repo.isin(OURS) & (d.problem_statement.str.len() > 200)
          & (d.requirements.fillna("").str.count("\n") < 80)]  # no giant dependency trees
    m = m.sort_values("created_at", ascending=False).groupby("repo").head(2)
    m = m.sample(frac=1, random_state=0).head(n)
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("snapshots", "wheels", "graphs", "embeddings"):
        (OUT / sub).mkdir(exist_ok=True)
    used: set[str] = set()
    with (OUT / "tasks.jsonl").open("w") as fh, (OUT / "meta.jsonl").open("w") as mf:
        for r in m.to_dict("records"):
            iid = short_id(r, used)
            fh.write(json.dumps({
                "instance_id": iid, "repo": r["repo"], "base_commit": r["base_commit"],
                "problem_statement": r["problem_statement"], "hints_text": "",
                "patch": r["patch"], "test_patch": r["test_patch"], "created_at": str(r["created_at"]).replace(" ", "T")[:19] + "Z",  # ISO, as install_deps expects
            }) + "\n")
            mf.write(json.dumps({"instance_id": iid, "source_id": r["instance_id"], "python": r["py"],
                                 "requirements": r["requirements"], "install": r["install_config"].get("install"),
                                 "license": r["license_name"]}) + "\n")
    print(f"selected {len(m)} tasks from {m.repo.nunique()} repos (of {len(d)}; {len(m)} after filters)")


def tasks() -> list[dict]:
    return [json.loads(ln) for ln in (OUT / "tasks.jsonl").open() if ln.strip()]


def snapshots() -> None:
    ok = fail = 0
    for t in tasks():
        dest = OUT / "snapshots" / f"{t['instance_id']}.tgz"
        if dest.exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            cmds = [["git", "init", "-q"], ["git", "remote", "add", "origin", f"https://github.com/{t['repo']}.git"],
                    ["git", "fetch", "-q", "--depth", "1", "origin", t["base_commit"]],
                    ["git", "checkout", "-q", "FETCH_HEAD"], ["git", "remote", "remove", "origin"]]
            try:
                for c in cmds:
                    subprocess.run(c, cwd=repo, check=True, capture_output=True, timeout=600)
                subprocess.run(["tar", "czf", str(dest), "-C", str(repo), "."], check=True, timeout=600)
                ok += 1
            except Exception as exc:  # unreachable repo or commit: drop the task later
                fail += 1
                print("snapshot failed", t["instance_id"], str(exc)[:120], flush=True)
    print(f"snapshots: {ok} new, {fail} failed", flush=True)


def wheels() -> None:
    wdir = OUT / "wheels"
    for w in (ROOT / "data/competition/wheels").iterdir():  # the public pool, unchanged
        if not (wdir / w.name).exists():
            os.link(w, wdir / w.name)
    meta = [json.loads(ln) for ln in (OUT / "meta.jsonl").open()]
    pins: dict[str, set[str]] = {}
    for m in meta:
        for ln in (m["requirements"] or "").splitlines():
            ln = ln.split(";")[0].strip()
            mm = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*==\s*([^\s#]+)", ln)
            if mm:
                pins.setdefault(mm.group(1).lower(), set()).add(mm.group(2))
    specs = sorted(set(BUILD) | set(pins))
    pip = [sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--only-binary=:all:",
           "--python-version", "3.13", "--implementation", "cp", "--abi", "cp313",
           "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux2014_x86_64",
           "--platform", "manylinux_2_17_x86_64", "--platform", "any", "-d", str(wdir)]
    miss = []
    for i, name in enumerate(specs):
        reqs = [name] + [f"{name}=={v}" for v in sorted(pins.get(name, ()))]
        got = False
        for r in reqs:
            got |= subprocess.run(pip + [r], capture_output=True, timeout=300).returncode == 0
        if not got:
            miss.append(name)
        if i % 50 == 0:
            print(f"wheels {i}/{len(specs)} missing {len(miss)}", flush=True)
    (OUT / "wheels_missing.txt").write_text("\n".join(miss))
    print(f"wheels: {len(specs)} names, {len(miss)} without a py3.13 wheel", flush=True)


def selfcheck(which: str, concurrency: int) -> None:
    import asyncio

    os.environ["DATA_ROOT"] = str(OUT.parent)
    from swebox import sandbox as sb
    from swebox.evaluate import verify_only
    from swebox.tasks import load_tasks

    out = OUT.parent / f"selfcheck_{which}.jsonl"
    done = {json.loads(ln)["instance_id"] for ln in out.open()} if out.exists() else set()
    todo = [t for t in load_tasks(OUT / "tasks.jsonl")
            if t.instance_id not in done and t.snapshot(OUT.parent / "competition").exists()]
    for i in range(0, len(todo), 8):  # chunks, so a crash keeps the rows so far
        rows = asyncio.run(verify_only(todo[i:i + 8], which, sb.Limits(8.0, 4.0),
                                       OUT.parent / "selfcheck_logs", concurrency))
        with out.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"selfcheck {which}: {i + len(rows)}/{len(todo)}", flush=True)


def keep() -> None:
    def res(which: str) -> dict[str, bool]:
        rows = [json.loads(ln) for ln in (OUT.parent / f"selfcheck_{which}.jsonl").open()]
        return {r["instance_id"]: r["resolved"] for r in rows}
    gold, empty = res("gold"), res("empty")
    good = [t for t in tasks() if gold.get(t["instance_id"]) and empty.get(t["instance_id"]) is False]
    shutil.copy(OUT / "tasks.jsonl", OUT / "tasks_all.jsonl")
    with (OUT / "tasks.jsonl").open("w") as fh:
        for t in good:
            fh.write(json.dumps(t) + "\n")
    print(f"kept {len(good)} of {len(gold)} (gold resolves, empty fails); "
          f"repos {len({t['repo'] for t in good})}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "snapshots", "wheels", "selfcheck", "keep"])
    ap.add_argument("--which", choices=["gold", "empty"], default="gold")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--n", type=int, default=320)
    a = ap.parse_args()
    {"select": lambda: select(a.n), "snapshots": snapshots, "wheels": wheels,
     "selfcheck": lambda: selfcheck(a.which, a.concurrency), "keep": keep}[a.step]()
