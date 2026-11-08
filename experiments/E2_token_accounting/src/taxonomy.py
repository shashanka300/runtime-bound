"""Failure taxonomy for swebox task runs (H3: does context loss cause failures?).

Every unresolved run on a gradable task gets exactly one PRIMARY label, assigned by the
first matching rule (precedence = the proximate cause of failure):

    context_exhausted    the session ended because a request exceeded the context window
    time_exhausted       the wall-clock budget ran out
    gave_up_no_patch     the session ended normally (submitted / nudges) with an empty patch
    patch_apply_failed   the patch could not be applied in the grading container
    mislocalized_file    the patch touches no source file the reference patch touches
    mislocalized_region  right file(s), but no hunk overlaps a reference hunk (±slack lines)
    wrong_edit           a hunk overlaps the reference region, yet the tests fail
    other                model/harness errors that fit none of the above

A second, PATCH-ONLY label ignores how the session ended and classifies whatever patch
exists (``no_patch``, ``mislocalized_file``, ``mislocalized_region``, ``right_region``). Cross-
tabulating the two shows, e.g., how many context-killed runs had already found the right spot.

Behavioural markers (multi-label, every run): repeated identical actions, re-reads (from the
token accounting segments), reverted edits, whether the agent ever viewed a reference file,
and scratch files left in the patch (new files the reference does not create; excluded from
localization). These operationalise the plan's "repeated actions, forgotten hypothesis, re-reading".
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import polars as pl

from swebox.tasks import Task

from .accounting import KEY, read_ranges

PRIMARY = [
    "context_exhausted",
    "time_exhausted",
    "gave_up_no_patch",
    "patch_apply_failed",
    "mislocalized_file",
    "mislocalized_region",
    "wrong_edit",
    "other",
]
FREE_TOOLS = {"get_status", "submit_patch"}
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.MULTILINE)
_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)


def is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        path.startswith(("tests/", "test/"))
        or "/tests/" in path
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def new_files(patch: str) -> set[str]:
    """Files the patch creates (``new file mode``)."""
    out = set()
    for block in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE):
        m = _FILE.search(block)
        if m and re.search(r"^new file mode", block, re.MULTILINE):
            out.add(m.group(2))
    return out


def patch_hunks(patch: str, exclude: set[str] | None = None) -> dict[str, list[tuple[int, int]]]:
    """Source file -> old-side line ranges touched (test files and ``exclude`` skipped)."""
    out: dict[str, list[tuple[int, int]]] = {}
    exclude = exclude or set()
    blocks = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    for block in blocks:
        m = _FILE.search(block)
        if not m:
            # tasks.jsonl patches start at '--- a/'; fall back to the +++ header
            plus = re.search(r"^\+\+\+ b/(\S+)", block, re.MULTILINE)
            if not plus:
                continue
            path = plus.group(1)
        else:
            path = m.group(2)
        if is_test_path(path) or path in exclude:
            continue
        ranges = [(int(a), int(a) + max(int(n or 1), 1) - 1) for a, n in _HUNK.findall(block)]
        out.setdefault(path, []).extend(ranges or [(1, 1)])
    return out


def scratch_files(agent_patch: str, gold_patch: str) -> set[str]:
    """Files the agent created that the reference patch does not: scratch/repro scripts.

    The spec's write_file strips a leading "/", so write_file("/tmp/repro.py") lands in
    /workspace/tmp/repro.py and enters the patch; such files are not a localization attempt.
    """
    gold_new = new_files(gold_patch)
    return {f for f in new_files(agent_patch) if f not in gold_new}


def localization(agent_patch: str, gold_patch: str, slack: int = 5) -> str:
    agent = patch_hunks(agent_patch, exclude=scratch_files(agent_patch, gold_patch))
    gold = patch_hunks(gold_patch)
    if not agent:
        return "no_patch"
    shared = set(agent) & set(gold)
    if not shared:
        return "mislocalized_file"
    for f in shared:
        for a0, a1 in agent[f]:
            for g0, g1 in gold[f]:
                if a0 <= g1 + slack and g0 <= a1 + slack:
                    return "right_region"
    return "mislocalized_region"


def primary_label(row: dict[str, object], loc: str) -> str:
    err = str(row.get("agent_error") or "")
    if "ContextWindowExceeded" in err:
        return "context_exhausted"
    if row.get("end_reason") == "time":
        return "time_exhausted"
    if loc == "no_patch":
        return "gave_up_no_patch" if row.get("end_reason") in ("submitted", "nudges") else "other"
    if row.get("patch_applied") is False:
        return "patch_apply_failed"
    return {
        "mislocalized_file": "mislocalized_file",
        "mislocalized_region": "mislocalized_region",
        "right_region": "wrong_edit",
    }.get(loc, "other")


# -- behavioural markers --------------------------------------------------------------------
def behaviour(steps: pl.DataFrame, gold_files: dict[str, set[str]]) -> pl.DataFrame:
    """Per run: repeated actions, reverted edits, whether a reference file was viewed."""
    tools = steps.filter(pl.col("kind") == "tool").sort([*KEY, "step"])
    rows = []
    for (tag, iid, seed), g in tools.group_by(KEY, maintain_order=True):
        calls = Counter()
        repeats = reverted = 0
        edits_old: dict[str, set[str]] = {}
        viewed_gold = False
        gold = gold_files.get(str(iid), set())
        for r in g.iter_rows(named=True):
            tool, args = r["tool"], r["args"] or "{}"
            if tool not in FREE_TOOLS:
                calls[(tool, args)] += 1
                if calls[(tool, args)] > 1:
                    repeats += 1
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                parsed = {}
            if tool == "edit_file" and isinstance(parsed, dict):
                path = str(parsed.get("filepath", "")).removeprefix("/workspace/")
                new = parsed.get("new_string")
                if new is not None and new in edits_old.get(path, set()):
                    reverted += 1  # puts back text an earlier edit removed
                edits_old.setdefault(path, set()).add(str(parsed.get("old_string")))
            for path, *_ in read_ranges(
                tool, args, r.get("start_line"), r.get("end_line"), r.get("filepath")
            ):
                viewed_gold |= path in gold
            if tool in ("edit_file", "write_file") and isinstance(parsed, dict):
                viewed_gold |= str(parsed.get("filepath", "")).removeprefix("/workspace/") in gold
        rows.append(
            {
                "tag": tag,
                "instance_id": iid,
                "seed": seed,
                "repeated_actions": repeats,
                "reverted_edits": reverted,
                "viewed_gold_file": viewed_gold,
                "tool_calls_total": g.height,
            }
        )
    return pl.DataFrame(rows)


# -- labelling ------------------------------------------------------------------------------
def label_runs(results: pl.DataFrame, dirs: list[Path], tasks: dict[str, Task]) -> pl.DataFrame:
    """Attach patch-only and primary labels to every run (resolved runs: ``resolved``)."""
    rows = []
    for r in results.iter_rows(named=True):
        key = f"{r['instance_id']}__s{r['seed']}"
        patch = ""
        for d in dirs:
            if d.name == r["tag"] and (d / "patches" / f"{key}.patch").exists():
                patch = (d / "patches" / f"{key}.patch").read_text(errors="replace")
        gold = tasks[r["instance_id"]].patch
        loc = localization(patch, gold)
        primary = "resolved" if r["resolved"] else primary_label(r, loc)
        rows.append(
            {
                "tag": r["tag"],
                "instance_id": r["instance_id"],
                "seed": r["seed"],
                "patch_only": loc,
                "primary": primary,
                "scratch_files": len(scratch_files(patch, gold)),
            }
        )
    return pl.DataFrame(rows)


def gold_source_files(tasks: dict[str, Task]) -> dict[str, set[str]]:
    return {i: set(patch_hunks(t.patch)) for i, t in tasks.items()}
