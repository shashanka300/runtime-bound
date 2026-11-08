"""Token accounting over swebox step traces (H2: where do the tokens go?).

Two complementary views, both computed per model call and aggregated per task:

Compute view: what the GPU processed on each call
    prompt = cached (prefix-cache hit) + new (tokens appended since the previous call in
    the same context) + recomputed (prefix the cache no longer held and prefill had to
    redo: KV eviction under memory pressure, or after compaction)
    decode = thinking + visible output / tool-call arguments

Occupancy view: what fills each prompt, by segment kind. Request characters are converted
to tokens with a per-agent linear fit ``prompt_tokens ~ a * chars + b`` (``b`` absorbs the
tool schemas and chat template). Tool outputs are tracked as segments that stay in the
context for every later call of the same context (no compaction fires in practice), flagged:
    reread     read of file lines already returned earlier in the task (read_file ranges
               and cat/sed/head/tail commands)
    truncated  output cut at a harness limit (read_file is_truncated; run_command > 5000
               chars, the stdout limit)
    oversized  a single output above ``oversized_chars`` (e.g. search_similar_code
               returning full source of 10 functions)
Stale thinking is past thinking text re-sent in the history.

Pre-registered H2 metric (docs/research_plan.md): waste = re-read + truncated segment tokens
(occupancy, summed over calls) + recomputed prefill tokens (compute view), as a share of all
processed tokens (sum of prompt + completion). The three can overlap (a recomputed token may
belong to a re-read segment), so the sum is an upper bound on their union; the largest single
category is the lower bound. Both bounds are reported.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass

import numpy as np
import polars as pl

KEY = ["tag", "instance_id", "seed"]
CTX = [*KEY, "agent", "invocation"]
CHAR_KINDS = ["system_chars", "user_text", "model_text", "thought", "tool_call", "tool_output"]
RUN_COMMAND_LIMIT = 5000


OPTIONAL = {
    "finish_reason": pl.String,
    "error": pl.String,
    "is_truncated": pl.Boolean,
    "filepath": pl.String,
    "start_line": pl.Int64,
    "end_line": pl.Int64,
    "total_lines": pl.Int64,
    "args": pl.String,
    "invocation": pl.String,
}


def ensure_columns(steps: pl.DataFrame) -> pl.DataFrame:
    """Add optional trace columns as nulls when a trace set never produced them."""
    return steps.with_columns(
        pl.lit(None, dtype=t).alias(c) for c, t in OPTIONAL.items() if c not in steps.columns
    )


# -- model calls --------------------------------------------------------------------------
def model_calls(steps: pl.DataFrame) -> pl.DataFrame:
    """One row per model call: request breakdown joined with its response usage.

    Requests rejected for exceeding the context window (``model_error``) are kept with
    ``overflow=True`` and no usage: vLLM rejects them before any computation.
    """
    s = ensure_columns(steps).sort([*KEY, "step"])
    req = s.filter(pl.col("kind") == "model_request")
    # the response/error that follows a request in the same context is its outcome
    out = s.filter(pl.col("kind").is_in(["model_response", "model_error"])).select(
        *KEY,
        "agent",
        pl.col("step").alias("out_step"),
        pl.col("kind").alias("out_kind"),
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "thought_tokens",
        "finish_reason",
        pl.col("error").alias("out_error"),
    )
    calls = req.select(*CTX, "step", "t_ns", "n_contents", *CHAR_KINDS).join_asof(
        out.sort("out_step"),
        left_on="step",
        right_on="out_step",
        by=[*KEY, "agent"],
        strategy="forward",
        check_sortedness=False,  # sorted by (KEY, step) above
    )
    calls = calls.with_columns(
        (pl.col("out_kind") == "model_error").alias("overflow"),
        pl.sum_horizontal(CHAR_KINDS).alias("chars"),
        pl.int_range(pl.len()).over(CTX).alias("call_idx"),
    )
    return calls.with_columns(
        pl.when(pl.col("overflow")).then(None).otherwise(pl.col(c)).alias(c)
        for c in ("prompt_tokens", "cached_tokens", "completion_tokens", "thought_tokens")
    )


@dataclass(frozen=True)
class CharFit:
    agent: str
    tokens_per_char: float
    intercept: float
    r2: float
    n: int


def fit_chars_to_tokens(calls: pl.DataFrame) -> dict[str, CharFit]:
    """Per-agent OLS ``prompt_tokens = a * chars + b`` on calls with known usage."""
    fits = {}
    ok = calls.filter(pl.col("prompt_tokens").is_not_null() & (pl.col("prompt_tokens") > 0))
    for (agent,), g in ok.group_by("agent"):
        x = g["chars"].to_numpy().astype(float)
        y = g["prompt_tokens"].to_numpy().astype(float)
        design = np.column_stack([x, np.ones_like(x)])
        (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ np.array([a, b])
        r2 = 1 - float(((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))
        fits[str(agent)] = CharFit(str(agent), float(a), float(b), r2, len(g))
    return fits


def compute_view(calls: pl.DataFrame) -> pl.DataFrame:
    """Split each call's prompt into cached / new / recomputed tokens."""
    c = calls.filter(~pl.col("overflow")).sort([*CTX, "call_idx"])
    prev = pl.col("prompt_tokens").shift(1).over(CTX)
    c = c.with_columns(
        (pl.col("prompt_tokens") - pl.col("cached_tokens")).alias("uncached"),
        pl.when(prev.is_null())
        .then(pl.col("prompt_tokens") - pl.col("cached_tokens"))
        .otherwise((pl.col("prompt_tokens") - prev).clip(lower_bound=0))
        .alias("new_raw"),
    )
    return (
        c.with_columns(
            pl.min_horizontal("new_raw", "uncached").alias("new"),
        )
        .with_columns(
            (pl.col("uncached") - pl.col("new")).alias("recomputed"),
            (pl.col("completion_tokens") - pl.col("thought_tokens"))
            .clip(lower_bound=0)
            .alias("output_tokens"),
        )
        .drop("new_raw")
    )


# -- tool-output segments ---------------------------------------------------------------------
_SED = re.compile(r"^(\d+)(?:,(\d+))?p$")


def read_ranges(
    tool: str, args_json: str, start: int | None, end: int | None, filepath: str | None
) -> list[tuple[str, int, int]]:
    """File line ranges a tool call returned: read_file, or cat/sed -n/head/tail commands.
    A whole-file read is (path, 1, 10**9)."""
    whole = 10**9
    if tool == "read_file" and filepath:
        return [(filepath.removeprefix("/workspace/"), int(start or 1), int(end or whole))]
    if tool != "run_command":
        return []
    try:
        cmd = json.loads(args_json).get("command", "")
        parts = shlex.split(cmd)
    except (ValueError, json.JSONDecodeError, AttributeError):
        return []
    if not parts:
        return []
    files = [p for p in parts[1:] if not p.startswith("-") and "." in p and "|" not in p]
    prog = parts[0]
    if prog == "cat" and files and "|" not in cmd:
        return [(f.removeprefix("/workspace/"), 1, whole) for f in files]
    if prog == "sed" and "-n" in parts and len(parts) >= 4:
        m = _SED.match(parts[parts.index("-n") + 1].strip("'\""))
        if m and files:
            a = int(m.group(1))
            b = int(m.group(2) or a)
            return [(files[-1].removeprefix("/workspace/"), a, b)]
    if prog in ("head", "tail") and files:
        return [(files[-1].removeprefix("/workspace/"), 1, whole)]
    return []


def _covered(ranges: list[tuple[int, int]], a: int, b: int) -> int:
    """Lines of [a, b] already covered by earlier ranges."""
    covered = 0
    for lo, hi in _merge(ranges):
        covered += max(0, min(hi, b) - max(lo, a) + 1)
    return covered


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def segments(
    steps: pl.DataFrame,
    calls: pl.DataFrame,
    fits: dict[str, CharFit],
    reread_overlap: float,
    oversized_chars: int,
) -> pl.DataFrame:
    """One row per tool output with flags, token size and lifetime in its context."""
    s = ensure_columns(steps).sort([*KEY, "step"])
    tools = s.filter(pl.col("kind") == "tool")
    # assign each tool output to the context of the model call that requested it
    owners = calls.select(*KEY, "agent", "invocation", pl.col("step").alias("call_step"))
    tools = tools.drop("invocation").join_asof(
        owners.sort("call_step"),
        left_on="step",
        right_on="call_step",
        by=[*KEY, "agent"],
        strategy="backward",
        check_sortedness=False,
    )
    rows = []
    for (tag, iid, seed), g in tools.group_by(KEY, maintain_order=True):
        seen: dict[str, list[tuple[int, int]]] = {}
        for r in g.iter_rows(named=True):
            ranges = read_ranges(
                r["tool"],
                r["args"] or "{}",
                r.get("start_line"),
                r.get("end_line"),
                r.get("filepath"),
            )
            reread = False
            for path, a, b in ranges:
                span_end = b if b < 10**9 else (r.get("total_lines") or a + 150)
                span = max(1, span_end - a + 1)
                if _covered(seen.get(path, []), a, span_end) / span >= reread_overlap:
                    reread = True
                seen.setdefault(path, []).append((a, span_end))
            chars = r["output_chars"] or 0
            truncated = bool(r.get("is_truncated")) or (
                r["tool"] == "run_command" and chars > RUN_COMMAND_LIMIT
            )
            fit = fits.get(r["agent"])
            rows.append(
                {
                    "tag": tag,
                    "instance_id": iid,
                    "seed": seed,
                    "agent": r["agent"],
                    "invocation": r["invocation"],
                    "step": r["step"],
                    "tool": r["tool"],
                    "chars": chars,
                    "tokens": chars * (fit.tokens_per_char if fit else 0.25),
                    "reread": reread,
                    "truncated": truncated,
                    "oversized": chars > oversized_chars,
                }
            )
    seg = pl.DataFrame(rows)
    if seg.is_empty():
        return seg
    # lifetime: later calls in the same context re-send the segment
    later = calls.filter(~pl.col("overflow")).select(*CTX, pl.col("step").alias("cstep"))
    life = (
        seg.join(later, on=CTX, how="left")
        .filter(pl.col("cstep") > pl.col("step"))
        .group_by([*CTX, "step"])
        .len()
        .rename({"len": "lifetime"})
    )
    seg = seg.join(life, on=[*CTX, "step"], how="left").with_columns(
        pl.col("lifetime").fill_null(0)
    )
    return seg.with_columns((pl.col("tokens") * pl.col("lifetime")).alias("token_steps"))


# -- per-task accounting ----------------------------------------------------------------------
def per_task(
    cv: pl.DataFrame, seg: pl.DataFrame, calls: pl.DataFrame, fits: dict[str, CharFit]
) -> pl.DataFrame:
    comp = cv.group_by(KEY).agg(
        pl.col("prompt_tokens").sum().alias("prompt"),
        pl.col("cached_tokens").sum().alias("cached"),
        pl.col("new").sum(),
        pl.col("recomputed").sum(),
        pl.col("thought_tokens").sum().alias("decode_thinking"),
        pl.col("output_tokens").sum().alias("decode_output"),
        pl.len().alias("calls"),
    )
    flags = seg.group_by(KEY).agg(
        pl.col("token_steps").filter(pl.col("reread")).sum().alias("reread"),
        pl.col("token_steps").filter(pl.col("truncated")).sum().alias("truncated"),
        pl.col("token_steps").filter(pl.col("oversized")).sum().alias("oversized"),
        pl.col("token_steps").sum().alias("tool_output_total"),
    )
    # stale thinking: thought characters re-sent in requests, converted per agent
    rate = pl.col("agent").replace_strict(
        {a: f.tokens_per_char for a, f in fits.items()}, default=0.25
    )
    stale = (
        calls.filter(~pl.col("overflow"))
        .group_by(KEY)
        .agg((pl.col("thought") * rate).sum().alias("stale_thinking"))
    )
    overflow = calls.group_by(KEY).agg(pl.col("overflow").sum().alias("overflow_calls"))
    t = (
        comp.join(flags, on=KEY, how="left")
        .join(stale, on=KEY, how="left")
        .join(overflow, on=KEY, how="left")
        .fill_null(0)
    )
    total = pl.col("prompt") + pl.col("decode_thinking") + pl.col("decode_output")
    upper = (pl.col("reread") + pl.col("truncated") + pl.col("recomputed")) / total
    lower = pl.max_horizontal("reread", "truncated", "recomputed") / total
    return t.with_columns(
        total.alias("total_tokens"),
        upper.alias("waste_share_upper"),
        lower.alias("waste_share_lower"),
        (pl.col("reread") / total).alias("reread_share"),
        (pl.col("truncated") / total).alias("truncated_share"),
        (pl.col("recomputed") / total).alias("recomputed_share"),
        (pl.col("cached") / pl.col("prompt")).alias("cache_hit_rate"),
    )
