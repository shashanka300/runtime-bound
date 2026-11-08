"""Three-tier resilient string replacement, per the harness spec (``apply_replacement``).

1. ``exact``    exact substring match (after CRLF -> LF normalisation);
2. ``flexible`` line-by-line match ignoring leading/trailing whitespace, re-indenting
                ``new`` to the matched block's baseline indentation;
3. ``regex``    tokens of ``old`` split around code delimiters ``( ) : [ ] { } > = <``,
                joined with ``\\s*`` so line-wrapping/spacing differences still match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DELIMS = re.compile(r"([()\[\]{}:>=<])")


class EditError(ValueError):
    pass


@dataclass(frozen=True)
class Replacement:
    content: str
    occurrences: int
    strategy: str


def apply_replacement(
    content: str, old: str, new: str, allow_multiple: bool = False
) -> Replacement:
    if not old:
        raise EditError("old_string must not be empty")
    content = content.replace("\r\n", "\n")
    old = old.replace("\r\n", "\n")
    new = new.replace("\r\n", "\n")
    for strategy in (_exact, _flexible, _regex):
        result = strategy(content, old, new, allow_multiple)
        if result is not None:
            return result
    raise EditError("old_string not found in file (tried exact, flexible and regex matching)")


def _check_count(n: int, allow_multiple: bool) -> None:
    if n > 1 and not allow_multiple:
        raise EditError(
            f"old_string matches {n} locations; add surrounding context or set allow_multiple"
        )


def _exact(content: str, old: str, new: str, allow_multiple: bool) -> Replacement | None:
    n = content.count(old)
    if n == 0:
        return None
    _check_count(n, allow_multiple)
    return Replacement(content.replace(old, new), n, "exact")


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _flexible(content: str, old: str, new: str, allow_multiple: bool) -> Replacement | None:
    lines = content.split("\n")
    old_lines = [ln.strip() for ln in old.strip("\n").split("\n")]
    k = len(old_lines)
    starts = [
        i for i in range(len(lines) - k + 1) if [ln.strip() for ln in lines[i : i + k]] == old_lines
    ]
    if not starts:
        return None
    _check_count(len(starts), allow_multiple)
    new_lines = new.strip("\n").split("\n")
    new_base = min((_indent(ln) for ln in new_lines if ln.strip()), key=len, default="")
    for i in reversed(starts):  # reverse keeps earlier indices valid
        base = _indent(lines[i])
        rebased = [
            base + ln[len(new_base) :] if ln.startswith(new_base) else ln for ln in new_lines
        ]
        lines[i : i + k] = rebased
    return Replacement("\n".join(lines), len(starts), "flexible")


def _regex(content: str, old: str, new: str, allow_multiple: bool) -> Replacement | None:
    tokens = [t for t in _DELIMS.split(old) for t in t.split() if t]
    if not tokens:
        return None
    pattern = re.compile(r"\s*".join(re.escape(t) for t in tokens))
    matches = list(pattern.finditer(content))
    if not matches:
        return None
    _check_count(len(matches), allow_multiple)
    out = pattern.sub(lambda _: new, content, count=0 if allow_multiple else 1)
    return Replacement(out, len(matches), "regex")
