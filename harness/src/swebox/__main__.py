"""swebox CLI (thin; experiments call the library through notebooks).

uv run python -m swebox selfcheck --subset fidelity10 --which gold
uv run python -m swebox eval --subset fidelity10 --submission <dir> --out <dir>
uv run python -m swebox prune-images
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from rbcore import logging as rblog
from rbcore.paths import competition_dir
from swebox import sandbox as sb
from swebox.evaluate import EvalSpec, evaluate, summarize, verify_only
from swebox.submission import ServingTarget
from swebox.tasks import load_tasks, select
from swebox.tools import Budget


def bundle_budget(submission: Path, time_scale: float = 1.0) -> Budget:
    """Budget from a submission's eval_config.yaml, read the way the scorer does (only
    timeout_seconds, max_tool_calls, max_time_minutes and max_turns; absent = no limit)."""
    cfg_path = submission / "eval_config.yaml"
    raw = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    ev = (raw or {}).get("evaluation", raw or {})
    return Budget(
        time_minutes=float(ev.get("max_time_minutes", 60.0)) * time_scale,
        tool_calls=ev.get("max_tool_calls"),
        turns=int(ev.get("max_turns") or 500),
        command_timeout_seconds=int(ev.get("timeout_seconds") or 300),
    )


def _subset(value: str) -> str | list[str]:
    return value.split(",") if "," in value or "_" in value else value


def main() -> None:
    ap = argparse.ArgumentParser(prog="swebox")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selfcheck", "eval"):
        p = sub.add_parser(name)
        p.add_argument(
            "--subset",
            default="fidelity10",
            help="all | dev40 | fidelity10 | comma-separated instance ids",
        )
        p.add_argument("--mem-gib", type=float, default=8.0)
        p.add_argument("--cpus", type=float, default=4.0)
        p.add_argument("--concurrency", type=int, default=2)
        p.add_argument("--out", type=Path, default=Path("swebox_out"))
    sub.choices["selfcheck"].add_argument("--which", choices=["gold", "empty"], default="gold")
    ev = sub.choices["eval"]
    ev.add_argument("--submission", type=Path, default=competition_dir() / "sample_submission")
    ev.add_argument("--time-minutes", type=float, default=10.0)
    ev.add_argument("--tool-calls", type=int, default=None)
    ev.add_argument("--seeds", default="0")
    ev.add_argument("--api-base", default=ServingTarget.api_base)
    ev.add_argument("--thinking-translation", choices=["budget", "official"], default="budget")
    ev.add_argument(
        "--bundle-budget",
        action="store_true",
        help="take budgets from the submission's eval_config.yaml (as the scorer does)",
    )
    ev.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="multiply the bundle's max_time_minutes (e.g. 1/1.6 for the slower 4xL4 scorer)",
    )
    sub.add_parser("prune-images")
    args = ap.parse_args()
    rblog.configure()

    if args.cmd == "prune-images":
        print(f"removed {sb.prune_task_images()} task images")
        return
    tasks = select(load_tasks(), _subset(args.subset))
    limits = sb.Limits(args.mem_gib, args.cpus)
    if args.cmd == "selfcheck":
        rows = asyncio.run(verify_only(tasks, args.which, limits, args.out, args.concurrency))
        for r in rows:
            print(
                f"{r['instance_id']:<22} resolved={r['resolved']!s:<5} exit={r['exit_code']} "
                f"{r['seconds']:>6}s {r['error'] or ''}"
            )
        print(f"{sum(r['resolved'] for r in rows)}/{len(rows)} resolved with {args.which} patch")
        return
    spec = EvalSpec(
        submission=args.submission,
        limits=limits,
        target=ServingTarget(
            api_base=args.api_base, thinking_translation=args.thinking_translation
        ),
        budget=bundle_budget(args.submission, args.time_scale)
        if args.bundle_budget
        else Budget(time_minutes=args.time_minutes, tool_calls=args.tool_calls),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        concurrency=args.concurrency,
    )
    rows = asyncio.run(evaluate(tasks, spec, args.out))
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
