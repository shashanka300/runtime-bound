"""Phase 2: grade a patch in a fresh Container B (spec section 8.2)."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

import docker

from swebox.sandbox import Limits, Sandbox, ensure_task_image
from swebox.tasks import Task

PATCH_PATH = "/tmp/agent.patch"
TEST_PATCH_PATH = "/tmp/test.patch"
PYTEST_TIMEOUT_S = 1800

# Spec 8.2 step 2: four passes, each trying its variants in order until one applies.
APPLY_PASSES: list[list[str]] = [
    [
        f"git apply --unsafe-paths -p1 {opt} {PATCH_PATH}".replace("  ", " ")
        for opt in ("", "-3", "--ignore-space-change --ignore-whitespace", "--recount")
    ],
    # Pass 2 in the spec normalises symlinked directory prefixes; snapshots here contain no
    # directory symlinks (removed at extraction), so pass 1 already covers it.
    [f"git apply --unsafe-paths -p0 {PATCH_PATH}"],
    [
        f"patch --batch --forward --dry-run -p{p} {flag} -i {PATCH_PATH} && "
        f"patch --batch --forward -p{p} {flag} -i {PATCH_PATH}"
        for p in (1, 0)
        for flag in ("", "-l")
    ],
]


@dataclass
class VerifyResult:
    resolved: bool
    applied: bool
    apply_method: str | None
    exit_code: int | None
    seconds: float
    log: str
    error: str | None = None


def _terminated(patch: str) -> str:
    """Every patch in tasks.jsonl lacks its final newline, which makes git reject the last
    hunk line ("corrupt patch"); diff text is newline-terminated by definition."""
    return patch if patch.endswith("\n") else patch + "\n"


def verify(dc: docker.DockerClient, task: Task, patch: str, limits: Limits) -> VerifyResult:
    image = ensure_task_image(dc, task, limits)
    box = Sandbox.start(dc, image, limits, "verify", task)
    try:
        return _verify_in(box, task, patch)
    finally:
        box.remove()


def _verify_in(box: Sandbox, task: Task, patch: str) -> VerifyResult:
    method = None
    if patch.strip():
        box.put_bytes(PATCH_PATH, _terminated(patch).encode())
        for variants in APPLY_PASSES:
            for cmd in variants:
                if box.exec(cmd, 120).exit_code == 0:
                    method = cmd
                    break
            if method:
                break
        if method is None:
            return VerifyResult(
                False,
                False,
                None,
                None,
                0.0,
                "",
                error="Failed to apply agent_patch: all 4 passes failed",
            )

    # Anti-tampering: restore target test files, then apply the hidden test patch.
    files = " ".join(shlex.quote(f) for f in task.test_files)
    box.exec(f"git checkout HEAD -- {files} 2>/dev/null || true")
    box.exec(f"git clean -f -- {files} 2>/dev/null || true")
    box.put_bytes(TEST_PATCH_PATH, _terminated(task.test_patch).encode())
    tp = box.exec(
        f"git apply --unsafe-paths -p1 {TEST_PATCH_PATH} || "
        f"patch --batch --forward -p1 -i {TEST_PATCH_PATH}",
        120,
    )
    if tp.exit_code != 0:
        return VerifyResult(
            False,
            method is not None,
            method,
            None,
            0.0,
            tp.stderr,
            error="test_patch failed to apply",
        )
    box.exec(f"python /sandbox/setup.py --fast-path {shlex.quote(task.repo)} >/dev/null 2>&1")

    cmd = (
        "PYTHONSAFEPATH=1 python3 -m pytest "
        + files
        + ' -p no:anyio -o timeout=0 -o norecursedirs=".* build dist venv"'
        ' -o python_classes="Test* *Test" -q'
    )
    r = box.exec(cmd, PYTEST_TIMEOUT_S)
    log = f"$ {cmd}\n[exit {r.exit_code}, {r.seconds:.1f}s]\n{r.stdout}\n{r.stderr}"
    return VerifyResult(
        r.exit_code == 0,
        method is not None or not patch.strip(),
        method,
        r.exit_code,
        round(r.seconds, 2),
        log,
        error="pytest timed out" if r.timed_out else None,
    )
