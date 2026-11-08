"""Docker sandboxes following the harness spec (section 4.1-4.2).

Image hierarchy:
- ``swebench-sandbox:latest``: built from the competition ``docker/Dockerfile.public``
  (python:3.13-slim, git, pytest, build backends, offline ``/wheels``).
- ``swebox-task:<instance_id>``: the base image plus the bootstrapped workspace
  (snapshot extracted, git excludes, editable install, test deps, pytest.ini/conftest.py,
  ``baseline`` commit). Bootstrapping is deterministic, so it runs once per task and
  is committed as an image. Container A (agent) and Container B (verification) then start
  from an identical fresh copy in about a second, instead of re-running pip each time. The
  official harness re-bootstraps per container; the resulting workspace is the same.

Every container runs with ``network_mode=none`` and the configured memory/CPU limits,
and is labelled so ``rbcore.hwmon.docker_stats`` can attribute its resource use.
"""

from __future__ import annotations

import contextlib
import io
import os
import shlex
import shutil
import socket
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker
import structlog
from docker.errors import ImageNotFound, NotFound

from rbcore.paths import competition_dir
from swebox.tasks import Task

log = structlog.get_logger(__name__)

BASE_IMAGE = "swebench-sandbox:latest"
TASK_IMAGE_REPO = "swebox-task"
LABEL = "swebox.sandbox"
OWNER_LABEL = "swebox.owner"
OWNER = f"{socket.gethostname()}:{os.getpid()}"
# Bumped whenever bootstrap() changes, so stale cached task images are never reused.
BOOTSTRAP_VERSION = "9"
INSTALL_DEPS = Path(__file__).parent / "assets" / "install_deps.py"

GIT_EXCLUDES = "__pycache__/\n*.pyc\n.pytest_cache/\n*.egg-info/\nbuild/\ndist/\n.coverage\n"


@dataclass(frozen=True)
class Limits:
    mem_gib: float
    cpus: float


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool


def client() -> docker.DockerClient:
    return docker.from_env(timeout=600)


# -- images -------------------------------------------------------------------------------
def ensure_base_image(dc: docker.DockerClient) -> None:
    try:
        dc.images.get(BASE_IMAGE)
        return
    except ImageNotFound:
        pass
    src = competition_dir()
    with tempfile.TemporaryDirectory() as tmp:
        ctx = Path(tmp)
        for name in ("Dockerfile.public", "imp.py", "telnetlib.py"):
            shutil.copy(src / "docker" / name, ctx / name)
        shutil.copytree(src / "wheels", ctx / "wheels")
        log.info("sandbox.build_base_image", tag=BASE_IMAGE)
        dc.images.build(path=str(ctx), dockerfile="Dockerfile.public", tag=BASE_IMAGE, rm=True)


def task_image_tag(task: Task) -> str:
    return f"{TASK_IMAGE_REPO}:{task.instance_id}-b{BOOTSTRAP_VERSION}"


def ensure_task_image(dc: docker.DockerClient, task: Task, limits: Limits) -> str:
    """Bootstrap the task workspace once and cache it as an image; returns the tag."""
    tag = task_image_tag(task)
    try:
        dc.images.get(tag)
        return tag
    except ImageNotFound:
        pass
    ensure_base_image(dc)
    t0 = time.monotonic()
    # Network only while bootstrapping, to fill gaps in the public wheel cache
    # (install_deps.py step 5). Agent/verify containers are always network_mode=none.
    box = Sandbox.start(dc, BASE_IMAGE, limits, role="bootstrap", task=task, network="bridge")
    try:
        bootstrap(box, task)
        repo, _, version = tag.partition(":")
        box.container.commit(
            repository=repo, tag=version, changes=["WORKDIR /workspace", "ENV TEST_TMPDIR=/tmp"]
        )
    finally:
        box.remove()
    log.info(
        "sandbox.task_image_built", task=task.instance_id, seconds=round(time.monotonic() - t0, 1)
    )
    return tag


def bootstrap(box: Sandbox, task: Task) -> None:
    """Spec section 4.2 steps 2-7 (step 1, /wheels, is baked into the base image)."""
    box.put_archive_file("/workspace", task.snapshot())  # 2. snapshot
    # put_archive keeps the tarball's uid/gid; the official harness extracts as root, and
    # git refuses a repository with "dubious ownership" otherwise.
    box.check("chown -R root:root /workspace")
    box.check("find /workspace -xtype l -delete")  # circular symlinks
    box.check(f"mkdir -p .git/info && printf {shlex.quote(GIT_EXCLUDES)} >> .git/info/exclude")
    box.put_bytes("/sandbox/setup.py", (competition_dir() / "sandbox" / "setup.py").read_bytes())
    box.put_bytes("/sandbox/install_deps.py", INSTALL_DEPS.read_bytes())
    # 5a. runtime + test deps with version pins, from the offline wheels (see install_deps.py)
    deps = box.check(
        f"SWEBOX_ALLOW_NET=1 SWEBOX_CUTOFF={shlex.quote(task.created_at)} "
        "python /sandbox/install_deps.py",
        timeout=1800,
    )
    log.info("sandbox.deps", task=task.instance_id, report=deps.stdout.strip()[-2000:])
    box.check(
        "pip install --no-index --find-links=/wheels --no-build-isolation --no-deps "
        "-e /workspace >/dev/null 2>&1 || true"
    )  # 4. editable
    # 5b-6. the competition script: path injection, pytest.ini + conftest.py hooks
    box.check(f"python /sandbox/setup.py --fast-path {shlex.quote(task.repo)}", timeout=600)
    box.check("git add -A && git commit -m baseline --allow-empty -q")  # 7. baseline


def remove_orphans(dc: docker.DockerClient) -> int:
    """Remove sandbox containers whose owning process no longer exists (a crashed or killed
    run leaves them idling forever). Containers of live processes are left alone."""
    removed = 0
    host = socket.gethostname()
    for c in dc.containers.list(all=True, filters={"label": LABEL}):
        owner = c.labels.get(OWNER_LABEL, "")
        owner_host, _, pid = owner.partition(":")
        alive = owner_host == host and pid.isdigit() and Path(f"/proc/{pid}").exists()
        if not alive:
            with contextlib.suppress(NotFound):
                c.remove(force=True)
                removed += 1
    if removed:
        log.warning("sandbox.orphans_removed", count=removed)
    return removed


def prune_task_images(dc: docker.DockerClient | None = None) -> int:
    dc = dc or client()
    removed = 0
    for img in dc.images.list(name=TASK_IMAGE_REPO):
        dc.images.remove(img.id, force=True)
        removed += 1
    return removed


# -- containers -----------------------------------------------------------------------------
class Sandbox:
    def __init__(self, container: Any) -> None:
        self.container = container

    @classmethod
    def start(
        cls,
        dc: docker.DockerClient,
        image: str,
        limits: Limits,
        role: str,
        task: Task,
        labels: dict[str, str] | None = None,
        network: str = "none",
    ) -> Sandbox:
        container = dc.containers.run(
            image,
            command=["sleep", "infinity"],
            detach=True,
            network_mode=network,
            mem_limit=f"{int(limits.mem_gib * 1024)}m",
            memswap_limit=f"{int(limits.mem_gib * 1024)}m",  # no swap: OOM -> exit 137 as spec
            nano_cpus=int(limits.cpus * 1e9),
            working_dir="/workspace",
            environment={"TEST_TMPDIR": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"},
            labels={
                LABEL: "1",
                OWNER_LABEL: OWNER,
                "swebox.role": role,
                "swebox.task": task.instance_id,
                **(labels or {}),
            },
        )
        return cls(container)

    def remove(self) -> None:
        with contextlib.suppress(NotFound):
            self.container.remove(force=True)

    # -- exec ----------------------------------------------------------------------------
    def exec(self, command: str, timeout: float = 300, workdir: str = "/workspace") -> ExecResult:
        """Run ``bash -c command``; the whole process tree is SIGKILLed after ``timeout``."""
        t = max(1, int(timeout))
        wrapped = f"timeout -s KILL {t} bash -c {shlex.quote(command)}"
        t0 = time.monotonic()
        res = self.container.exec_run(["bash", "-c", wrapped], workdir=workdir, demux=True)
        seconds = time.monotonic() - t0
        out, err = res.output if res.output else (b"", b"")
        timed_out = res.exit_code == 137 and seconds >= t - 0.5
        return ExecResult(
            res.exit_code,
            (out or b"").decode(errors="replace"),
            (err or b"").decode(errors="replace"),
            seconds,
            timed_out,
        )

    def check(self, command: str, timeout: float = 600) -> ExecResult:
        r = self.exec(command, timeout)
        if r.exit_code != 0:
            raise RuntimeError(
                f"sandbox command failed ({r.exit_code}): {command}\n{r.stderr[-2000:]}"
            )
        return r

    # -- files ---------------------------------------------------------------------------
    def put_archive_file(self, dest: str, tgz: Path) -> None:
        """Extract a (gzip) tar archive into ``dest``; the daemon decompresses it."""
        with tgz.open("rb") as fh:
            if not self.container.put_archive(dest, fh):
                raise RuntimeError(f"put_archive failed for {tgz}")

    def put_bytes(self, path: str, data: bytes, mode: int = 0o644) -> None:
        p = Path(path)
        self.exec(f"mkdir -p {shlex.quote(str(p.parent))}", workdir="/")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(p.name)
            info.size = len(data)
            info.mode = mode
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        if not self.container.put_archive(str(p.parent), buf.getvalue()):
            raise RuntimeError(f"put_archive failed for {path}")

    def read_bytes(self, path: str) -> bytes:
        stream, _ = self.container.get_archive(path)
        buf = io.BytesIO(b"".join(stream))
        with tarfile.open(fileobj=buf) as tar:
            member = tar.next()
            if member is None or not member.isfile():
                raise IsADirectoryError(path)
            f = tar.extractfile(member)
            assert f is not None
            return f.read()

    def git_diff(self) -> str:
        """Spec 8.1: intent-to-add new files so the diff includes them, excludes honoured."""
        self.exec("git add -N .")
        return self.exec("git diff HEAD", timeout=120).stdout
