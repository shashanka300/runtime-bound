#!/usr/bin/env python3
"""Install a repository's runtime + test dependencies from the offline /wheels cache.

Runs INSIDE the sandbox container (stdlib only). The competition's sandbox/setup.py reads
only PEP 621 ``[project]`` dependencies, which misses repos declaring them in setup.py or
setup.cfg (e.g. requests); the organizers' harness instead streams pre-built
site-packages that we do not have. This script reconstructs an equivalent environment:

1. collect requirement strings *with version specifiers* from pyproject (PEP 621 deps,
   optional deps, PEP 735 dependency groups, PDM dev deps, Poetry dependency tables),
   setup.cfg, setup.py (install_requires / extras_require / tests_require) and
   requirements*.txt;
2. add distributions whose top-level module is imported by the repo's code or tests
   (from /wheels; or from PyPI when also declared, e.g. python-multipart), plus declared
   ``pytest-*`` plugins, which are never imported but supply fixtures (pytest-httpbin);
3. keep only names available in /wheels, drop pytest/build backends (image-provided);
4. install them from /wheels only, WITHOUT dependency resolution, so the repo's own pins
   pick the versions (all at once; one at a time as fallback);
5. the public wheel cache is incomplete (e.g. pydantic 2.13 needs typing-inspection,
   which is absent), so ``pip check`` lists what is still missing and, when
   ``SWEBOX_ALLOW_NET=1`` (image bootstrap only), exactly those are fetched from PyPI.
   Agent and grading containers never have network access.
6. era pinning (wheel-cache packages only): when a specifier admits several cached
   versions, pick the newest one released on PyPI before the task's creation date
   (``SWEBOX_CUTOFF``) within the newest allowed major version, i.e. what the repo's CI ran
   when the task was made (a newer Pygments changes rich's highlighting output). Packages
   fetched from PyPI are NOT era-pinned: old releases of test plugins break on Python 3.13
   (e.g. pytest-httpbin's old bundled certificate fails 3.13's strict X.509 checks).

The harness self-check (gold patch must pass, empty patch must fail) validates the result.
"""

from __future__ import annotations

import ast
import configparser
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.request
from datetime import datetime
from functools import cache
from pathlib import Path

from pip._vendor.packaging.specifiers import InvalidSpecifier, SpecifierSet
from pip._vendor.packaging.version import InvalidVersion, Version

WHEELS = Path("/wheels")
WS = Path("/workspace")
SKIP = {
    "pytest",
    "setuptools",
    "wheel",
    "pip",
    "hatchling",
    "flit-core",
    "poetry-core",
    "pdm-backend",
    "editables",
}
# import name -> distribution name, where they differ
IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "attr": "attrs",
    "python_multipart": "python-multipart",
    "jinja2": "jinja2",
    "markdown_it": "markdown-it-py",
    "dateutil": "python-dateutil",
    "multipart": "python-multipart",
    "email_validator": "email-validator",
    "typing_extensions": "typing-extensions",
    "annotated_types": "annotated-types",
    "pydantic_core": "pydantic-core",
    "charset_normalizer": "charset-normalizer",
}
NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def available() -> set[str]:
    return {norm(p.name.split("-")[0]) for p in WHEELS.glob("*.whl")}


def clean(req: str) -> str | None:
    req = req.split("#")[0].strip()
    if not req or req.startswith(("-", "git+", "http")):
        return None
    req = re.sub(r"\[.*?\]", "", req)  # extras: their deps are collected separately
    req = req.split(";")[0].strip()  # env markers: python is fixed at 3.13
    return req or None


def from_pyproject() -> list[str]:
    path = WS / "pyproject.toml"
    if not path.exists():
        return []
    data = tomllib.loads(path.read_text(errors="replace"))
    out = list(data.get("project", {}).get("dependencies", []))
    for deps in data.get("project", {}).get("optional-dependencies", {}).values():
        out += deps
    for deps in data.get("dependency-groups", {}).values():
        out += [d for d in deps if isinstance(d, str)]
    for deps in data.get("tool", {}).get("pdm", {}).get("dev-dependencies", {}).values():
        out += deps
    return out


def poetry_spec(value: object) -> str:
    """Poetry constraint -> PEP 440 (caret/tilde ranges; tables use their ``version``)."""
    spec = value.get("version", "") if isinstance(value, dict) else str(value)
    spec = spec.strip()
    if spec in ("", "*"):
        return ""
    if spec[0] in "^~":
        base = spec[1:].strip()
        nums = [int(x) for x in re.findall(r"\d+", base)][:3] or [0]
        if spec[0] == "^":
            i = next((k for k, n in enumerate(nums) if n), len(nums) - 1)
        else:  # ~X.Y -> <X.(Y+1); ~X -> <X+1
            i = min(1, len(nums) - 1)
        upper = [*nums[:i], nums[i] + 1]
        return f">={base},<{'.'.join(map(str, upper))}"
    return spec


def from_poetry() -> list[str]:
    path = WS / "pyproject.toml"
    if not path.exists():
        return []
    poetry = tomllib.loads(path.read_text(errors="replace")).get("tool", {}).get("poetry", {})
    tables = [poetry.get("dependencies", {}), poetry.get("dev-dependencies", {})]
    tables += [g.get("dependencies", {}) for g in poetry.get("group", {}).values()]
    # optional = true marks an extra (e.g. rich's ipywidgets -> IPython, whose lexer
    # plugin changes rich's lexer guessing); extras are not part of the test environment.
    return [
        f"{name}{poetry_spec(v)}"
        for t in tables
        for name, v in t.items()
        if name != "python" and not (isinstance(v, dict) and v.get("optional"))
    ]


def from_setup_cfg() -> list[str]:
    path = WS / "setup.cfg"
    if not path.exists():
        return []
    cp = configparser.ConfigParser()
    cp.read_string(path.read_text(errors="replace"))
    out: list[str] = []
    if cp.has_option("options", "install_requires"):
        out += cp.get("options", "install_requires").splitlines()
    if cp.has_section("options.extras_require"):
        for _, v in cp.items("options.extras_require"):
            out += v.splitlines()
    return out


def from_setup_py() -> list[str]:
    path = WS / "setup.py"
    if not path.exists():
        return []
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in (
            "install_requires",
            "tests_require",
            "extras_require",
        ):
            out += [
                c.value
                for c in ast.walk(node.value)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            ]
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and "requires" in t.id.lower() for t in node.targets
        ):
            out += [
                c.value
                for c in ast.walk(node.value)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            ]
    return out


def from_requirements() -> list[str]:
    out: list[str] = []
    for f in list(WS.glob("**/requirements*.txt")) + list(WS.glob("**/*requirements*.txt")):
        if any(part.startswith(".") for part in f.relative_to(WS).parts) or "docs_src" in f.parts:
            continue
        out += f.read_text(errors="replace").splitlines()
    return out


def imported_dists() -> set[str]:
    mods: set[str] = set()
    for f in WS.rglob("*.py"):
        if any(p.startswith(".") for p in f.relative_to(WS).parts):
            continue
        try:
            tree = ast.parse(f.read_text(errors="replace"))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    return {norm(IMPORT_TO_DIST.get(m, m)) for m in mods}


def own_packages() -> set[str]:
    """The repo's own top-level packages: never replace the checkout with a wheel."""
    roots = [WS, WS / "src"]
    return {
        norm(d.name)
        for r in roots
        if r.is_dir()
        for d in r.iterdir()
        if (d / "__init__.py").exists()
    }


CONSTRAINTS = Path("/tmp/swebox-constraints.txt")


def freeze_constraints(own: set[str]) -> None:
    """Pin every installed distribution (and forbid the repo's own), so online fills can
    only ADD packages; they can never move a version the repo's pins selected."""
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    pins = [ln for ln in out if "==" in ln and norm(ln.split("==")[0]) not in own]
    # "<0" is unsatisfiable: pip may not pull a wheel of the project under test.
    CONSTRAINTS.write_text("\n".join(pins + [f"{o}<0" for o in sorted(own)]) + "\n")


def pip(*reqs: str, offline: bool = True, deps: bool = False, constrained: bool = True) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        f"--find-links={WHEELS}",
        "--no-build-isolation",
    ]
    if offline:
        cmd += ["--no-index"]
    else:
        cmd += ["--prefer-binary"] + (["-c", str(CONSTRAINTS)] if constrained else [])
    cmd += [] if deps else ["--no-deps"]
    return subprocess.run([*cmd, *reqs], capture_output=True, text=True).returncode == 0


# -- era pinning --------------------------------------------------------------------------
def cutoff() -> datetime | None:
    raw = os.environ.get("SWEBOX_CUTOFF")
    return datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None


@cache
def release_dates(name: str) -> dict[str, datetime]:
    """Version -> first upload time on PyPI ({} without network)."""
    if os.environ.get("SWEBOX_ALLOW_NET") != "1":
        return {}
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=20) as r:
            data = json.load(r)
    except Exception:
        return {}
    out = {}
    for version, files in data.get("releases", {}).items():
        times = [f["upload_time_iso_8601"] for f in files if f.get("upload_time_iso_8601")]
        if times:
            out[version] = datetime.fromisoformat(min(times).replace("Z", "+00:00"))
    return out


def wheel_versions() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in WHEELS.glob("*.whl"):
        name, version = p.name.split("-")[:2]
        out.setdefault(norm(name), []).append(version)
    return out


def era_pin(req: str, candidates: list[str] | None) -> tuple[str, bool]:
    """``name<spec>`` -> (``name==<version>``, from_pypi).

    Picks the newest allowed version released before the cutoff, within the newest allowed
    major. Candidates are the cached wheels; if none of them predates the task, the
    era-correct release is taken from PyPI instead (``from_pypi=True``).
    """
    when = cutoff()
    m = NAME.match(req)
    if when is None or m is None:
        return req, False
    name = m.group(1)
    try:
        spec = SpecifierSet(req[m.end() :].strip().replace(" ", ""))
    except InvalidSpecifier:
        return req, False
    dates = release_dates(norm(name))

    def allowed_in(versions: list[str]) -> list[Version]:
        out = []
        for v in set(versions):
            try:
                ver = Version(v)
            except InvalidVersion:
                continue
            if not ver.is_prerelease and spec.contains(ver, prereleases=False):
                out.append(ver)
        return out

    allowed = allowed_in(candidates if candidates is not None else list(dates))
    if not allowed:
        return req, False
    # Stay within the newest allowed major: the sparse wheel cache may hold only an old
    # major (pydantic 1.x) that predates the task but cannot run on Python 3.13.
    newest = max(allowed)
    same_major = [v for v in allowed if v.major == newest.major]
    before = [v for v in same_major if str(v) in dates and dates[str(v)] <= when]
    if before:
        return f"{name}=={max(before)}", False
    # No cached version is old enough: the era release from PyPI, same major.
    era = [v for v in allowed_in(list(dates)) if v.major == newest.major and dates[str(v)] <= when]
    if candidates is not None and era:
        return f"{name}=={max(era)}", True
    return f"{name}=={newest}", False


MISSING = re.compile(r"requires (\S+?), which is not installed")


CONFLICT = re.compile(r"has requirement (\S+?), but you have")


def _check() -> str:
    return subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    ).stdout


def missing_deps() -> list[str]:
    return sorted({m.group(1) for m in MISSING.finditer(_check())})


def repair_conflicts(allow_net: bool, rounds: int = 3) -> list[str]:
    """Install exactly what installed packages require where versions disagree.

    Pinning packages one at a time can split coupled pairs (pydantic pins pydantic-core
    exactly); the dependent's requirement wins. Binary wheels only."""
    fixed: list[str] = []
    for _ in range(rounds):
        wanted = sorted({m.group(1) for m in CONFLICT.finditer(_check())})
        if not wanted:
            break
        for req in wanted:
            if pip(req) or (
                allow_net and pip(req, "--only-binary=:all:", offline=False, constrained=False)
            ):
                fixed.append(req)
    return fixed


def main() -> None:
    avail = available()
    wheels = wheel_versions()
    own = own_packages()
    imported = imported_dists() - SKIP - own
    declared: dict[str, str] = {}
    sources = from_pyproject() + from_poetry() + from_setup_cfg() + from_setup_py()
    for raw in sources + from_requirements():
        req = clean(raw)
        if req and (m := NAME.match(req)):
            name = norm(m.group(1))
            if name not in SKIP | own:
                declared.setdefault(name, req)  # first declaration (runtime deps) wins
    pinned = [era_pin(declared[n], wheels[n]) for n in sorted(declared) if n in avail]
    pinned += [era_pin(n, wheels[n]) for n in sorted(imported) if n in avail and n not in declared]
    offline = [r for r, from_pypi in pinned if not from_pypi]
    era_online = [r for r, from_pypi in pinned if from_pypi]
    ok = pip(*offline) if offline else True
    failed = [] if ok else [r for r in offline if not pip(r)]
    era_fetched: list[str] = []
    for r in era_online:
        # binary wheels only (no source builds); on failure use the cached version
        if pip(r, "--only-binary=:all:", offline=False, constrained=False):
            era_fetched.append(r)
        else:
            name = NAME.match(r).group(1)  # type: ignore[union-attr]
            pip(f"{name}=={max(wheels[norm(name)], key=Version)}")
    # Declared AND (imported OR a pytest plugin) but absent from the public cache.
    wanted_online = [
        declared[n]
        for n in sorted(declared)
        if n not in avail and (n in imported or n.startswith("pytest-"))
    ]
    repaired = repair_conflicts(os.environ.get("SWEBOX_ALLOW_NET") == "1")
    missing = missing_deps()
    fetched: list[str] = []
    if os.environ.get("SWEBOX_ALLOW_NET") == "1":
        freeze_constraints(own)
        # deps=True: a fetched package may itself need further missing packages. A package
        # whose requirements conflict with the constraints fails and is simply skipped.
        fetched = [r for r in wanted_online + missing if pip(r, offline=False, deps=True)]
    print(
        json.dumps(
            {
                "offline": offline,
                "era_from_pypi": era_fetched,
                "repaired": repaired,
                "failed": failed,
                "wanted_online": wanted_online,
                "missing_after_wheels": missing,
                "fetched_from_pypi": fetched,
                "still_missing": missing_deps(),
            }
        )
    )


if __name__ == "__main__":
    main()
