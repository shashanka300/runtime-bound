from pathlib import Path

from rbcore.hwmon.docker_stats import CgroupV1, CgroupV2, detect

CID = "abc123"


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_v2_layout(tmp_path: Path) -> None:
    _w(tmp_path / "cgroup.controllers", "cpu memory")
    d = tmp_path / "system.slice" / f"docker-{CID}.scope"
    _w(d / "cpu.stat", "usage_usec 2500000\nuser_usec 1\n")
    _w(d / "memory.current", str(3 * 2**30))
    _w(d / "memory.peak", str(4 * 2**30))
    _w(d / "memory.events", "low 0\nhigh 0\nmax 2\noom 1\noom_kill 1\n")
    cg = detect(tmp_path)
    assert isinstance(cg, CgroupV2)
    located = cg.locate(CID)
    assert located == d
    s = cg.read(located)
    assert (s.cpu_usec, s.mem_bytes, s.peak_bytes, s.oom_kills) == (
        2_500_000,
        3 * 2**30,
        4 * 2**30,
        1,
    )


def test_v1_cgroupfs_layout(tmp_path: Path) -> None:
    mem = tmp_path / "memory" / "docker" / CID
    _w(mem / "memory.usage_in_bytes", str(2**30))
    _w(mem / "memory.max_usage_in_bytes", str(2 * 2**30))
    _w(mem / "memory.oom_control", "oom_kill_disable 0\nunder_oom 0\noom_kill 3\n")
    _w(tmp_path / "cpu,cpuacct" / "docker" / CID / "cpuacct.usage", "7000000000")
    cg = detect(tmp_path)
    assert isinstance(cg, CgroupV1)
    located = cg.locate(CID)
    assert located == mem
    s = cg.read(located)
    assert (s.cpu_usec, s.mem_bytes, s.peak_bytes, s.oom_kills) == (7_000_000, 2**30, 2 * 2**30, 3)
    assert cg.locate("missing") is None
