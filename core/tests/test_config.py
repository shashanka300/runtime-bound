from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from rbcore.config import SandboxConfig, load_config


class Params(BaseModel):
    concurrency: list[int] = [1]
    name: str = "x"


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body)
    return p


def test_load_and_override(tmp_path: Path) -> None:
    path = _write(tmp_path, "experiment: E0\nparams:\n  concurrency: [1, 4]\n")
    cfg = load_config(path, Params, {"params.concurrency": [8], "seed": 3})
    assert cfg.params.concurrency == [8]
    assert cfg.seed == 3
    assert cfg.flat()["params.concurrency"] == [8]


def test_json_string_overrides(tmp_path: Path) -> None:
    path = _write(tmp_path, "experiment: E0\nparams: {}\n")
    assert load_config(path, Params, '{"params.name": "y"}').params.name == "y"


def test_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    path = _write(tmp_path, "experiment: E0\nparams: {}\n")
    a = load_config(path, Params)
    assert a.hash() == load_config(path, Params).hash()
    assert a.hash() != load_config(path, Params, {"seed": 1}).hash()


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "experiment: E0\nparams: {}\ntypo_section: 1\n")
    with pytest.raises(ValidationError):
        load_config(path, Params)


def test_sandbox_profiles() -> None:
    assert SandboxConfig(profile="eval").matches_eval
    dev = SandboxConfig(profile="dev")
    assert (dev.mem_gib, dev.cpus) == (8.0, 4.0) and not dev.matches_eval
    assert SandboxConfig(profile="custom", mem_gib=16, cpus=8).cpus == 8
    with pytest.raises(ValidationError):
        SandboxConfig(profile="eval", mem_gib=16)
    with pytest.raises(ValidationError):
        SandboxConfig(profile="custom")
