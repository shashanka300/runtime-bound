"""Typed experiment configuration.

Each experiment has a ``config.yaml`` with shared sections (serving, sandbox, monitor,
tracking) plus an experiment-specific ``params`` block validated by its own pydantic
model. Papermill parameters arrive as a flat dict of dotted keys and are merged on top.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServingConfig(_Strict):
    base_url: str = "http://127.0.0.1:8000"
    model: str = "gemma-4-31b-it"
    """served-model-name exposed by vLLM (see infra/vllm/serve.yaml)."""
    max_model_len: int = 32_768
    request_timeout_s: float = 600.0


# (memory GiB, vCPUs). "eval" reproduces the official sandbox and is required for
# headline numbers; "dev" gives containers more room for faster iteration.
SANDBOX_PROFILES: dict[str, tuple[float, float]] = {"eval": (4.0, 2.0), "dev": (8.0, 4.0)}


class SandboxConfig(_Strict):
    profile: Literal["eval", "dev", "custom"] = "dev"
    mem_gib: float | None = None
    cpus: float | None = None
    network: Literal["none"] = "none"

    @model_validator(mode="after")
    def _resolve_profile(self) -> SandboxConfig:
        if self.profile == "custom":
            if self.mem_gib is None or self.cpus is None:
                raise ValueError("custom sandbox profile needs mem_gib and cpus")
            return self
        mem, cpus = SANDBOX_PROFILES[self.profile]
        if self.mem_gib not in (None, mem) or self.cpus not in (None, cpus):
            raise ValueError(f"profile {self.profile!r} fixes mem={mem} cpus={cpus}; use 'custom'")
        object.__setattr__(self, "mem_gib", mem)
        object.__setattr__(self, "cpus", cpus)
        return self

    @property
    def matches_eval(self) -> bool:
        return (self.mem_gib, self.cpus) == SANDBOX_PROFILES["eval"]


class MonitorConfig(_Strict):
    interval_s: float = Field(1.0, gt=0)
    flush_every_s: float = Field(30.0, gt=0)
    nvml: bool = True
    host: bool = True
    docker: bool = True
    vllm: bool = True
    codecarbon: bool = True
    codecarbon_country: str | None = None
    """ISO-3166 alpha-3 grid region for offline CodeCarbon (e.g. "USA"); None = auto-detect."""


class TrackingConfig(_Strict):
    mlflow_uri: str | None = "http://127.0.0.1:5000"
    """None disables MLflow; files under runs/<run_id>/ are always written."""
    phoenix_endpoint: str | None = "http://127.0.0.1:6006/v1/traces"


class ExperimentConfig[P: BaseModel](_Strict):
    experiment: str
    description: str = ""
    seed: int = 0
    serving: ServingConfig = ServingConfig()
    sandbox: SandboxConfig = SandboxConfig()
    monitor: MonitorConfig = MonitorConfig()
    tracking: TrackingConfig = TrackingConfig()
    params: P

    def hash(self) -> str:
        """Stable short hash of the fully resolved config (goes into run ids and manifests)."""
        canon = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:12]

    def flat(self) -> dict[str, Any]:
        """Dotted-key view for MLflow params."""
        return _flatten(self.model_dump(mode="json"))


def load_config[P: BaseModel](
    path: str | Path,
    params_model: type[P],
    overrides: Mapping[str, Any] | str | None = None,
) -> ExperimentConfig[P]:
    """Load YAML, apply dotted-key overrides (e.g. ``{"params.concurrency": [1, 4]}``), validate."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if isinstance(overrides, str):  # papermill `-p CONFIG_OVERRIDES '{...}'` passes a string
        overrides = json.loads(overrides) if overrides.strip() else {}
    for key, value in (overrides or {}).items():
        _set_dotted(raw, key, value)
    model: type[ExperimentConfig[P]] = ExperimentConfig[params_model]  # type: ignore[valid-type]
    return model.model_validate(raw)


def _set_dotted(tree: dict[str, Any], key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    node = tree
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def _flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in tree.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out
