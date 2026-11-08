from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: str = "phase_v2"
    """Named variant in src/build_submission.py (VARIANTS)."""
    subset: str | list[str] = "dev40"
    seeds: list[int] = Field(default=[0, 1])
    time_minutes: float = 6.0
    output_reserve: int = 4096
    concurrency: int = 2
    baseline_dir: str | None = None
    """Results of the tuned monolithic agent at the same budget (E1b arm B): the
    comparison baseline. Only this experiment's (task, seed) pairs are used."""
    baseline_label: str = "B (monolithic + reserve fix)"
    g3_min_gain: float = 0.05
    """Pre-registered G3/H4: gain >= 5 points with a paired CI above 0."""
    results_dir: str | None = None
    """Resume into an earlier run's results root."""
    adapter: str | None = "main_lora"
    """LoRA adapter named in every phase (starter default). None = base model, as the scorer
    effectively runs (its vLLM 0.19.1 wipes LoRA weights)."""
    thinking_translation: Literal["budget", "official"] = "budget"
    """swebox ServingTarget.thinking_translation ("official" = the scorer's mapping)."""
    label: str = ""
    """Output name suffix (e.g. seed1): runs of one variant never overwrite each other's tables."""
