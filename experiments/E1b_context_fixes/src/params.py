from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_submission: str = "competition/sample_submission"
    subset: str | list[str] = "dev40"
    seeds: list[int] = Field(default=[0, 1])
    arms: list[str] = Field(default=["B", "C", "D"])
    """Arms to run now; arm A comes from ``reuse_arm_a`` when given, else it is run too."""
    reuse_arm_a: str | None = None
    """A results dir with the unmodified starter at the same budget (e.g. E1's
    artifacts/swebox/t6min); only the (task, seed) pairs of this experiment are used."""
    reuse: dict[str, str] = Field(default_factory=dict)
    """Any other arm -> an earlier results dir for it (same pairing rule as ``reuse_arm_a``)."""
    time_minutes: float = 6.0
    output_reserve: int = 4096
    """B/C/D: max_output_tokens = thinking budget + this."""
    d_context_retries: int = 2
    concurrency: int = 2
    results_dir: str | None = None
    """Resume into an earlier run's results root."""
    comparisons: list[list[str]] = Field(default=[["A", "B"], ["B", "C"], ["B", "D"], ["A", "C"]])
    """Pre-registered paired comparisons (second minus first)."""
    thinking_translation: Literal["budget", "official"] = "budget"
    """swebox ServingTarget.thinking_translation: "official" = the scorer's mapping (no thinking
    budget reaches vLLM; thinking is bounded by max_output_tokens only)."""
    label: str = ""
    """Suffix for exported tables/figures, so follow-up runs never overwrite earlier outputs."""
