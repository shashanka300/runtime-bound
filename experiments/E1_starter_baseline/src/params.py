from pydantic import BaseModel, ConfigDict, Field


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    submission: str = "competition/sample_submission"
    """Submission directory, relative to DATA_ROOT."""
    subset: str | list[str] = "fidelity10"
    """all | dev40 | fidelity10 | explicit instance ids."""
    seeds: list[int] = [0]
    time_budgets_min: list[float] = Field(default=[6.0, 10.0])
    """Per-task agent wall-clock budgets (the plan's realistic 12 h / ~120 tasks regime)."""
    max_tool_calls: int | None = None
    concurrency: int = 2
    selfcheck: bool = True
    """Grade gold and empty patches on the subset first (harness validity, no model)."""
    min_gold_rate: float = 0.9
    results_dir: str | None = None
    """Resume: an earlier run's ``artifacts/swebox`` directory; finished (task, seed, budget)
    results there are skipped and new ones appended. None = this run's own artifacts."""
    """Harness gate: at least this fraction of gold patches must resolve."""
