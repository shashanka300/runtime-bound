from pydantic import BaseModel, ConfigDict, Field


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    """Short label used in tags and figures (e.g. 31b-int4)."""
    hf_id: str
    served_name: str
    alias: str
    """Model alias written into the submission (must be in the eval registry)."""
    concurrency: int = 2
    extra_args: list[str] = Field(default_factory=list)
    """Extra `vllm serve` flags for this model."""


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_submission: str = "competition/sample_submission"
    subset: str | list[str] = "dev40"
    models: list[ModelCfg] = Field(
        default_factory=lambda: [
            ModelCfg(
                key="31b-int4",
                hf_id="google/gemma-4-31B-it-qat-w4a16-ct",
                served_name="gemma-4-31b-it",
                alias="gemma-4-31b-it-qat-w4a16-ct",
            ),
        ]
    )
    thinking_budgets: list[int] = Field(default=[2048, 4096, 8192])
    attempts: int = 4
    """Independent fresh-context sessions per (model, thinking, task); k in `ks` <= this."""
    ks: list[int] = Field(default=[1, 2, 4])
    output_reserve: int = 4096
    """max_output_tokens = thinking budget + this (see src/variants.py)."""
    time_minutes: float = 6.0
    """Per-attempt budget (as E1)."""
    recommend_budget_s: float = 360.0
    """Mean per-task seconds a recommended configuration must fit (12 h / 120 tasks)."""
    fresh_server: bool = True
    """Restart vLLM before the first model group so every group runs on an identical,
    adapter-free server (a server left by another experiment may have LoRA enabled)."""
    results_dir: str | None = None
    """Resume into an earlier run's results root (finished work is skipped)."""
