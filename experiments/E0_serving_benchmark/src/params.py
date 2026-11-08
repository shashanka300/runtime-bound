from pydantic import BaseModel, ConfigDict, Field


class E0Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tokenizer: str = "google/gemma-4-31B-it-qat-w4a16-ct"
    concurrency: list[int] = [1, 4, 8]
    context_lens: list[int] = [8192, 16384, 32768]
    prefix_lens: list[int] = Field(
        default=[0, 4096],
        description="Shared-prefix length per request; >0 exercises prefix caching (C2).",
    )
    output_len: int = 512
    prompts_per_slot: int = 4
    """num_prompts = max(min_prompts, concurrency * prompts_per_slot)."""
    min_prompts: int = 8
    warmup_prompts: int = 2
    cooldown_s: float = 5.0
    """Idle gap between cells so power/temperature readings do not bleed across cells."""
    vllm_log: str = "logs/vllm_latest.log"
    """Relative to DATA_ROOT; written by infra/vllm/serve.sh."""
    g1_min_output_tok_s: float = 15.0
    """Gate G1: single-stream decode must at least reach this (catches broken kernels)."""
