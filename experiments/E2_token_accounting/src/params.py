from pydantic import BaseModel, ConfigDict, Field


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = ""
    """Suffix for table/figure names (e.g. e1_full, e1b_armB) so analyses of different
    inputs never overwrite each other."""
    input_dirs: list[str] = Field(default_factory=list)
    """swebox results directories to replay (each holds traces/ and task_results.jsonl),
    e.g. an E1 run's artifacts/swebox/t6min. Empty: produce a small input run first."""
    fallback_subset: str = "fidelity10"
    fallback_minutes: float = 6.0
    fallback_submission: str = "competition/sample_submission"
    """Used only when input_dirs is empty (keeps E2 runnable without E1)."""
    reread_overlap: float = 0.5
    """A read counts as a re-read if at least this fraction of its lines was returned before."""
    oversized_chars: int = 20_000
    reserved_output_tokens: int = 16_384
    """max_output_tokens of the replayed submission (starter: 16384). vLLM rejects a request
    when prompt + max_tokens > max_model_len, so the usable prompt is 32768 - this."""
    h2_supported_above: float = 0.40
    h2_falsified_below: float = 0.20
    """Pre-registered H2 thresholds on the waste share (docs/research_plan.md)."""
    gradability_csv: str | None = None
    """Per-task gradability from a harness self-check (swebox.analysis.classify_selfcheck);
    runs on non-gradable tasks are excluded from the H3 failure ranking. None: keep all."""
    h3_top_k: int = 3
    """H3 supported if context_exhausted ranks within the top k failure modes."""
    spot_check_n: int = 12
