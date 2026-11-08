"""Scaffold E1–E7 from the research plan (docs/research_plan.md). Idempotent."""

from new_experiment import scaffold

REPLAY_INPUTS = (
    "`params.input_run`: path to a finished run directory to replay (e.g. an E1 or E5 run).\n"
    "If empty, the notebook produces its own 10-task input run first, so this experiment\n"
    "never depends on another experiment having been run."
)

SPECS = {
    "E1_starter_baseline": dict(
        title="Starter baseline under realistic budgets",
        question="How does the organizers' starter agent behave on the 129 public tasks "
        "under 6 and 10 minute per-task budgets, and does our harness reproduce it?",
        feeds="H2, H3",
        design="per-task time {6, 10} min; 129 tasks x 3 seeds",
        budget="~60 GPU-h",
        exit="G1b: harness reproduces starter behaviour on a 10-task fidelity subset.",
    ),
    "E2_token_accounting": dict(
        title="Token accounting and failure taxonomy",
        question="Where do the tokens go (thinking, tool output, re-reads, re-prefill after "
        "compaction), and which failure modes dominate?",
        feeds="H2, H3",
        design="offline replay of trajectory logs",
        budget="~0 GPU-h (replay)",
        exit="G2: context-loss is a top-3 failure mode (else promote E4).",
        inputs=REPLAY_INPUTS,
    ),
    "E3_frontier_sweep": dict(
        title="Wall-clock frontier sweep",
        question="Under a fixed wall-clock budget, is 31B-int4 always the best base, or does "
        "12B with more attempts match it?",
        feeds="H1",
        design="model {31B-int4, 12B} x thinking {2k, 4k, 8k} x attempts {1, 2, 4}; dev-40 x 1",
        budget="~180 GPU-h",
        exit="Frontier plot drafted.",
    ),
    "E4_localization_tooling": dict(
        title="Localization tooling: graph tools vs in-sandbox skill scripts",
        question="Do in-sandbox BM25 + AST skill scripts beat the provided code-graph tools "
        "for localization with small models?",
        feeds="H6",
        design="{graph tools, skill scripts, both, none}; dev-40 x 2",
        budget="~60 GPU-h",
        exit="Localization default chosen.",
    ),
    "E5_context_architecture": dict(
        title="Prefix-stable phase-decomposed context architecture",
        question="Does a phase-decomposed agent with a pinned prefix raise resolved rate at "
        "equal wall-clock over the tuned monolithic agent?",
        feeds="H4",
        design="{monolithic, phase-decomposed, phase + pinned prefix}; dev-40 x 2, then 129 x 3",
        budget="~140 GPU-h",
        exit="G3: >= 5-point gain at equal wall-clock.",
    ),
    "E6_oracle_replay": dict(
        title="Offline context-eviction oracle",
        question="How far is the online context policy from a prefix-closed eviction oracle "
        "with a value term, computed on logged trajectories?",
        feeds="H5",
        design="offline oracle over logged trajectories",
        budget="~10 GPU-h",
        exit="Oracle gap reported with CIs.",
        inputs=REPLAY_INPUTS,
    ),
    "E7_rft_ablation": dict(
        title="Self-bootstrap rejection-sampling fine-tuning (optional)",
        question="Does one round of QLoRA RFT on Gemma's own successful trajectories transfer "
        "to held-out repos?",
        feeds="H7",
        design="base vs 1 round QLoRA RFT; leave-one-repo-out",
        budget="~180 GPU-h",
        exit="Gain on held-out repos, not only fastapi/rich.",
    ),
}

if __name__ == "__main__":
    for name, spec in SPECS.items():
        print(scaffold(name, **spec))
