# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: challeenge (py3.12)
#     language: python
#     name: challeenge
# ---

# %% [markdown]
# # E0 — Serving benchmark (gate G1)
#
# **Question.** Can one RTX 5090 serve `gemma-4-31B-it-qat-w4a16-ct` at 32k context
# correctly, and what are its throughput, latency, KV headroom and energy per token?
# Every later wall-clock number is scaled against this.
#
# **Grid.** shared-prefix {0, 4096} × context {8k, 16k, 32k} × concurrency {1, 4, 8},
# random tokens, `--ignore-eos`, fixed output length, driven by `vllm bench serve`.
#
# **Exit criterion (G1).** Smoke test is correct *and* every 32k cell completes *and*
# single-stream decode ≥ `g1_min_output_tok_s`.
#
# **Prerequisite.** `bash infra/vllm/serve.sh` running in another terminal;
# optionally `bash infra/services.sh start` for MLflow / Phoenix.
#
# Each section below can be run and inspected on its own.

# %% [markdown]
# ## 0 · Parameters
# Papermill injects overrides here (dotted keys, e.g. `{"params.concurrency": [1]}`).

# %% tags=["parameters"]
CONFIG_OVERRIDES: dict = {}

# %% [markdown]
# ## 1 · Setup & provenance

# %%
from pathlib import Path

import polars as pl
from IPython.display import Markdown, display
from src import bench
from src.params import E0Params

from rbcore import RunContext, load_config, plotting
from rbcore.paths import data_root
from rbcore.run import phase_windows
from rbcore.windows import counter_delta, gpu_energy_j, per_window

EXP_DIR = Path.cwd()
cfg = load_config(EXP_DIR / "config.yaml", E0Params, CONFIG_OVERRIDES)
P = cfg.params
plotting.use_style()
cells = bench.grid(P)
display(Markdown(f"config hash `{cfg.hash()}` · **{len(cells)} cells**"))
pl.DataFrame(
    [{**c.__dict__, "input_len": c.input_len, "prompts": bench.num_prompts(c, P)} for c in cells]
)

# %% [markdown]
# ## 2 · Preflight
# Server health, KV capacity from the vLLM startup log, and a correctness smoke test.
# A wrong answer here almost always means a quantized-kernel problem on Blackwell.

# %%
import httpx

models = httpx.get(f"{cfg.serving.base_url}/v1/models", timeout=5).raise_for_status().json()
served = {m["id"]: m.get("max_model_len") for m in models["data"]}
assert cfg.serving.model in served, f"served models: {served}"
assert served[cfg.serving.model] == cfg.serving.max_model_len, served

kv = bench.kv_capacity(data_root() / P.vllm_log)
smoke = bench.smoke_test(cfg.serving)
display(Markdown(f"served: `{served}`  \nKV capacity: `{kv}`  \nsmoke: `{smoke}`"))
assert smoke["ok"], "smoke test failed: model output is wrong, check kernels/quantization"
assert bench.VLLM_BIN.exists(), "run: (cd infra/vllm && uv sync)"

# %% [markdown]
# ## 3 · Run the grid
# Everything below runs inside one `RunContext`: hardware monitors at 1 Hz, spans,
# events and MLflow. Each cell is a `phase`, so telemetry is attributed per cell.
# A live table updates as cells finish.

# %%
run = RunContext(cfg, exp_dir=EXP_DIR).start()  # closed in section 8 (run.finish())
run.event("preflight", kv=kv, smoke=smoke)
bench_dir = run.artifacts / "bench"
bench_dir.mkdir()
print(run.dir)

# %%
import time

progress = display(pl.DataFrame(), display_id=True)
with run.phase("warmup"):
    warm = bench.Cell(0, 4096, 1, 64)
    bench.run_cell(
        bench.bench_cmd(warm, P, cfg.serving, bench_dir / "warmup", cfg.seed, P.warmup_prompts),
        bench_dir / "warmup.log",
    )

for i, cell in enumerate(cells):
    time.sleep(P.cooldown_s)
    n = bench.num_prompts(cell, P)
    with run.phase(
        "cell",
        key=cell.key,
        prefix_len=cell.prefix_len,
        context_len=cell.context_len,
        concurrency=cell.concurrency,
        prompts=n,
    ):
        bench.run_cell(
            bench.bench_cmd(cell, P, cfg.serving, bench_dir, cfg.seed, n),
            bench_dir / f"{cell.key}.log",
        )
    res = bench.load_results(bench_dir, cells)
    run.log_metrics({"cells_done": i + 1})
    progress.update(res.select("key", "output_throughput", "median_ttft_ms", "p99_itl_ms"))

# %% [markdown]
# ## 4 · Live hardware view
# Re-run this cell at any time (also mid-run from a second kernel via `read_hw`).

# %%
gpu = run.hw("gpu")
fig, axes = plotting.figure("double", aspect=0.45, ncols=3)
for ax, col, label in zip(
    axes,
    ["power_w", "util_gpu_pct", "mem_used_mib"],
    ["Power (W)", "GPU util (%)", "Memory (MiB)"],
    strict=True,
):
    ax.plot(gpu["time"], gpu[col], lw=0.6)
    ax.set_ylabel(label)
    ax.tick_params(axis="x", labelrotation=30)
fig.tight_layout()

# %% [markdown]
# ## 5 · Ingest: benchmark results ⨝ telemetry per cell

# %%
results = bench.load_results(bench_dir, cells)
windows = (
    phase_windows(run.events())
    .filter(pl.col("phase") == "cell")
    .select("key", "t_start_ns", "t_end_ns")
)
vllm = run.hw("vllm")

energy = gpu_energy_j(run.hw("gpu"), windows)
power = per_window(
    run.hw("gpu"),
    windows,
    [
        pl.col("power_w").mean().alias("mean_power_w"),
        pl.col("temp_c").max().alias("max_temp_c"),
        pl.col("sm_clock_mhz").mean().alias("mean_sm_mhz"),
    ],
)
kv_use = per_window(
    vllm.filter(pl.col("metric") == "vllm:kv_cache_usage_perc"),
    windows,
    [pl.col("value").max().alias("peak_kv_usage")],
)
hits = counter_delta(vllm, windows, "vllm:prefix_cache_hits_total")
queries = counter_delta(vllm, windows, "vllm:prefix_cache_queries_total")

table = (
    results.join(energy.drop("t_start_ns", "t_end_ns"), on="key", how="left")
    .join(power.drop("t_start_ns", "t_end_ns"), on="key", how="left")
    .join(kv_use.drop("t_start_ns", "t_end_ns"), on="key", how="left")
    .join(hits.drop("t_start_ns", "t_end_ns"), on="key", how="left")
    .join(queries.drop("t_start_ns", "t_end_ns"), on="key", how="left")
    .with_columns(
        (pl.col("gpu_energy_j") / pl.col("total_output_tokens")).alias("j_per_out_tok"),
        (
            pl.col("gpu_energy_j") / (pl.col("total_input_tokens") + pl.col("total_output_tokens"))
        ).alias("j_per_tok"),
        (pl.col("vllm:prefix_cache_hits_total") / pl.col("vllm:prefix_cache_queries_total")).alias(
            "prefix_hit_rate"
        ),
    )
)
table.write_parquet(run.dir / "results.parquet")
table.select(
    "key",
    "output_throughput",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_itl_ms",
    "mean_power_w",
    "j_per_out_tok",
    "peak_kv_usage",
    "prefix_hit_rate",
)

# %% [markdown]
# ## 6 · Gate G1

# %%
single = table.filter((pl.col("concurrency") == 1) & (pl.col("prefix_len") == 0))
expected = pl.max_horizontal(pl.lit(P.min_prompts), pl.col("concurrency") * P.prompts_per_slot)
full_ctx = table.filter(pl.col("context_len") == max(P.context_lens))
g1 = {
    "smoke_ok": smoke["ok"],
    "all_32k_cells_complete": full_ctx.height
    == sum(c.context_len == max(P.context_lens) for c in cells)
    and bool(full_ctx.select((pl.col("completed") == expected).all()).item()),
    "min_single_stream_tok_s": float(single["output_throughput"].min()),
}
g1["pass"] = (
    g1["smoke_ok"]
    and g1["all_32k_cells_complete"]
    and g1["min_single_stream_tok_s"] >= P.g1_min_output_tok_s
)
run.event("gate_g1", **g1)
run.log_metrics(
    {"g1_pass": float(g1["pass"]), "single_stream_tok_s": g1["min_single_stream_tok_s"]}
)
g1

# %% [markdown]
# ## 7 · Paper figures
# (a) output throughput vs concurrency · (b) median TTFT vs context · (c) energy per
# output token · (d) prefix-cache effect on TTFT.

# %%
fig, axes = plotting.figure("double", aspect=0.30, ncols=4)
base = table.filter(pl.col("prefix_len") == 0)
for ctx in P.context_lens:
    d = base.filter(pl.col("context_len") == ctx).sort("concurrency")
    lbl = f"{ctx // 1024}k"
    axes[0].plot(d["concurrency"], d["output_throughput"], marker="o", label=lbl)
    axes[2].plot(d["concurrency"], d["j_per_out_tok"], marker="o", label=lbl)
for conc in P.concurrency:
    d = base.filter(pl.col("concurrency") == conc).sort("context_len")
    axes[1].plot(d["context_len"] / 1024, d["median_ttft_ms"] / 1000, marker="o", label=f"n={conc}")
for pre in P.prefix_lens:
    d = table.filter((pl.col("prefix_len") == pre) & (pl.col("concurrency") == max(P.concurrency)))
    d = d.sort("context_len")
    axes[3].plot(
        d["context_len"] / 1024, d["median_ttft_ms"] / 1000, marker="o", label=f"prefix {pre}"
    )
axes[0].set(xlabel="Concurrency", ylabel="Output tok/s", title="(a) Throughput")
axes[1].set(xlabel="Context (k tokens)", ylabel="Median TTFT (s)", title="(b) Prefill latency")
axes[2].set(xlabel="Concurrency", ylabel="GPU J / output token", title="(c) Energy")
axes[3].set(xlabel="Context (k tokens)", ylabel="Median TTFT (s)", title="(d) Shared prefix")
for ax in axes:
    ax.legend()
fig.tight_layout()
plotting.save_fig(fig, "e0_serving_frontier", exp_dir=EXP_DIR, run=run)

# %%
# Timeline figure: power and KV usage over the whole run with cells shaded — an
# observability figure for the appendix (shows the monitor captured every cell).
fig, ax = plotting.figure("double", aspect=0.28)
g = run.hw("gpu")
ax.plot(g["time"], g["power_w"], lw=0.5, label="GPU power (W)")
kvt = vllm.filter(pl.col("metric") == "vllm:kv_cache_usage_perc")
ax2 = ax.twinx()
ax2.plot(kvt["time"], kvt["value"] * 100, lw=0.5, color=plotting.PALETTE[1], label="KV usage (%)")
spans = windows.select(pl.from_epoch("t_start_ns", "ns"), pl.from_epoch("t_end_ns", "ns"))
for start, end in spans.iter_rows():
    ax.axvspan(start, end, alpha=0.08, color="grey")
ax.set_ylabel("Power (W)")
ax2.set_ylabel("KV usage (%)")
ax2.grid(False)
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
plotting.save_fig(fig, "e0_timeline", exp_dir=EXP_DIR, run=run)

# %% [markdown]
# ## 8 · Teardown
# Stops monitors, flushes spans, writes `summary.json`, closes the MLflow run.

# %%
tables = EXP_DIR / "tables"
tables.mkdir(exist_ok=True)
table.write_csv(tables / "e0_serving.csv")
run.finish()
print((run.dir / "summary.json").read_text())
