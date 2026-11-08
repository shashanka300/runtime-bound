"""One figure style for the whole paper, and provenance-stamped figure export.

    from rbcore import plotting
    plotting.use_style()
    fig, ax = plotting.figure("single")
    ...
    plotting.save_fig(fig, "e0_throughput", exp_dir=EXP_DIR, run=run)

``save_fig`` writes ``figures/<name>.pdf`` (vector, for the paper) and ``.png`` (for
the notebook / Kaggle writeup), plus ``<name>.json`` recording which run(s), config
hash and git sha produced it, so every figure in the paper is traceable.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from rbcore.run import RunContext

# Okabe-Ito: colour-blind safe, prints acceptably in greyscale.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
# Two-column conference layout (NeurIPS/ICML-like): 3.25in column, 6.75in text width.
WIDTHS = {"single": 3.25, "double": 6.75}


def use_style() -> None:
    import scienceplots  # noqa: F401  (registers styles)

    plt.style.use(["science", "no-latex"])
    mpl.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=PALETTE),
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # embed TrueType: editable text, accepted by publishers
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def figure(
    width: Literal["single", "double"] = "single", aspect: float = 0.62, **kwargs: Any
) -> tuple[Figure, Any]:
    w = WIDTHS[width]
    return plt.subplots(figsize=(w, w * aspect), **kwargs)


def save_fig(
    fig: Figure,
    name: str,
    *,
    exp_dir: str | Path,
    run: RunContext | None = None,
    source_runs: list[str] | None = None,
    note: str = "",
) -> Path:
    """Write ``figures/<name>.{pdf,png,json}`` (the latest version, what the paper cites) and,
    when the figure has a run, a permanent copy ``figures/runs/<name>__<run_id>.*`` that later
    runs never overwrite."""
    out = Path(exp_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(out / f"{name}.png")
    runs = source_runs or ([run.run_id] if run else [])
    meta: dict[str, Any] = {
        "figure": name,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_runs": runs,
        "note": note,
    }
    if run is not None:
        meta |= {"config_hash": run.cfg.hash(), "git": run.manifest.get("git")}
        run.log_artifact(pdf)
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
    if runs:
        archive = out / "runs"
        archive.mkdir(exist_ok=True)
        for ext in ("pdf", "png", "json"):
            shutil.copy2(out / f"{name}.{ext}", archive / f"{name}__{runs[0]}.{ext}")
    return pdf
