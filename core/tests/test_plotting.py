import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from rbcore import plotting


def test_save_fig_writes_pdf_png_and_provenance(tmp_path: Path) -> None:
    plotting.use_style()
    fig, ax = plotting.figure("single")
    ax.plot([0, 1], [0, 1], label="a")
    ax.legend()
    pdf = plotting.save_fig(fig, "t", exp_dir=tmp_path, source_runs=["r1"])
    assert pdf.read_bytes()[:4] == b"%PDF"
    assert (tmp_path / "figures/t.png").stat().st_size > 0
    meta = json.loads((tmp_path / "figures/t.json").read_text())
    assert meta["source_runs"] == ["r1"]
    w, _ = fig.get_size_inches()
    assert w == plotting.WIDTHS["single"]


def test_save_fig_keeps_every_runs_version(tmp_path: Path) -> None:
    fig, _ = plotting.figure("single")
    plotting.save_fig(fig, "t", exp_dir=tmp_path, source_runs=["r1"])
    plotting.save_fig(fig, "t", exp_dir=tmp_path, source_runs=["r2"])
    runs = tmp_path / "figures" / "runs"
    for r in ("r1", "r2"):
        for ext in ("pdf", "png", "json"):
            assert (runs / f"t__{r}.{ext}").stat().st_size > 0
    assert json.loads((tmp_path / "figures/t.json").read_text())["source_runs"] == ["r2"]
