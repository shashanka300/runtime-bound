"""Paper figures (paper/figures/). Run from the repo root:
    .venv/bin/python paper/make_figures.py
Fig 1 and fig 3 use numbers recorded in docs/design_changes.md (sources inline); fig 2 is
computed from the lb04 screen traces (data/screen_nov/lb04_public_7min)."""
import ast
import collections
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#c9c8c2"  # categorical slots 1-2 + neutral
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def as_dict(x):
    return x if isinstance(x, dict) else ast.literal_eval(x) if x else {}


# ---------- Fig 1: fidelity ladder ----------
# design_changes.md 2026-09-27 (ladder result, CORRECTION, clean engine effect); lb01 board 2/58.
steps = [
    ("Local engine (vLLM 0.30), thinking budget sent", 30.0, "ok", ""),
    ("+ scorer's thinking translation", 30.0, "ok", ""),
    ("+ scorer's engine (vLLM 0.19.1), concurrency 2", 17.5, "artefact",
     "  local KV-reservation stall (artefact)"),
    ("Scorer engine, stall-free, L4 speed", 27.5, "ok", ""),
    ("Leaderboard (lb01, same agent family)", 3.4, "board", "  public split, 2 of 58"),
]
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ys = list(range(len(steps)))[::-1]
for y, (lab, v, kind, note) in zip(ys, steps):
    color = {"ok": BLUE, "artefact": NEUTRAL, "board": ORANGE}[kind]
    ax.barh(y, v, height=0.56, color=color, edgecolor=SURF, linewidth=2)
    ax.text(v + 0.6, y, f"{v:.1f}%{note}", va="center", color=INK, fontsize=9)
ax.set_yticks(ys, [s[0] for s in steps])
ax.set_xlim(0, 55)
ax.set_xlabel("Tasks solved, % (local: dev-40, seed 0; leaderboard: public score)")
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.set_title("Moving local runs toward the scorer one condition at a time", loc="left",
             fontsize=11, color=INK, pad=10)
save(fig, "fig1_fidelity_ladder")

# ---------- Fig 2: where the time goes in lb04 (computed from traces) ----------
run = "data/screen_nov/lb04_public_7min"
res = {json.loads(l)["instance_id"]: json.loads(l) for l in open(f"{run}/results.jsonl")}


def is_loop(msg):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", msg or "") if len(x.strip()) > 20]
    c = collections.Counter(s)
    return bool(s) and sum(v - 1 for v in c.values() if v > 1) / len(s) > 0.25


rows = []
for iid, r in res.items():
    f = glob.glob(f"{run}/harness/{iid}/traces/*.json")
    if not f:
        continue
    d = json.load(open(f[0]))
    prev = float(as_dict(d["steps"][0]["extra"])["elapsed_s"])
    loop_t = delib_t = other_t = 0.0
    for s in d["steps"]:
        if s.get("source") != "agent":
            continue
        t = float(as_dict(s["extra"])["elapsed_s"])
        dt, prev = t - prev, t
        if as_dict(s.get("metrics")).get("completion_tokens", 0) >= 1500:
            if is_loop(s.get("message")):
                loop_t += dt
            else:
                delib_t += dt
        else:
            other_t += dt
    timed_out = "timeout" in (r.get("error") or "")
    # time after the last logged turn: a final model call still generating when the cap hit
    unfinished = max(0.0, float(r["duration_s"]) - (loop_t + delib_t + other_t)) if timed_out else 0.0
    if r["resolved"] or timed_out:
        rows.append([iid, "Timed out" if timed_out else "Solved", loop_t, delib_t, other_t, unfinished])
rows.sort(key=lambda x: (x[1] != "Timed out", -sum(x[2:])))
json.dump(rows, open(OUT / "fig2_data.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(7.4, 7.8))
ypos, y, last = [], 0.0, None
for row in rows:
    if last is not None and row[1] != last:
        y += 1.4
    ypos.append(y)
    y += 1
    last = row[1]
top = max(ypos)
ypos = [top - v for v in ypos]
for yy, (_, _, lt, dl, ot, un) in zip(ypos, rows):
    left = 0.0
    for val, col in ((lt, ORANGE), (dl, BLUE), (ot, NEUTRAL)):
        if val > 0:
            ax.barh(yy, val, left=left, height=0.72, color=col, edgecolor=SURF, linewidth=1.5)
        left += val
    XCLIP = 320
    if un > 1:
        shown = min(un, XCLIP - left)
        ax.barh(yy, shown, left=left, height=0.72, color=SURF, edgecolor=MUTED, linewidth=0.8,
                hatch="////")
        if left + un > XCLIP:
            ax.text(XCLIP + 2, yy, f"→ {left + un:.0f} s", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(ypos, [r[0] for r in rows], fontsize=7.5)
ax.tick_params(axis="y", length=0)
xmax = 320
for grp in ("Timed out", "Solved"):
    g = [yy for yy, r in zip(ypos, rows) if r[1] == grp]
    ax.text(xmax * 1.08, (max(g) + min(g)) / 2, f"{grp}\n({len(g)} tasks)", va="center",
            fontsize=9, color=INK2)
cap = 7 * 60 * 0.625
ax.axvline(cap, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
ax.text(cap - 3, max(ypos) + 1.1, f"time cap {cap:.0f} s", fontsize=8, color=MUTED, ha="right")
ax.set_xlim(0, xmax * 1.25)
ax.set_xticks(range(0, 301, 50))
ax.set_xlabel("Seconds of the task spent in each kind of turn")
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=ORANGE, label="Turn of 1,500+ tokens: repetition loop"),
                   Patch(color=BLUE, label="Turn of 1,500+ tokens: long deliberation"),
                   Patch(color=NEUTRAL, label="All other turns"),
                   Patch(facecolor=SURF, edgecolor=MUTED, hatch="////",
                         label="Last turn, unfinished at the cap")],
          loc="upper center", bbox_to_anchor=(0.45, -0.06), ncol=2, frameon=False, fontsize=8.5)
ax.set_title("lb04: timed-out tasks lose their time in a few very long turns", loc="left",
             fontsize=11, color=INK, pad=16)
save(fig, "fig2_timeout_anatomy")

# ---------- Fig 3: local screen vs leaderboard ----------
# design_changes.md / leaderboard/README.md: local rates and public scores per submission.
subs = [("lb01", 31.2, 0.03, "dev-40, September harness"),
        ("lb03", 32.5, 0.06, "dev-40, fidelity mode"),
        ("lb04", 41.0, 0.15, "39 gradable tasks"),
        ("lb10", 38.5, 0.10, "39 gradable tasks")]
fig, ax = plt.subplots(figsize=(6.0, 3.6))
for name, loc, board, note in subs:
    ax.scatter(loc, board * 100, s=70, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
    ax.annotate(f"{name} ({note})", (loc, board * 100), xytext=(8, -3),
                textcoords="offset points", fontsize=8.5, color=INK)
ax.set_xlim(28, 52)
ax.set_ylim(0, 20)
ax.set_xlabel("Local solve rate (%)")
ax.set_ylabel("Public leaderboard score (%)")
ax.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.set_title("Local solve rates overstate the leaderboard 3-10x", loc="left",
             fontsize=10.5, color=INK, pad=10)
save(fig, "fig3_local_vs_board")
print("figures:", sorted(p.name for p in OUT.iterdir()))
