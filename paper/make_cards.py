"""Kaggle Writeup card images (560x280 at 2x = 1120x560). Run from the repo root."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("paper/figures")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#c9c8c2"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF})


def card(name, title, line1, line2, draw):
    fig = plt.figure(figsize=(5.6, 2.8), dpi=200)
    fig.text(0.06, 0.80, title, fontsize=20, weight="bold", color=INK)
    fig.text(0.06, 0.67, line1, fontsize=10.5, color=INK2)
    fig.text(0.06, 0.59, line2, fontsize=10.5, color=INK2)
    ax = fig.add_axes([0.06, 0.15, 0.88, 0.37])
    draw(ax)
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def ladder(ax):
    labels = ["local", "+ thinking\ntranslation", "+ scorer\nengine", "stall-free\nreplica", "leaderboard"]
    vals = [30.0, 30.0, 17.5, 27.5, 3.4]
    cols = [BLUE, BLUE, NEUTRAL, BLUE, ORANGE]
    ax.bar(range(5), vals, color=cols, width=0.62, edgecolor=SURF, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(5), labels, fontsize=7.2, color=INK2)
    ax.set_ylim(0, 38)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", length=0)


def failures(ax):
    items = [("adapter\nwiped", BLUE), ("thinking\nbudget\nnot sent", BLUE),
             ("earlier\nreasoning\ndropped", BLUE), ("config\nflag\ndropped", ORANGE),
             ("poisoned\nwheel\ncache", ORANGE), ("41 of 129\ntasks\ngradable", NEUTRAL)]
    for i, (lab, col) in enumerate(items):
        ax.add_patch(plt.Rectangle((i + 0.08, 0.12), 0.84, 0.78, color=col, alpha=0.14, lw=0))
        ax.add_patch(plt.Rectangle((i + 0.08, 0.12), 0.84, 0.78, fill=False, ec=col, lw=1.4))
        ax.text(i + 0.5, 0.51, lab, ha="center", va="center", fontsize=7.4, color=INK, linespacing=1.15)
    ax.text(1.5, 0.0, "in the scorer, fixed by the organizers", ha="center", va="top", fontsize=7.2, color=BLUE)
    ax.text(4.0, 0.0, "in our own setup", ha="center", va="top", fontsize=7.2, color=ORANGE)
    ax.text(5.5, 0.0, "public test env", ha="center", va="top", fontsize=7.2, color=INK2)
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.25, 1)
    ax.axis("off")


card("card_runtime_bound", "Runtime-Bound",
     "Our agent solved 31% of tasks locally and 3% on the leaderboard.",
     "Where the gap went, and what really limits a local 31B coding agent.", ladder)
card("card_scorer_in_a_box", "Scorer in a Box",
     "A one-GPU replica of the Gemma 4 agent scorer.",
     "Six silent failures it caught, none of which raised an error.", failures)
print("cards written")
