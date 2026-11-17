"""Y3 seed figure: which Y3 numbers survive a different seed?

The Y3 writeup makes three claims of decreasing strength:
  (1) the grid is largely additive, with a measurable interaction;
  (2) the wA axis keeps <60% of its slope once wB is loaded;
  (3) A loses its leverage *before* B does (the .09 vs .51 asymmetry).

Only (1) and (3) are in question here, and they fail in different ways, which is
why they get different panels:

  (a) per-cell shrink, seed 11 vs seed 22.  Levels are reproducible in rank
      (rho .68) but the *magnitude* moves by a factor of several, and 4/12 cells
      sit outside [0,1] on both seeds -- the axis is a non-monotone mess, so its
      median is a summary, not a level.
  (b) the asymmetry, same cells, same axes of the plot.  If "A collapses first"
      were structural, this scatter would lie along the diagonal.  It does not
      (r = -.14): the asymmetry is a single-seed artefact and is dropped.
  (c) the count form, which is what is left after (a) forbids quoting a level:
      how often each axis keeps <60% of its w = 0 slope.  Both axes, both seeds,
      ~2/3 of cells -- a symmetric statement that survives the seed change.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y")
SHRINK_LINE = 0.6


def load(tag, fn):
    D = pd.read_csv(os.path.join(OUT, fn))
    D["cell"] = D.prompt_id + "@s" + D.s.astype(str).str.replace(".0", "", regex=False)
    return D.set_index("cell")


def main():
    a = load("s11", "an_y3_additivity.csv")
    b = load("s22", "an_y3_s22_additivity.csv")
    common = [c for c in a.index if c in b.index]
    a, b = a.loc[common], b.loc[common]

    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.5))

    # ---- (a) shrink level, seed 11 vs seed 22 ---------------------------
    p = ax[0]
    # one cell reaches +3.44 on seed 22; letting it set the axis squashes the
    # other eleven into the origin, so it is clipped and marked instead.
    TOP = 2.15
    n_off = 0
    for pid, col in [("obj00", "#2c3e50"), ("obj01", "#c0392b"),
                     ("obj02", "#1e8449"), ("obj03", "#8e44ad")]:
        sel = [c for c in common if c.startswith(pid)]
        xv = a.loc[sel, "wA_shrink"].to_numpy(float)
        yv = b.loc[sel, "wA_shrink"].to_numpy(float)
        up = yv > TOP
        n_off += int(up.sum())
        p.scatter(xv[~up], yv[~up], s=56, color=col, edgecolor="white",
                  linewidth=.7, zorder=3, label=pid)
        if up.any():
            p.scatter(xv[up], np.full(up.sum(), TOP - .07), s=76, color=col,
                      marker="^", edgecolor="white", linewidth=.7, zorder=4)
            for xx, yy in zip(xv[up], yv[up]):
                p.annotate(f"{yy:.2f}", (xx, TOP - .07), textcoords="offset points",
                           xytext=(9, -3), ha="left", va="center",
                           fontsize=7.6, color=col, clip_on=False)
    lim = [-1.35, 2.15]
    p.plot(lim, lim, color="#888", lw=1, ls="--", zorder=1)
    p.axhline(SHRINK_LINE, color="#b03030", lw=1, ls=":", zorder=1)
    p.axvline(SHRINK_LINE, color="#b03030", lw=1, ls=":", zorder=1)
    p.set_xlim(*lim); p.set_ylim(*lim)
    p.set_aspect("equal")
    r = stats.pearsonr(a.wA_shrink, b.wA_shrink)[0]
    rho = stats.spearmanr(a.wA_shrink, b.wA_shrink)[0]
    p.text(.04, .96, f"$r$ = {r:+.2f}   $\\rho$ = {rho:+.2f}\n"
                     f"med {np.median(a.wA_shrink):.2f} → {np.median(b.wA_shrink):.2f}\n"
                     f"4/12 outside [0,1] on both seeds\n"
                     f"$\\triangle$ = clipped above {TOP:.1f}",
           transform=p.transAxes, va="top", fontsize=8.4, color="#333",
           bbox=dict(fc="white", ec="#ccc", lw=.6, alpha=.93))
    p.set_xlabel("wA shrink, seed 11"); p.set_ylabel("wA shrink, seed 22")
    p.set_title("(a) the level moves, the rank partly holds", fontsize=10.5)
    p.legend(fontsize=7.4, loc="lower right", frameon=False)

    # ---- (b) the asymmetry, seed 11 vs seed 22 --------------------------
    p = ax[1]
    x = (a.wA_shrink - a.wB_shrink).to_numpy(float)
    y = (b.wA_shrink - b.wB_shrink).to_numpy(float)
    p.scatter(x, y, s=54, color="#2f6f9f", edgecolor="white", linewidth=.7, zorder=3)
    p.axhline(0, color="#555", lw=1)
    p.axvline(0, color="#555", lw=1)
    rr, pp = stats.pearsonr(x, y)
    p.text(.04, .95, f"$r$ = {rr:+.2f}  ($p$ = {pp:.2f})\n"
                     f"A < B in {int((x < 0).sum())}/12 and {int((y < 0).sum())}/12 cells",
           transform=p.transAxes, va="top", fontsize=8.6, color="#333",
           bbox=dict(fc="white", ec="#ccc", lw=.6, alpha=.92))
    p.set_xlabel("wA−wB shrink, seed 11"); p.set_ylabel("wA−wB shrink, seed 22")
    p.set_title("(b) 'A collapses first' does not replicate", fontsize=10.5)
    p.text(.98, .04, "quadrants are the claim;\nthe cloud ignores them",
           transform=p.transAxes, ha="right", va="bottom", fontsize=8,
           color="#b03030", style="italic")

    # ---- (c) the count form ---------------------------------------------
    p = ax[2]
    labels = ["wA", "wB"]
    cols = ["wA_shrink", "wB_shrink"]
    xs = np.arange(2)
    for k, (tag, D, col) in enumerate([("seed 11", a, "#4a6fa5"),
                                       ("seed 22", b, "#c98a3c")]):
        vals = [(D[c].to_numpy(float) < SHRINK_LINE).mean() * 12 for c in cols]
        bars = p.bar(xs + (k - .5) * .36, vals, .34, color=col, label=tag,
                     edgecolor="white", linewidth=.8, zorder=3)
        for bb, v in zip(bars, vals):
            p.text(bb.get_x() + bb.get_width() / 2, v + .18, f"{v:.0f}/12",
                   ha="center", fontsize=8.6, color="#333")
    p.set_xticks(xs); p.set_xticklabels(labels)
    p.set_ylim(0, 12.4); p.set_yticks(range(0, 13, 2))
    p.set_ylabel("cells (out of 12) keeping < 60% of slope")
    p.set_title("(c) both axes collapse in ~2/3 of cells", fontsize=10.5)
    p.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)

    for q in ax:
        q.grid(axis="y", color="#e6e6e6", lw=.7, zorder=0)
        q.set_axisbelow(True)
    fig.suptitle("Y3 fixed per-seed: the asymmetry is an artefact, the collapse is not",
                 fontsize=11.5, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = os.path.join(HERE, "fig_y3_seed.png")
    fig.savefig(out, dpi=170, facecolor="white")
    print("->", out)


if __name__ == "__main__":
    main()
