"""Y7 dose figure: how much of the kappa drop is the first negative word?

Panel (a) is the ladder itself: kappa against the number of negative terms, both
estimators, one point per rung, all on the same 5w x 9s x 4-prompt x 1-seed
subset so the ladder is a paired comparison.

Panel (b) is why a level change alone is not enough: the shape of s*(w).  A
negative prompt could lower kappa by moving the whole curve down (no effect on
the shape) or by flattening it (the collinearity weakening).  The drop from
w=0.2 to w=2.0 in s* is the second reading.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y")


def main():
    A = pd.read_csv(os.path.join(OUT, "an_y7_dose.csv")).sort_values("n_neg_words")
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2))

    p = ax[0]
    p.plot(A.n_neg_words, A.k_grid, "-o", color="#1f3864", lw=2, ms=8,
           label="grid estimator", zorder=3)
    p.plot(A.n_neg_words, A.k_local, "-s", color="#c0392b", lw=2, ms=7.5,
           label="local (3-point) estimator", zorder=3)
    # values go in a small table rather than next to the markers: the grid and
    # local estimates cross between rungs, so point labels collide wherever the
    # two lines are close (x=1 and x=4).
    hdr = "terms   grid    local\n"
    body = "\n".join(f"{int(r.n_neg_words):>4}   {r.k_grid:.3f}   {r.k_local:.3f}"
                      for _, r in A.iterrows())
    p.text(.98, .97, hdr + body, transform=p.transAxes, ha="right", va="top",
           fontsize=8.3, family="monospace", color="#333",
           bbox=dict(fc="white", ec="#ccc", lw=.6, alpha=.93))
    p.axhline(0, color="#333", lw=1)
    p.set_xticks(A.n_neg_words)
    p.set_xlim(-0.7, 10.7)
    p.set_xlabel("number of terms in the negative prompt")
    p.set_ylabel("$\\kappa$  (slope of $\\log s^*$ on $\\log w$, sign-flipped)")
    p.set_title("(a) no dose-response: 10 terms rebounds to the empty level", fontsize=10.5)
    p.legend(fontsize=8.4, frameon=False, loc="lower right")
    p.set_ylim(min(-0.02, A.k_local.min() - .05), max(A.k_grid.max(), A.k_local.max()) + .09)

    p = ax[1]
    p.bar(np.arange(len(A)), A.sstar_drop * 100, .55,
          color=["#8fa5c4"] + ["#c0392b"] * (len(A) - 1),
          edgecolor="white", linewidth=.8, zorder=3)
    for i, (v, e) in enumerate(zip(A.sstar_drop * 100, A.edge_share)):
        p.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8.6, color="#333")
    p.set_xticks(np.arange(len(A)))
    p.set_xticklabels([f"{int(n)}" for n in A.n_neg_words])
    p.set_xlabel("number of terms in the negative prompt")
    p.set_ylabel("drop in $s^*$ from w=0.2 to w=2.0  (%)")
    p.set_title("(b) the $s^*(w)$ drop does not track it either", fontsize=10.5)
    p.text(.05, .95, "edge-share ($s^*$ pinned to a grid end)\n"
                     f"= {A.edge_share.max():.2f} at every rung -> not a fake flat",
           transform=p.transAxes, va="top", fontsize=8.2, color="#333",
           bbox=dict(fc="white", ec="#ccc", lw=.6, alpha=.92))
    p.set_ylim(0, max(A.sstar_drop * 100) * 1.32)

    for q in ax:
        q.grid(axis="y", color="#e9e9e9", lw=.7, zorder=0)
        q.set_axisbelow(True)
    fig.suptitle("Y7 dose: kappa against how negative the prompt is "
                 "(same grid for every rung)", fontsize=11.2, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .94])
    out = os.path.join(HERE, "fig_y7_dose.png")
    fig.savefig(out, dpi=175, facecolor="white")
    print("->", out)


if __name__ == "__main__":
    main()
