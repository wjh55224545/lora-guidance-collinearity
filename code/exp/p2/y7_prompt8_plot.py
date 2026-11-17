"""Figure for Y7p8: the four fresh prompts are where the claim dies.

One arm (the four-term quality string), eight prompts, split into the four that
were used to make the original claim and the four that were not.  The left panel
is the per-prompt kappa pair (joined lines, so a prompt that moved is visible as
a segment that is not vertical) and the right panel is the difference against the
empty baseline grouped by estimator.

The point of drawing it at all: a mean of -.081 with p=.077 looks like "almost
there", and only the per-prompt view shows that the four fresh prompts contribute
-.030 with one of four negative -- i.e. the mean is carried by obj00 and obj03.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y")
C_E, C_N = "#6b6b6b", "#d1622b"


def main():
    C = pd.read_csv(os.path.join(OUT, "an_y7_prompt8.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9),
                             gridspec_kw=dict(width_ratios=[1.2, 1]))

    # ---- (a) per-prompt pair, grid estimator
    ax = axes[0]
    D = C[C.estimator == "grid"].sort_values("prompt_id")
    x = np.arange(len(D))
    for xi, (_, r) in zip(x, D.iterrows()):
        ax.plot([xi, xi], [r["empty"], r["neg"]], color="#bbb", lw=1.4, zorder=1)
    ax.scatter(x, D["empty"], s=42, color=C_E, zorder=3, label="empty negative")
    ax.scatter(x, D["neg"], s=42, color=C_N, zorder=3, label="4-term quality string")
    ax.axhline(0, color="#999", lw=.8)
    ax.axvline(3.5, color="#bbb", lw=1.0, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("obj", "") for p in D.prompt_id], fontsize=8)
    lo = min(D["neg"].min(), D["empty"].min()) - .05
    hi = max(D["neg"].max(), D["empty"].max()) + .05
    ax.set_ylim(lo, hi)
    ax.text(3.4, hi - .02, "original 4", ha="right", fontsize=7.8, color="#666",
            va="top")
    ax.text(3.6, hi - .02, "fresh 4", ha="left", fontsize=7.8, color="#666",
            va="top")
    ax.set_xlabel("prompt id")
    ax.set_ylabel("κ  (grid estimator)")
    ax.set_title("(a) one arm, eight prompts", fontsize=9.8)
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    # ---- (b) group means with the per-prompt values
    ax = axes[1]
    groups = [("all 8", lambda p: True), ("original 4", lambda p: p <= "obj03"),
              ("fresh 4", lambda p: p >= "obj04")]
    pos, labs = [], []
    for gi, (gname, sel) in enumerate(groups):
        for ei, (est, off, hatch) in enumerate([("grid", -.28, ""),
                                               ("local", .28, "///")]):
            d = C[C.estimator == est]
            m = d.prompt_id.map(sel).to_numpy(bool)
            v = d.loc[m, "diff"].to_numpy(float)
            xi = gi * 1.0 + off
            ax.bar(xi, v.mean(), .5, color=C_N if est == "grid" else "#e8a37e",
                   alpha=.85, hatch=hatch, edgecolor="white", zorder=2)
            ax.scatter(np.full(len(v), xi), v, s=20, color="#222", alpha=.6,
                       zorder=4, lw=0)
            ax.text(xi, v.mean() + (.008 if v.mean() >= 0 else -.008),
                    f"{v.mean():+.3f}", ha="center",
                    va="bottom" if v.mean() >= 0 else "top", fontsize=7.6,
                    color="#333")
            pos.append(xi)
            labs.append(est)
        ax.text(gi, -.30, gname, ha="center", fontsize=8.2, color="#444")
    ax.set_ylim(-.33, None)
    ax.set_xlim(-.62, 2.62)
    ax.axhline(0, color="#444", lw=1.0)
    ax.set_xticks(pos)
    ax.set_xticklabels(labs, fontsize=7.8)
    ax.set_ylabel("Δκ vs empty negative")
    ax.set_title("(b) the fresh four are flat", fontsize=9.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(HERE, "fig_y7_prompt8.png")
    fig.savefig(p, dpi=190)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
