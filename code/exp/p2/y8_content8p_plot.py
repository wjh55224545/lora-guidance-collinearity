"""Figure for Y8-content-8p: the effect that survived.

The negative-prompt story went through three stages and the figure has to show
the third one, because the first two were wrong:

  Y7   "a four-term negative prompt halves kappa"      -- 4 prompts, p=.13
  Y7p8 the same string does nothing on four fresh ones -- so it was dropped
  Y8   four concrete nouns drop kappa as much         -- so it is not the
       quality vocabulary                                   vocabulary
  here  the content arm on all eight prompts          -- -0.143 (grid, 7/8,
                                                          p=.004), and on the
                                                          fresh four it is
                                                          STRONGER than the
                                                          quality string

Left   per-prompt kappa, empty vs content vs quality, eight prompts, sorted by
       the empty value so the ordering is readable; the two fresh prompts are
       marked, because the whole dispute is about them.
Right  mean difference against empty with the per-prompt values drawn as ticks,
       split into the original four and the fresh four.  If a bar is long but its
       ticks straddle zero, the mean is being carried by one prompt -- that is the
       failure mode this experiment exists to rule out.

Grid estimator only.  The local estimator agrees in sign everywhere
(-.085 overall, 6/8) and the numbers are in an_y8_content8p.csv.
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
C_EMPTY, C_CONT, C_QUAL = "#6b6b6b", "#d1622b", "#1f6fb4"


def main():
    C = pd.read_csv(os.path.join(OUT, "an_y8_content8p.csv"))
    C = C[C.estimator == "grid"]
    W = C.pivot_table(index="prompt_id", columns="arm", values="diff")
    # kappa_arm, not kappa_empty: the empty column is repeated for every arm, so
    # pivoting it puts the same curve in every cell and the arm lines vanish.
    E = C.pivot_table(index="prompt_id", columns="arm", values="kappa_arm")
    base = C.pivot_table(index="prompt_id", columns="arm",
                         values="kappa_empty").iloc[:, 0]
    # prompt order, not sorted by kappa: the original/fresh split has to stay a
    # clean group boundary or the divider line means nothing
    order = sorted(base.index)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.1),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))

    # ---- (a) per-prompt levels
    ax = axes[0]
    x = np.arange(len(order))
    ax.plot(x, base.loc[order], "o-", color=C_EMPTY, lw=1.4, ms=5,
            label="empty negative", zorder=4)
    for col, col_c, lab in [("y8_neg_content4", C_CONT, "content ×4"),
                            ("y7_neg", C_QUAL, "quality ×4")]:
        if col not in E:
            continue
        ax.plot(x, E.loc[order, col], "s--", color=col_c, lw=1.3, ms=4.5,
                alpha=.9, label=lab)
    ax.axhline(0, color="#999", lw=.8)
    split = 3.5
    ax.axvline(split, color="#bbb", lw=1.0, ls=":")
    lo = ax.get_ylim()[0]
    ytxt = lo + (ax.get_ylim()[1] - lo) * .04
    ax.text(split - .2, ytxt, "original 4", ha="right", fontsize=7.8,
            color="#666", va="bottom")
    ax.text(split + .2, ytxt, "fresh 4", ha="left", fontsize=7.8,
            color="#666", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("obj", "") for p in order], fontsize=8)
    ax.set_xlabel("prompt id")
    ax.set_ylabel("κ  (grid estimator)")
    ax.set_title("(a) per-prompt κ on all eight prompts", fontsize=9.8)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left",
              bbox_to_anchor=(.01, .90))
    ax.spines[["top", "right"]].set_visible(False)

    # ---- (b) differences, split by prompt group
    ax = axes[1]
    groups = [("original 4", [p for p in order if p <= "obj03"]),
              ("fresh 4", [p for p in order if p >= "obj04"])]
    arms = [("y8_neg_content4", "content ×4", C_CONT),
            ("y7_neg", "quality ×4", C_QUAL)]
    xs, labels = [], []
    for gi, (gname, sel) in enumerate(groups):
        for ai, (col, lab, col_c) in enumerate(arms):
            if col not in W:
                continue
            v = W.loc[sel, col].to_numpy()
            pos = gi * 2.6 + ai * 1.0
            ax.bar(pos, v.mean(), .62, color=col_c, alpha=.8, zorder=2)
            ax.scatter(np.full(len(v), pos), v, s=22, color="#222", alpha=.6,
                       zorder=4, lw=0)
            ax.text(pos, v.mean() + (.012 if v.mean() >= 0 else -.012),
                    f"{v.mean():+.3f}", ha="center",
                    va="bottom" if v.mean() >= 0 else "top", fontsize=7.6,
                    color=col_c)
            xs.append(pos)
            labels.append(lab.replace(" ×4", ""))
        ax.set_ylim(-.30, None)
        ax.text(gi * 2.6 + .5, -.265, gname, ha="center",
                fontsize=8.2, color="#444")
    ax.axhline(0, color="#444", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δκ vs empty negative")
    ax.set_title("(b) the fresh four decide it — and content wins there",
                 fontsize=9.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(HERE, "fig_y8_content8p.png")
    fig.savefig(p, dpi=190)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
