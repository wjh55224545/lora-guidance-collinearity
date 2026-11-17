"""Figure for Y8: is the negative-prompt effect the vocabulary or just a
non-empty unconditional branch?

Left   kappa for every rung on the same 5w x 9s subset, grid and local
       estimators side by side, with the four per-prompt values drawn as small
       markers.  The bars are means over four prompts and four prompts is not a
       lot -- showing the dots is the whole point, because the mean moves mostly
       by obj00/obj03 dropping while obj02 rises.
Right  the paired difference against the empty baseline, one line per prompt,
       for the three four-word arms (quality-original / quality-other / content).
       If the three arms sit at the same height, the cause is "the unconditional
       branch is non-empty"; if the content arm sits at zero while the two
       quality arms sit below, the cause is the quality vocabulary.

Colours: one hue per vocabulary family, grey for the baseline, so the 2x2
(presence x vocabulary) is readable without a legend lookup table.
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

ORDER = [("y2_full", "empty", "base"),
         ("y7_neg_short_9s", "quality x1", "qual"),
         ("y7_neg", "quality x4\n(original)", "qual"),
         ("y8_neg_quality4b", "quality x4\n(other words)", "qual2"),
         ("y8_neg_content4", "content x4", "cont"),
         ("y7_neg_long_9s", "quality x10", "qual")]
COL = {"base": "#6b6b6b", "qual": "#1f6fb4", "qual2": "#3aa0c8", "cont": "#d1622b"}
FOUR = [("y7_neg", "quality (original)", "qual"),
        ("y8_neg_quality4b", "quality (other words)", "qual2"),
        ("y8_neg_content4", "content nouns", "cont")]


def main():
    A = pd.read_csv(os.path.join(OUT, "an_y8_neg_kind.csv"))
    C = pd.read_csv(os.path.join(OUT, "an_y8_neg_cells.csv"))
    A = A.set_index("tag")
    tags = [t for t, *_ in ORDER if t in A.index]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.1),
                             gridspec_kw=dict(width_ratios=[1.45, 1]))

    # ---- (a) ladder
    ax = axes[0]
    x = np.arange(len(tags))
    W = 0.36
    for j, (est, off, hatch) in enumerate([("grid", -W / 2, ""),
                                           ("local", W / 2, "///")]):
        col = f"k_{est}"
        if col not in A:
            continue
        vals = A.loc[tags, col].to_numpy(float)
        colors = [COL[key] for tag_, _, key in ORDER if tag_ in tags]
        ax.bar(x + off, vals, W, color=colors, alpha=.85, hatch=hatch,
               edgecolor="#ffffff", linewidth=.6, zorder=2,
               label=f"{est} estimator")
        for xi, v in zip(x + off, vals):
            # tags are 0.36 apart; ".16" twice at the same height collides, so
            # push each label outward and give it an opaque pad
            dx = -.012 if off < 0 else .012
            ax.text(xi + dx, v + .010, f"{v:.2f}",
                    ha="right" if off < 0 else "left", va="bottom",
                    fontsize=7.2, color="#333",
                    bbox=dict(fc="white", ec="none", alpha=.75, pad=.8))
        # per-prompt dots
        for xi, t in zip(x + off, tags):
            d = C[(C.tag == t) & (C.estimator == est)]
            ax.scatter(np.full(len(d), xi), d.kappa, s=13, color="#222",
                       alpha=.55, zorder=5, lw=0)
    ax.axhline(0, color="#888", lw=.9)
    ax.set_xticks(x)
    ax.set_xticklabels([dict((t, l) for t, l, _ in ORDER)[t] for t in tags],
                       fontsize=7.6)
    ax.set_ylabel("κ  (log-log slope of s* on w)")
    ax.set_title("(a) same 5w × 9s subset, seed 11, pix_l2 — dots are the four prompts",
                 fontsize=9.6)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ---- (b) paired differences for the four-word arms
    ax = axes[1]
    base = C[(C.tag == "y2_full")].set_index(["estimator", "prompt_id"]).kappa
    xpos = np.arange(len(FOUR))
    for k, (tag, lab, key) in enumerate(FOUR):
        if tag not in set(C.tag):
            continue
        for est, mk in [("grid", "o"), ("local", "s")]:
            d = C[(C.tag == tag) & (C.estimator == est)].set_index("prompt_id").kappa
            idx = [p for p in d.index if (est, p) in base.index]
            if not idx:
                continue
            diff = d.loc[idx].to_numpy() - base.loc[[(est, p) for p in idx]].to_numpy()
            jit = np.linspace(-.10, .10, len(idx))
            ax.scatter(np.full(len(idx), k) + jit + (.03 if est == "local" else -.03),
                       diff, s=26, marker=mk, color=COL[key],
                       alpha=.85 if est == "grid" else .5, lw=.8,
                       facecolor=COL[key] if est == "grid" else "none")
            ax.plot([k - .26, k + .26], [diff.mean()] * 2, color=COL[key],
                    lw=2.2, solid_capstyle="butt")
            ax.text(k + .30, diff.mean(), f"{diff.mean():+.3f}", fontsize=7.6,
                    va="center", color=COL[key])
    ax.axhline(0, color="#444", lw=1.1)
    ax.set_xticks(xpos)
    ax.set_xticklabels([l for _, l, _ in FOUR if _ in set(C.tag)], fontsize=8)
    ax.set_xlim(-.55, len(FOUR) - .25)
    ax.set_ylabel("Δκ vs empty  (per prompt)")
    ax.set_title("(b) fixed at four words, only the vocabulary changes", fontsize=9.6)
    ax.text(.44, .02, "filled ○ = grid      hollow □ = local      bar = mean of 4",
            transform=ax.transAxes, fontsize=7.2, color="#444")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(HERE, "fig_y8_neg.png")
    fig.savefig(p, dpi=190)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
