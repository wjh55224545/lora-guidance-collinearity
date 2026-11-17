"""Y3 figure: with two adapters on, how much of the first one's leverage is left?

Three panels, ordered by how much they can be argued with:

  (a) the wA dose-response curve at wB = 0 and at wB = 1.5, each cell
      normalised by its own value at (wA = 1.5, wB = 0).  This is the raw shape
      the shrink number is computed from -- if the red curve is flat, the second
      adapter really has taken the first one's leverage away, and no fitting
      choice can hide that.
  (b) the shrink ratios, per prompt, with the 0.6 line that was written into the
      judgement *before* the run.  Two wA variants are shown because the 1.5
      endpoint sits on a flat or non-monotone tail in 3 of 12 cells: the full
      contrast (0 -> 1.5) and the interior contrast (0 -> 1.0, at wB = 0 vs 1.0).
      If both are below the line the conclusion does not depend on the endpoint.
  (c) joint versus sum.  Sub-additive (< 1) is "they fight"; super-additive
      would falsify the community advice outright.

All levels live on pix_l2_base, whose reference is generated at S0 = 7.5 for
every row, so a constant offset exists per s slice; every statistic here is a
difference or a ratio inside one slice, so the offset cancels.
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

SHRINK_LINE = 0.6      # written into the judgement before the grid ran
PROMPT_C = {"obj00": "#2c3e50", "obj01": "#c0392b",
            "obj02": "#1e8449", "obj03": "#8e44ad"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="y3")
    ap.add_argument("--out", default="fig_y3.png")
    a = ap.parse_args()
    D = pd.read_csv(os.path.join(OUT, a.tag, "images.csv"))
    D = D[D.w.notna() & D.w2.notna()]
    R = pd.read_csv(os.path.join(OUT, f"an_{a.tag}_additivity.csv"))

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14.6, 4.5),
                                  gridspec_kw=dict(width_ratios=[1, 1, 1]))

    # ---- (a) normalised wA profile -----------------------------------------
    ws = sorted(D.w.unique())
    wmax = max(D.w2.unique())
    prof = {0.0: [], wmax: []}
    for (s, pid), h in D.groupby(["s", "prompt_id"]):
        piv = h.pivot_table(index="w", columns="w2", values="pix_l2_base")
        piv = piv.reindex(index=ws, columns=sorted(D.w2.unique()))
        if piv.isna().any().any():
            continue
        ref = piv.loc[ws[-1], 0.0]
        if not np.isfinite(ref) or ref == 0:
            continue
        for wb in (0.0, wmax):
            prof[wb].append(piv[wb].values / ref)
    for wb, col, lab, ls in ((0.0, "#2c3e50", r"$w_B=0$  (B absent)", "-"),
                             (wmax, "#c0392b", rf"$w_B={wmax}$  (B at max)",
                              "-")):
        M = np.vstack(prof[wb])
        med = np.median(M, axis=0)
        lo, hi = np.percentile(M, [25, 75], axis=0)
        a.fill_between(ws, lo, hi, color=col, alpha=.13, lw=0)
        a.plot(ws, med, ls, marker="o", ms=5, color=col, lw=1.8, label=lab)
        for row in M:
            a.plot(ws, row, "-", color=col, lw=.55, alpha=.28)
    a.set_xlabel("$w_A$  (weight of the first adapter)")
    a.set_ylabel("drift, normalised to $w_A{=}1.5,\\,w_B{=}0$")
    a.set_title("(a) the first adapter's dose-response flattens", fontsize=10.5)
    a.set_ylim(top=3.7)
    a.legend(fontsize=8.5, loc="upper left", framealpha=.95)
    a.grid(alpha=.25)

    # ---- (b) shrink per prompt ---------------------------------------------
    pids = sorted(R.prompt_id.unique())
    x = np.arange(len(pids))
    wid = .26
    for k, (col, lab) in enumerate((("wA_shrink", r"$w_A$  $0\to1.5$"),
                                    ("wA_shrink_int", r"$w_A$  $0\to1.0$"),
                                    ("wB_shrink", r"$w_B$  $0\to1.5$"))):
        v = [R[R.prompt_id == p][col].median() for p in pids]
        b.bar(x + (k - 1) * wid, v, wid, color=["#1f3864", "#5b7db1",
                                                "#95a5a6"][k],
              edgecolor="white", lw=.6, label=lab)
    b.axhline(SHRINK_LINE, color="#c0392b", lw=1.2, ls="--",
              label=f"pre-registered line {SHRINK_LINE}")
    b.axhline(1.0, color="#2c3e50", lw=1.0, ls=":")
    b.axhline(0, color="#2c3e50", lw=.9)
    b.set_xticks(x)
    b.set_xticklabels(pids, fontsize=9)
    b.set_ylabel("leverage of this axis at max of the other / at zero")
    b.set_title("(b) how much leverage survives (median over $s$)", fontsize=10.5)
    b.set_ylim(-0.35, 1.28)
    b.legend(fontsize=7.6, loc="upper center", ncol=2, framealpha=.92)
    b.grid(axis="y", alpha=.25)
    for i, p in enumerate(pids):
        neg = (R[(R.prompt_id == p)].wA_slope_wBmax < 0).sum()
        if neg:
            b.annotate("non-monotone\nat 1.5", (i - wid, 0), fontsize=7,
                       color="#c0392b", ha="center", va="bottom",
                       xytext=(0, 6), textcoords="offset points")

    # ---- (c) joint vs sum --------------------------------------------------
    ss = sorted(R.s.unique())
    wid2 = .8 / len(pids)
    for i, p in enumerate(pids):
        v = [R[(R.prompt_id == p) & (R.s == s)].joint_over_sum.iloc[0]
             for s in ss]
        c.bar(np.arange(len(ss)) + (i - 1.5) * wid2, v, wid2,
              color=PROMPT_C.get(p, "#7f8c8d"), edgecolor="white", lw=.6,
              label=p)
    c.axhline(1.0, color="#2c3e50", lw=1.2, ls="--")
    c.annotate("1 = exactly additive", (len(ss) - .45, 1.0), fontsize=8,
               color="#2c3e50", ha="right", va="bottom")
    c.set_xticks(np.arange(len(ss)))
    c.set_xticklabels([f"s={s:g}" for s in ss], fontsize=9)
    c.set_ylim(0, 1.32)
    c.set_ylabel(r"$D(1,1)\,/\,[D(1,0)+D(0,1)]$")
    c.set_title("(c) both at full strength vs the sum", fontsize=10.5)
    c.legend(fontsize=8, ncol=4, loc="upper center", framealpha=.92)
    c.grid(axis="y", alpha=.25)

    fig.suptitle("Y3  two adapters loaded at once:  additivity of the drift, "
                 "and who loses leverage", fontsize=11.5, y=1.01)
    fig.tight_layout()
    p = os.path.join(OUT, a.out)
    fig.savefig(p, dpi=190, bbox_inches="tight")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
