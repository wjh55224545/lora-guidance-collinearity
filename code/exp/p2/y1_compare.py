"""Y1 compare: is kappa a property of *a* LoRA or of LoRAs?

Y1 exists for one reason.  Everything reported so far comes from a single
adapter (`full`, made by E2c) or from surgical variants of it, so "kappa = 0.20"
could be a number about that one checkpoint.  Two independently trained
adapters (A, B -- same prompt pair style, independent seeds and step streams)
give the first honest test.

The judgement is written down before the grids finish (research log 15.2):
    * kappa(A) and kappa(B) both inside [0.11, 0.27]  -> cross-training robust,
      the interval may be quoted as a property of the fine-tuning procedure
    * A and B differ by more than the surgical variants differ from `full`
      -> kappa moves with training; only "within one LoRA" may be claimed

Two panels, because the second one is what makes the first believable:
  (a) kappa per adapter (and per surgical variant, greyed) with bootstrap CI,
      against the published band and the kappa = 1 line that "equal w*s" needs
  (b) s*(w) curves overlaid -- if the three adapters produce the same *shape*
      with a different level, the collinearity is structural; if the shapes
      differ, the level agreement in (a) is a coincidence of averaging.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y")

# what is already written into the draft; the band is what Y6 narrowed it to
BAND = (0.16, 0.28)
PUBLISHED = 0.211

INSTANCES = [("y2_full", "full\n(E2c, r=8)"), ("y1_A", "A\n(own seed)"),
             ("y1_B", "B\n(own seed)"), ("y2_A", "A\ndense-s")]


def load_est(tag, est="local"):
    p = os.path.join(OUT, f"an_{tag}_{est}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def sstar_by_w(D):
    """Median s*(w) over cells, same local three-point estimator as the kappas."""
    sys.path.insert(0, HERE)
    import y_analyze as YA
    per = {}
    for (_pid, _sd), h in D.groupby(["prompt_id", "seed"]):
        for w, s, _e in YA.s_star_curve(h, "local"):
            per.setdefault(w, []).append(s)
    w = np.array(sorted(per), float)
    s = np.array([np.median(per[k]) for k in w], float)
    return pd.DataFrame(dict(w=w, s=s))


def boot_mean_ci(v, n=4000, seed=0):
    """CI for the *mean over cells*, not the spread of single cells.

    The per-cell kappa already carries a regression CI, but what the paper
    quotes is a mean across prompts/seeds, so the relevant uncertainty is the
    between-cell one.  Resampling the 8 cells is the cheap honest version.
    """
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rows, curves, cells = [], {}, []
    for tag, label in INSTANCES:
        K = load_est(tag)
        if K is None or not len(K):
            continue
        lo, hi = boot_mean_ci(K.kappa.values)
        rows.append(dict(tag=tag, label=label, n=len(K),
                         kappa=K.kappa.mean(), sd=K.kappa.std(ddof=1),
                         lo=lo, hi=hi))
        cells.append(K.assign(inst=label.split("\n")[0]))
        p = os.path.join(OUT, tag, "images.csv")
        if os.path.exists(p):
            curves[label.replace("\n", " ")] = sstar_by_w(pd.read_csv(p))
    T = pd.DataFrame(rows)
    if not len(T):
        print("no tags analysed yet")
        return
    T.to_csv(os.path.join(OUT, "an_y1_compare.csv"), index=False)
    print(T.round(3).to_string(index=False))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.4),
                               gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = a
    ax.axhspan(*BAND, color="#cfe3f7", alpha=.55, zorder=0,
               label=f"draft band [{BAND[0]:.2f}, {BAND[1]:.2f}]")
    ax.axhline(PUBLISHED, color="#2b6cb0", lw=1.1, ls="--",
               label=f"published $\\kappa$ = {PUBLISHED:.3f}")
    ax.axhline(1.0, color="#c0392b", lw=1.0, ls=":",
               label=r"$\kappa$ = 1 (equal $w\cdot s$)")
    ys = np.arange(len(T))[::-1]
    rng = np.random.default_rng(1)
    for y, r in zip(ys, T.itertuples()):
        cs = [c.kappa.values for c in cells if c.inst.iloc[0] == r.tag or
              c.inst.iloc[0] == r.label.split("\n")[0]]
        if cs:
            jit = rng.normal(0, .06, len(cs[0]))
            ax.plot(cs[0], y + jit, "o", ms=3.4, color="#8fa5c4", alpha=.75,
                    zorder=2)
    ax.errorbar(T.kappa, ys, xerr=[T.kappa - T.lo, T.hi - T.kappa],
                fmt="o", color="#1f3864", ms=8, capsize=4, lw=1.6, zorder=3)
    for y, r in zip(ys, T.itertuples()):
        ax.annotate(f"{r.kappa:.3f}", (r.kappa, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels(T.label, fontsize=9)
    ax.set_xlim(0, 1.22)
    ax.set_xlabel(r"$\kappa$  (per adapter; grey = cells, bar = 95% boot CI of mean)")
    ax.set_title("(a) one adapter, or adapters in general?", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right", framealpha=.9)
    ax.grid(axis="x", alpha=.25)

    for i, (k, S) in enumerate(curves.items()):
        c = ["#2c3e50", "#c0392b", "#1e8449", "#8e44ad"][i % 4]
        b.plot(S.w, S.s, "-o", color=c, ms=4, lw=1.5, label=k)
    b.set_xlabel("$w$  (adapter weight)")
    b.set_ylabel("$s^*$  (guidance at minimum drift)")
    b.set_title("(b) same shape?  $s^*(w)$ per adapter", fontsize=10.5)
    b.legend(fontsize=8.5)
    b.grid(alpha=.25)

    fig.suptitle("Y1  cross-training check of the collinearity slope",
                 fontsize=11.5, y=1.00)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_y1_compare.png")
    fig.savefig(p, dpi=190, bbox_inches="tight")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
