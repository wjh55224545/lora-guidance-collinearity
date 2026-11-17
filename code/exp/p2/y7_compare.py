"""Y7 compare: is the collinearity about (w, s), or about *this* CFG variant?

Everything reported so far lives inside one guidance operator: plain CFG with
an empty negative prompt.  That is one line of the pipeline, and it is the line
a reviewer of an image-generation paper pokes first.  Two alternative operators
on the same grid, same prompts, same seeds, same anchors:

    y7_rescale   variance-normalised CFG (guidance_rescale = 0.7).  The guided
                 prediction's std is rescaled back to the conditional one, so by
                 construction the *magnitude* channel through which a stronger
                 adapter inflates guidance is removed.
    y7_neg       a real negative prompt, so the unconditional branch is no
                 longer the empty string and the adapter also acts on the
                 "what not to draw" side.

Judgement written before seeing the numbers
-------------------------------------------
  rescale  kappa near 0 and a flat s*(w)  ->  the degeneracy belongs to
           *unnormalised* CFG.  The paper's claim has to be scoped to standard
           CFG, and the practical advice gains a second half: the fix removes
           the degeneracy together with the control.
  rescale  kappa unchanged                ->  the degeneracy is in the (w, s)
           pair itself, independent of how the guidance gap is assembled.  That
           is the stronger version of the claim.
  neg      kappa inside [0.13, 0.25]      ->  operator-robust; the interval may
           be quoted without naming CFG.
  neg      kappa moves > 0.05             ->  report per operator, do not pool.

Panel (b) is the reason not to stop at kappa: a rescale can flatten s*(w) by
pushing the minimum to the edge of the grid, which would masquerade as "no
collinearity".  D_min(w) in panel (c) is what distinguishes the two.
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
sys.path.insert(0, HERE)

OPS = [("y2_full", "plain CFG\n(dense $s$)", "#1f3864", "-"),
       ("y7_rescale", "CFG + rescale 0.7", "#c0392b", "--"),
       ("y7_neg", "CFG + negative prompt", "#1e8449", ":")]


def boot_mean_ci(v, n=4000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sstar_and_dmin(D):
    """Median s*(w) and D_min(w) over cells, local estimator."""
    import y_analyze as YA
    per, dmin = {}, {}
    for (_pid, _sd), h in D.groupby(["prompt_id", "seed"]):
        for w, s, _e in YA.s_star_curve(h, "local"):
            per.setdefault(w, []).append(s)
        for w, g in h.groupby("w"):
            if w == 0.0:
                continue
            dmin.setdefault(w, []).append(g.pix_l2.min())
    w = np.array(sorted(per), float)
    S = pd.DataFrame(dict(w=w,
                          s=[np.median(per[k]) for k in w],
                          dmin=[np.median(dmin.get(k, [np.nan])) for k in w]))
    return S


def main():
    rows, curves = [], {}
    for tag, label, col, ls in OPS:
        p = os.path.join(OUT, tag, "images.csv")
        if not os.path.exists(p):
            print(f"  [skip] {tag}: not generated yet")
            continue
        D = pd.read_csv(p)
        for est in ("grid", "local"):
            q = os.path.join(OUT, f"an_{tag}_{est}.csv")
            if not os.path.exists(q):
                print(f"  [skip] {tag}/{est}: run y_analyze.py --tag {tag}")
                continue
            K = pd.read_csv(q)
            lo, hi = boot_mean_ci(K.kappa.values)
            rows.append(dict(tag=tag, op=label.split("\n")[0], estimator=est,
                             n=len(K), kappa=K.kappa.mean(),
                             sd=K.kappa.std(ddof=1), lo=lo, hi=hi,
                             r2_med=K.r2.median()))
        S = sstar_and_dmin(D)
        curves[label.split("\n")[0]] = (S, col, ls)

    T = pd.DataFrame(rows)
    if not len(T):
        print("nothing to compare")
        return
    T.to_csv(os.path.join(OUT, "an_y7_operators.csv"), index=False)
    print(T.round(3).to_string(index=False))

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14.2, 4.3))
    ys = {op: i for i, op in enumerate(dict.fromkeys(T.op))}
    for est, off, mk in (("grid", +.16, "s"), ("local", -.16, "o")):
        d = T[T.estimator == est]
        y = np.array([ys[o] for o in d.op]) + off
        a.errorbar(d.kappa, y, xerr=[d.kappa - d.lo, d.hi - d.kappa],
                   fmt=mk, color="#1f3864" if est == "local" else "#8fa5c4",
                   ms=8, capsize=4, lw=1.5, label=est)
    a.axvspan(0.16, 0.28, color="#cfe3f7", alpha=.55, label="draft band")
    a.axvline(1.0, color="#c0392b", ls=":", lw=1, label=r"$\kappa$=1")
    a.set_yticks(list(ys.values()))
    a.set_yticklabels(list(ys.keys()), fontsize=9)
    a.set_xlim(0, 1.22)
    a.set_xlabel(r"$\kappa$")
    a.set_title("(a) kappa under each guidance operator", fontsize=10)
    a.legend(fontsize=8, loc="lower right")
    a.grid(axis="x", alpha=.25)

    for name, (S, col, ls) in curves.items():
        b.plot(S.w, S.s, ls, marker="o", ms=4, color=col, lw=1.6, label=name)
        c.plot(S.w, S.dmin, ls, marker="o", ms=4, color=col, lw=1.6, label=name)
    b.set_xlabel("$w$")
    b.set_ylabel("$s^*$  (median over cells)")
    b.set_title("(b) does the optimum stay interior?", fontsize=10)
    b.grid(alpha=.25); b.legend(fontsize=8)
    c.set_xlabel("$w$")
    c.set_ylabel("$D_{min}$  (best pix_l2 at that $w$)")
    c.set_title("(c) within-operator $D_{min}$\n(levels are NOT comparable across"
                " operators)", fontsize=10)
    # the curves climb to ~10 on the right, so the legend goes to the empty
    # lower-right corner rather than over the last two points
    c.set_ylim(1.4, 10.9)
    c.grid(alpha=.25); c.legend(fontsize=8, loc="lower right", framealpha=.93)
    fig.suptitle("Y7  the collinearity is scoped by the guidance operator",
                 fontsize=11.5)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_y7.png")
    fig.savefig(p, dpi=190, bbox_inches="tight")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
