"""Y7 dose: does kappa keep falling as the negative prompt gets longer?

Why this file exists.  The two dose runs (y7_neg_short / y7_neg_long) were run on
a deliberately small grid -- 5 w x 5 s x 4 prompts x 1 seed = 100 images -- while
y7_neg (the 4-term string) used the default 8 w x 9 s x 4 prompts x 2 seeds = 576.
Comparing those two kappas directly would confound "the operator got more
negative" with "the grid got coarser", and the grid resolution is known to move
kappa (Y2 / Y2-A: dense vs coarse s grid moves the estimate).

So every operator is re-scored on the SAME subset:

    seed 11, w in {0.2, 0.6, 1.0, 1.5, 2.0}, s in {5, 6.5, 7.5, 8.5, 10}

which is exactly the grid the two dose runs used.  The naive baseline (y2_full)
and the variance-normalised run (y7_rescale) are subset the same way, so the
whole operator ladder becomes one paired comparison:

    naive  ->  rescale (no negation, magnitude removed)  ->  1 word  ->  4 words  ->  10 words

What is being asked (written before seeing the numbers):
  monotone decrease in word count   -> report kappa as decreasing in "how much
                                       the unconditional branch is specified"
  a step at empty vs non-empty      -> the effect is *whether* there is a
                                       negative prompt, not how strong; then the
                                       paper reports two levels, not a dose
  no ordering                       -> the y7_neg result was noise at n=4 prompts

Metric: pix_l2, NOT pix_l2_base.  These runs are --mode single with the base
reference taken at w = W0 = 1.0, so pix_l2_base is zero *at w = 1.0* (checked:
min .919 at w=1.0, rising to 16.5 at w=2.0) -- feeding it to an argmin over s
throws away the middle of the w axis and inflates kappa from .185 to .306 on the
same rows.  pix_l2 is also what an_y7_* and an_y2_* were computed with, so this
keeps the ladder on the same scale as the published operator table.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import y_analyze as YA  # noqa: E402

OUT = os.path.join(HERE, "out_y")

W_SUB = [0.2, 0.6, 1.0, 1.5, 2.0]
# the s values every rung can be scored at: y_gen's default S_GRID.  The first
# attempt used a finer 5-point grid (5..10, spacing 1.5), but on 5 s points the
# local estimator returns sd .67 for the naive baseline -- it is not identifiable
# there -- so the ladder moved to the resolution the operators already used.
S_SUB = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]
SEED = 11

VAL = "pix_l2"

OPS = [("y2_full", "naive CFG, empty negative", 0),
       ("y7_neg_short_9s", "'blurry'", 1),
       ("y7_neg", "'blurry, low quality, watermark, text'", 4),
       ("y7_neg_long_9s", "10-term string", 10)]


def subset(D):
    m = (D.seed == SEED) & D.w.isin(W_SUB) & D.s.isin(S_SUB)
    return D[m].copy()


def main():
    rows = []
    for tag, label, nword in OPS:
        p = os.path.join(OUT, tag, "images.csv")
        if not os.path.exists(p):
            print(f"[miss] {tag}")
            continue
        D = subset(YA.load(tag))
        seen_w = sorted(D.w.unique())
        seen_s = sorted(D.s.unique())
        if len(seen_w) != len(W_SUB) or len(seen_s) != len(S_SUB):
            print(f"[skip] {tag}: subset has {len(seen_w)}w x {len(seen_s)}s, "
                  f"not the {len(W_SUB)}x{len(S_SUB)} ladder grid")
            continue
        out = dict(tag=tag, operator=label, n_neg_words=nword, n_img=len(D))
        for est in ("grid", "local"):
            K = YA.kappa_cells(D, method=est, val=VAL)
            if not len(K):
                continue
            out[f"k_{est}"] = K.kappa.mean()
            out[f"k_{est}_med"] = K.kappa.median()
            out[f"k_{est}_sd"] = K.kappa.std(ddof=1)
            if est == "local":
                # shape of s*(w): does the minimum drift, and by how much
                cur = YA.s_star_curve(D, "local", VAL)
                w = np.array([c[0] for c in cur], float)
                ss = np.array([c[1] for c in cur], float)
                out["sstar_at_wmin"] = float(ss[np.argmin(w)])
                out["sstar_at_wmax"] = float(ss[np.argmax(w)])
                out["sstar_drop"] = float(1 - ss[np.argmax(w)] / ss[np.argmin(w)])
                out["sstar_min"] = float(ss.min())
                out["sstar_max"] = float(ss.max())
                # is the argmin pinned to a grid edge?  (the "fake flat" trap)
                out["edge_share"] = float(np.mean((ss <= S_SUB[0] + 1e-9) |
                                                  (ss >= S_SUB[-1] - 1e-9)))
        rows.append(out)

    A = pd.DataFrame(rows)
    cols = ["tag", "operator", "n_neg_words", "n_img", "k_grid", "k_grid_med",
            "k_local", "k_local_med", "k_local_sd", "sstar_drop", "edge_share"]
    if not len(A):
        raise SystemExit("no rung had a complete ladder grid -- run the y7_neg_*_9s "
                         "segments first")
    A = A[[c for c in cols if c in A.columns]]
    p = os.path.join(OUT, "an_y7_dose.csv")
    A.to_csv(p, index=False)

    pd.set_option("display.width", 180)
    print(f"\n=== Y7 dose ladder, same {len(W_SUB)}w x {len(S_SUB)}s subset (seed 11) ===")
    print(A.round(3).to_string(index=False))

    lat = A.sort_values("n_neg_words")
    if len(lat) >= 3:
        for est in ("k_grid", "k_local"):
            if est not in lat:
                continue
            r = np.corrcoef(lat.n_neg_words, lat[est])[0, 1]
            print(f"\n  corr(word count, {est}) = {r:+.3f}")
            v = lat[est].to_numpy()
            drops = np.diff(v)
            print(f"  step sizes along the ladder: {np.round(drops, 3)}  "
                  f"(all same sign -> {'monotone' if np.all(drops < 0) or np.all(drops > 0) else 'NOT monotone'})")
            e0 = v[lat.n_neg_words.to_numpy() == 0]
            e1 = v[lat.n_neg_words.to_numpy() > 0]
            if len(e0) and len(e1):
                print(f"  mean |empty| {e0.mean():.3f} vs non-empty {e1.mean():.3f}  "
                      f"gap {e1.mean()-e0.mean():+.3f};  "
                      f"spread *within* non-empty {e1.max()-e1.min():.3f}")
    # ---- per-prompt table + paired tests.  n=4 prompts is small, so the
    # aggregate means above are not the whole story: a rung can look lower on
    # average because three prompts dropped and one rose.  This block is what
    # the writeup quotes.
    cells = {}
    for tag, label, nword in OPS:
        q = os.path.join(OUT, tag, "images.csv")
        if not os.path.exists(q):
            continue
        D = subset(YA.load(tag))
        if len(D) < len(W_SUB) * len(S_SUB) * 3:
            continue
        for est in ("grid", "local"):
            K = YA.kappa_cells(D, method=est, val=VAL)
            if len(K):
                for _, r in K.iterrows():
                    cells[(nword, est, r.prompt_id)] = r.kappa
    if cells:
        nws = sorted({k[0] for k in cells})
        proms = sorted({k[2] for k in cells})
        Ct = pd.DataFrame([{**{"n_neg_words": nw, "estimator": est,
                               "prompt_id": pr},
                            "kappa": cells.get((nw, est, pr), np.nan)}
                           for nw in nws for est in ("grid", "local")
                           for pr in proms])
        Ct.to_csv(os.path.join(OUT, "an_y7_dose_cells.csv"), index=False)
        for est in ("grid", "local"):
            E = Ct[Ct.estimator == est].pivot_table(index="prompt_id",
                                                    columns="n_neg_words",
                                                    values="kappa")
            print(f"\n  per-prompt kappa, {est}:")
            print("    " + E.round(3).to_string().replace("\n", "\n    "))
            base = E[nws[0]]
            for nw in nws[1:]:
                d = (E[nw] - base).dropna()
                if len(d) < 3:
                    continue
                tt = stats.ttest_rel(E[nw].dropna(), base.dropna())
                print(f"    {int(nw):>2} vs 0 words: mean {d.mean():+.3f}  "
                      f"per-prompt {np.round(d.values, 3)}  paired t p={tt.pvalue:.2f}"
                      f"  ({int((d < 0).sum())}/{len(d)} negative)")

    print(f"\n-> {p}")
    print("  naive vs dense-grid value:  y2_full on its own 8x9 grid gives kappa "
          ".159/.165 (grid/local); the subset number above is what is comparable "
          "to the dose runs.")


if __name__ == "__main__":
    main()
