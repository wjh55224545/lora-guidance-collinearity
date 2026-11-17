"""Y8: the negative-prompt effect -- is it the vocabulary, or just a non-empty
unconditional branch?

The dose ladder (Y7 dose) killed "more negative words -> weaker collinearity":
kappa went .206 (empty) / .101 (1 word) / .074 (4 words) / .158 (10 words), which
is not monotone and peaks back at the empty level for the longest string.  What
survives from Y7 is exactly one contrast -- empty vs 4 words, -0.10 to -0.13,
p=.13 at n=4, and still about -.10 at n=8.  That contrast has two separable
causes:

  (i)  the unconditional branch is no longer the empty string.  In CFG the LoRA
       acts on *both* branches, so specifying a negative prompt changes what the
       adapter's low-rank direction is being applied to, regardless of what the
       words mean.  If this is the cause, ANY four words do it.
  (ii) it is the quality-attribute vocabulary specifically ("blurry", "low
       quality") pointing CFG away from the region where w and s trade off.

Fix the word count at four and change only the vocabulary:

  y8_neg_content4     "cat, dog, tree, car"                        -> (i)
  y8_neg_quality4b    "grainy, jpeg artifacts, oversaturated, cartoon" -> (ii)
  y7_neg              "blurry, low quality, watermark, text"      (already run)

All rungs are scored on the same 5w x 9s grid, seed 11, metric pix_l2, estimators
grid and local -- identical to an_y7_dose.csv, so this file's table is a strict
extension of that one rather than a parallel experiment.

Read it as a 2x2:

    same vocabulary, fewer words   -> answered by the dose ladder (non-monotone)
    different vocabulary, same words -> THIS file.  content4 near .20 means (i),
                                        content4 near .10 means (ii) is what
                                        carries it, and the paper must then say
                                        "quality vocabulary", not "negative
                                        prompts".

n=4 prompts, so every cell also gets a per-prompt paired test; a rung that only
moves the mean without moving the per-prompt ranks is not written up.
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
S_SUB = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]
SEED = 11
VAL = "pix_l2"

OPS = [
    ("y2_full", "empty negative", "none", 0),
    ("y7_neg_short_9s", "'blurry'", "quality", 1),
    ("y7_neg", "'blurry, low quality, watermark, text'", "quality", 4),
    ("y8_neg_quality4b", "'grainy, jpeg artifacts, oversaturated, cartoon'",
     "quality", 4),
    ("y8_neg_content4", "'cat, dog, tree, car'", "content", 4),
    ("y7_neg_long_9s", "10-term quality string", "quality", 10),
]


def subset(D):
    return D[(D.seed == SEED) & D.w.isin(W_SUB) & D.s.isin(S_SUB)].copy()


def main():
    rows, cells = [], {}
    for tag, label, kind, nword in OPS:
        p = os.path.join(OUT, tag, "images.csv")
        if not os.path.exists(p):
            print(f"[miss] {tag}")
            continue
        D = subset(YA.load(tag))
        if (len(D.w.unique()) != len(W_SUB)) or (len(D.s.unique()) != len(S_SUB)):
            print(f"[skip] {tag}: {len(D.w.unique())}w x {len(D.s.unique())}s, "
                  f"not the {len(W_SUB)}x{len(S_SUB)} ladder grid")
            continue
        out = dict(tag=tag, operator=label, kind=kind, n_img=len(D))
        for est in ("grid", "local"):
            K = YA.kappa_cells(D, method=est, val=VAL)
            if not len(K):
                continue
            out[f"k_{est}"] = K.kappa.mean()
            out[f"k_{est}_sd"] = K.kappa.std(ddof=1)
            for _, r in K.iterrows():
                cells[(tag, est, r.prompt_id)] = r.kappa
        if "k_local" in out:
            cur = YA.s_star_curve(D, "local", VAL)
            w = np.array([c[0] for c in cur], float)
            ss = np.array([c[1] for c in cur], float)
            out["sstar_drop"] = float(1 - ss[np.argmax(w)] / ss[np.argmin(w)])
            out["edge_share"] = float(np.mean((ss <= S_SUB[0] + 1e-9) |
                                              (ss >= S_SUB[-1] - 1e-9)))
        rows.append(out)

    A = pd.DataFrame(rows)
    if not len(A):
        raise SystemExit("no rung available -- run the y8_neg_* segments first")
    A.to_csv(os.path.join(OUT, "an_y8_neg_kind.csv"), index=False)

    pd.set_option("display.width", 190)
    print(f"\n=== Y8 negative-prompt kind, same {len(W_SUB)}w x {len(S_SUB)}s "
          f"subset (seed 11, pix_l2) ===")
    print(A.round(3).to_string(index=False))

    # ---- per-prompt paired table + tests
    proms = sorted({k[2] for k in cells})
    tags = [t for t, *_ in OPS if any(k[0] == t for k in cells)]
    Ct = pd.DataFrame([{**{"tag": t, "estimator": est, "prompt_id": pr},
                        "kappa": cells.get((t, est, pr), np.nan)}
                       for t in tags for est in ("grid", "local") for pr in proms])
    Ct.to_csv(os.path.join(OUT, "an_y8_neg_cells.csv"), index=False)

    for est in ("grid", "local"):
        E = Ct[Ct.estimator == est].pivot_table(index="prompt_id", columns="tag",
                                                values="kappa")
        if "y2_full" not in E:
            continue
        print(f"\n  per-prompt kappa, {est} estimator:")
        print("    " + E.round(3).to_string().replace("\n", "\n    "))
        base = E["y2_full"]
        for t in tags:
            if t == "y2_full":
                continue
            # align first: rungs differ in how many prompts they cover (the
            # 5w x 9s ladder has 4, the dense-grid baseline has 8), so
            # dropping NaN on each side separately hands ttest_rel two
            # different-length vectors.
            d = (E[t] - base).dropna()
            if len(d) < 3:
                continue
            idx = d.index
            tt = stats.ttest_rel(E.loc[idx, t], base.loc[idx])
            w = stats.wilcoxon(E.loc[idx, t], base.loc[idx]) if len(d) >= 5 else None
            print(f"    {t:<20} vs empty: mean {d.mean():+.3f}  "
                  f"per-prompt {np.round(d.values, 3)}  t p={tt.pvalue:.3f}"
                  + (f"  wilcoxon p={w.pvalue:.3f}" if w else "")
                  + f"  ({int((d < 0).sum())}/{len(d)} negative)")
        # the decisive comparison: same word count, different vocabulary
        # two-way layout (rung x prompt) on the prompts every rung shares:
        # if the operator mattered, the rung term would carry the variance;
        # if the estimate is prompt-limited, the prompt term does.
        shared = [tp for tp in E.index
                  if E.loc[tp, [c for c in E.columns if c in tags]].notna().all()]
        if len(shared) >= 3:
            cols = [c for c in E.columns if c in tags]
            for name, keep in [("all rungs", cols),
                               ("non-empty only", [c for c in cols if c != "y2_full"])]:
                if len(keep) < 3:
                    continue
                sub = E.loc[shared, keep]
                y = sub.to_numpy(float).ravel()
                rung = np.tile(np.array(sub.columns), len(shared))
                prom = np.repeat(np.array(shared), sub.shape[1])
                ss_tot = ((y - y.mean()) ** 2).sum()

                def ss_of(grp):
                    out = 0.0
                    for g in np.unique(grp):
                        m = grp == g
                        out += m.sum() * (y[m].mean() - y.mean()) ** 2
                    return out

                ss_r, ss_p = ss_of(rung), ss_of(prom)
                print(f"    variance split ({name}, {len(shared)} prompts x "
                      f"{sub.shape[1]} rungs): rung R2={ss_r / ss_tot:.2f}  "
                      f"prompt R2={ss_p / ss_tot:.2f}  "
                      f"residual={1 - (ss_r + ss_p) / ss_tot:.2f}")
        if {"y8_neg_content4", "y7_neg"} <= set(E.columns):
            d = (E["y8_neg_content4"] - E["y7_neg"]).dropna()
            tt = stats.ttest_rel(E.loc[d.index, "y8_neg_content4"],
                                 E.loc[d.index, "y7_neg"])
            print(f"    content4 - quality4: mean {d.mean():+.3f}  "
                  f"t p={tt.pvalue:.3f}  ({int((d > 0).sum())}/{len(d)} "
                  f"content higher)")
        if {"y8_neg_quality4b", "y8_neg_content4"} <= set(E.columns):
            d = (E["y8_neg_quality4b"] - E["y8_neg_content4"]).dropna()
            tt = stats.ttest_rel(E.loc[d.index, "y8_neg_quality4b"],
                                 E.loc[d.index, "y8_neg_content4"])
            print(f"    quality4b - content4: mean {d.mean():+.3f}  "
                  f"t p={tt.pvalue:.3f}")

    print(f"\n-> {os.path.join(OUT, 'an_y8_neg_kind.csv')}")
    print("  reference: naive on its own dense grid gives .159/.165 "
          "(grid/local); only the subset numbers above are comparable.")


if __name__ == "__main__":
    main()
