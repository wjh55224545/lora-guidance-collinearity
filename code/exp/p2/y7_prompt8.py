"""Y7 with 8 prompts: was the negative-prompt effect under-powered or absent?

The dose ladder gave neg-vs-empty = -0.10 to -0.13 with a paired t of p=.13 over
4 prompts.  With 4 pairs a "p=.13" is uninformative: it is one prompt away from
significance either way.  This reruns the two rungs that carry the headline claim
(empty vs the 4-term negative prompt) on prompts obj00-obj07 -- all "object"
class, so pclass and the measured rho stay fixed -- and then does three tests:

    all 8 prompts      the pooled paired test
    obj00-03           reproduces the original 4, as a sanity check
    obj04-07           four prompts never used in any operator comparison, so
                       this is an independent replication rather than more of
                       the same

Same grid as the ladder (5 w x 9 s, seed 11), same metric (pix_l2), same
estimators.  If the effect is real, obj04-07 must show it too; if obj04-07 is
flat or reversed, the honest conclusion is that the effect is prompt-conditional
and 4 prompts were never enough to say anything.
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
W5 = [0.2, 0.6, 1.0, 1.5, 2.0]
S9 = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]
VAL = "pix_l2"


def per_prompt(tag):
    D = YA.load(tag)
    D = D[(D.seed == 11) & D.w.isin(W5) & D.s.isin(S9)]
    out = {}
    for est in ("grid", "local"):
        K = YA.kappa_cells(D, method=est, val=VAL)
        if len(K):
            out[est] = K.set_index("prompt_id").kappa
    return out, len(D)


def main():
    a, na = per_prompt("y2_full")
    b, nb = per_prompt("y7_neg")
    print(f"empty: {na} images, {len(a['grid'])} prompts   "
          f"neg: {nb} images, {len(b['grid'])} prompts")

    rows = []
    for est in ("grid", "local"):
        x, y = a[est], b[est]
        idx = [p for p in x.index if p in y.index]
        x, y = x.loc[idx], y.loc[idx]
        T = pd.DataFrame({"empty": x, "neg": y})
        # T.empty is the DataFrame attribute (a bool), not the column --
        # using it here silently makes diff == neg.  Always bracket.
        T["diff"] = T["neg"] - T["empty"]
        T["estimator"] = est
        rows.append(T.assign(estimator=est).reset_index().rename(
            columns={"index": "prompt_id"}))
        print(f"\n--- {est} ---")
        print("    " + T.round(3).to_string().replace("\n", "\n    "))
        groups = {"all": idx,
                  "obj00-03 (original 4)": [p for p in idx if p <= "obj03"],
                  "obj04-07 (new 4)": [p for p in idx if p >= "obj04"]}
        for name, sel in groups.items():
            if len(sel) < 3:
                continue
            d = T.loc[sel, "diff"]
            tt = stats.ttest_rel(T.loc[sel, "neg"], T.loc[sel, "empty"])
            w = stats.wilcoxon(T.loc[sel, "neg"], T.loc[sel, "empty"]) \
                if len(sel) >= 5 else None
            print(f"    {name:<22} n={len(sel)}  mean diff {d.mean():+.3f}  "
                  f"t p={tt.pvalue:.3f}" +
                  (f"  wilcoxon p={w.pvalue:.3f}" if w else "") +
                  f"  negative in {int((d < 0).sum())}/{len(sel)}")
    C = pd.concat(rows)
    p = os.path.join(OUT, "an_y7_prompt8.csv")
    C.to_csv(p, index=False)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
