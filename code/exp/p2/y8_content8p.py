"""Does the *content* negative prompt also vanish on the fresh four prompts?

Y8 found that four concrete nouns ("cat, dog, tree, car") lower kappa exactly as
much as the four-term quality string on obj00-obj03 (-.106 vs -.132, p=.59), and
that a different quality string does not (-.021).  That killed "it is the quality
vocabulary".  The remaining candidate is "it is whether the unconditional branch
is empty".

But obj00-obj03 are also the only prompts where the effect exists at all: Y7p8
showed the quality string does nothing on obj04-obj07 (-.030 grid, -.000 local).
So the two candidate mechanisms and the prompt-dependence are confounded, and
this run separates them: the content arm on the same eight prompts.

  content effect on obj00-03 and obj04-07 both present  -> "non-empty branch" is
        a mechanism, and Y7p8's null is specific to the quality string
  content effect on obj00-03 only                        -> the effect is
        prompt-specific, not operator-specific; the paper reports the direction
        as an observation and stops there

Same grid, same estimator, same metric as y8_neg_kind.py, so the numbers stack.

Run:  python y8_content8p.py
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

ARMS = [("y2_full", "empty"),
        ("y8_neg_content4", "content x4"),
        ("y7_neg", "quality x4")]


def per_prompt(tag):
    D = YA.load(tag)
    D = D[(D.seed == 11) & D.w.isin(W5) & D.s.isin(S9)]
    out = {}
    for est in ("grid", "local"):
        K = YA.kappa_cells(D, method=est, val=VAL)
        if len(K):
            out[est] = K.set_index("prompt_id").kappa
    return out


def main():
    tabs = {t: per_prompt(t) for t, _ in ARMS}
    n = {t: len(v.get("grid", [])) for t, v in tabs.items()}
    print("prompts per arm: " + "  ".join(f"{t}={n[t]}" for t, _ in ARMS))
    base = tabs["y2_full"]

    rows = []
    for est in ("grid", "local"):
        print(f"\n--- {est} ---")
        for tag, lab in ARMS[1:]:
            if est not in tabs[tag] or est not in base:
                continue
            x = tabs[tag][est]
            b = base[est]
            idx = [p for p in x.index if p in b.index]
            d = (x.loc[idx] - b.loc[idx])
            T = pd.DataFrame({"prompt_id": idx, "arm": tag,
                              f"kappa_empty": b.loc[idx].to_numpy(),
                              f"kappa_arm": x.loc[idx].to_numpy(),
                              "diff": d.to_numpy()})
            rows.append(T.assign(estimator=est))
            print(f"  {lab:<12} n={len(idx)}  mean κ {x.loc[idx].mean():.3f}  "
                  f"vs empty {d.mean():+.3f}")
            for gname, sel in [("obj00-03", [p for p in idx if p <= "obj03"]),
                               ("obj04-07", [p for p in idx if p >= "obj04"]),
                               ("all", idx)]:
                if len(sel) < 3:
                    continue
                dd = d.loc[sel]
                tt = stats.ttest_rel(x.loc[sel], b.loc[sel])
                print(f"      {gname:<9} n={len(sel)}  Δ {dd.mean():+.3f}  "
                      f"t p={tt.pvalue:.3f}  negative {int((dd < 0).sum())}/{len(sel)}")
    C = pd.concat(rows) if rows else pd.DataFrame()
    p = os.path.join(OUT, "an_y8_content8p.csv")
    C.to_csv(p, index=False)
    print(f"\n-> {p}")

    # the decisive cross-arm check on the fresh prompts only
    if "grid" in tabs["y8_neg_content4"] and "grid" in tabs["y7_neg"]:
        for est in ("grid", "local"):
            if est not in tabs["y8_neg_content4"] or est not in tabs["y7_neg"]:
                continue
            a = tabs["y8_neg_content4"][est]
            q = tabs["y7_neg"][est]
            idx = [p_ for p_ in a.index if p_ in q.index and p_ >= "obj04"]
            if len(idx) >= 3:
                dd = a.loc[idx] - q.loc[idx]
                print(f"  [fresh prompts] content - quality ({est}): "
                      f"{dd.mean():+.3f}  p="
                      f"{stats.ttest_rel(a.loc[idx], q.loc[idx]).pvalue:.3f}")


if __name__ == "__main__":
    main()
