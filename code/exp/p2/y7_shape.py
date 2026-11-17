"""Y7 shape table: s*(w) and D_min(w) for the three guidance operators.

The kappa forest says *whether* the degeneracy changes with the operator.  This
saves the two curves that say *how*: where the drift-minimising guidance sits at
each adapter weight, and how deep that minimum is.

Two caveats that travel with these numbers:

  * D_min levels are NOT comparable across operators.  Each operator generates
    its own no-LoRA reference image, so the absolute distance to it carries an
    operator-specific offset.  Only the *shape* over w is comparable.
  * a "flat" s*(w) can be an artefact of the minimum being pushed to the edge of
    the s grid.  The edge share column is the check; it is .04 for all three.

Run:  python y7_shape.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import y_analyze as YA

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y")
OPS = ["y2_full", "y7_rescale", "y7_neg"]


def main():
    rows = []
    for tag in OPS:
        p = os.path.join(OUT, tag, "images.csv")
        if not os.path.exists(p):
            print(f"[skip] {tag}")
            continue
        D = pd.read_csv(p)
        per, dmin = {}, {}
        for (_pid, _sd), h in D.groupby(["prompt_id", "seed"]):
            for w, s, _e in YA.s_star_curve(h, "local"):
                per.setdefault(w, []).append(s)
            for w, g in h.groupby("w"):
                if w == 0.0:
                    continue
                dmin.setdefault(w, []).append(g.pix_l2.min())
        alls = [s for w in per for s in per[w]]
        lo, hi = min(alls), max(alls)
        edge = float(np.mean([s in (lo, hi) for s in alls]))
        for w in sorted(per):
            rows.append(dict(tag=tag, w=w,
                             s_star=float(np.median(per[w])),
                             dmin=float(np.median(dmin[w])),
                             edge_share=edge))
    T = pd.DataFrame(rows)
    piv = T.pivot(index="w", columns="tag", values="s_star")
    print("\ns*(w) per operator (median over cells)")
    print(piv.round(2).to_string())
    print("\ndrop from w=0.2 to w=2 (relative)")
    for c in piv.columns:
        v = piv[c]
        print(f"  {c:12s} {v.iloc[0]:.2f} -> {v.iloc[-1]:.2f}   "
              f"{(v.iloc[-1]/v.iloc[0]-1):+.1%}")
    print("\nedge share of s*:", T.groupby("tag").edge_share.first().round(2).to_dict())
    p = os.path.join(OUT, "an_y7_shape.csv")
    T.to_csv(p, index=False)
    print(f"\n  -> {p}")


if __name__ == "__main__":
    main()
