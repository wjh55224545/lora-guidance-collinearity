"""Y3-swap check: does the A/B asymmetry follow the adapter, or the load slot?

Y3 found the first adapter's leverage collapses to 0.09x while the second keeps
0.51x.  Two readings, and they are not distinguishable from one run:

  * the second adapter loaded *wins* because of how the deltas are composed
    (slot effect)  -> rerunning with B loaded first should reproduce 0.09 / 0.51
  * the two adapters simply differ in how much leverage they have
    (adapter effect) -> rerunning with B first should give 0.51 / 0.09, i.e. the
    shrink numbers follow the adapter through the swap

The grid also transposes: in the swap run axis 1 is B and axis 2 is A, so
d_swap[i, j] = d_orig[j, i].  Therefore

  corr(vec(d_swap), vec(d_orig.T)) near 1  -> composition is order-free, the
                                              asymmetry is a property of the
                                              adapters
  corr(vec(d_swap), vec(d_orig))   near 1  -> the order itself carries the effect

Both are printed; neither is assumed.

Run:  python y3_swap_check.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y")
VAL = "pix_l2_base"


def matrices(tag):
    D = pd.read_csv(os.path.join(OUT, tag, "images.csv"))
    D = D[D.w.notna() & D.w2.notna()]
    ws = sorted(D.w.unique())
    w2s = sorted(D.w2.unique())
    out = {}
    for (s, pid), h in D.groupby(["s", "prompt_id"]):
        piv = h.pivot_table(index="w", columns="w2", values=VAL)
        piv = piv.reindex(index=ws, columns=w2s)
        if piv.isna().any().any():
            continue
        out[(s, pid)] = piv.values.astype(float)
    return out


def shrink_of(tag):
    """Median shrink of axis1 and axis2, same definition as y3_analyze."""
    p = os.path.join(OUT, tag, "images.csv")
    if not os.path.exists(p):
        return None
    D = pd.read_csv(p)
    D = D[D.w.notna() & D.w2.notna()]
    wv = np.array(sorted(D.w.unique()), float)
    w2v = np.array(sorted(D.w2.unique()), float)
    M = matrices(tag)
    s1, s2 = [], []
    for Mx in M.values():
        a, ah = stats.linregress(wv, Mx[:, 0]).slope, \
                stats.linregress(wv, Mx[:, -1]).slope
        b, bh = stats.linregress(w2v, Mx[0, :]).slope, \
                stats.linregress(w2v, Mx[-1, :]).slope
        if a:
            s1.append(ah / a)
        if b:
            s2.append(bh / b)
    return (np.median(s1) if s1 else np.nan,
            np.median(s2) if s2 else np.nan)


def main():
    A = matrices("y3")
    B = matrices("y3_swap")
    keys = sorted(set(A) & set(B))
    if not keys:
        raise SystemExit("swap run not finished (or grids differ)")

    rT, rI = [], []
    for k in keys:
        a, b = A[k], B[k]
        rT.append(stats.pearsonr(b.ravel(), a.T.ravel()).statistic)
        rI.append(stats.pearsonr(b.ravel(), a.ravel()).statistic)
    rT, rI = np.array(rT), np.array(rI)
    print(f"Y3-swap: {len(keys)} matched (s, prompt) grids")
    print(f"  corr(d_swap, d_orig^T) = {np.median(rT):.4f}  "
          f"[{rT.min():.4f}, {rT.max():.4f}]")
    print(f"  corr(d_swap, d_orig  ) = {np.median(rI):.4f}  "
          f"[{rI.min():.4f}, {rI.max():.4f}]")
    if np.median(rT) > np.median(rI) + .05:
        print("  -> order-free composition: the asymmetry belongs to the "
              "ADAPTERS, not to the loading slot")
    elif np.median(rI) > np.median(rT) + .05:
        print("  -> the loading slot itself carries the effect; report it as an "
              "artefact of stacking order, not as an adapter property")
    else:
        print("  -> neither reading wins; the two runs disagree in a way this "
              "test cannot resolve -- report both grids, claim nothing")

    # the direct reading: shrink per axis, in both runs
    for tag in ("y3", "y3_swap"):
        sh = shrink_of(tag)
        if sh is not None:
            print(f"  {tag:9s} axis1 shrink {sh[0]:+.3f}   axis2 shrink {sh[1]:+.3f}")

if __name__ == "__main__":
    main()
