"""Y3 analysis (paper 2): with two LoRAs loaded, how many knobs are left?

The claim under test is the one every practitioner community repeats -- "stacking
LoRAs means turning the guidance down, and total strength above ~1.5-2 makes them
fight".  Nobody publishes a number for it, and the reason is that the right
number is not a coefficient but a *separability* test:

  if  d(wA, wB, s) = f(wA, s) + g(wB, s) + const
  then the two adapters act on independent directions and there really are two
  knobs, so collinearity between w and s is all that matters.
  If instead the wA axis *shrinks* as wB comes up, then wB has eaten part of
  wA's leverage: the effective number of knobs is below 2, which is the
  composition-side version of the same phenomenon I1 describes for (w, s).

Measurements, all on pix_l2_base (distance to the no-LoRA image, generated in the
same run so the whole 5x5 grid sits in one coordinate system):

  1  additive fit  D[i,j] = mu + a_i + b_j  -> R^2 and the residual scale.
     R^2 near 1 means separable; the residual is the interaction.
  2  axis shrinkage  slope of D along wA computed at wB = 0 versus at wB = 1.5.
     ratio < 1 means the second adapter eats the first one's leverage.
  3  joint versus sum  D(1,1) against D(1,0) and D(0,1).  Sub-additive is the
     quantitative form of "they fight"; super-additive would mean mutual
     amplification and would falsify the community advice.

Note on the reference: the base image is generated at S0=7.5 for every row, so
pix_l2_base carries a constant offset within each s slice.  Constants drop out of
all three statistics above, which is why the analysis is done *per s slice* and
never pooled across s.

Run:  python y3_analyze.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y")
TAG = "y3"
VAL = "pix_l2_base"


def fit_additive(D):
    """D[i,j] = mu + a_i + b_j, least squares; returns R^2 and residual sd."""
    n, m = D.shape
    A = np.zeros((n * m, 1 + n + m))
    y = D.reshape(-1)
    for i in range(n):
        for j in range(m):
            r = i * m + j
            A[r, 0] = 1.0
            A[r, 1 + i] = 1.0
            A[r, 1 + n + j] = 1.0
    # drop one column per factor to remove the degeneracy
    keep = [0] + list(range(1, n)) + list(range(1 + n, 1 + n + m - 1))
    coef, *_ = np.linalg.lstsq(A[:, keep], y, rcond=None)
    pred = A[:, keep] @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return (1 - ss_res / ss_tot if ss_tot > 0 else np.nan), \
           float(np.sqrt(ss_res / max(1, len(y) - len(keep))))


def slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(np.unique(x)) < 2:
        return np.nan
    return float(stats.linregress(x, y).slope)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG)
    a = ap.parse_args()
    tag = a.tag
    p = os.path.join(OUT, tag, "images.csv")
    if not os.path.exists(p):
        raise SystemExit(f"not finished yet: {p}")
    D = pd.read_csv(p)
    D = D[D.w.notna() & D.w2.notna()]
    ws = sorted(D.w.unique())
    print(f"Y3: {len(D)} rows, {D.prompt_id.nunique()} prompts, "
          f"wA={ws}, wB={sorted(D.w2.unique())}, s={sorted(D.s.unique())}")

    rows = []
    for s in sorted(D.s.unique()):
        for pid in sorted(D.prompt_id.unique()):
            h = D[(D.s == s) & (D.prompt_id == pid)]
            piv = h.pivot_table(index="w", columns="w2", values=VAL)
            piv = piv.reindex(index=ws, columns=sorted(D.w2.unique()))
            if piv.isna().any().any():
                continue
            M = piv.values.astype(float)
            r2, rsd = fit_additive(M)
            i1 = ws.index(1.0) if 1.0 in ws else len(ws) - 1
            i0, ihi = 0, len(ws) - 1
            j1 = list(piv.columns).index(1.0) if 1.0 in piv.columns else -1
            # 1) shrink of the wA axis as wB goes 0 -> max
            sA_low = slope(ws, M[:, 0])
            sA_high = slope(ws, M[:, -1])
            # 1b) sensitivity: same contrast on the interior sub-range (wA 0->1.0)
            # with wB at 0 vs at 1.0.  The 1.5 endpoint can sit on a flat or
            # non-monotone tail, in which case the full-range ratio flips sign
            # for a reason that has nothing to do with composition.
            sA_low_i = slope(ws[:i1 + 1], M[:i1 + 1, 0])
            sA_high_i = slope(ws[:i1 + 1], M[:i1 + 1, j1])
            # 2) same for wB as wA goes 0 -> max
            sB_low = slope(piv.columns.values.astype(float), M[0, :])
            sB_high = slope(piv.columns.values.astype(float), M[-1, :])
            # 3) joint vs sum
            d10 = M[i1, 0]
            d01 = M[0, j1 if j1 >= 0 else -1]
            d11 = M[i1, j1 if j1 >= 0 else -1]
            sub = (d11 / (d10 + d01)) if (d10 + d01) > 0 else np.nan
            rows.append(dict(s=s, prompt_id=pid, r2_add=r2, resid_sd=rsd,
                             AxB=float(M[-1, -1]),
                             wA_slope_wB0=sA_low, wA_slope_wBmax=sA_high,
                             wA_shrink=(sA_high / sA_low if sA_low else np.nan),
                             wA_shrink_int=(sA_high_i / sA_low_i if sA_low_i else np.nan),
                             wB_slope_wA0=sB_low, wB_slope_wAmax=sB_high,
                             wB_shrink=(sB_high / sB_low if sB_low else np.nan),
                             d10=d10, d01=d01, d11=d11, joint_over_sum=sub))
    R = pd.DataFrame(rows)
    if R.empty:
        raise SystemExit("no complete 5x5 grid yet")

    print(f"\n{'s':>5s} {'n':>3s} {'R2 additive':>12s} {'resid sd':>9s} "
          f"{'wA shrink':>10s} {'wB shrink':>10s} {'joint/sum':>10s}")
    for s, g in R.groupby("s"):
        print(f"{s:5g} {len(g):3d} {g.r2_add.median():12.3f} "
              f"{g.resid_sd.median():9.2f} {g.wA_shrink.median():10.2f} "
              f"{g.wB_shrink.median():10.2f} {g.joint_over_sum.median():10.2f}")

    allr = R.r2_add.dropna().values
    shA = R.wA_shrink.dropna().values
    shAi = R.wA_shrink_int.dropna().values
    shB = R.wB_shrink.dropna().values
    js = R.joint_over_sum.dropna().values
    # how many cells have a *negative* wA slope once the other adapter is at max
    # -- those cells make the ratio meaningless and must be reported, not hidden
    nneg = int((R.wA_slope_wBmax < 0).sum())
    print("\n=== verdict ===")
    print(f"  additive R^2 median {np.median(allr):.3f} "
          f"(IQR {np.percentile(allr, 25):.3f}-{np.percentile(allr, 75):.3f})")
    print(f"  wA shrink median {np.median(shA):.2f}  "
          f"[interior 0-1.0 contrast: {np.median(shAi):.2f}]   "
          f"wB shrink median {np.median(shB):.2f}")
    print(f"  cells with negative wA slope at wB=max: {nneg}/{len(R)} "
          f"({', '.join(sorted(set(R[R.wA_slope_wBmax < 0].prompt_id)))})")
    if np.median(allr) > 0.95 and np.median(shA) > 0.8:
        print("  -> separable: the two adapters act on independent directions, "
              "so 'two knobs' survives composition.  Do NOT claim interference.")
    elif np.median(shA) < 0.6:
        print("  -> the wA axis shrinks to "
              f"{np.median(shA):.2f}x once the other adapter is at full strength: "
              "composition eats leverage, which is the quantitative form of the "
              "community rule.  Report the number, not the rule.")
    else:
        print("  -> partially separable; report R^2 and shrink, claim neither.")
    print("  NOTE: R^2 and shrink do not have to agree -- R^2 is variance-weighted "
          "over the whole 5x5 grid, a shrinking derivative is an end-point "
          "quantity.  High R^2 + low shrink = 'mostly additive, but the second "
          "adapter takes away the first one's leverage where it matters most'.")
    print(f"  joint/sum median {np.median(js):.2f} "
          f"(<1 = sub-additive, i.e. 'they fight'; >1 = amplification)")
    tt = stats.ttest_1samp(js, 1.0) if len(js) > 2 else None
    if tt is not None:
        print(f"    test against 1: t={tt.statistic:+.2f}, p={tt.pvalue:.3g}")

    R.to_csv(os.path.join(OUT, f"an_{tag}_additivity.csv"), index=False)
    print(f"\n  -> {OUT}/an_{tag}_additivity.csv")


if __name__ == "__main__":
    main()
