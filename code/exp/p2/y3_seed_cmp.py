"""Y3 seed 11 vs seed 22: which of the Y3 conclusions replicate?

Three distinct quantities, deliberately kept apart (this is the X6 lesson:
level correlation and per-cell slope correlation are different animals):

  1. level   -- corr(d11[cell], d11_swap_or_s22[cell])   "same magnitude at same cell"?
  2. rank    -- Spearman on the same pairs
  3. asym    -- per-cell (wA_shrink - wB_shrink), i.e. does A lose its leverage
                earlier than B?  Tested by sign test + corr across seeds.

Reads only files already on disk; writes an_y3_seed_cmp.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y")

P = {}
for tag, fn in [("s11", "an_y3_additivity.csv"), ("s22", "an_y3_s22_additivity.csv")]:
    P[tag] = pd.read_csv(os.path.join(OUT, fn))
    P[tag]["cell"] = P[tag].prompt_id + "@s" + P[tag].s.astype(str).str.replace(".0", "", regex=False)

KEY = ["cell"]
a, b = P["s11"].set_index(KEY), P["s22"].set_index(KEY)
common = [c for c in a.index if c in b.index]
a, b = a.loc[common], b.loc[common]

rows = []
for col in ["d11", "r2_add", "resid_sd", "AxB", "joint_over_sum",
            "wA_shrink", "wA_shrink_int", "wB_shrink",
            "wA_slope_wB0", "wB_slope_wA0"]:
    x, y = a[col].to_numpy(float), b[col].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    r = stats.pearsonr(x[ok], y[ok])
    rho = stats.spearmanr(x[ok], y[ok])
    rows.append(dict(quantity=col, n=int(ok.sum()),
                     pearson=r[0], p_pearson=r[1],
                     spearman=rho[0], p_spearman=rho[1],
                     med_s11=float(np.median(x[ok])), med_s22=float(np.median(y[ok])),
                     ratio_med=float(np.median(y[ok]) / np.median(x[ok]))
                     if abs(np.median(x[ok])) > 1e-9 else np.nan))
C = pd.DataFrame(rows)
C.to_csv(os.path.join(OUT, "an_y3_seed_cmp.csv"), index=False)

# ---- asymmetry: is A weaker than B, and is that stable across seeds?
A11 = (a.wA_shrink - a.wB_shrink).to_numpy(float)
A22 = (b.wA_shrink - b.wB_shrink).to_numpy(float)
print("=" * 78)
print("Y3  seed 11 vs seed 22   (12 cells, same grid)")
print("=" * 78)
print(f"{'quantity':<16}{'med s11':>9}{'med s22':>9}{'pearson':>9}{'p':>8}{'spearman':>10}{'p':>8}")
for _, r in C.iterrows():
    print(f"{r.quantity:<16}{r.med_s11:>9.3f}{r.med_s22:>9.3f}"
          f"{r.pearson:>9.3f}{r.p_pearson:>8.3f}{r.spearman:>10.3f}{r.p_spearman:>8.3f}")

print("\n-- asymmetry  wA_shrink - wB_shrink  (negative = A loses leverage first)")
for tag, v in [("s11", A11), ("s22", A22)]:
    nneg = int((v < 0).sum())
    print(f"   {tag}: median {np.median(v):+.3f}   mean {v.mean():+.3f}   "
          f"A<B in {nneg}/12 cells   sign-test p={stats.binomtest(nneg, len(v), .5).pvalue:.3f}")
print(f"   corr(asym_s11, asym_s22) = {stats.pearsonr(A11, A22)[0]:+.3f} "
      f"(p={stats.pearsonr(A11, A22)[1]:.3f})")

print("\n-- the two hard-coded thresholds, per seed")
for tag, D in [("s11", a), ("s22", b)]:
    m, mi = np.median(D.wA_shrink), np.median(D.wA_shrink_int)
    mB = np.median(D.wB_shrink)
    print(f"   {tag}: wA_shrink {m:.3f} (<0.6 -> {'HIT' if m < .6 else 'MISS'})   "
          f"interior {mi:.3f} (<0.6 -> {'HIT' if mi < .6 else 'MISS'})   wB {mB:.3f}")

# ---- how big is the seed-to-seed cell difference, relative to the noise floor
# that the additive fit already reports as resid_sd?  If |d(s11)-d(s22)| is of
# the same order as resid_sd, a single cell value is not a transportable number.
dd = (b.d11 - a.d11).to_numpy(float)
lev = a.d11.to_numpy(float)
floor = np.sqrt(0.5 * (a.resid_sd.to_numpy(float) ** 2 + b.resid_sd.to_numpy(float) ** 2))
print("\n-- per-cell D(1,1)  difference between seeds, against the within-fit noise floor")
print(f"   median |d_s11 - d_s22| = {np.median(np.abs(dd)):.2f}   "
       f"median level = {np.median(lev):.2f}   "
       f"median relative = {np.median(np.abs(dd)/lev)*100:.1f}%")
print(f"   median resid_sd (pooled) = {np.median(floor):.2f}   "
       f"median |diff| / floor = {np.median(np.abs(dd)/floor):.2f}")
print(f"   |diff| > floor in {int((np.abs(dd) > floor).sum())}/12 cells   "
       f"(sign test on dd: {(dd > 0).sum()}/12 positive, "
       f"p={stats.binomtest(int((dd > 0).sum()), 12, .5).pvalue:.3f}")

print("\n-- dispersion of the wA axis (is 'shrink' even a level on seed 22?)")
for tag, D in [("s11", a), ("s22", b)]:
    v = D.wA_shrink.to_numpy(float)
    print(f"   {tag}: range [{v.min():+.2f}, {v.max():+.2f}]  sd {v.std(ddof=1):.3f}  "
          f"IQR {np.percentile(v,75)-np.percentile(v,25):.3f}  "
          f"n_outside_[0,1] {int(((v<0)|(v>1)).sum())}/12")

# the replicable form of "the axis collapses": a COUNT, not a level.  A median of
# a quantity whose sd is twice its own median is not a level, it is a summary of
# a bimodal mess -- so count how often each axis loses more than 40% of its slope.
print("\n-- count form: cells where an axis keeps <0.6 of its w=0 slope")
for tag, D in [("s11", a), ("s22", b)]:
    for ax, col in [("wA", "wA_shrink"), ("wB", "wB_shrink")]:
        v = D[col].to_numpy(float)
        print(f"   {tag} {ax}: {int((v < .6).sum())}/12 keep <0.6   "
              f"{int((v < 0).sum())}/12 keep <0 (slope reverses)")
print("=" * 78)
