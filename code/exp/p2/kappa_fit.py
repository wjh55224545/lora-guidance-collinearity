"""Fit kappa from a y_gen grid (paper 2 definition).

s*(w) = argmin_s pix_l2(w, s) per (prompt, seed); kappa = -slope of log s* on log w.
Reports the full-grid kappa and the two segments used in the paper (w<=0.8, w>=1.2).

Usage: python kappa_fit.py <images.csv> [label]
"""
import io
import sys

import numpy as np
import pandas as pd

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path
d = pd.read_csv(path)

rows = []
for (pid, seed), g in d.groupby(["prompt_id", "seed"]):
    piv = g.pivot_table(index="w", columns="s", values="pix_l2", aggfunc="median")
    if piv.shape[0] < 5:
        continue
    s_star = piv.idxmin(axis=1)
    x = np.log(piv.index.values.astype(float))
    y = np.log(s_star.values.astype(float))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        continue
    slope, inter = np.polyfit(x[ok], y[ok], 1)
    rows.append(dict(prompt_id=pid, seed=seed, kappa=-slope, n_w=int(ok.sum())))

r = pd.DataFrame(rows)
out = []
out.append("== %s ==" % label)
if len(r) == 0:
    out.append("no fits")
else:
    out.append("fits %d | kappa median %.3f | IQR %.3f-%.3f | min %.3f max %.3f"
               % (len(r), r.kappa.median(), r.kappa.quantile(.25),
                  r.kappa.quantile(.75), r.kappa.min(), r.kappa.max()))
    out.append("fraction above 0.5: %.1f%%" % (100 * (r.kappa > 0.5).mean()))

# piecewise, using the w grid actually present
w_lo = sorted([w for w in d.w.unique() if w <= 0.8])
w_hi = sorted([w for w in d.w.unique() if w >= 1.2])
for nm, ws in (("w<=0.8", w_lo), ("w>=1.2", w_hi)):
    if len(ws) < 3:
        continue
    kk = []
    for (pid, seed), g in d[d.w.isin(ws)].groupby(["prompt_id", "seed"]):
        piv = g.pivot_table(index="w", columns="s", values="pix_l2", aggfunc="median")
        if piv.shape[0] < 3:
            continue
        s_star = piv.idxmin(axis=1)
        x = np.log(piv.index.values.astype(float))
        y = np.log(s_star.values.astype(float))
        kk.append(-np.polyfit(x, y, 1)[0])
    if kk:
        out.append("%s: kappa median %.3f (n=%d)" % (nm, float(np.median(kk)), len(kk)))

txt = "\n".join(out)
print(txt)
base = path.replace(".csv", "_kappa.txt")
io.open(base, "w", encoding="utf-8").write(txt)
# per-fit table, so a later run does not have to re-derive the spread from the text
if len(r):
    r.to_csv(path.replace(".csv", "_kappa_fits.csv"), index=False, float_format="%.6f")
