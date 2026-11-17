"""Paper 2, T2: kappa on the SDXL grid, judged the same way as the SD2.1-base run.

Why a separate script rather than kappa_panel.py: kappa_panel.py computes a slope
unconditionally, and on this grid one cell has s* constant across all eight w
values.  np.polyfit happily returns a slope of 0 for constant y, which would be
reported as "kappa = 0" -- an unmeasurable cell masquerading as a measured zero.
That is the exact failure the SD2.1 paragraph in §5.8 avoids by requiring
R^2 >= 0.25 before a cell counts.  This script applies the same pre-registered
rule and reports the excluded cells explicitly.

Pre-registered criteria (written in T2_SDXL_2026-11-15/summary.md before the
numbers existed):
  C1  panel median kappa > 0
  C2  kappa inside [0.045, 0.204], the span of the five SD1.5 subject adapters
  C3  high-w segment kappa below whole-grid kappa

Usage:  python xl_kappa.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "out_xl_xl_S1", "images.csv")
PANEL = ["obj00", "obj01", "obj02", "obj03"]
W_LO, W_HI = 0.8, 1.2
R2_MIN = 0.25          # same threshold as the SD2.1 run in §5.8
SD15_SPAN = (0.045, 0.204)
SD15_BAND = 0.211      # reference-adapter kappa quoted in §4


def fit(g: pd.DataFrame, w_sel=None):
    """Per-cell kappa, or a reason it cannot be measured."""
    d = g if w_sel is None else g[g.w.isin(w_sel)]
    piv = d.pivot_table(index="w", columns="s", values="pix_l2", aggfunc="median")
    need = 5 if w_sel is None else 3
    if piv.shape[0] < need or piv.shape[1] < 5:
        return None, None, "too few cells"
    sstar = piv.idxmin(axis=1)
    x = np.log(piv.index.values.astype(float))
    y = np.log(sstar.values.astype(float))
    if np.ptp(y) == 0.0:
        return None, None, "s* constant over w (no slope to estimate)"
    k = -float(np.polyfit(x, y, 1)[0])
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    if not np.isfinite(r2) or r2 < R2_MIN:
        return None, r2, "R2 below %.2f" % R2_MIN
    return k, r2, "ok"


def main():
    d = pd.read_csv(CSV)
    d = d[d.prompt_id.isin(PANEL)]
    print("== T2: SDXL 1.0, %d images, %d prompts ==" % (len(d), d.prompt_id.nunique()))

    rows = []
    for pid, g in d.groupby("prompt_id"):
        k, r2, why = fit(g)
        piv = g.pivot_table(index="w", columns="s", values="pix_l2", aggfunc="median")
        sstar = piv.idxmin(axis=1).values
        kh, r2h, whyh = fit(g, sorted(w for w in g.w.unique() if w >= W_HI))
        kl, _, _ = fit(g, sorted(w for w in g.w.unique() if w <= W_LO))
        rows.append(dict(cell=pid, kappa=k, r2=r2, status=why,
                         sstar=" ".join(str(int(v)) for v in sstar),
                         kappa_hi=kh, r2_hi=r2h,
                         strength=float(g[np.isclose(g.w, 1.0) & np.isclose(g.s, 7.5)]
                                        .pix_l2_base.median())))
    t = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format=lambda v: "%.3f" % v))

    ok = t.dropna(subset=["kappa"])
    n_id = len(ok)
    print("\n-- identifiable cells: %d/%d (R2 >= %.2f)" % (n_id, len(t), R2_MIN))
    for r in t.itertuples():
        if r.kappa is None or (isinstance(r.kappa, float) and np.isnan(r.kappa)):
            print("   %s excluded: %s" % (r.cell, r.status))

    if n_id:
        med = float(ok.kappa.median())
        print("\n-- C1  panel median kappa = %+.3f  (all identifiable cells positive: %s)"
              % (med, bool((ok.kappa > 0).all())))
        print("-- C2  inside SD1.5 span [%.3f, %.3f]: %s" % (SD15_SPAN[0], SD15_SPAN[1],
              bool(SD15_SPAN[0] <= med <= SD15_SPAN[1])))
        print("     inside reference band %.3f +- 0.10: %s"
              % (SD15_BAND, bool(abs(med - SD15_BAND) <= 0.10)))
        print("     per-cell values: " + ", ".join("%.3f" % v for v in ok.kappa))
        hi = ok.kappa_hi.dropna()
        if len(hi):
            print("-- C3  high-w segment median kappa = %+.3f vs whole-grid %+.3f -> %s"
                  % (hi.median(), med, "lower" if hi.median() < med else "NOT lower"))
        print("-- R2 on identifiable cells: " + ", ".join("%.2f" % v for v in ok.r2))

    out = os.path.join(HERE, "out_xl_kappa.csv")
    t.to_csv(out, index=False, float_format="%.6f")
    print("\n-> " + out)


if __name__ == "__main__":
    main()
