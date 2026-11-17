"""Phase-Y analysis (paper 2): kappa under a dense s grid, three estimators.

The question this answers, stated so it can fail:

  The published kappa (~0.19-0.27) comes from argmin over 9 s values.  In log
  space that grid is quantised at ~0.18, and the movement w produces is of the
  same size, so "kappa is small" and "kappa is invisible to this grid" predict
  the same number.  Y2 re-measures the same 3..14 range at 0.5 spacing (23
  points, quantisation 0.069).  Pre-committed: a kappa above 0.5 kills the
  interval and every kappa in the draft gets re-reported; a kappa that stays
  where it was means the grid was never the binding constraint.

Three estimators, all reported, because agreement between them is the only
defence against another estimator artefact (E2c-2 already lost a conclusion to
exactly that):
  grid    argmin over the 23 values                     -- comparable to L2
  local   parabola through argmin +/- 2 neighbours      -- removes quantisation
  global  parabola through all 23 points                -- most smoothing, most
                                                          bias if the curve is
                                                          asymmetric

Identifiability gate (same rule as E2c-2, so the two cannot be read differently):
  ratio = sd_w(s*) / median bootstrap error of a single s*
  ratio <= 1 means the w-driven movement of s* is smaller than s*'s own error,
  i.e. the slope is being fitted to noise and the kappa has to be reported as
  "not measurable", never as "small".

Run:
    python y_analyze.py --tag y2_full
    python y_analyze.py --tag y1_A
    python y_analyze.py --tag all --compare
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y")
W0, S0 = 1.0, 7.5
REF = {  # already published, for the comparison table
    "L2 (coarse, 91 cells)": 0.211,
    "L2 quad (91 cells)": 0.249,
    "E2c-2 full (coarse, 16 cells)": 0.189,
    "E2c-2 full quad": 0.219,
}


def load(tag):
    p = os.path.join(OUT, tag, "images.csv")
    if not os.path.exists(p):
        raise SystemExit(f"no such tag: {p}")
    D = pd.read_csv(p)
    D = D[D.w.notna()]
    return D


def _fit_local(s, y, i0, half=2):
    lo, hi = max(0, i0 - half), min(len(s), i0 + half + 1)
    ss, yy = s[lo:hi], y[lo:hi]
    if len(ss) < 3 or len(np.unique(ss)) < 3:
        return None
    c = np.polyfit(ss, yy, 2)
    if c[0] <= 0:
        return None
    return c, ss, yy


def s_star_curve(h, method="local", val="pix_l2", boot=0, rng=None):
    """[(w, s*, err)] for one (prompt, seed) cell."""
    out = []
    for w in sorted(h.w.unique()):
        if w == W0:
            continue
        g = h[h.w == w].groupby("s")[val].mean().dropna().sort_index()
        if len(g) < 5:
            continue
        s = g.index.values.astype(float)
        y = g.values.astype(float)
        i0 = int(np.argmin(y))
        if method == "grid":
            out.append((w, float(s[i0]), np.nan))
            continue
        if method == "global":
            c = np.polyfit(s, y, 2)
            ss, yy, cc = s, y, (c if c[0] > 0 else None)
        else:
            r = _fit_local(s, y, i0)
            if r is None:
                out.append((w, float(s[i0]), np.nan))
                continue
            cc, ss, yy = r
        if cc is None or cc[0] <= 0:
            out.append((w, float(s[i0]), np.nan))
            continue
        sst = float(-cc[1] / (2 * cc[0]))
        err = np.nan
        if boot and rng is not None:
            fit = np.polyval(cc, ss)
            res = yy - fit
            bs = []
            for _ in range(boot):
                yb = fit + rng.choice(res, size=len(res), replace=True)
                cb = np.polyfit(ss, yb, 2)
                if cb[0] > 0:
                    bs.append(-cb[1] / (2 * cb[0]))
            if bs:
                err = float(np.std(bs))
        out.append((w, sst, err))
    return out


def kappa_cells(D, method="local", val="pix_l2", boot=0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for (pid, sd), h in D.groupby(["prompt_id", "seed"]):
        cur = s_star_curve(h, method, val, boot, rng)
        if len(cur) < 4:
            continue
        w = np.array([c[0] for c in cur], float)
        ss = np.array([c[1] for c in cur], float)
        ok = (ss > 0) & np.isfinite(ss)
        if ok.sum() < 4:
            continue
        r = stats.linregress(np.log(w[ok] / W0), np.log(ss[ok] / S0))
        errs = np.array([c[2] for c in cur], float)
        rows.append(dict(prompt_id=pid, seed=sd, pclass=h.pclass.iloc[0],
                         kappa=-r.slope, r2=r.rvalue ** 2,
                         k_lo=-r.slope - 1.96 * r.stderr,
                         k_hi=-r.slope + 1.96 * r.stderr,
                         n_pts=int(ok.sum()),
                         sd_sstar=float(np.std(ss[ok])),
                         med_err=float(np.nanmedian(errs)) if np.isfinite(errs).any()
                         else np.nan,
                         span=float(ss.max() - ss.min())))
    return pd.DataFrame(rows)


def lev(D, val="pix_l2"):
    gw = D.groupby(["prompt_id", "seed", "s"])[val].agg(lambda v: v.max() - v.min())
    gs = D.groupby(["prompt_id", "seed", "w"])[val].agg(lambda v: v.max() - v.min())
    return float(gw.median()), float(gs.median())


def boot_ci(v, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(D, tag):
    print(f"\n{'=' * 78}\n{tag}: {len(D)} rows, "
          f"{D.prompt_id.nunique()} prompts x {D.seed.nunique()} seeds, "
          f"{D.w.nunique()} w x {D.s.nunique()} s, "
          f"s spacing {np.diff(sorted(D.s.unique())).min():g}")
    lw, ls = lev(D)
    print(f"  leverage: along w {lw:.2f}, along s {ls:.2f} "
          f"(ratio {lw / ls:.2f})")
    print(f"  {'estimator':9s} {'n':>3s} {'mean':>7s} {'med':>7s} {'sd':>6s} "
          f"{'95% CI':>18s} {'sd_w(s*)':>9s} {'s* err':>7s} {'ratio':>6s}")

    res = {}
    for m in ("grid", "local", "global"):
        K = kappa_cells(D, m, boot=200 if m != "grid" else 0)
        if K.empty:
            print(f"  {m:9s}  -- no cells")
            continue
        res[m] = K
        lo, hi = boot_ci(K.kappa.values)
        ratio = (K.sd_sstar / K.med_err).median() if m != "grid" else np.nan
        print(f"  {m:9s} {len(K):3d} {K.kappa.mean():7.3f} "
              f"{K.kappa.median():7.3f} {K.kappa.std(ddof=1):6.3f} "
              f"  [{lo:+.3f},{hi:+.3f}]  "
              f"{K.sd_sstar.median():9.3f} {K.med_err.median():7.3f} "
              f"{ratio:6.2f}")

    if "grid" in res and "local" in res:
        a = res["grid"].set_index(["prompt_id", "seed"]).kappa
        b = res["local"].set_index(["prompt_id", "seed"]).kappa
        ix = a.index.intersection(b.index)
        d = (b.loc[ix] - a.loc[ix]).values
        tt = stats.ttest_1samp(d, 0.0) if len(d) > 2 else None
        if tt is not None:
            print(f"  grid -> local shift: {d.mean():+.3f} "
                  f"(sd {d.std(ddof=1):.3f}, t={tt.statistic:+.2f}, "
                  f"p={tt.pvalue:.3g})")

    best = res.get("local", res.get("grid"))
    if best is not None:
        k = best.kappa.mean()
        print(f"\n  VERDICT  kappa(local) = {k:.3f}")
        if k > 0.5:
            print("    > 0.5  ->  the published interval [0.17,0.27] is a grid "
                  "artefact.  STOP and re-report every kappa in the draft.")
        elif k > 0.30:
            print("    in (0.3, 0.5]  ->  the interval has to be widened "
                  "upward; the 'equal w*s is 4-5x wrong' sentence must be "
                  "recomputed.")
        else:
            print("    <= 0.30  ->  the grid was not the binding constraint. "
                  "The interval survives; say by how much the two estimators "
                  "agree.")
        print(f"    distance to kappa=0 {k:.3f}, to kappa=1 (equal w*s) "
              f"{1 - k:.3f} -> equal w*s is {1 / k:.1f}x wrong")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    tags = ([t for t in ("y2_full", "y1_A", "y1_B", "y2_A")
             if os.path.exists(os.path.join(OUT, t, "images.csv"))]
            if a.tag == "all" else [a.tag])
    if not tags:
        raise SystemExit("nothing to analyse yet")

    rows = []
    for t in tags:
        D = load(t)
        res = report(D, t)
        for m, K in res.items():
            K.assign(tag=t, estimator=m).to_csv(
                os.path.join(OUT, f"an_{t}_{m}.csv"), index=False)
            rows.append(dict(tag=t, estimator=m, n=len(K),
                             kappa=K.kappa.mean(),
                             kappa_med=K.kappa.median(),
                             kappa_sd=K.kappa.std(ddof=1),
                             ratio=(K.sd_sstar / K.med_err).median()))
    S = pd.DataFrame(rows)
    if len(S):
        print(f"\n{'=' * 78}\nsummary")
        print(S.round(3).to_string(index=False))
        if a.compare:
            print("\nagainst what is already written down:")
            for k, v in REF.items():
                print(f"  {k:32s} {v:.3f}")
        S.to_csv(os.path.join(OUT, "an_kappa_dense_summary.csv"), index=False)
        print(f"\n  -> {OUT}/an_kappa_dense_summary.csv")


if __name__ == "__main__":
    main()
