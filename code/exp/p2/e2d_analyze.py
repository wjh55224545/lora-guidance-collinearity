"""E2c-2 analysis (paper 2): does kappa depend on *where* the LoRA sits?

Same estimator as L2 so the numbers are comparable: per (prompt, seed), find
s*(w) = argmin_s pix_l2 at each w, then regress log(s*/S0) on log(w/W0); the
slope is -kappa.  kappa=1 means "equal w*s", kappa=0 means "s does not matter".

Two things are reported that L2 alone could not:
  1  kappa per variant, i.e. is the collinearity a property of the *update
     direction* (same kappa everywhere) or of the *path* (te vs unet differ)
  2  the boundary-hit rate.  argmin over a finite grid sits on an edge whenever
     the loss is monotone in s; a variant whose identity barely responds to w
     will pin s* to 3.0 or 14.0 and produce a meaningless slope.  Any variant
     with >40% edge hits gets flagged rather than reported.

Run:  python e2d_analyze.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "out_e2d")
OUT = os.path.join(HERE, "out_e2d")
W0, S0 = 1.0, 7.5
W_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
S_GRID = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]
ORDER = ["full", "te_only", "unet_only", "no_attn2", "r4"]


def load(with_l2_full=True):
    """All variant grids, plus the `full` baseline borrowed from L2.

    The `full` rows are the same LoRA on the same 8 prompts and seeds 11/22, so
    L2's own grid already contains them; re-using it keeps the comparison on
    identical prompts instead of on whatever the re-run happened to finish.
    The re-run (out_e2d/full) is kept as a determinism check, not as the
    baseline.
    """
    fs = [p for p in glob.glob(os.path.join(SRC, "*", "images.csv"))
          if not os.path.basename(os.path.dirname(p)).startswith("_")]
    if not fs:
        raise SystemExit("no out_e2d/*/images.csv yet")
    D = pd.concat([pd.read_csv(p) for p in fs], ignore_index=True)
    if with_l2_full and "full" not in set(D.tag):
        p = os.path.join(HERE, "out_long2", "L2_images.csv")
        if os.path.exists(p):
            L = pd.read_csv(p)
            L = L[L.segment == "G"]
            keep = sorted(L.prompt_id.unique())[:8]
            L = L[L.prompt_id.isin(keep) & L.seed.isin([11, 22])].copy()
            L["tag"] = "full"
            D = pd.concat([D, L[D.columns]], ignore_index=True)
            print(f"  (full baseline: {len(L)} rows borrowed from L2, "
                  f"{L.prompt_id.nunique()} prompts x {L.seed.nunique()} seeds)")
    return D


def kappa_of(g, val="pix_l2", method="grid"):
    """g: rows of one variant.  Returns (per-fit table, edge-hit rate).

    method="grid" is the L2 estimator: s*(w) = argmin over the 9 s values.
    method="quad" interpolates that minimum with a parabola fitted to the whole
    s-curve.  The two are reported together because the grid estimator can only
    return 9 distinct s* values, so log s* is quantised to steps of ~0.18 while
    the w-driven movement is often smaller than that -- the quantisation alone
    can inflate or flatten a slope.  Agreement between the two is the check.
    """
    fits, edges = [], 0
    for (pid, sd), h in g.groupby(["prompt_id", "seed"]):
        s_star, ws = [], []
        for w in sorted(h.w.unique()):
            if w == W0:
                continue
            v = h[h.w == w].groupby("s")[val].mean().dropna()
            if len(v) < 3:
                continue
            s_best = float(v.idxmin())
            if s_best in (min(S_GRID), max(S_GRID)):
                edges += 1
            if method == "quad":
                c = np.polyfit(v.index.values.astype(float),
                               v.values.astype(float), 2)
                if c[0] > 0:
                    st = -c[1] / (2 * c[0])
                    if 2.0 <= st <= 16.0:
                        s_best = float(st)
            s_star.append(s_best)
            ws.append(w)
        if len(ws) < 4:
            continue
        x = np.log(np.array(ws) / W0)
        y = np.log(np.array(s_star) / S0)
        r = stats.linregress(x, y)
        fits.append(dict(prompt_id=pid, pclass=h.pclass.iloc[0], seed=sd,
                         kappa=-r.slope, r2=r.rvalue ** 2,
                         kappa_lo=-r.slope - 1.96 * r.stderr,
                         kappa_hi=-r.slope + 1.96 * r.stderr,
                         n_pts=len(ws)))
    K = pd.DataFrame(fits)
    n_edge = edges / max(1, len(K) * (len(W_GRID) - 1))
    return K, float(n_edge)


def piecewise_of(g, val="pix_l2"):
    lo, hi = [], []
    for (pid, sd), h in g.groupby(["prompt_id", "seed"]):
        d = {}
        for part, ws in (("lo", [.2, .4, .6, .8]), ("hi", [1.2, 1.5, 2.0])):
            xs, ys = [], []
            for w in ws:
                v = h[h.w == w].groupby("s")[val].mean().dropna()
                if len(v) < 3:
                    continue
                xs.append(np.log(w / W0))
                ys.append(np.log(float(v.idxmin()) / S0))
            if len(xs) >= 2:
                d[part] = -stats.linregress(xs, ys).slope
        if "lo" in d and "hi" in d:
            lo.append(d["lo"])
            hi.append(d["hi"])
    return np.array(lo), np.array(hi)


def leverage_of(g, val="pix_l2"):
    """Range of the metric along each axis -- which knob moves the image more."""
    gw = g.groupby(["prompt_id", "seed", "s"])[val].agg(lambda v: v.max() - v.min())
    gs = g.groupby(["prompt_id", "seed", "w"])[val].agg(lambda v: v.max() - v.min())
    return float(gw.median()), float(gs.median())


def star_error_of(g, n_boot=200, seed=0, val="pix_l2"):
    """How much s* moves across w, versus how tightly s* itself is pinned.

    kappa is a slope through s*(w), so it is only meaningful when the w-driven
    movement of s* exceeds the uncertainty of one s* estimate.  That uncertainty
    is obtained by bootstrapping the residuals of the local quadratic fit of
    metric(s) inside each (prompt, seed, w) cell -- no repeats are needed, the
    lack of fit plays the role of noise.

    ratio < 1  =>  the apparent slope is inside the estimator's own error and
    the variant must be reported as "not identified", not as "kappa smaller".
    """
    rng = np.random.default_rng(seed)
    out = {}
    for (pid, sd), d in g.groupby(["prompt_id", "seed"]):
        pts = []
        for w, x in d.groupby("w"):
            if len(x) < 5:
                continue
            s = x.s.values.astype(float)
            y = x[val].values.astype(float)
            c = np.polyfit(s, y, 2)
            if c[0] <= 0:                      # not convex: argmin undefined
                continue
            st = -c[1] / (2 * c[0])
            if not (2.0 <= st <= 16.0):        # minimum outside the grid
                continue
            fit = np.polyval(c, s)
            res = y - fit
            bs = []
            for _ in range(n_boot):
                cb = np.polyfit(s, fit + rng.choice(res, len(res), True), 2)
                if cb[0] > 0:
                    bs.append(-cb[1] / (2 * cb[0]))
            pts.append((w, st, float(np.std(bs)) if bs else np.nan))
        if len(pts) >= 6:
            S = np.array([p[1] for p in pts])
            E = np.array([p[2] for p in pts])
            me = float(np.nanmedian(E))
            out[(pid, sd)] = dict(
                n_w=len(pts), sd_sstar=float(np.std(S)), med_err=me,
                ratio=float(np.std(S) / me) if me > 0 else np.nan,
                span=float(S.max() - S.min()))
    if not out:
        return pd.DataFrame(columns=["n_w", "sd_sstar", "med_err", "ratio",
                                     "span"])
    return pd.DataFrame(out).T


def main():
    D = load()
    tags = [t for t in ORDER if t in set(D.tag)]
    print(f"{len(D)} image rows, variants: "
          + ", ".join(f"{t}({int((D.tag == t).sum())})" for t in tags))

    rows, allk = [], []
    for method in ("grid", "quad"):
        for t in tags:
            g = D[D.tag == t]
            K, edge = kappa_of(g, method=method)
            if K.empty:
                print(f"  !! {t}: not enough points")
                continue
            klo, khi = piecewise_of(g)
            lev_w, lev_s = leverage_of(g)
            allk.append(K.assign(tag=t, method=method))
            rows.append(dict(method=method, tag=t, n_fit=len(K),
                             kappa=K.kappa.mean(),
                             kappa_sd=K.kappa.std(), kappa_med=K.kappa.median(),
                             r2=K.r2.median(), edge_hit=edge,
                             k_lo=(klo.mean() if len(klo) else np.nan),
                             k_hi=(khi.mean() if len(khi) else np.nan),
                             lev_w=lev_w, lev_s=lev_s))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "an_kappa_variants.csv"), index=False)
    KA = pd.concat(allk)
    KA.to_csv(os.path.join(OUT, "an_kappa_fits.csv"), index=False)

    pd.set_option("display.width", 160)
    print("\n=== kappa per variant, two s* estimators ===")
    print(S.pivot_table(index="tag", columns="method",
                        values="kappa").round(3).to_string())
    print("  grid = argmin over the 9 s values (L2 estimator); "
          "quad = parabolic interpolation")
    print("\n  full table:")
    print(S.round(3).to_string(index=False))
    print("\n  0 = s irrelevant, 1 = equal w*s; edge_hit > .40 = estimator pinned")

    print("\n=== pairwise: is kappa stable across variants? ===")
    for method in ("grid", "quad"):
        K = KA[KA.method == method]
        piv = K.pivot_table(index=["prompt_id", "seed"], columns="tag",
                            values="kappa").dropna()
        if piv.shape[1] < 2 or len(piv) < 4:
            continue
        print(f"\n  [{method}] paired over {len(piv)} prompt-seed cells:")
        groups = [piv[c].values for c in piv.columns]
        F, p = stats.f_oneway(*groups)
        print(f"    ANOVA on the variant factor: F={F:.2f}, p={p:.3g}")
        for c in piv.columns:
            if c == "full":
                continue
            a, b = piv["full"].values, piv[c].values
            tt = stats.ttest_rel(a, b)
            print(f"    full={a.mean():+.3f}  {c:10s}={b.mean():+.3f}  "
                  f"diff={b.mean() - a.mean():+.3f}  p={tt.pvalue:.4f}")

    # ---- can kappa be measured at all in each variant? --------------------
    print("\n=== identifiability: does s* move across w by more than its own error? ===")
    IE = pd.concat([star_error_of(D[D.tag == t]).assign(tag=t) for t in tags])
    IE.to_csv(os.path.join(OUT, "identifiability.csv"))
    agg = (IE.groupby("tag")
             .agg(n=("ratio", "size"), sd_sstar=("sd_sstar", "median"),
                  med_err=("med_err", "median"), ratio=("ratio", "median"),
                  frac_lt1=("ratio", lambda v: float((v < 1).mean()))))
    print(agg.round(3).to_string())
    ok = set(agg[agg.ratio >= 1.5].index)
    print(f"\n  identified (ratio >= 1.5): {sorted(ok)}")
    print("  ratio < 1  => the w-driven movement of s* sits inside the")
    print("  estimator's own error; report 'not identified', not 'kappa lower'.")

    for method in ("grid", "quad"):
        K = KA[KA.method == method]
        piv = K.pivot_table(index=["prompt_id", "seed"], columns="tag",
                            values="kappa").dropna()
        cols = [c for c in piv.columns if c in ok]
        if "full" not in cols or len(cols) < 2:
            continue
        print(f"\n  [{method}] paired vs full, identified variants only "
              f"(n={len(cols)}):")
        for c in cols:
            if c == "full":
                continue
            a, b = piv["full"].values, piv[c].values
            tt = stats.ttest_rel(a, b)
            print(f"    full={a.mean():+.3f}  {c:10s}={b.mean():+.3f}  "
                  f"diff={b.mean() - a.mean():+.3f}  p={tt.pvalue:.4f}")

    print("\n=== leverage (median max-min of pix_l2) ===")
    print(S[S.method == "grid"][["tag", "lev_w", "lev_s"]]
          .round(2).to_string(index=False))

    print("\n=== bimodality / spread sanity: overall pix_l2 by variant ===")
    print(D.groupby("tag").pix_l2.describe()[["mean", "50%", "max"]]
          .round(2).to_string())


if __name__ == "__main__":
    main()
