"""L2 analysis (paper 2): does a scalar g_eff actually let s replace w?

Three readouts, each answering one question the reviewer will ask.

  1  leverage   inside a fixed (prompt, seed, w), how far does s move the image
                over its 4.7x range?  Inside a fixed (prompt, seed, s), how far
                does w move it over 10x?  If s barely matters, "compensating w
                with s" is dead and the whole section is moot.  If s matters as
                much as w, the compensation question is real.
  2  iso-line   for each w, which s makes the image closest to the reference
                (w0=1, s0=7.5)?  That empirical s*(w) is the answer the three
                routes are trying to guess.  Fit s* = S0 (w/W0)^-kappa and
                report kappa: kappa=1 is "equal w*s", kappa->0 is "s irrelevant",
                and the geff_const route is the kappa implied by
                s = S0 (1+W0 rho)/(1+w rho), which is NOT a power law -- so we
                also report how well that closed form tracks the empirical curve.
  3  routes     three routes at matched intent: fixed_s, geff_const (closed form
                with the rho measured in E3a), ws_equal (the naive equal-product
                rule).  Report mean/max pix_l2 against w, and the low-w end
                separately, because the smoke test flagged w<=0.4 as the place
                where geff_const stops beating fixed_s.

Everything is per (prompt, seed); prompts are grouped by class because E2c
measured a different rho per class (object .208 / person .295 / style .200).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_long2")
W0, S0 = 1.0, 7.5
RHO = {"object": 0.208, "person": 0.295, "style": 0.200}
RHO_DEFAULT = 0.234
W_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
S_GRID = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]


def wide(df, val="pix_l2"):
    """(prompt_id, seed, w) x s -> value, from the G grid."""
    return df[df.segment == "G"].pivot_table(
        index=["prompt_id", "pclass", "seed", "w"], columns="s", values=val)


def part1_leverage(g):
    gr = g.groupby(["prompt_id", "seed", "w"]).pix_l2.agg(["min", "max"])
    gr["rng"] = gr["max"] - gr["min"]
    gw = g.groupby(["prompt_id", "seed", "s"]).pix_l2.agg(["min", "max"])
    gw["rng"] = gw["max"] - gw["min"]
    tab = pd.DataFrame([
        dict(sweep="s (3.0-14.0, 4.7x), w fixed", n=len(gr),
             mean=gr.rng.mean(), med=gr.rng.median(),
             p90=gr.rng.quantile(.9)),
        dict(sweep="w (0.2-2.0, 10x), s fixed", n=len(gw),
             mean=gw.rng.mean(), med=gw.rng.median(),
             p90=gw.rng.quantile(.9)),
    ])
    return tab, gr, gw


def part2_isoline(G):
    """Empirical s*(w) per (prompt, seed): the s minimising pix_l2 at each w."""
    rows = []
    for (pid, pcl, sd, w), row in G.iterrows():
        v = row.dropna()
        if v.empty:
            continue
        s_best = float(v.idxmin())
        rows.append(dict(prompt_id=pid, pclass=pcl, seed=sd, w=w,
                         s_star=s_best, l_at_best=float(v.min()),
                         l_flat=float(np.median(v))))
    E = pd.DataFrame(rows)

    # fit kappa on each (prompt, seed): log(s*/S0) = -kappa log(w/W0)
    fits = []
    for (pid, sd), g in E.groupby(["prompt_id", "seed"]):
        g = g[(g.w != W0)]
        if len(g) < 4:
            continue
        x = np.log(g.w.values / W0)
        y = np.log(g.s_star.values / S0)
        r = stats.linregress(x, y)
        fits.append(dict(prompt_id=pid, pclass=g.pclass.iloc[0], seed=sd,
                         kappa=-r.slope, r2=r.rvalue ** 2,
                         kappa_lo=-r.slope - 1.96 * r.stderr,
                         kappa_hi=-r.slope + 1.96 * r.stderr))
    K = pd.DataFrame(fits)

    # what kappa does each route imply?  geff_const is not a power law, so we
    # evaluate its implied local kappa at each grid w and average over the range.
    imp = []
    for pcl, rho in list(RHO.items()) + [("all", RHO_DEFAULT)]:
        ks = []
        for w in W_GRID:
            if w == W0:
                continue
            s_geff = S0 * (1 + W0 * rho) / (1 + w * rho)
            ks.append(-np.log(s_geff / S0) / np.log(w / W0))
        imp.append(dict(route="geff_const", pclass=pcl, kappa=np.mean(ks)))
    imp.append(dict(route="ws_equal", pclass="all", kappa=1.0))
    imp.append(dict(route="fixed_s", pclass="all", kappa=0.0))
    return E, K, pd.DataFrame(imp)


def part3_routes(t):
    piv = t.pivot_table(index=["prompt_id", "pclass", "seed", "w"],
                        columns="route", values="pix_l2").reset_index()
    summ = []
    for route in ("fixed_s", "geff_const", "ws_equal"):
        v = piv[route]
        summ.append(dict(route=route, n=len(v), mean=v.mean(),
                         med=v.median(), p90=v.quantile(.9), max=v.max(),
                         mean_loww=v[piv.w <= 0.4].mean(),
                         mean_highw=v[piv.w >= 1.5].mean()))
    S = pd.DataFrame(summ)

    byw = piv.groupby("w")[["fixed_s", "geff_const", "ws_equal"]].mean()
    # paired win rate: does geff_const beat fixed_s per image?
    d = piv.geff_const - piv.fixed_s
    wins = dict(geff_beats_fixed=float((d < 0).mean()),
                mean_delta=float(d.mean()),
                median_delta=float(d.median()))
    return S, byw, wins, piv


def main():
    d = pd.read_csv(os.path.join(OUT, "L2_images.csv"))
    g, t = d[d.segment == "G"], d[d.segment == "T"]
    print(f"G rows {len(g)} ({g.prompt_id.nunique()} prompts x "
          f"{g.seed.nunique()} seeds), T rows {len(t)}")

    print("\n=== 1  leverage of s vs w on pix_l2 ===")
    lev, gr, gw = part1_leverage(g)
    print(lev.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("ratio of w-leverage to s-leverage: %.2f"
          % (lev['mean'].iloc[1] / lev['mean'].iloc[0]))
    gr.to_csv(os.path.join(OUT, "an_s_range.csv"))
    gw.to_csv(os.path.join(OUT, "an_w_range.csv"))

    print("\n=== 2  empirical iso-line s*(w) ===")
    E, K, imp = part2_isoline(wide(d))
    E.to_csv(os.path.join(OUT, "an_isoline.csv"), index=False)
    K.to_csv(os.path.join(OUT, "an_kappa.csv"), index=False)
    imp.to_csv(os.path.join(OUT, "an_kappa_implied.csv"), index=False)
    print("kappa (s* = S0 (w/W0)^-kappa), per prompt-seed:")
    print(K[["kappa", "kappa_lo", "kappa_hi", "r2"]].describe()
          .loc[["mean", "50%", "min", "max"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nkappa by class:")
    print(K.groupby("pclass").kappa.agg(["mean", "std", "count"])
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nthe kappa each route implies:")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\ns*(w) averaged over prompts (the target the routes should hit):")
    print(E.groupby("w").s_star.agg(["mean", "std"]).to_string(
        float_format=lambda v: f"{v:.2f}"))

    print("\n=== 3  three routes ===")
    S, byw, wins, piv = part3_routes(t)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nby w:")
    print(byw.to_string(float_format=lambda v: f"{v:.2f}"))
    print("\npaired geff_const vs fixed_s:", wins)
    S.to_csv(os.path.join(OUT, "an_routes.csv"), index=False)
    byw.to_csv(os.path.join(OUT, "an_routes_byw.csv"))
    piv.to_csv(os.path.join(OUT, "an_routes_paired.csv"), index=False)


if __name__ == "__main__":
    main()


# ------------------------------------------------- piecewise kappa (extra)
def piecewise():
    """The compensation is NOT a single power law: fit kappa separately below
    and above w=1.  This is the number that replaces "one rho for the whole
    range" -- and it is why a constant closed-form only pays off above w~1.2."""
    E = pd.read_csv(os.path.join(OUT, "an_isoline.csv"))
    rows = []
    for (pid, sd), g in E.groupby(["prompt_id", "seed"]):
        g = g.set_index("w").sort_index()
        if not {0.2, 1.0, 2.0}.issubset(g.index):
            continue
        lo = stats.linregress(np.log(g.loc[[0.2, 0.4, 0.6, 0.8]].index.values),
                              np.log(g.loc[[0.2, 0.4, 0.6, 0.8]].s_star.values))
        hi = stats.linregress(np.log(g.loc[[1.2, 1.5, 2.0]].index.values),
                              np.log(g.loc[[1.2, 1.5, 2.0]].s_star.values))
        rows.append(dict(prompt_id=pid, pclass=g.pclass.iloc[0], seed=sd,
                         kappa_lo=-lo.slope, kappa_hi=-hi.slope,
                         r2_lo=lo.rvalue ** 2, r2_hi=hi.rvalue ** 2,
                         s_at_02=g.loc[0.2, "s_star"], s_at_20=g.loc[2.0, "s_star"]))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, "an_kappa_piecewise.csv"), index=False)
    print("\n=== piecewise kappa: w<=0.8 vs w>=1.2 ===")
    print(P[["kappa_lo", "kappa_hi"]].describe().loc[["mean", "50%", "min", "max"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nby class:")
    print(P.groupby("pclass")[["kappa_lo", "kappa_hi"]].mean()
          .to_string(float_format=lambda v: f"{v:.3f}"))
    t = stats.wilcoxon(P.kappa_lo, P.kappa_hi)
    print("\nkappa_hi vs kappa_lo: median diff %.3f, Wilcoxon p=%.4g"
          % ((P.kappa_hi - P.kappa_lo).median(), t.pvalue))
    print("s* at w=0.2: %.2f ; at w=2.0: %.2f  (S0=%.1f)"
          % (P.s_at_02.mean(), P.s_at_20.mean(), S0))
    return P


# --------------------------------------- grid as a route evaluator (extra2)
def evaluate_routes_on_grid(d=None):
    """The 8x9 grid is a surrogate for ANY compensation curve.

    For a candidate s(w) we look up the grid value at (w, s(w)) by bilinear
    interpolation in (log w, log s).  That lets us score curves we never
    rendered -- including the piecewise rule the iso-lines suggest -- and it is
    validated against segment T, where three of those curves were rendered for
    real.
    """
    from scipy.interpolate import RegularGridInterpolator
    if d is None:
        d = pd.read_csv(os.path.join(OUT, "L2_images.csv"))
    G = d[d.segment == "G"]
    lw, ls = np.log(W_GRID), np.log(S_GRID)

    def curve(route, w, rho=RHO_DEFAULT, kappa=None, klo=0.0, khi=0.9):
        if route == "fixed_s":
            return S0
        if route == "geff_const":
            return S0 * (1 + W0 * rho) / (1 + w * rho)
        if route == "ws_equal":
            return W0 * S0 / w
        if route == "power":
            return S0 * (w / W0) ** (-kappa)
        if route == "piecewise":
            return S0 if w <= W0 else S0 * (w / W0) ** (-khi)
        if route == "piecewise_sym":
            return S0 * (w / W0) ** (-(klo if w <= W0 else khi))
        raise ValueError(route)

    cands = [("fixed_s", {}), ("geff_const", {}), ("ws_equal", {}),
             ("power", dict(kappa=0.25)), ("power", dict(kappa=0.5)),
             ("power", dict(kappa=0.75)), ("power", dict(kappa=1.0)),
             ("piecewise", dict(khi=0.7)), ("piecewise", dict(khi=0.9)),
             ("piecewise", dict(khi=1.0)),
             ("piecewise_sym", dict(klo=0.0, khi=0.9))]

    rows = []
    for (pid, pcl, sd), g in G.groupby(["prompt_id", "pclass", "seed"]):
        piv = g.pivot_table(index="w", columns="s", values="pix_l2")
        piv = piv.reindex(index=W_GRID, columns=S_GRID)
        vals = piv.values
        if np.isnan(vals).any():
            continue
        f = RegularGridInterpolator((lw, ls), vals, bounds_error=False,
                                    fill_value=None)
        for tag, kw in cands:
            rho = RHO.get(pcl, RHO_DEFAULT)
            for w in W_GRID:
                s = curve(tag, w, rho=rho, **kw)
                if not (S_GRID[0] <= s <= S_GRID[-1]):
                    rows.append(dict(prompt_id=pid, seed=sd, w=w, tag=tag,
                                     kw=str(kw), s=s, pred=np.nan, extrap=True))
                    continue
                v = float(f([[np.log(w), np.log(s)]]))
                rows.append(dict(prompt_id=pid, seed=sd, w=w, tag=tag,
                                 kw=str(kw), s=s, pred=v, extrap=False))
    E = pd.DataFrame(rows)
    E["label"] = E.tag + E.kw.map(lambda k: "" if k == "{}" else
                                  " k=" + str(list(eval(k).values())[0]) if k != "{}" else "")
    E.to_csv(os.path.join(OUT, "an_grid_routes.csv"), index=False)

    print("\n=== grid-evaluated routes (pix_l2 vs the w=1,s=S0 reference) ===")
    tab = (E.groupby("label")
            .agg(mean=("pred", "mean"), med=("pred", "median"),
                 p90=("pred", lambda v: v.quantile(.9)),
                 cov=("extrap", lambda v: 1 - v.mean()))
            .sort_values("mean"))
    print(tab.to_string(float_format=lambda v: f"{v:.2f}"))

    # hold-out check: does the grid reproduce segment T where T was rendered?
    T = d[d.segment == "T"].rename(columns={"pix_l2": "obs"})
    chk = T.merge(E[["prompt_id", "seed", "w", "tag", "pred", "extrap"]],
                  left_on=["prompt_id", "seed", "w", "route"],
                  right_on=["prompt_id", "seed", "w", "tag"], how="inner")
    chk = chk[~chk.extrap]
    if len(chk):
        r = stats.spearmanr(chk.obs, chk.pred)
        print(f"\ngrid vs rendered (segment T): n={len(chk)}  "
              f"Spearman={r.statistic:.3f}  mean|err|={np.abs(chk.obs-chk.pred).mean():.2f}")
        print(chk.groupby("route").apply(
            lambda g: pd.Series({"obs": g.obs.mean(), "pred": g.pred.mean(),
                                 "n": len(g)}), include_groups=False
        ).to_string(float_format=lambda v: f"{v:.2f}"))
    return E, tab
