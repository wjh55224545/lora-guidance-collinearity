"""Y4: where do real-world (w, s) settings sit on the s*(w) plane, and in which
w-interval does the compensation demand actually appear?

Pure analysis on L2_2026-11-06/csv/L2_images.csv.  No GPU, no images.

Why this exists
---------------
The paper reports a single global kappa = 0.211.  A global slope hides the fact
that s* only leaves the grid centre for part of the range.  If the demand for
s-compensation is concentrated at the ends, the actionable advice is not
"multiply s by w^-0.211" but "inside the usable band, w and s behave as separate
knobs -- outside it, they do not".  That is a much sharper deployment statement
and it is testable on data we already have.

Method
------
s*(w) is estimated exactly as in e2d_analyze.kappa_of(method="grid"): for each
(prompt, seed) group, argmin over the 9 s-values of the mean displacement, with
edge hits dropped.  Then:

  * the group-median curve s*_med(w) is the descriptive object,
  * local kappa over adjacent w-pairs = -d log s* / d log w, with a
    group-level bootstrap CI (resample the 91 groups, not the rows),
  * practice settings are placed on the plane; for each we report the local
    kappa of the band it sits in.

Red line carried over from the log: these are positions, not verdicts.  We never
write that a cited work configured anything wrong.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "results", "论文二_LoRA共线",
                   "L2_2026-11-06", "csv", "L2_images.csv")
OUT = os.path.join(HERE, "out_y", "y4")

W0, S0 = 1.0, 7.5
W_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
S_GRID = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]

# literature / community settings.  "s" is guidance_scale; for the FLUX-based
# entry the absolute value is not transferable, so it is marked direction-only.
PRACTICE = [
    (1.0, 7.5, "SD1.5 ecosystem default (also this paper's ref)"),
    (0.7, 7.5, "community: lower w only"),
    (1.0, 6.0, "community: lower s by 20%"),
    (0.7, 6.0, "community: lower w and s together"),
    (1.0, 3.5, "LoRAShop gs=3.5 (FLUX; direction only)"),
    (2.0, 7.5, "strong LoRA, s left alone"),
]


def load():
    d = pd.read_csv(SRC)
    d = d[d.route == "grid"].copy()
    d = d[d.s.isin(S_GRID) & d.w.isin(W_GRID)]
    return d


def groups(d):
    # one group holds 8 w x 9 s = 72 cells, not ~100
    return [(k, g) for k, g in d.groupby(["prompt_id", "seed"]) if len(g) >= 60]


def sstar_curve(gs, w_list=W_GRID, method="grid"):
    """per-group s*(w).

    method="grid"  argmin over the 9 s-values (the L2 estimator).  Its output is
                   quantised to 9 levels, so a bootstrap CI collapses to a point
                   whenever the median lands on the same level -- that is a
                   resolution limit, not precision.
    method="local" parabola through the argmin and its two neighbours.  Continuous,
                   so the bootstrap is informative, and local (a global parabola
                   on a V-shaped curve is biased by the far shoulders).
    NaN when the argmin sits on an edge or data is missing.
    """
    rows = []
    for (pid, sd), h in gs:
        for w in w_list:
            v = h[h.w == w].groupby("s")["pix_l2"].mean().dropna()
            v = v.reindex(S_GRID)
            if v.notna().sum() < 3:
                rows.append(dict(prompt_id=pid, seed=sd, w=w, s_star=np.nan))
                continue
            s_best = float(v.idxmin())
            if s_best in (min(S_GRID), max(S_GRID)):
                rows.append(dict(prompt_id=pid, seed=sd, w=w, s_star=np.nan))
                continue
            if method == "local":
                i = S_GRID.index(s_best)
                xs = np.array(S_GRID[i - 1:i + 2], float)
                ys = np.array([v[S_GRID[i - 1]], v[s_best], v[S_GRID[i + 1]]], float)
                if np.all(np.isfinite(ys)):
                    c = np.polyfit(xs, ys, 2)
                    if c[0] > 0:
                        vert = -c[1] / (2 * c[0])
                        if xs[0] <= vert <= xs[-1]:
                            s_best = float(vert)
            rows.append(dict(prompt_id=pid, seed=sd, w=w, s_star=s_best))
    return pd.DataFrame(rows)


def med_curve(tab):
    return tab.groupby("w")["s_star"].median().reindex(W_GRID)


def local_kappa(curve, w_list=W_GRID):
    """adjacent-pair log-log slope; -inf guards removed."""
    out = []
    for a, b in zip(w_list[:-1], w_list[1:]):
        ya, yb = curve.get(a, np.nan), curve.get(b, np.nan)
        if not (np.isfinite(ya) and np.isfinite(yb)) or ya <= 0 or yb <= 0:
            out.append(np.nan)
            continue
        out.append(-(np.log(yb) - np.log(ya)) / (np.log(b) - np.log(a)))
    return np.array(out, float)


def boot_local(tab, n_boot=2000, seed=7):
    """resample whole groups, so the CI reflects prompt-to-prompt spread."""
    gk = list(tab.groupby(["prompt_id", "seed"]))
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(gk), len(gk))
        sub = pd.concat([gk[i][1] for i in pick])
        acc.append(local_kappa(med_curve(sub)))
    return np.array(acc)


def surf(d):
    return d.groupby(["w", "s"])["pix_l2"].median().unstack().reindex(
        index=W_GRID, columns=S_GRID)


def interp_D(S, w, s):
    """log-log bilinear on the 8x9 median surface."""
    lw = np.log(W_GRID)
    ls = np.log(S_GRID)
    x, y = np.log(w), np.log(s)
    x = float(np.clip(x, lw[0], lw[-1]))
    y = float(np.clip(y, ls[0], ls[-1]))
    i = int(np.clip(np.searchsorted(lw, x) - 1, 0, len(lw) - 2))
    j = int(np.clip(np.searchsorted(ls, y) - 1, 0, len(ls) - 2))
    fx = (x - lw[i]) / (lw[i + 1] - lw[i])
    fy = (y - ls[j]) / (ls[j + 1] - ls[j])
    z = S.values
    return float(z[i, j] * (1 - fx) * (1 - fy) + z[i + 1, j] * fx * (1 - fy)
                 + z[i, j + 1] * (1 - fx) * fy + z[i + 1, j + 1] * fx * fy)


def band_of(w):
    if w <= 0.8:
        return "low (w<=0.8)"
    if w >= 1.5:
        return "high (w>=1.5)"
    return "usable band (0.8<w<1.5)"


def s_stack(curve, w):
    """log-log linear read-off of s*(w) at an off-grid w (e.g. the 0.7 used by
    community guides); returns NaN if either bracketing point is missing."""
    ws = [x for x in W_GRID if np.isfinite(curve.get(x, np.nan))]
    if not ws:
        return np.nan
    if w <= min(ws) or w >= max(ws):
        return float(curve[min(ws, key=lambda x: abs(x - w))])
    for a, b in zip(ws[:-1], ws[1:]):
        if a <= w <= b:
            ya, yb = curve[a], curve[b]
            t = (np.log(w) - np.log(a)) / (np.log(b) - np.log(a))
            return float(np.exp(np.log(ya) * (1 - t) + np.log(yb) * t))
    return np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    gs = groups(d)
    print(f"grid rows {len(d)}, groups {len(gs)}", flush=True)

    S = surf(d)
    S.round(4).to_csv(os.path.join(OUT, "y4_surface.csv"), index_label="w")

    tab = sstar_curve(gs, method="grid")
    tab.to_csv(os.path.join(OUT, "y4_sstar_rows.csv"), index=False)
    tab_q = sstar_curve(gs, method="local")
    tab_q.to_csv(os.path.join(OUT, "y4_sstar_rows_local.csv"), index=False)

    curve = med_curve(tab)
    share75 = tab.groupby("w")["s_star"].apply(
        lambda v: float((v == 7.5).mean())).reindex(W_GRID)

    # the continuous estimator, used for the local kappa and its CI
    curve_q = tab_q.groupby("w")["s_star"].median().reindex(W_GRID)

    lk = local_kappa(curve_q)
    bl = boot_local(tab_q)
    lo = np.nanpercentile(bl, 2.5, axis=0)
    hi = np.nanpercentile(bl, 97.5, axis=0)

    lk_tab = pd.DataFrame(dict(
        w_lo=W_GRID[:-1], w_hi=W_GRID[1:],
        s_lo=curve.values[:-1], s_hi=curve.values[1:],
        s_lo_local=curve_q.values[:-1], s_hi_local=curve_q.values[1:],
        kappa_grid=local_kappa(curve),
        kappa_local=lk, kappa_lo=lo, kappa_hi=hi,
        share_s_at_7p5_lo=share75.values[:-1],
        share_s_at_7p5_hi=share75.values[1:]))
    lk_tab.to_csv(os.path.join(OUT, "y4_local_kappa.csv"), index=False)

    print("\n--- s*(w) ---")
    for w in W_GRID:
        print(f"  w={w:4.1f}  s*={curve[w]:5.2f} (local {curve_q[w]:5.2f})   "
              f"P(s*=7.5)={share75[w]:.2f}   band={band_of(w)}")
    print("\n--- local kappa (adjacent pairs, continuous estimator) ---")
    for _, r in lk_tab.iterrows():
        print(f"  w {r.w_lo:.1f}->{r.w_hi:.1f}  kappa={r.kappa_local:+.3f} "
              f"[{r.kappa_lo:+.3f},{r.kappa_hi:+.3f}]  "
              f"(grid est {r.kappa_grid:+.3f})  band={band_of(r.w_hi)}")

    # global, for the record
    for nm, cv in (("grid", curve), ("local", curve_q)):
        xs, ys = [], []
        for w in W_GRID:
            if w == W0 or not np.isfinite(cv[w]):
                continue
            xs.append(np.log(w / W0))
            ys.append(np.log(cv[w] / S0))
        print(f"\nglobal kappa on the {nm} median curve = "
              f"{-np.polyfit(xs, ys, 1)[0]:.3f}  (L2 reported 0.211)")

    # ---- practice settings -------------------------------------------------
    print("\n--- practice settings on the plane ---")
    rows = []
    for w, s, label in PRACTICE:
        sstar = s_stack(curve_q, w)
        dmeas = interp_D(S, w, s)
        dx = np.log(s / sstar) if np.isfinite(sstar) and sstar > 0 else np.nan
        band = band_of(w)
        # which local kappa governs this point?
        pairing = lk_tab[(lk_tab.w_lo <= w) & (lk_tab.w_hi >= w)]
        if not len(pairing):
            pairing = lk_tab.iloc[[int(np.argmin(np.abs(lk_tab.w_hi - w)))]]
        k_loc = float(pairing.kappa_local.mean())
        rows.append(dict(w=w, s=s, label=label, pix_l2=dmeas,
                         s_star=sstar, log_s_over_sstar=dx, band=band,
                         kappa_local=k_loc))
        print(f"  w={w:4.1f} s={s:5.2f}  D={dmeas:6.2f}  s*={sstar:5.2f}  "
              f"log(s/s*)={dx:+.3f}  band={band}  kappa_local={k_loc:+.3f}"
              f"   {label}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "y4_practice.csv"), index=False)

    with open(os.path.join(OUT, "y4_readme.txt"), "w", encoding="utf-8") as f:
        f.write(f"source {SRC}\ngrid rows {len(d)} groups {len(gs)}\n"
                f"W0={W0} S0={S0}\nbootstrap {bl.shape[0]} x group resample\n"
                "estimator: grid = argmin over 9 s-levels (L2), "
                "local = parabola through argmin +-1 level (continuous)\n"
                "local kappa and its CI use the continuous estimator; "
                "the grid estimator's CI degenerates because s* is quantised.\n")
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
