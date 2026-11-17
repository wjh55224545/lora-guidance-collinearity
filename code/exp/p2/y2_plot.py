"""Y2 figure: does a 0.5-step s grid change kappa?

Three things this figure has to make visible at once, because they are the whole
argument:

(a) the s*(w) curve on the dense grid against the coarse grid.  If the coarse
    grid were censoring the slope, the dense curve would be visibly steeper.
    It is not -- it is flatter, and the two agree to within the marker size for
    w <= 1.2.
(b) kappa per cell.  Eight cells is few, so the spread matters as much as the
    mean: the two flattest-fitted cells (R^2 ~ .03) are flagged.
(c) grid estimator against parabola-refined estimator on the same cells.  On the
    coarse grid these two differed by 0.038; if the grid were binding, the dense
    grid would leave them different.  They collapse onto the identity line.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out_y")
W0, S0 = 1.0, 7.5

plt.rcParams.update({
    "figure.dpi": 190, "savefig.dpi": 190, "font.size": 8.3,
    "axes.edgecolor": "#444", "axes.grid": True, "grid.color": "#e8e8e8",
    "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
})
C = {"dense": "#1f6fb4", "coarse": "#9a9a9a", "ref": "#c0392b"}


def sstar_table(tag):
    p = os.path.join(OUT, tag, "images.csv")
    D = pd.read_csv(p)
    D = D[D.w.notna() & D.s.notna()]
    rows = []
    for (pid, sd), h in D.groupby(["prompt_id", "seed"]):
        for w in sorted(h.w.unique()):
            if w == W0:
                continue
            v = h[h.w == w].groupby("s")["pix_l2"].mean().dropna().sort_index()
            if len(v) < 5:
                continue
            rows.append(dict(prompt_id=pid, seed=sd, w=w,
                             s_star=float(v.idxmin())))
    return pd.DataFrame(rows)


def main():
    S = sstar_table("y2_full")
    S.to_csv(os.path.join(OUT, "an_y2_full_sstar.csv"), index=False)
    med = S.groupby("w").s_star.median()
    q1 = S.groupby("w").s_star.quantile(.25)
    q3 = S.groupby("w").s_star.quantile(.75)

    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.5),
                           gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0]))

    # ---- (a) s*(w): dense vs coarse -------------------------------------
    a = ax[0]
    xg = np.linspace(-1.75, 0.80, 100)
    yq = np.log(S0 * W0 / np.exp(xg))
    vis = (np.exp(yq) >= 3.0) & (np.exp(yq) <= 12.0)
    a.plot(xg[vis], yq[vis], "--", lw=1.3, color=C["coarse"],
           label=r"equal $w\cdot s$  ($\kappa=1$)")

    y4 = os.path.join(HERE, "out_y", "y4", "y4_sstar_rows_local.csv")
    if os.path.exists(y4):
        c = pd.read_csv(y4).groupby("w").s_star.median()
        a.plot(np.log(c.index / W0), np.log(c.values), "-o", lw=1.2, ms=3.4,
               color=C["coarse"], alpha=0.9,
               label="coarse grid, 9 s-levels (Y4)")

    xw = np.log(med.index / W0)
    a.fill_between(xw, np.log(q1.values), np.log(q3.values),
                   color=C["dense"], alpha=0.18, lw=0)
    a.plot(xw, np.log(med.values), "-o", lw=1.9, ms=5, color=C["dense"],
           label="dense grid, 23 s-levels (Y2)")

    a.set_xlabel(r"$\log\,(w/W_0)$")
    a.set_ylabel(r"$\log\,s$")
    a.set_title("(a)  refining the s grid does not steepen $s^*(w)$",
                fontsize=8.6, loc="left")
    a.set_ylim(np.log(3.0), np.log(12.0))
    yt = [3, 4, 5, 6, 7.5, 9, 10, 12]
    a.set_yticks(np.log(yt))
    a.set_yticklabels([f"{v:g}" for v in yt])
    a.legend(fontsize=6.3, loc="lower left", framealpha=0.93)

    # ---- (b) per-cell kappa ---------------------------------------------
    b = ax[1]
    L = pd.read_csv(os.path.join(OUT, "an_y2_full_local.csv"))
    G = pd.read_csv(os.path.join(OUT, "an_y2_full_grid.csv"))
    m = L.merge(G, on=["prompt_id", "seed"], suffixes=("_loc", "_grid"))
    lowr2 = m.r2_loc < 0.15
    b.scatter(m.kappa_loc[~lowr2], m.r2_loc[~lowr2], s=46, color=C["dense"],
              zorder=5, edgecolor="white", lw=0.8, label="fitted (R² ≥ .15)")
    b.scatter(m.kappa_loc[lowr2], m.r2_loc[lowr2], s=52, facecolor="none",
              edgecolor=C["ref"], lw=1.3, zorder=5, label="barely fitted (R² < .15)")
    b.axvline(m.kappa_loc.mean(), ls="-", lw=1.3, color=C["dense"], alpha=0.6)
    b.text(0.02, 0.97,
           f"mean {m.kappa_loc.mean():.3f}\n"
           f"median {m.kappa_loc.median():.3f}",
           transform=b.transAxes, fontsize=6.5, color=C["dense"], va="top")
    b.axvline(0.211, ls="--", lw=1.2, color=C["ref"])
    b.annotate("L2 coarse, 91 cells: 0.211", xy=(0.211, 0.30),
               xytext=(0.335, 0.44), fontsize=6.2, color=C["ref"],
               ha="right", va="center",
               arrowprops=dict(arrowstyle="-", lw=0.7, color=C["ref"]))
    b.set_xlabel(r"$\kappa$ (parabola-refined, per cell)")
    b.set_ylabel(r"fit $R^2$")
    b.set_title("(b)  eight cells, two of them flat", fontsize=8.6, loc="left")
    b.set_xlim(-0.02, 0.34)
    b.set_ylim(-0.03, 1.0)
    b.legend(fontsize=6.0, loc="lower right", framealpha=0.9)

    # ---- (c) estimator agreement ----------------------------------------
    c = ax[2]
    lim = (0, 0.36)
    c.plot(lim, lim, "-", lw=1.0, color="#bbb", zorder=1)
    c.scatter(m.kappa_grid, m.kappa_loc, s=46, color=C["dense"], zorder=5,
              edgecolor="white", lw=0.8)
    d = m.kappa_loc - m.kappa_grid
    c.text(0.04, 0.96,
           f"shift = {d.mean():+.3f}  (sd {d.std(ddof=1):.3f})\n"
           f"all |Δ| ≤ {d.abs().max():.3f}\n"
           "coarse grid had 0.038 between these",
           transform=c.transAxes, fontsize=6.5, va="top", color="#333")
    c.set_xlim(*lim)
    c.set_ylim(*lim)
    c.set_xlabel(r"$\kappa$ (grid argmin)")
    c.set_ylabel(r"$\kappa$ (parabola-refined)")
    c.set_title("(c)  the two estimators collapse", fontsize=8.6, loc="left")

    fig.tight_layout()
    p = os.path.join(OUT, "fig_y2_full.png")
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)
    print(med.round(2).to_string())
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
