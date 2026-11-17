"""L2 figures (paper 2).  English labels, dpi=170.

fig_L2_a  leverage      : pix_l2 range when s moves 4.7x vs when w moves 10x
fig_L2_b  iso-line      : empirical s*(w) with the three route curves overlaid
fig_L2_c  routes        : pix_l2 vs w for the three routes, plus the piecewise
fig_L2_d  piecewise kappa: kappa below and above w=1, paired per prompt-seed
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_long2")
DPI = 170
W0, S0 = 1.0, 7.5
W_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
RHO = {"object": 0.208, "person": 0.295, "style": 0.200, "_all": 0.234}
CC = {"object": "#2b6cb0", "person": "#c05621", "style": "#6b46c1"}
RC = {"fixed_s": "#a0aec0", "geff_const": "#2b6cb0", "ws_equal": "#c53030",
      "piecewise": "#2f855a"}


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=.18, lw=.6)


def fig_a():
    sr = pd.read_csv(os.path.join(OUT, "an_s_range.csv"))
    wr = pd.read_csv(os.path.join(OUT, "an_w_range.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.5))
    for a, df, key, title, xlab in [
            (ax[0], sr, "w", r"(a) sweep s while w is fixed", "w"),
            (ax[1], wr, "s", r"(b) sweep w while s is fixed", "s")]:
        data = [df[df[key] == k].rng.values for k in sorted(df[key].unique())]
        bp = a.boxplot(data, positions=range(len(data)), widths=.6,
                       patch_artist=True, showfliers=False)
        for p in bp["boxes"]:
            p.set_facecolor("#bee3f8"); p.set_edgecolor("#2b6cb0"); p.set_alpha(.85)
        for m in bp["medians"]:
            m.set_color("#c53030"); m.set_lw(1.4)
        a.set_xticks(range(len(data)))
        a.set_xticklabels([f"{v:g}" for v in sorted(df[key].unique())], fontsize=7.5)
        a.set_xlabel(xlab, fontsize=9)
        a.set_title(title, fontsize=9.5)
        a.set_ylim(0, 26)
        _clean(a)
    ax[0].set_ylabel(r"range of pix$_{L2}$ inside the cell", fontsize=9)
    ax[0].axhline(sr.rng.mean(), color="#2b6cb0", ls="--", lw=1)
    ax[1].axhline(wr.rng.mean(), color="#c53030", ls="--", lw=1)
    ax[0].text(.02, .95, f"mean {sr.rng.mean():.2f}", transform=ax[0].transAxes,
               fontsize=8, color="#2b6cb0", va="top")
    ax[1].text(.02, .95, f"mean {wr.rng.mean():.2f}", transform=ax[1].transAxes,
               fontsize=8, color="#c53030", va="top")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_L2_a_leverage.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_b():
    E = pd.read_csv(os.path.join(OUT, "an_isoline.csv"))
    K = pd.read_csv(os.path.join(OUT, "an_kappa.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6))
    for pcl in ("object", "person", "style"):
        g = E[E.pclass == pcl].groupby("w").s_star.agg(["mean", "std"])
        ax[0].errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=4.5,
                       capsize=2.5, lw=1.4, color=CC[pcl], label=pcl)
    ws = np.linspace(S0 / 14., 2, 60)
    ax[0].plot(ws, S0 / ws, color=RC["ws_equal"], ls="--", lw=1.3,
               label=r"equal $w\cdot s$ ($\kappa$=1)")
    ax[0].set_ylim(1.5, 15.5)
    for pcl in ("object", "person", "style"):
        rho = RHO[pcl]
        ax[0].plot(ws, S0 * (1 + W0 * rho) / (1 + ws * rho), color=CC[pcl],
                   ls=":", lw=1.2, alpha=.9)
    ax[0].plot(ws, S0 * (ws / W0) ** -0.213, color="black", lw=1.4,
               label=r"fitted $\kappa$=0.21")
    ax[0].axhline(S0, color=RC["fixed_s"], ls="-.", lw=1.2, label="fixed s")
    ax[0].set_xscale("log"); ax[0].set_xticks(W_GRID)
    ax[0].set_xticklabels([f"{v:g}" for v in W_GRID], fontsize=7.5)
    ax[0].set_xlabel("LoRA weight scale w", fontsize=9)
    ax[0].set_ylabel(r"best-guess $s^*(w)$ (closest to the reference)", fontsize=9)
    ax[0].set_title("(a) the iso-line is flatter than equal-$w\\cdot s$", fontsize=9.5)
    ax[0].legend(fontsize=6.6, frameon=False, ncol=2)
    _clean(ax[0])

    ax[1].hist(K.kappa, bins=np.arange(-.2, .55, .05), color="#2b6cb0",
               alpha=.8, edgecolor="white")
    ax[1].set_xlim(-.25, 1.12)
    ax[1].set_ylim(0, 21)
    for x, c, lab, dy in [(K.kappa.median(), "#c53030",
                           f"measured {K.kappa.median():.2f}", 19.4),
                          (.177, "#2f855a", "closed form .18", 16.4),
                          (1.0, "#718096", r"equal $w\cdot s$ = 1.0", 13.4)]:
        ax[1].axvline(x, color=c, lw=1.4, ls="--")
        ax[1].annotate(lab, (x, dy), fontsize=7, color=c,
                       xytext=(4 if x > .9 else 4, 0),
                       textcoords="offset points", va="center")
    ax[1].set_xlabel(r"fitted $\kappa$ per prompt-seed", fontsize=9)
    ax[1].set_ylabel("count", fontsize=9)
    ax[1].set_title(r"(b) 70 independent fits, 4.7x away from equal $w\cdot s$",
                    fontsize=9.5)
    _clean(ax[1])
    fig.tight_layout()
    p = os.path.join(OUT, "fig_L2_b_isoline.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_c():
    byw = pd.read_csv(os.path.join(OUT, "an_routes_byw.csv"))
    Pi = pd.read_csv(os.path.join(OUT, "an_kappa_piecewise.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6))
    for r, lab in [("fixed_s", "fixed s (status quo)"),
                   ("geff_const", r"$g_{eff}$ const (measured $\rho$)"),
                   ("ws_equal", r"equal $w\cdot s$")]:
        ax[0].plot(byw.w, byw[r], marker="o", ms=5, lw=1.5, color=RC[r], label=lab)
    pw = S0 if True else None
    # the piecewise rule rendered on the grid: w<=1 keeps s, w>1 drops with k=.7
    E = pd.read_csv(os.path.join(OUT, "an_grid_routes.csv"))
    lab = "piecewise " + str({"khi": 0.7})
    g = E[E.label == "piecewise k=0.7"]
    if len(g):
        s = g.groupby("w").pred.mean()
        ax[0].plot(s.index, s.values, marker="^", ms=5, lw=1.3, ls=":",
                   color=RC["piecewise"], label="piecewise (grid)")
    ax[0].set_xscale("log"); ax[0].set_xticks(W_GRID)
    ax[0].set_xticklabels([f"{v:g}" for v in W_GRID], fontsize=7.5)
    ax[0].set_xlabel("w", fontsize=9)
    ax[0].set_ylabel(r"mean pix$_{L2}$ vs the reference image", fontsize=9)
    ax[0].set_title("(c) compensation helps a little, equal-$w\\cdot s$ hurts a lot",
                    fontsize=9.5)
    ax[0].legend(fontsize=6.8, frameon=False)
    _clean(ax[0])

    ax[1].scatter(Pi.kappa_lo, Pi.kappa_hi, s=34, c=[CC.get(c, "#4a5568")
                                                    for c in Pi.pclass],
                  alpha=.75, edgecolor="white", lw=.6)
    lim = (-.45, 2.2)
    ax[1].plot(lim, lim, color="#a0aec0", ls="--", lw=1)
    ax[1].axhline(1.0, color=RC["ws_equal"], ls=":", lw=1)
    ax[1].axhline(0.0, color=RC["fixed_s"], ls=":", lw=1)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel(r"$\kappa$ for w $\leq$ 0.8", fontsize=9)
    ax[1].set_ylabel(r"$\kappa$ for w $\geq$ 1.2", fontsize=9)
    ax[1].set_title(r"(d) two regimes: flat below, near-linear above "
                    r"(p=8.7e-13)", fontsize=9.5)
    for pcl in ("object", "person", "style"):
        ax[1].scatter([], [], s=30, c=CC[pcl], label=pcl)
    ax[1].legend(fontsize=6.8, frameon=False, loc="lower right")
    _clean(ax[1])
    fig.tight_layout()
    p = os.path.join(OUT, "fig_L2_c_routes.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    for f in (fig_a, fig_b, fig_c):
        print("wrote", f())
