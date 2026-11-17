"""E2c-2 figures (paper 2): the same w,s grid under five different LoRAs.

The point of the figure is NOT "kappa differs by location".  It is:

  (a) every configuration whose kappa can be measured at all sits at 0.11-0.22,
      i.e. 5-9x away from the equal-w*s anchor kappa=1, and the two s*
      estimators agree on the ordering but not on the significance of the gaps
  (b) the contours they sit on, which for the identified variants are the same
      shallow line, and for te_only is flat in w
  (c) why two variants cannot be measured: the w-driven movement of s* is
      smaller than the bootstrap error of a single s*, so their low kappa is
      absence of signal, not absence of collinearity
  (d) the pix_l2 surfaces behind (b), so the flat-in-w case is visible

Usage:  python e2d_plot.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from e2d_analyze import kappa_of, load, star_error_of, S0, W0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_e2d")
DPI = 170
ORDER = ["full", "unet_only", "no_attn2", "r4", "te_only"]
LAB = {"full": "full", "unet_only": "UNet\nonly", "no_attn2": "no\nattn",
       "r4": "rank 4", "te_only": "text\nenc only"}
CL = {"full": "#1a202c", "te_only": "#2b6cb0", "unet_only": "#c53030",
      "no_attn2": "#2f855a", "r4": "#b7791f"}
RATIO_OK = 1.5


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=.18, lw=.6)


def main():
    D = load()
    tags = [t for t in ORDER if t in set(D.tag)]
    KG = {t: kappa_of(D[D.tag == t], method="grid")[0] for t in tags}
    KQ = {t: kappa_of(D[D.tag == t], method="quad")[0] for t in tags}
    IE = {t: star_error_of(D[D.tag == t]) for t in tags}
    ratio = {t: float(IE[t].ratio.median()) for t in tags}
    ident = {t: ratio[t] >= RATIO_OK for t in tags}

    fig = plt.figure(figsize=(15.8, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, .92], hspace=.36,
                          wspace=.30)

    # ------------------------------------------------------------------ (a)
    ax = fig.add_subplot(gs[0, 0])
    for i, t in enumerate(tags):
        if not ident[t]:
            ax.axvspan(i - .45, i + .45, color="#edf2f7", zorder=0)
    for i, t in enumerate(tags):
        for K, mk, fc in ((KG[t], "o", CL[t]), (KQ[t], "s", "white")):
            v = K.kappa.values
            se = v.std() / max(1, np.sqrt(len(v)))
            ax.errorbar(i + (-.16 if mk == "o" else .16), v.mean(),
                        yerr=1.96 * se, fmt=mk, ms=5.4, color=CL[t],
                        mfc=fc, mew=1.2, capsize=2.6, lw=1.1, zorder=4)
    ax.axhline(1, color="#c53030", lw=.9, ls="--")
    ax.text(-.42, 1.03, r"equal $w\cdot s$  ($\kappa=1$)", fontsize=7.2,
            color="#c53030", va="bottom")
    ax.axhline(0, color="#718096", lw=.9, ls=":")
    ax.text(-.42, .02, r"$s$ irrelevant", fontsize=7.0, color="#718096",
            va="bottom")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([LAB[t] for t in tags], fontsize=7.4)
    for i, t in enumerate(tags):
        if not ident[t]:
            ax.text(i, .70, "not\nidentified", fontsize=6.4, color="#718096",
                    ha="center", va="top", style="italic")
    ax.set_ylim(-.16, 1.22)
    ax.set_xlim(-.55, len(tags) - .45)
    ax.set_ylabel(r"$\hat\kappa$   ($s^*=S_0(w/W_0)^{-\kappa}$)")
    ax.set_title(r"(a)  All measurable $\kappa$ sit in 0.11-0.22, not at 1",
                 fontsize=10.6, loc="left")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#4a5568",
                                  ms=5, label=r"argmin over $s$ grid"),
                       plt.Line2D([], [], marker="s", ls="", color="#4a5568",
                                  mfc="white", ms=5,
                                  label="parabolic interpolation")],
              frameon=False, fontsize=6.8, loc="upper right")
    _clean(ax)

    # ------------------------------------------------------------------ (b)
    ax = fig.add_subplot(gs[0, 1])
    for t in tags:
        g = D[D.tag == t]
        cur = []
        for w in sorted(g.w.unique()):
            v = g[g.w == w].groupby("s").pix_l2.mean()
            if len(v):
                cur.append((w, float(v.idxmin())))
        if not cur:
            continue
        w, s = zip(*cur)
        ax.plot(w, s, "-o" if ident[t] else "--",
                color=CL[t] if ident[t] else "#a0aec0",
                ms=3.2, lw=1.6 if ident[t] else 1.1,
                alpha=1.0 if ident[t] else .85, label=t)
    ww = np.linspace(.2, 2.0, 50)
    ax.plot(ww, S0 * (ww / W0) ** -1.0, ":", color="#c53030", lw=1.0,
            label=r"$\kappa=1$")
    ax.plot(ww, np.full_like(ww, S0), ":", color="#4a5568", lw=1.0,
            label=r"$\kappa=0$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([.2, .4, .8, 1.2, 2.0])
    ax.set_xticklabels([".2", ".4", ".8", "1.2", "2.0"])
    ax.minorticks_off()
    ax.set_yticks([3, 5, 7.5, 10.5, 14])
    ax.set_yticklabels(["3", "5", "7.5", "10.5", "14"])
    ax.set_xlabel("LoRA weight  w")
    ax.set_ylabel(r"empirical  $s^*(w)$")
    ax.set_title("(b)  The contour each variant sits on", fontsize=10.6,
                 loc="left")
    ax.legend(frameon=False, fontsize=6.6, ncol=2, loc="upper right")
    _clean(ax)

    # ------------------------------------------------------------------ (c)
    ax = fig.add_subplot(gs[0, 2])
    ys = np.arange(len(tags))[::-1]
    levw = {t: float(D[D.tag == t]
                     .groupby(["prompt_id", "seed", "s"]).pix_l2
                     .agg(lambda v: v.max() - v.min()).median()) for t in tags}
    for y, t in zip(ys, tags):
        ax.barh(y, ratio[t], height=.58,
                color=CL[t] if ident[t] else "#cbd5e0",
                edgecolor=CL[t] if ident[t] else "#a0aec0",
                lw=1.0, hatch=None if ident[t] else "///")
        ax.text(ratio[t] + .07, y, f"{ratio[t]:.2f}", va="center",
                fontsize=7.0, color="#2d3748")
    ax.axvline(1.0, color="#c53030", lw=1.0, ls="--")
    ax.axvline(RATIO_OK, color="#2d3748", lw=1.0, ls=":")
    ax.text(1.02, -.72, "1.0", fontsize=6.6, color="#c53030")
    ax.text(RATIO_OK + .05, -.72, "gate 1.5", fontsize=6.6, color="#2d3748")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{t}  (lev$_w$={levw[t]:.1f})" for t in tags],
                       fontsize=7.0)
    ax.set_xlabel(r"$SD_w(s^*)\;/\;$bootstrap error of one $s^*$")
    ax.set_title("(c)  Is the slope measurable at all?", fontsize=10.6,
                 loc="left")
    ax.set_xlim(0, 4.3)
    _clean(ax)
    ax.legend(handles=[Patch(facecolor="#cbd5e0", edgecolor="#a0aec0",
                             hatch="///", label="ratio < 1.5: not identified")],
              frameon=False, fontsize=6.6, loc="lower right")

    # ------------------------------------------------------------------ (d)
    show = [t for t in ("full", "unet_only", "te_only") if t in tags]
    if show:
        gsc = gs[1, :].subgridspec(2, len(show), height_ratios=[.10, 1],
                                   hspace=.03, wspace=.10)
        tit = fig.add_subplot(gsc[0, :])
        tit.axis("off")
        tit.set_title("(d)  pix_l2 surface: te_only is almost flat along w, "
                      "which is why its kappa cannot be estimated",
                      fontsize=10.6, loc="left")
        vmax = float(np.percentile(
            np.concatenate([D[D.tag == t].pix_l2.values for t in show]), 95))
        W = sorted(D.w.unique())
        axes = []
        for k, t in enumerate(show):
            sub = fig.add_subplot(gsc[1, k], sharey=axes[0] if axes else None)
            g = D[D.tag == t]
            M = (g.groupby(["w", "s"]).pix_l2.mean()
                 .unstack().reindex(index=W, columns=sorted(g.s.unique())))
            im = sub.imshow(M.values, aspect="auto", origin="lower",
                            cmap="magma_r", vmin=0, vmax=vmax)
            sub.set_xticks(range(len(M.columns)))
            sub.set_xticklabels([f"{s:g}" for s in M.columns], fontsize=5.8,
                                rotation=90)
            sub.set_yticks(range(len(W)))
            if k == 0:
                sub.set_yticklabels([f"{w:g}" for w in W], fontsize=6)
                sub.set_ylabel("LoRA weight  w", fontsize=8)
            else:
                sub.set_yticklabels([])
            sub.set_xlabel("guidance  s", fontsize=8)
            sub.set_title(f"{t}   (kappa={KG[t].kappa.mean():.3f})",
                          fontsize=8.4)
            sub.grid(False)
            axes.append(sub)
        cb = fig.colorbar(im, ax=axes, fraction=.020, pad=.010)
        cb.set_label("pix_l2", fontsize=7.4)

    p = os.path.join(OUT, "fig_E2c2_variants.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    print("wrote", p)
    print("\nkappa(grid/quad) and identifiability ratio:")
    for t in tags:
        print(f"  {t:10s} grid={KG[t].kappa.mean():+.3f}  "
              f"quad={KQ[t].kappa.mean():+.3f}  ratio={ratio[t]:.2f}  "
              f"{'ok' if ident[t] else 'NOT IDENTIFIED'}")


if __name__ == "__main__":
    main()
