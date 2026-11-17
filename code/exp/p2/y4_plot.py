"""Y4 figures: the s*(w) plane, and where the compensation demand actually lives.

Two panels, one message each.

(a) How much guidance does each w need?  The measured s*(w) never leaves 4.9-9.0.
    The equal-w*s rule, by contrast, asks for s=3.75 at w=2 and s=37.5 at w=0.2 --
    the second of which is above every guidance grid used in this paper, so the
    rule is not merely wrong here, it is not executable.  Panel (a) is drawn on
    the range where measurements exist; the off-scale demand is annotated rather
    than drawn, otherwise one curve would flatten the other into a line.

    Well depth D_min(w) is printed along the top: s* is only an equal-response
    coordinate where a well exists.  At w=0.2 and w=2.0 the best s still leaves
    the image 6.7 / 9.1 L2 away from the reference, so s* there is "least bad",
    not "equivalent" -- those points are ringed in red.

(b) local kappa per adjacent w-pair with the group bootstrap CI.  Grey marks the
    pairs whose endpoints sit in a shallow well; their numbers are reported but
    must not be used to argue either way.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y", "y4")

W0, S0 = 1.0, 7.5
DEEP = 4.0
YLO, YHI = 3.0, 12.6

plt.rcParams.update({
    "figure.dpi": 190, "savefig.dpi": 190, "font.size": 8.5,
    "axes.edgecolor": "#444", "axes.labelcolor": "#111",
    "text.color": "#111", "xtick.color": "#333", "ytick.color": "#333",
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

C_MEAS = "#1f6fb4"
C_EQ = "#8a8a8a"
C_BAD = "#d1495b"


def main():
    S = pd.read_csv(os.path.join(OUT, "y4_surface.csv"), index_col=0)
    S.columns = [float(c) for c in S.columns]
    S.index = [float(i) for i in S.index]
    lk = pd.read_csv(os.path.join(OUT, "y4_local_kappa.csv"))
    sq = pd.read_csv(os.path.join(OUT, "y4_sstar_rows_local.csv"))
    pr = pd.read_csv(os.path.join(OUT, "y4_practice.csv"))

    dmin = S.min(axis=1)
    cg = sq.groupby("w")["s_star"].median()
    W = list(cg.index)

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.3),
                           gridspec_kw=dict(width_ratios=[1.10, 1.0]))
    fig.subplots_adjust(wspace=0.28)

    # ---------- (a) how much s does each w need ---------------------------
    a = ax[0]
    xw = np.log(np.array(W) / W0)

    xr = np.linspace(-1.78, 0.82, 200)
    y_eq = np.log(S0 * W0 / np.exp(xr))
    vis = np.exp(y_eq) <= YHI
    a.plot(xr[vis], y_eq[vis], "--", lw=1.5, color=C_EQ, zorder=2,
           label=r"equal $w\cdot s$:  $s=S_0W_0/w$")
    i0 = int(np.argmax(vis))
    a.plot([xr[i0], xr[i0]], [y_eq[i0], np.log(YHI)], ":", lw=1.2,
           color=C_EQ, zorder=2)

    a.plot(xw, np.log(cg.values), "-", lw=1.9, color=C_MEAS, zorder=4,
           label=r"measured $s^*(w)$  (parabola-refined)")
    a.scatter(xw, np.log(cg.values), s=34, color=C_MEAS, zorder=5,
              edgecolor="white", linewidth=0.9)
    a.scatter([], [], s=130, facecolor="none", edgecolor=C_BAD, lw=1.4,
              label=r"shallow well ($D_{\min}>%g$)" % DEEP)
    for w in W:
        if dmin[w] > DEEP:
            a.scatter([np.log(w / W0)], [np.log(cg[w])], s=165,
                      facecolor="none", edgecolor=C_BAD, lw=1.4, zorder=6)

    styles = [("SD1.5", "#2a9d8f", "o", r"SD1.5 default (= $W_0,S_0$)"),
              ("community: lower w only", "#e08a1e", "s", r"lower $w$ only"),
              ("community: lower s by 20%", "#e08a1e", "^", r"lower $s$ 20%"),
              ("community: lower w and s together", "#e08a1e", "D",
               r"lower both"),
              ("LoRAShop", "#8d6bb5", "v", "LoRAShop gs=3.5 (FLUX, dir. only)"),
              ("strong LoRA", "#444444", "P", r"strong LoRA, $s$ untouched")]
    for _, r in pr.iterrows():
        for key, col, mk, lab in styles:
            if r.label.startswith(key):
                a.scatter([np.log(r.w / W0)], [np.log(r.s)], s=52, marker=mk,
                          color=col, zorder=8, edgecolor="white",
                          linewidth=0.7, label=lab)
                break

    a.text(-0.18, np.log(11.6),
           r"$\uparrow$ off scale: the rule asks for $s=37.5$ at $w=0.2$,"
           "\n" r"above every guidance grid in this paper",
           ha="left", va="top", fontsize=6.4, color="#7a7a7a")

    a.text(0.995, 0.985,
           r"$D_{\min}$:  " + "  ".join(f"{dmin[w]:.1f}" for w in W),
           transform=a.transAxes, ha="right", va="top", fontsize=5.6,
           color="#9a9a9a")

    a.set_xlabel(r"$\log\,(w/W_0)$")
    a.set_ylabel(r"$\log\,s$")
    a.set_title(r"(a)  what each $w$ actually needs", fontsize=8.8, loc="left")
    a.set_xlim(-1.78, 0.82)
    a.set_ylim(np.log(YLO), np.log(YHI))
    yt = [3, 4, 5, 6, 7.5, 9, 10, 12]
    a.set_yticks(np.log(yt))
    a.set_yticklabels([f"{v:g}" for v in yt])
    a.legend(fontsize=6.1, loc="lower left", framealpha=0.95,
             bbox_to_anchor=(-0.012, -0.012), borderpad=0.45,
             labelspacing=0.30, handletextpad=0.5)

    # ---------- (b) local kappa ------------------------------------------
    b = ax[1]
    n = len(lk)
    bad = np.array([(dmin[a_] > DEEP) or (dmin[b_] > DEEP)
                    for a_, b_ in zip(lk.w_lo, lk.w_hi)])
    for i, (k, lo, hi, isbad) in enumerate(zip(lk.kappa_local, lk.kappa_lo,
                                               lk.kappa_hi, bad)):
        col = "#bdbdbd" if isbad else C_MEAS
        b.plot([i, i], [lo, hi], "-", lw=2.4, color=col, solid_capstyle="round")
        b.plot([i - 0.16, i + 0.16], [lo, lo], "-", lw=0.9, color=col)
        b.plot([i - 0.16, i + 0.16], [hi, hi], "-", lw=0.9, color=col)
        b.scatter([i], [k], s=46, color=col, zorder=5,
                  edgecolor="#555" if isbad else "white", linewidth=0.8)
        b.annotate(f"{k:.2f}", (i, hi + 0.04), ha="center", va="bottom",
                   fontsize=6.0, color=col if not isbad else "#8f8f8f")
    b.axhline(1.0, ls="--", lw=1.2, color=C_EQ)
    b.text(0.02, 1.015, r"$\kappa=1$ (equal $w\cdot s$)", fontsize=7,
           color="#888", transform=b.get_yaxis_transform())
    b.axhline(0.0, ls=":", lw=1.0, color="#c8c8c8")

    b.set_xticks(np.arange(n))
    b.set_xticklabels([f"{a_:g}→{c:g}" for a_, c in zip(lk.w_lo, lk.w_hi)],
                      rotation=38, ha="right", fontsize=7.2)
    b.set_xlim(-0.6, n - 0.4)
    b.set_ylabel(r"local $\kappa=-\Delta\log s^*\,/\,\Delta\log w$")
    b.set_xlabel(r"$w$ pair")
    b.set_title("(b)  the slope is not one number", fontsize=8.8, loc="left")
    b.set_ylim(-0.18, 1.34)
    b.grid(True, axis="y")
    b.grid(False, axis="x")
    b.text(0.985, 0.02, "grey: an endpoint sits in a shallow well;\n"
                        "reported, not evidence either way",
           transform=b.transAxes, fontsize=6.2, color="#999", va="bottom",
           ha="right")

    p = os.path.join(OUT, "fig_Y4.png")
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    main()
