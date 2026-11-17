"""Figure for the E-metric arbitration (paper 2, section 8.7).

Two panels:

(a) SSIM vs w for the three routes.  This is the panel that settles the
    argument: `ws_equal` is not merely off-tone, its *structure* collapses
    (SSIM .36 at w=.2 vs .74 for the two truthful routes), while `geff_const`
    sits just above `fixed_s` everywhere.  The 1.0 anchor at w=w0 is the
    reference image itself, so it is plotted but not interpreted.

(b) Effect size of `geff_const` relative to `fixed_s`, per metric, ordered from
    the most permissive (thumbnail pixels) to the strictest (text-image CLIP).
    The monotone decline is the point: compensation buys low-level appearance,
    not semantics.  Only +9.8% would be a cherry-pick.

Run:  python e_metric_plot.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "out_emetric", "emetric_images.csv")
OUT = os.path.join(HERE, "out_emetric", "fig_Emetric_arbitration.png")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
})

C = {"fixed_s": "#4C72B0", "geff_const": "#C44E52", "ws_equal": "#55A868"}
LBL = {"fixed_s": "fixed s (status quo)", "geff_const": "geff compensation",
       "ws_equal": "equal w·s"}
ORDER = ["fixed_s", "geff_const", "ws_equal"]


def main():
    d = pd.read_csv(CSV)

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.5),
                               gridspec_kw={"width_ratios": [1.25, 1]})

    # ---------------------------------------------------------------- panel a
    piv = d.pivot_table(index="w", columns="route", values="ssim", aggfunc="mean")
    for r in ORDER:
        a.plot(piv.index, piv[r], "o-", color=C[r], label=LBL[r],
               ms=4, lw=1.6, zorder=3 if r == "ws_equal" else 2)
    a.axvline(1.0, color="0.6", ls=":", lw=.9, zorder=1)
    a.annotate("reference\n(w₀, s₀)", xy=(1.0, 1.005), xytext=(1.28, .93),
               fontsize=7, color="0.35",
               arrowprops=dict(arrowstyle="->", color="0.55", lw=.8))
    a.set_xlabel("LoRA weight scale w")
    a.set_ylabel("SSIM vs reference (higher = closer)")
    a.set_title("(a) equal w·s breaks structure, not just tone", fontsize=9)
    a.legend(frameon=False, fontsize=7, loc="lower left")
    a.set_ylim(.32, 1.06)
    a.set_xticks(sorted(d.w.unique()))
    a.grid(alpha=.25, lw=.6)

    # annotate the w=.2 gap
    y_ws = piv.loc[.2, "ws_equal"]
    y_fx = piv.loc[.2, "fixed_s"]
    a.annotate(f"−{100*(y_fx-y_ws)/y_fx:.0f}%",
               xy=(.2, (y_ws + y_fx) / 2), xytext=(.38, .52),
               fontsize=8, color="#55A868",
               arrowprops=dict(arrowstyle="->", color="#55A868", lw=.9))

    # ---------------------------------------------------------------- panel b
    gm = d.groupby("route")[["pix64", "pix512", "ssim", "clip_i_ref"]].mean()
    metrics = ["pix64", "pix512", "ssim", "clip_i_ref"]
    names = ["pix 64²\n(thumbnail)", "pix 512²\n(full res.)",
             "SSIM 512²\n(structure)", "CLIP-I\n(semantics)"]
    gain, loss = [], []
    for m in metrics:
        base = gm.loc["fixed_s", m]
        g, w = gm.loc["geff_const", m], gm.loc["ws_equal", m]
        if m.startswith("pix"):
            gain.append(-100 * (g - base) / base)
            loss.append(-100 * (w - base) / base)
        else:                       # ssim / cos: larger is closer
            gain.append(100 * (g - base) / abs(base))
            loss.append(100 * (w - base) / abs(base))

    x = np.arange(len(metrics))
    b.bar(x - .19, gain, .36, color="#C44E52", label="geff vs fixed (gain)")
    b.bar(x + .19, loss, .36, color="#55A868", label="ws vs fixed (loss)")
    b.axhline(0, color="0.3", lw=.8)
    for i, v in enumerate(gain):
        b.annotate(f"{v:+.1f}%", (i - .19, v), ha="center",
                   va="bottom" if v >= 0 else "top", fontsize=7, color="#C44E52")
    for i, v in enumerate(loss):
        # deep bars get white text *inside* the bar (green-on-green is unreadable),
        # shallow ones stay outside
        if v < -20:
            b.annotate(f"{v:+.1f}%", (i + .19, v), ha="center", va="bottom",
                       xytext=(0, 2), textcoords="offset points",
                       fontsize=7, color="white", fontweight="bold")
        else:
            b.annotate(f"{v:+.1f}%", (i + .19, v), ha="center",
                       va="top" if v < 0 else "bottom",
                       xytext=(0, -3 if v < 0 else 3), textcoords="offset points",
                       fontsize=7, color="#2F7A4A")
    b.set_xticks(x)
    b.set_xticklabels(names, fontsize=7)
    b.set_ylabel("% change vs fixed s\n(positive = gain, negative = loss)")
    b.set_title("(b) the gain shrinks as the metric gets stricter", fontsize=9)
    b.legend(frameon=False, fontsize=7, loc="upper right",
             bbox_to_anchor=(1.0, 1.0))
    b.grid(axis="y", alpha=.25, lw=.6)
    b.set_ylim(-88, 26)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
