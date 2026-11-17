"""E3a analysis: turn the (w,s) surface into the two numbers the paper needs.

1. Coupling strength.  Define the effective-guidance inflation

       infl(w) = g_eff(w, s0) / s0 - 1

   This is how much guidance a LoRA silently adds at a fixed slider value.  It
   is the direct quantitative statement of "the two knobs are not independent",
   and unlike a correlation it is in the units of the interface (CFG scale).

2. Coupling slope.  From two neighbouring s values,

       1 + w*rho  ~=  (g_eff(s+ds) - g_eff(s)) / ds

   so w*rho is recoverable without a separate measurement.  The log's
   condition "only claim monotonicity when w*rho >~ 1" can then be checked
   against the actual w*rho of each grid point instead of being asserted.

3. Route comparison in that coordinate: does the advantage of matching g_eff
   over leaving s alone grow with w*rho?  If yes, the paper gets a quantitative
   applicability condition instead of a hand-wave.

Writes out/e3a_wrho.csv and fig_e3a.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
S0 = 7.5
DPI = 200
COL = {"geff": "#1f6feb", "geff_const": "#7aa7f0", "ws": "#d1495b", "s": "#4c956c"}
LBL = {"geff": "match $g_{eff}$ per step", "geff_const": "match $g_{eff}$, const $s$",
       "ws": "match $w\\cdot s$", "s": "leave $s$ alone"}


def main():
    g = pd.read_csv(os.path.join(OUT, "e3a_grid.csv"))
    r = pd.read_csv(os.path.join(OUT, "e3a_routes.csv"))

    # ---- w*rho from the slope in s over the whole measured range.  A 0.5-wide
    # stencil is too noisy here: the whole change across s=7.0..7.5 is only
    # ~0.01, which is the same order as the fp16 error on g_eff.
    rows = []
    for (p, w, t), v in g.groupby(["prompt", "w", "t"]):
        v = v.sort_values("s")
        s = v.s.values
        ge = v.g_eff.values
        i0 = int(np.argmin(np.abs(s - 3.0)))
        i1 = int(np.argmin(np.abs(s - 12.0)))
        if i1 <= i0:
            continue
        slope = (ge[i1] - ge[i0]) / (s[i1] - s[i0])
        rows.append(dict(prompt=p, w=w, t=t, slope=slope,
                         wrho=max(slope - 1.0, 0.0)))
    wr = pd.DataFrame(rows)

    # inflation at the reference slider value.  Keep this in two separate
    # statements -- an earlier version divided by S0 twice and produced -1.
    ref = (g[np.abs(g.s - S0) < 1e-9]
           .groupby(["prompt", "w"]).g_eff.median()
           .reset_index().rename(columns={"g_eff": "geff_at_s0"}))
    ref["infl"] = ref.geff_at_s0 / S0 - 1.0

    # joint w*rho per (prompt, w)
    wm = wr.groupby(["prompt", "w"]).wrho.median().reset_index()
    wm = wm.merge(ref, on=["prompt", "w"])
    wm.to_csv(os.path.join(OUT, "e3a_wrho.csv"), index=False)

    print("coupling strength by adapter scale (median over prompts and steps)")
    print(f"{'w':>5s} {'w*rho':>8s} {'g_eff/s0-1':>11s}")
    for w in sorted(wm.w.unique()):
        v = wm[wm.w == w]
        print(f"{w:5.2f} {v.wrho.median():8.3f} {v.infl.median():11.3f}")

    # ---- routes vs coupling
    rr = r.merge(wm[["prompt", "w", "wrho", "infl"]], on=["prompt", "w"])
    rr.to_csv(os.path.join(OUT, "e3a_routes_wrho.csv"), index=False)
    print("\nroute error split by coupling regime (x0 relative L2)")
    bins = [-1, 0.05, 0.15, 0.35, 99]
    names = ["wρ<0.05", "0.05-0.15", "0.15-0.35", ">0.35"]
    print(f"{'regime':10s} {'n':>4s} " + "".join(f"{k:>12s}" for k in
                                                 ["geff", "geff_const", "ws", "s"]))
    for lo, hi, nm in zip(bins[:-1], bins[1:], names):
        v = rr[(rr.wrho > lo) & (rr.wrho <= hi)]
        if not len(v):
            continue
        line = f"{nm:10s} {v.groupby('route').size().iloc[0]:4d} "
        for k in ["geff", "geff_const", "ws", "s"]:
            u = v[v.route == k]
            line += f"{u.x0_rel_l2.mean():12.4f}" if len(u) else f"{'-':>12s}"
        print(line)

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))

    for sref, st in [(3.0, ":"), (S0, "-"), (10.5, "--")]:
        v = g[np.abs(g.s - sref) < 1e-9].groupby("w").g_eff.median()
        ax[0].plot(v.index, v.values, st, marker="o", ms=3.5,
                   label=f"$s={sref:g}$")
    ax[0].plot([0.1, 2.05], [0.1, 2.05], "k-", lw=.6, alpha=.4)
    ax[0].text(1.6, 1.85, "$g_{eff}=s$", fontsize=7)
    ax[0].set_xlabel("adapter scale $w$")
    ax[0].set_ylabel("effective guidance $g_{eff}$")
    ax[0].set_title("(a) a LoRA adds guidance you did not ask for", fontsize=9.5)
    ax[0].legend(fontsize=7, frameon=False)

    w = wm.groupby("w").infl.median()
    ax[1].plot(w.index, 100 * w.values, "o-", color="#1f6feb", ms=4)
    ax[1].axhline(0, color="k", lw=.6)
    for x, y in [(1.0, w.loc[1.0]), (2.0, w.loc[2.0])]:
        ax[1].annotate(f"{100*y:+.1f}%", (x, 100 * y), fontsize=8,
                       textcoords="offset points", xytext=(4, -10))
    ax[1].set_xlabel("adapter scale $w$")
    ax[1].set_ylabel("$(g_{eff}(w,s_0)/s_0-1)$  [%]")
    ax[1].set_title("(b) inflation at the default slider $s_0=7.5$", fontsize=9.5)

    for k in ["geff", "geff_const", "ws", "s"]:
        v = rr[(rr.route == k) & (np.abs(rr.w - 1.0) > 1e-9)]
        v = v.groupby("w").x0_rel_l2.mean()
        ax[2].plot(v.index, v.values, "o-", ms=3.5, color=COL[k], label=LBL[k])
    ax[2].axvline(1.0, color="k", lw=.6, ls=":")
    ax[2].text(1.02, 0.92, "reference\n$w_0=1$", fontsize=6.5, va="top",
               transform=ax[2].get_xaxis_transform())
    ax[2].set_xlabel("adapter scale $w$")
    ax[2].set_ylabel("relative $L_2$ of $x_0$ vs reference")
    ax[2].set_title("(c) how far each rule lands from $(w_0,s_0)$", fontsize=9.5)
    ax[2].legend(fontsize=7, frameon=False)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_e3a.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
