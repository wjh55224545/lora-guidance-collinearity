"""Plot E2c (real LoRA) and compare its alpha_t with E2b (synthetic rank-4)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
real = pd.read_csv(os.path.join(OUT, "e2c_real_lora.csv"))
syn = pd.read_csv(os.path.join(OUT, "e2b_rel.csv"))

CP = {"person": "#1f77b4", "style": "#d62728", "object": "#2ca02c"}
ws = sorted(real.w.unique())
LS = {ws[0]: ":", ws[1]: "--", ws[2]: "-", ws[3]: "-."}


def alpha_curve(df, xcol, p):
    out = []
    for t in sorted(df.t_frac.unique()):
        s = df[(df.prompt == p) & (np.abs(df.t_frac - t) < 1e-9)].sort_values(xcol)
        a, _ = np.polyfit(np.log(s[xcol].values), np.log(s.rho.values), 1)
        out.append((t, a))
    return np.array(out)


fig, ax = plt.subplots(2, 3, figsize=(13.4, 7.0))

for p in CP:
    for w in ws:
        s = real[(real.prompt == p) & (real.w == w)].sort_values("t_frac")
        ax[0, 0].plot(s.t_frac, s.rho, ls=LS[w], color=CP[p], lw=1.5, marker="o",
                      ms=2.2, label=f"{p} w={w:g}" if w in (ws[0], ws[-1]) else None)
ax[0, 0].set_xlabel(r"$t/1000$"); ax[0, 0].set_ylabel(r"$\rho_t$")
ax[0, 0].set_title(r"(a) real SD1.5 LoRA (rank 8, rel$_{w=1}$=1.2%)", fontsize=9.5)
ax[0, 0].legend(fontsize=6.5, frameon=False, ncol=2)

for p in CP:
    a = alpha_curve(real, "w", p)
    ax[0, 1].plot(a[:, 0], a[:, 1], "o-", ms=3, lw=1.5, color=CP[p], label=p)
ax[0, 1].axhline(1.0, ls=":", c="0.35", lw=1)
ax[0, 1].set_xlabel(r"$t/1000$")
ax[0, 1].set_ylabel(r"$\alpha_t=\mathrm{d}\log\rho/\mathrm{d}\log w$")
ax[0, 1].set_title(r"(b) same L-shape: mid $\approx$ 1, ends 0.75", fontsize=9.5)
ax[0, 1].legend(fontsize=7, frameon=False)

for p in CP:
    mm = [real[(real.prompt == p) & (real.w == w)].rho.values for w in ws]
    ax[0, 2].plot(ws, [v.max() / v.min() for v in mm], "o-", ms=4, lw=1.5, color=CP[p], label=p)
ax[0, 2].axhline(1.0, ls=":", c="0.35", lw=1)
ax[0, 2].set_xlabel("LoRA weight $w$")
ax[0, 2].set_ylabel(r"$\max_t\rho_t/\min_t\rho_t$")
ax[0, 2].set_title(r"(c) swing 2.0-4.1x, flat in $w$", fontsize=9.5)
ax[0, 2].legend(fontsize=7, frameon=False)

for j, p in enumerate(CP):
    ar = alpha_curve(real, "w", p)
    asy = alpha_curve(syn, "rel", p)
    ax[1, j].plot(asy[:, 0], asy[:, 1], "s--", ms=3, lw=1.4, color="0.45",
                  label="synthetic rank-4 (rel 2-20%)")
    ax[1, j].plot(ar[:, 0], ar[:, 1], "o-", ms=3.5, lw=1.7, color=CP[p],
                  label="real LoRA (w 0.25-1.5)")
    ax[1, j].axhline(1.0, ls=":", c="0.35", lw=1)
    ax[1, j].set_ylim(0.4, 1.12)
    ax[1, j].set_xlabel(r"$t/1000$")
    if j == 0:
        ax[1, j].set_ylabel(r"$\alpha_t$")
    ax[1, j].set_title(f"(d{j+1}) {p}: same shape, real LoRA is milder", fontsize=9)
    ax[1, j].legend(fontsize=6.5, frameon=False, loc="lower center")

for a in ax.ravel():
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)
fig.tight_layout()
p = os.path.join(OUT, "fig_e2c_real_lora.png")
fig.savefig(p, dpi=175, bbox_inches="tight")
print("wrote", p)
