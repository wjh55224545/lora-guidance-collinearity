"""Plot E2b: rho_t at four LoRA scales, and the local exponent alpha_t."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
df = pd.read_csv(os.path.join(OUT, "e2b_rel.csv"))

CP = {"person": "#1f77b4", "style": "#d62728", "object": "#2ca02c"}
rels = sorted(df.rel.unique())
LS = {rels[0]: ":", rels[1]: "--", rels[2]: "-", rels[3]: "-."}

fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.7))

# (a) rho_t at four scales
for p in CP:
    for rel in rels:
        s = df[(df.prompt == p) & (df.rel == rel)].sort_values("t_frac")
        ax[0].plot(s.t_frac, s.rho, ls=LS[rel], color=CP[p], lw=1.5,
                   marker="o", ms=2.2,
                   label=f"{p} w={rel:g}" if rel in (rels[0], rels[-1]) else None)
ax[0].set_xlabel(r"$t/1000$")
ax[0].set_ylabel(r"$\rho_t$")
ax[0].set_title(r"(a) $\rho_t$ vs noise level, four LoRA scales", fontsize=9.5)
ax[0].legend(fontsize=6.5, frameon=False, ncol=2)

# (b) local exponent alpha_t
for p in CP:
    al, ts = [], []
    for t in sorted(df.t_frac.unique()):
        s = df[(df.prompt == p) & (np.abs(df.t_frac - t) < 1e-9)].sort_values("rel")
        a, _ = np.polyfit(np.log(s.rel.values), np.log(s.rho.values), 1)
        al.append(a); ts.append(t)
    ax[1].plot(ts, al, "o-", ms=3, lw=1.5, color=CP[p], label=p)
ax[1].axhline(1.0, ls=":", c="0.35", lw=1)
ax[1].text(0.02, 1.02, r"linear ($\alpha=1$)", fontsize=7, color="0.35")
ax[1].set_xlabel(r"$t/1000$")
ax[1].set_ylabel(r"$\alpha_t=\mathrm{d}\log\rho/\mathrm{d}\log w$")
ax[1].set_title(r"(b) response is sub-linear at both ends", fontsize=9.5)
ax[1].legend(fontsize=7, frameon=False)

# (c) swing vs scale
for p in CP:
    mm = [df[(df.prompt == p) & (df.rel == r)].rho.values for r in rels]
    ax[2].plot(rels, [v.max() / v.min() for v in mm], "o-", ms=4, lw=1.5,
               color=CP[p], label=p)
ax[2].axhline(1.0, ls=":", c="0.35", lw=1)
ax[2].set_xlabel(r"LoRA scale $w$  ($=\|\Delta W\|_F/\|W\|_F$)")
ax[2].set_ylabel(r"$\max_t\rho_t / \min_t\rho_t$")
ax[2].set_title(r"(c) swing grows with scale, never below 2.2", fontsize=9.5)
ax[2].legend(fontsize=7, frameon=False)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)
fig.tight_layout()
p = os.path.join(OUT, "fig_e2b_rel.png")
fig.savefig(p, dpi=175, bbox_inches="tight")
print("wrote", p)
