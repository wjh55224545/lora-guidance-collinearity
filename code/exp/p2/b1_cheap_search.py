"""Cheap-search baseline for paper 2: golden-section on s, from the 21-point grid.

Reviewer question we answer here: the paper compares calibration (0 generations) with a
72-configuration exhaustive grid, which is a strawman, because a 1-D search on s needs
only a handful of generations. Using the already-generated 21-point grid we can simulate
a golden-section search exactly, since every s it visits is a grid point.

Output: results/论文二_LoRA共线/Z5_2026-11-11/z5_gs_baseline.csv + printed summary.
"""
import io
import os

import numpy as np
import pandas as pd

SRC = r"E:\研究\results\论文二_LoRA共线\Z5_2026-11-11\z5_raw.csv"
PAIRS = r"E:\研究\results\论文二_LoRA共线\Z5_2026-11-11\z5_pairs.csv"
OUT = r"E:\研究\results\论文二_LoRA共线\Z5_2026-11-11\z5_gs_baseline.csv"

raw = pd.read_csv(SRC)
pairs = pd.read_csv(PAIRS)

LO, HI = 5.0, 15.0
R = (np.sqrt(5) - 1) / 2


def golden(grid, n_eval):
    """Golden-section on the grid-valued function; returns (best_s, best_v, n_used)."""
    s_lo, s_hi = LO, HI
    a = s_hi - R * (s_hi - s_lo)
    b = s_lo + R * (s_hi - s_lo)
    used = []

    def f(x):
        s = round(min(max(x, LO), HI) * 2) / 2.0
        used.append(s)
        return float(grid.loc[s])

    fa, fb = f(a), f(b)
    while len(used) < n_eval:
        if fa < fb:
            s_hi, b, fb = b, a, fa
            a = s_hi - R * (s_hi - s_lo)
            fa = f(a)
        else:
            s_lo, a, fa = a, b, fb
            b = s_lo + R * (s_hi - s_lo)
            fb = f(b)
    cand = [(v, s) for s, v in zip(used, [grid.loc[s] for s in used])]
    v, s = min(cand)
    return s, float(v), len(set(used))


rows = []
for (pid, seed, w), g in raw.groupby(["prompt_id", "seed", "w"]):
    grid = g.set_index("s")["pix_l2"].astype(float)
    if len(grid) < 15:
        continue
    for n_eval in (4, 6, 8):
        s_gs, v_gs, used = golden(grid, n_eval)
        rows.append(dict(prompt_id=pid, seed=seed, w=w, n_eval=n_eval,
                         gs_s=s_gs, gs_pix=v_gs, gs_used=used,
                         grid_oracle=float(grid.min()),
                         grid_oracle_s=float(grid.idxmin())))
d = pd.DataFrame(rows)
d = d.merge(pairs[["prompt_id", "seed", "w", "cal_exact_pix", "cal_exact_s"]],
            on=["prompt_id", "seed", "w"], how="left")
d.to_csv(OUT, index=False)

lines = []
lines.append("cells %d  (rows %d)" % (d[["prompt_id", "seed", "w"]].drop_duplicates().shape[0], len(d)))
for n in (4, 6, 8):
    sub = d[d.n_eval == n]
    lines.append(
        "\n== golden-section with %d generations ==\n"
        "median gs_pix %.3f | median cal_exact %.3f | median grid oracle %.3f\n"
        "gs beats calibration on %.1f%% of cells; calibration beats gs on %.1f%%\n"
        "gs finds a value within 3%% of the 21-point oracle on %.1f%% of cells"
        % (n,
           sub.gs_pix.median(), sub.cal_exact_pix.median(), sub.grid_oracle.median(),
           100 * (sub.gs_pix < sub.cal_exact_pix).mean(),
           100 * (sub.cal_exact_pix < sub.gs_pix).mean(),
           100 * ((sub.gs_pix - sub.grid_oracle).abs() / sub.grid_oracle < 0.03).mean()))
txt = "\n".join(lines)
print(txt)
io.open(os.path.join(os.path.dirname(OUT), "z5_gs_baseline.txt"), "w", encoding="utf-8").write(txt)
