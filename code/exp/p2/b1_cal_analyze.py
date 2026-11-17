import pandas as pd, numpy as np, os, io

B = pd.read_csv(r"E:\研究\exp\p2\out_b1_grid\images.csv")
R = pd.read_csv(r"E:\研究\exp\p2\out_b1_grid_real\images.csv")
B["adapter"] = "B(rho=0.699)"
R["adapter"] = "REAL(rho=0.385)"
d = pd.concat([B, R], ignore_index=True)

RHO = {"B(rho=0.699)": 0.699, "REAL(rho=0.385)": 0.385}
S0, W0 = 7.5, 1.0

rows = []
for ad, g in d.groupby("adapter"):
    for w, gw in g.groupby("w"):
        if abs(float(w) - 1.0) < 1e-9:
            continue  # (W0,S0) is the reference cell: pix_l2 == 0 by construction
        # median over prompts for each s
        m = gw.groupby("s")[["pix_l2", "clip_t"]].median()
        fixed = float(m.loc[7.5, "pix_l2"]) if 7.5 in m.index else np.nan
        best_s = float(m["pix_l2"].idxmin())
        best = float(m["pix_l2"].min())
        rho = RHO[ad]
        s_cal = S0 * (W0 + rho) / (w + rho) if False else S0 * (1 + W0 * rho) / (1 + w * rho)
        s_snap = float(min(m.index, key=lambda x: abs(x - s_cal)))
        cal = float(m.loc[s_snap, "pix_l2"])
        rows.append(dict(adapter=ad, w=w, s_best=best_s, s_cal=round(s_cal, 2), s_snap=s_snap,
                         pix_fixed=round(fixed, 2), pix_best=round(best, 2), pix_cal=round(cal, 2),
                         gain_cal_vs_fixed=round(100 * (fixed - cal) / fixed, 1),
                         gain_oracle=round(100 * (fixed - best) / fixed, 1),
                         clip_t_at_cal=round(float(m.loc[s_snap, "clip_t"]), 4)))

out = pd.DataFrame(rows).sort_values(["adapter", "w"])
txt = out.to_string(index=False)
print(txt)
io.open(r"E:\研究\exp\p2\out_b1_grid\b1_cal_summary.txt", "w", encoding="utf-8").write(txt)
