# -*- coding: utf-8 -*-
"""ImageReward 三路对照分析（论文二，IR_2026-11-14）。

要回答的问题只有一个：把 s 从参考值 7.5 压到标定预测的 s*(w)，人偏好模型是变好还是变差？
参考点 w=1（s=7.5）是“未膨胀”的锚，因此三路是
    reference   w=1,  s=7.5
    uncomp.     w,    s=7.5        （按 w 放大 LoRA 但不动 CFG）
    comp.       w,    s*(w)        （按 g* = s(1+w rho) 反解）

配对单位是 (prompt, seed)：同一对提示词与种子下比较，消掉画面内容的方差。
统计量用中位与符号计数，不用均值 —— 8 个配对、分数是 logit，均值会被单张异常图拖走。

运行： D:\\python\\python.exe E:\\研究\\exp\\p2\\ir_analyze.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = r"E:\研究\results\论文二_LoRA共线\IR_2026-11-14"
PANEL = os.path.join(ROOT, "panel", "ir_scores.csv")
BASE = os.path.join(ROOT, "base", "ir_scores.csv")
RHO = 0.208          # reference (detail) 适配器的 L2 共线指数，正文 §5 值
S_REF = 7.5


def nearest(grid, v):
    return float(min(grid, key=lambda g: abs(g - v)))


def main():
    p = pd.read_csv(PANEL)
    b = pd.read_csv(BASE)
    grid = sorted(p.s.unique())
    ws = sorted(p.w.unique())

    out_rows, sum_rows = [], []
    for w in ws:
        s_star = S_REF / (1.0 + w * RHO)
        s_pick = nearest(grid, s_star)
        for pid, sd in sorted(set(zip(p.prompt_id, p.seed))):
            cur = p[(p.prompt_id == pid) & (p.seed == sd) & (p.w == w)]
            if cur.empty:
                continue
            unc = cur[cur.s == S_REF]
            com = cur[cur.s == s_pick]
            ref = b[(b.prompt_id == pid) & (b.seed == sd)]
            if unc.empty or com.empty or ref.empty:
                continue
            ir_unc = float(unc.image_reward.iloc[0])
            ir_com = float(com.image_reward.iloc[0])
            ir_ref = float(ref.image_reward.iloc[0])
            out_rows.append(dict(
                prompt_id=pid, seed=sd, w=w, s_star_pred=round(s_star, 3),
                s_comp=s_pick, ir_ref=ir_ref, ir_uncomp=ir_unc, ir_comp=ir_com,
                d_comp_minus_uncomp=ir_com - ir_unc,
                d_comp_minus_ref=ir_com - ir_ref,
                d_uncomp_minus_ref=ir_unc - ir_ref,
            ))
    d = pd.DataFrame(out_rows)
    d.to_csv(os.path.join(ROOT, "ir_pairs.csv"), index=False, float_format="%.6f")

    for w, g in d.groupby("w"):
        n = len(g)
        dc = g.d_comp_minus_uncomp
        du = g.d_uncomp_minus_ref
        sum_rows.append(dict(
            w=w, n=n, s_comp=g.s_comp.iloc[0],
            s_star_pred=round(g.s_star_pred.iloc[0], 3),
            ir_ref=round(g.ir_ref.median(), 4),
            ir_uncomp=round(g.ir_uncomp.median(), 4),
            ir_comp=round(g.ir_comp.median(), 4),
            d_med=round(dc.median(), 4),
            d_pos=int((dc > 0).sum()),
            d_uncomp_vs_ref=round(du.median(), 4),
            u_pos=int((du > 0).sum()),
        ))
    s = pd.DataFrame(sum_rows)
    s.to_csv(os.path.join(ROOT, "ir_summary.csv"), index=False)

    # 每行 IR 最优点在哪一格 —— 与预测 s* 对照
    best = (p.loc[p.groupby(["w", "prompt_id", "seed"]).image_reward.idxmax()]
            .groupby("w").s.agg(["median", "min", "max", "count"])
            .rename(columns={"median": "s_best_med", "min": "s_best_min",
                             "max": "s_best_max", "count": "n"}))
    best["s_star_pred"] = [round(S_REF / (1 + w * RHO), 3) for w in best.index]

    lines = ["== ImageReward: compensated vs uncompensated ==",
             f"rho={RHO}  s_ref={S_REF}  pairs per cell = {s.n.iloc[0]}", ""]
    for r in s.itertuples():
        lines.append(
            f"w={r.w}  s_pred={r.s_star_pred} (grid {r.s_comp})  "
            f"IR ref {r.ir_ref:.3f} | uncomp {r.ir_uncomp:.3f} | comp {r.ir_comp:.3f}  "
            f"| comp-uncomp med {r.d_med:+.3f} ({r.d_pos}/{r.n} positive)  "
            f"| uncomp-ref med {r.d_uncomp_vs_ref:+.3f} ({r.u_pos}/{r.n} positive)")
    lines += ["", "== best s under ImageReward =="]
    for w, r in best.iterrows():
        lines.append(f"w={w}  best-s median {r.s_best_med:.1f} "
                     f"(range {r.s_best_min:.1f}-{r.s_best_max:.1f}, n={r.n})  "
                     f"predicted {r.s_star_pred}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(ROOT, "ir_readout.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
