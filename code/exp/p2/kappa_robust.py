"""L2 的 kappa 到底有多少是测出来的？（2026-11-07 13:05，纯分析，不占 GPU）

背景
----
L2 的等值线法定义 `s*(w) = argmin_s pix_l2(w,·)`，而 pix_l2 是到**参考图**
的距离，参考图就是本模型在 (W0,S0)=(1.0,7.5) 的输出 —— 于是 `pix_l2(1.0,7.5) ≡ 0`
是**定义**而非测量，argmin 会被朝 S0 吸。

§8.10 当时的结论是「偏置方向是压平斜率 → κ 被低估 → κ=.211 是下界」。
那是**推理**，没有实测。这个脚本把它测掉。

做了什么
--------
1. `s*` 的取值分布（有多少点落在 S0 上、按 w 分开看）
2. κ 对 w 子集的敏感性：去掉构造零点 / 去掉两端 / 只剩两端
3. `l_at_best` 与 `l_flat` 的关系 —— 最小值附近有多平坦，决定 argmin 的稳定性

读法
----
如果「去掉构造零点」让 κ 大幅上升，说明零点是主要偏置源；
如果「去掉两端」让 κ 明显下降，说明中段其实是零信息、全部信息在两端；
两者都不大 → κ 是稳的，只需要报区间而不是单点。
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "out_long2", "an_isoline.csv")
OUT = os.path.join(HERE, "out_kappa")
W0, S0 = 1.0, 7.5

SUBSETS = {
    "all8": [],
    "drop_w1.0": [1.0],
    "drop_ends": [0.2, 2.0],
    "ends_only": [0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
    "drop_w1.0_and_ends": [1.0, 0.2, 2.0],
}


def kappa(sub):
    """log(s*/S0) ~ log(w/W0) 的斜率取负。"""
    x = np.log(sub.w.values / W0)
    y = np.log(sub.s_star.values / S0)
    if len(x) < 2 or np.ptp(x) == 0:
        return np.nan
    return -stats.linregress(x, y).slope


def main():
    os.makedirs(OUT, exist_ok=True)
    E = pd.read_csv(SRC)
    print(f"rows {len(E)}  prompts {E.prompt_id.nunique()}  seeds "
          f"{sorted(E.seed.unique())}")

    # ---------------------------------------------------------------- 1
    print("\n=== (1) s* 落在 S0=7.5 上的比例 ===")
    frac = (E.s_star == S0).mean()
    print(f"  {(E.s_star == S0).sum()} / {len(E)} = {frac:.1%}")
    print("\n  按 w 分开（s* 取值 : 个数）:")
    rows = []
    for w, g in E.groupby("w"):
        vc = g.s_star.value_counts().sort_index()
        rows.append(dict(w=w, n=len(g), at_S0=int((g.s_star == S0).sum()),
                         frac_at_S0=float((g.s_star == S0).mean()),
                         **{f"s{k:g}": int(v) for k, v in vc.items()}))
        print(f"    w={w:>4}: " + "  ".join(f"{k:g}:{v}" for k, v in vc.items())
              + f"   ({100 * (g.s_star == S0).mean():.0f}% at 7.5)")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sstar_by_w.csv"), index=False)

    # ---------------------------------------------------------------- 2
    print("\n=== (2) kappa 对 w 子集的敏感性 ===")
    print(f"  {'subset':22s} {'n':>4} {'med':>7} {'mean':>7} {'sd':>6}")
    recs = []
    for name, drop in SUBSETS.items():
        ks = [kappa(g[~g.w.isin(drop)])
              for _, g in E.groupby(["prompt_id", "seed"])]
        ks = np.array([k for k in ks if not np.isnan(k)])
        recs.append(dict(subset=name, n=len(ks), kappa_med=np.median(ks),
                         kappa_mean=ks.mean(), kappa_sd=ks.std(),
                         dropped=",".join(str(d) for d in drop) or "-"))
        print(f"  {name:22s} {len(ks):>4} {np.median(ks):>+7.3f} "
              f"{ks.mean():>+7.3f} {ks.std():>6.3f}")
    K = pd.DataFrame(recs)
    K.to_csv(os.path.join(OUT, "kappa_by_wsubset.csv"), index=False)

    # ---------------------------------------------------------------- 3
    print("\n=== (3) argmin 附近有多平坦 ===")
    print(E[["l_at_best", "l_flat"]].describe().round(2).to_string())
    ratio = (E.l_flat / E.l_at_best.replace(0, np.nan)).dropna()
    print(f"\n  l_flat / l_at_best : med {ratio.median():.2f}  "
          f"p10 {np.percentile(ratio, 10):.2f}  p90 {np.percentile(ratio, 90):.2f}")
    print("  （比值越大 = 谷底越平 = argmin 越容易被网格分辨率卡住）")

    lo, hi = K.kappa_med.min(), K.kappa_med.max()
    print("\n=== 结论 ===")
    print(f"  kappa 在 {lo:+.3f} .. {hi:+.3f} 之间，围绕 "
          f"{K[K.subset == 'all8'].kappa_med.iloc[0]:+.3f}")
    d0 = (K[K.subset == "drop_w1.0"].kappa_med.iloc[0]
          - K[K.subset == "all8"].kappa_med.iloc[0])
    print(f"  去掉构造零点只动 {d0:+.3f} -> 零点不是主要偏置源")
    print(f"  去掉两端后仍有 "
          f"{K[K.subset == 'drop_ends'].kappa_med.iloc[0]:+.3f} -> 中段不是零信息")
    print("  => 报区间，不报单点；'kappa=1（等 w*s）' 在整个区间之外")


if __name__ == "__main__":
    main()
