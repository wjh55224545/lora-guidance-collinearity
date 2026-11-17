"""L2 kappa 稳健性配图：s* 的分辨率限制 + kappa 对 w 子集的不敏感。

两格：
  (a) 每个 w 下 s* 的取值构成（横向堆叠条）—— 一眼看出 w=0.8/1.0/1.2 被
      参考点吸住（集中在 7.5），而两端有真实分散。
  (b) kappa 在五个 w 子集上的估计 —— 点 + 标准误，配 kappa=1（等 w*s）参考线。

结论要落在图上：曲线几乎平（子集不影响 kappa），但误差棒说明该报区间。
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_kappa")
S0 = 7.5

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#cbd5e0", "axes.linewidth": .8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# 标签必须短：第一版写了 "all 8 w (used in §8.5)" 这类长串，y 轴标签整体
# 溢出到左子图上，把 (a) 的条形盖住了。
LAB = {"all8": "all 8 w",
       "drop_w1.0": "drop w=1.0",
       "drop_ends": "drop ends",
       "ends_only": "ends only",
       "drop_w1.0_and_ends": "drop w=1.0+ends"}


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, color="#a0aec0", labelsize=8)
    ax.grid(axis="x", color="#edf2f7", lw=.8)
    ax.set_axisbelow(True)


def main():
    E = pd.read_csv(os.path.join(HERE, "out_long2", "an_isoline.csv"))
    K = pd.read_csv(os.path.join(OUT, "kappa_by_wsubset.csv"))

    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.3),
                            gridspec_kw=dict(width_ratios=[1, .92],
                                             wspace=.42))

    # ------------------------------------------------------------- (a)
    ax = axs[0]
    ws = sorted(E.w.unique())
    svals = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]
    cmap = plt.get_cmap("YlGnBu")
    for i, w in enumerate(ws):
        g = E[E.w == w]
        left = 0.0
        for sv in svals:
            frac = (g.s_star == sv).mean() * 100
            if frac <= 0:
                continue
            c = cmap(.18 + .72 * svals.index(sv) / (len(svals) - 1))
            ax.barh(i, frac, left=left, height=.68, color=c,
                    edgecolor="white", lw=.6)
            if frac >= 9:
                ax.text(left + frac / 2, i, f"{sv:g}", ha="center",
                        va="center", fontsize=6.8,
                        color="white" if svals.index(sv) > 3 else "#2d3748")
            left += frac
    ax.set_yticks(range(len(ws)))
    ax.set_yticklabels([f"w={w:g}" for w in ws], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of 91 (prompt, seed) fits with $s^*$ at this value")
    ax.set_title("(a)  argmin is pinned by grid, not by physics",
                 fontsize=10.5, loc="left")
    ii = ws.index(1.0)
    ax.text(50, ii - .42, r"$s^*\equiv 7.5$ by construction",
            ha="center", va="bottom", fontsize=7.4, color="white")
    clean(ax)

    # ------------------------------------------------------------- (b)
    ax = axs[1]
    order = ["all8", "drop_w1.0", "ends_only", "drop_ends",
             "drop_w1.0_and_ends"]
    K = K.set_index("subset").loc[order].reset_index()
    se = K.kappa_sd / np.sqrt(K.n)
    cols = ["#2b6cb0" if s == "all8" else "#718096" for s in K.subset]
    ax.errorbar(K.kappa_med, range(len(K)), xerr=1.96 * se, fmt="none",
                ecolor="#a0aec0", elinewidth=1.6, capsize=3.2, zorder=1)
    ax.scatter(K.kappa_med, range(len(K)), s=58, c=cols, zorder=3,
               edgecolor="white", lw=1.0)
    for i, r in K.iterrows():
        below = i < len(K) - 1
        ax.text(r.kappa_med, i + (.30 if below else -.30),
                f"{r.kappa_med:+.3f}", ha="center",
                va="top" if below else "bottom", fontsize=7.2,
                color="#2d3748")
    ax.axvline(0, color="#4a5568", lw=.8, ls=":")
    ax.axvline(1, color="#c53030", lw=1.1, ls="--")
    # y 用 axes 分数：invert_yaxis 之后数据坐标 -0.62 会跑到图顶上去。
    ax.text(.975, .52, "equal $w\\cdot s$  ($\\kappa=1$)", fontsize=7.6,
            color="#c53030", ha="right", va="center",
            transform=ax.get_xaxis_transform())
    ax.axvspan(K.kappa_med.min(), K.kappa_med.max(), color="#ebf4ff", zorder=0)
    ax.set_yticks(range(len(K)))
    ax.set_yticklabels([LAB[s] for s in K.subset], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(-.05, 1.12)
    ax.set_xlabel(r"$\hat{\kappa}$   (bars = 95% spread across fits)")
    ax.set_title("(b)  every subset lands at 0.17-0.27, none near 1",
                 fontsize=10.5, loc="left")
    clean(ax)

    fig.suptitle(r"Paper 2 · $\kappa\approx0.21$ is a measurement, not an "
                 r"artefact of the anchor point", fontsize=11.5, y=1.005)
    p = os.path.join(OUT, "fig_kappa_robust.png")
    fig.savefig(p, dpi=190, bbox_inches="tight", facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    main()
