# -*- coding: utf-8 -*-
"""论文二《LoRA 权重与引导强度的共线标定方法》全部插图重绘。

对标《计算机工程》投稿指南：黑白印刷（纯灰度+图案+标记编码）、
中文华文中宋 / 英文 Times New Roman、图宽 15.2 cm、600 dpi。

六张图与正文图号一一对应：
  图1  fig_L2_a_leverage.png   w 与 s 在同一网格单元内的作用范围
  图2  fig_L2_b_isoline.png    等响应线与 κ 的分布
  图3  fig_L2_c_routes.png     四条补偿路线的误差与两段式 κ
  图4  fig_Y4.png              每个 w 需要多少引导 + 局部 κ
  图5  fig_y10_sat.png         饱和像素率与锐度随 s 的变化
  图6  fig_y11_kappa.png       SD2.1-base 可辨识格的 s*(w) 与 κ

运行： D:\\python\\python.exe E:\\研究\\_figs_p2.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, r"E:\研究")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _figlib import setup, save, W_FULL, K0, K1, K2, K3, K4

ROOT = r"E:\研究"
L2 = os.path.join(ROOT, "exp", "p2", "out_long2")
Y4 = os.path.join(ROOT, "exp", "p2", "out_y", "y4")
RES = os.path.join(ROOT, "results", "论文二_LoRA共线")
Y10 = os.path.join(RES, "Y10_2026-11-08")
Y11 = os.path.join(RES, "Y11_2026-11-08")

W0, S0 = 1.0, 7.5
W_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
RHO = {"object": 0.208, "person": 0.295, "style": 0.200}
PCL_COL = {"object": K0, "person": K1, "style": K2}
PCL_MK = {"object": "o", "person": "s", "style": "^"}

H1, H2 = 5.6 / 2.54, 6.2 / 2.54


# ------------------------------------------------------------------ 图 1
def fig1():
    sr = pd.read_csv(os.path.join(L2, "an_s_range.csv"))
    wr = pd.read_csv(os.path.join(L2, "an_w_range.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, H1))

    for a, df, key, title, xlab in [
            (ax[0], sr, "w", "(a) 固定 w、扫 s", "w"),
            (ax[1], wr, "s", "(b) 固定 s、扫 w", "s")]:
        ks = sorted(df[key].unique())
        data = [df[df[key] == k].rng.values for k in ks]
        bp = a.boxplot(data, positions=range(len(data)), widths=.55,
                       patch_artist=True, showfliers=False)
        for p in bp["boxes"]:
            p.set_facecolor("white")
            p.set_edgecolor(K0)
            p.set_linewidth(.6)
        for m in bp["medians"]:
            m.set_color(K0)
            m.set_lw(1.1)
        for w in ("whiskers", "caps"):
            for e in bp[w]:
                e.set_color(K1)
                e.set_lw(.6)
        a.set_xticks(range(len(data)))
        a.set_xticklabels([f"{v:g}" for v in ks], fontsize=6.0)
        a.set_xlabel(xlab)
        a.set_title(title)
        a.set_ylim(0, 26)

    ax[0].set_ylabel("同一 (w, s) 格内像素 L2 跨度")
    ax[0].axhline(sr.rng.mean(), color=K1, ls="--", lw=.8)
    ax[1].axhline(wr.rng.mean(), color=K1, ls="--", lw=.8)
    ax[0].text(.03, .96, f"均值 {sr.rng.mean():.2f}", transform=ax[0].transAxes,
               fontsize=6.2, va="top", ha="left")
    ax[1].text(.03, .96, f"均值 {wr.rng.mean():.2f}", transform=ax[1].transAxes,
               fontsize=6.2, va="top", ha="left")

    fig.subplots_adjust(wspace=.24)
    return save(fig, os.path.join(RES, "L2_2026-11-06", "fig",
                                  "fig_L2_a_leverage.png"))


# ------------------------------------------------------------------ 图 2
def fig2():
    E = pd.read_csv(os.path.join(L2, "an_isoline.csv"))
    K = pd.read_csv(os.path.join(L2, "an_kappa.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, H2))
    a, b = ax

    # ---- (a) 等响应线 -----------------------------------------------------
    for pcl in ("object", "person", "style"):
        g = E[E.pclass == pcl].groupby("w").s_star.agg(["mean", "std"])
        a.errorbar(g.index, g["mean"], yerr=g["std"], marker=PCL_MK[pcl],
                   ms=3.0, capsize=1.8, capthick=.5, lw=.9,
                   color=PCL_COL[pcl], label={"object": "object",
                                              "person": "person",
                                              "style": "style"}[pcl])
    ws = np.linspace(S0 / 14., 2, 80)
    a.plot(ws, S0 / ws, color=K3, ls="--", lw=1.0, label="等 w·s（κ=1）")
    a.plot(ws, S0 * (ws / W0) ** -0.213, color=K0, lw=1.2,
           label="拟合 κ=0.21")
    a.axhline(S0, color=K1, ls="-.", lw=.9, label="固定 s")
    a.set_xscale("log")
    a.set_xticks(W_GRID)
    a.set_xticklabels([f"{v:g}" for v in W_GRID], fontsize=6.0)
    a.set_ylim(1.0, 15.5)
    a.set_xlabel("LoRA 权重 w")
    a.set_ylabel("最佳 s*(w)（最接近参考图）")
    a.set_title("(a) 等响应线比等 w·s 平得多")
    a.legend(loc="lower left", ncol=2, fontsize=5.6, columnspacing=.7)

    # ---- (b) κ 的分布 -----------------------------------------------------
    b.hist(K.kappa, bins=np.arange(-.2, .55, .05), color="#d9d9d9",
           edgecolor=K0, lw=.35)
    b.set_xlim(-.25, 1.12)
    b.set_ylim(0, 24)
    med = K.kappa.median()
    for x, col, ls, lab, dy, ha, dx in [
            (med, K0, "-", f"实测 {med:.2f}", 21.2, "left", 3),
            (.177, K1, "--", "闭式 0.18", 18.4, "right", -4),
            (1.0, K2, ":", "等 w·s = 1.0", 15.6, "right", -4)]:
        b.axvline(x, color=col, lw=.9, ls=ls)
        b.annotate(lab, (x, dy), fontsize=6.0, color=col, xytext=(dx, 0),
                   textcoords="offset points", va="center", ha=ha)
    b.set_xlabel("每提示词-种子拟合的 κ")
    b.set_ylabel("计数")
    b.set_title("(b) 70 次独立拟合，偏离等 w·s 达 4.7 倍")

    fig.subplots_adjust(wspace=.24)
    return save(fig, os.path.join(RES, "L2_2026-11-06", "fig",
                                  "fig_L2_b_isoline.png"))


# ------------------------------------------------------------------ 图 3
def fig3():
    byw = pd.read_csv(os.path.join(L2, "an_routes_byw.csv"))
    Pi = pd.read_csv(os.path.join(L2, "an_kappa_piecewise.csv"))
    E = pd.read_csv(os.path.join(L2, "an_grid_routes.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, H2))
    a, b = ax

    # ---- (c) 路线误差 -----------------------------------------------------
    for r, lab, col, mk, ls in [
            ("fixed_s", "固定 s（现状）", K2, "h", ":"),
            ("geff_const", "常数补偿 g_eff", K0, "o", "-"),
            ("ws_equal", "等 w·s", K1, "s", "--")]:
        a.plot(byw.w, byw[r], marker=mk, ms=3.4, lw=1.1, color=col, ls=ls,
               label=lab, markerfacecolor="white" if r == "fixed_s" else col)
    g = E[E.label == "piecewise k=0.7"].groupby("w").pred.mean()
    a.plot(g.index, g.values, marker="^", ms=3.4, lw=.9, ls="-.",
           color=K2, label="分段（网格）")
    a.set_xscale("log")
    a.set_xticks(W_GRID)
    a.set_xticklabels([f"{v:g}" for v in W_GRID], fontsize=6.0)
    a.set_xlabel("w")
    a.set_ylabel("相对参考图的平均像素 L2")
    a.set_title("(c) 常数补偿有效，等 w·s 严重变差")
    a.legend(loc="upper right", fontsize=5.8)

    # ---- (d) 两段式 κ -----------------------------------------------------
    for pcl in ("object", "person", "style"):
        s = Pi[Pi.pclass == pcl]
        b.scatter(s.kappa_lo, s.kappa_hi, s=11, marker=PCL_MK[pcl],
                  color=PCL_COL[pcl], edgecolor=K0, linewidth=.35,
                  zorder=3, label=pcl)
    lim = (-.45, 2.2)
    b.plot(lim, lim, color=K3, ls="--", lw=.8)
    b.axhline(1.0, color=K1, ls=":", lw=.8)
    b.axhline(0.0, color=K3, ls=":", lw=.8)
    b.set_xlim(lim)
    b.set_ylim(lim)
    b.set_xlabel("w≤0.8 段的 κ")
    b.set_ylabel("w≥1.2 段的 κ")
    b.set_title("(d) 两段式：低段平、高段近线性")
    b.legend(loc="lower right", fontsize=5.8)

    fig.subplots_adjust(wspace=.26)
    return save(fig, os.path.join(RES, "L2_2026-11-06", "fig",
                                  "fig_L2_c_routes.png"))


# ------------------------------------------------------------------ 图 4
def fig4():
    S = pd.read_csv(os.path.join(Y4, "y4_surface.csv"), index_col=0)
    S.columns = [float(c) for c in S.columns]
    S.index = [float(i) for i in S.index]
    lk = pd.read_csv(os.path.join(Y4, "y4_local_kappa.csv"))
    sq = pd.read_csv(os.path.join(Y4, "y4_sstar_rows_local.csv"))
    pr = pd.read_csv(os.path.join(Y4, "y4_practice.csv"))

    dmin = S.min(axis=1)
    cg = sq.groupby("w")["s_star"].median()
    W = list(cg.index)
    DEEP = 4.0
    YLO, YHI = 3.0, 12.6

    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, H2),
                           gridspec_kw=dict(width_ratios=[1.12, 1.0]))
    a, b = ax

    # ---- (a) 每个 w 需要多少 s -------------------------------------------
    xw = np.log(np.array(W) / W0)
    xr = np.linspace(-1.78, .82, 200)
    yr = np.log(S0 * W0 / np.exp(xr))
    vis = np.exp(yr) <= YHI
    a.plot(xr[vis], yr[vis], "--", lw=1.0, color=K3, zorder=2,
           label="等 w·s：s=S0·W0/w")
    i0 = int(np.argmax(vis))
    a.plot([xr[i0], xr[i0]], [yr[i0], np.log(YHI)], ":", lw=.8, color=K3)

    a.plot(xw, np.log(cg.values), "-", lw=1.2, color=K0, zorder=4,
           label="实测 s*(w)（抛物线细化）")
    a.scatter(xw, np.log(cg.values), s=13, color=K0, zorder=5,
              edgecolor="white", linewidth=.5)
    for w in W:
        if dmin[w] > DEEP:
            a.scatter([np.log(w / W0)], [np.log(cg[w])], s=52, facecolor="none",
                      edgecolor=K0, lw=.8, zorder=6)
    a.scatter([], [], s=44, facecolor="none", edgecolor=K0, lw=.8,
              label=f"浅井（D_min>{DEEP:g}）")

    styles = [("SD1.5", K0, "o", "SD1.5 默认（=W0, S0）"),
              ("community: lower w only", K1, "s", "仅降 w"),
              ("community: lower s by 20%", K1, "^", "仅降 s 20%"),
              ("community: lower w and s together", K1, "D", "同时降 w 与 s"),
              ("LoRAShop", K2, "v", "LoRAShop gs=3.5（FLUX，仅方向）"),
              ("strong LoRA", K0, "P", "强 LoRA，s 不动")]
    for _, r in pr.iterrows():
        for key, col, mk, lab in styles:
            if r.label.startswith(key):
                a.scatter([np.log(r.w / W0)], [np.log(r.s)], s=20, marker=mk,
                          color=col, zorder=8, edgecolor="white",
                          linewidth=.4, label=lab)
                break

    a.text(.985, .975,
           "↑ 等 w·s 在 w=0.2\n要求 s=37.5",
           transform=a.transAxes, ha="right", va="top", fontsize=5.8, color=K1)

    a.set_xlabel("lg(w/W0)")
    a.set_ylabel("lg s")
    a.set_title("(a) 每个 w 实际需要多少引导")
    a.set_xlim(-1.78, .82)
    a.set_ylim(np.log(YLO), np.log(YHI))
    yt = [3, 4, 5, 6, 7.5, 9, 10, 12]
    a.set_yticks(np.log(yt))
    a.set_yticklabels([f"{v:g}" for v in yt])
    a.legend(loc="lower left", ncol=2, fontsize=5.6, labelspacing=.20,
             handletextpad=.35, borderpad=.30, columnspacing=.6)

    # ---- (b) 局部 κ -------------------------------------------------------
    n = len(lk)
    bad = np.array([(dmin[x] > DEEP) or (dmin[y] > DEEP)
                    for x, y in zip(lk.w_lo, lk.w_hi)])
    for i, (k, lo, hi, isbad) in enumerate(zip(lk.kappa_local, lk.kappa_lo,
                                               lk.kappa_hi, bad)):
        col = K3 if isbad else K0
        b.plot([i, i], [lo, hi], "-", lw=1.4, color=col, solid_capstyle="round")
        b.plot([i - .16, i + .16], [lo, lo], "-", lw=.6, color=col)
        b.plot([i - .16, i + .16], [hi, hi], "-", lw=.6, color=col)
        b.scatter([i], [k], s=13, color="white" if isbad else col, zorder=5,
                  edgecolor=col, linewidth=.8)
        b.annotate(f"{k:.2f}", (i, hi + .045), ha="center", va="bottom",
                   fontsize=5.4, color=col)
    b.axhline(1.0, ls="--", lw=.9, color=K2)
    b.text(.02, 1.02, "κ=1（等 w·s）", fontsize=5.8, color=K2,
           transform=b.get_yaxis_transform())
    b.axhline(0.0, ls=":", lw=.7, color=K3)
    b.set_xticks(np.arange(n))
    b.set_xticklabels([f"{x:g}→{y:g}" for x, y in zip(lk.w_lo, lk.w_hi)],
                      rotation=38, ha="right", fontsize=5.8)
    b.set_xlim(-.6, n - .4)
    b.set_ylabel("局部 κ = −Δlg s*/Δlg w")
    b.set_xlabel("w 相邻对")
    b.set_title("(b) 斜率不是一个数")
    b.set_ylim(-.18, 1.34)
    b.grid(True, axis="y")
    b.grid(False, axis="x")
    b.text(.985, .02, "灰：端点落在浅井内，只报数值\n不作证据",
           transform=b.transAxes, fontsize=5.6, color=K2, va="bottom",
           ha="right")

    fig.subplots_adjust(wspace=.28)
    return save(fig, os.path.join(RES, "Y4_2026-11-08", "fig", "fig_Y4.png"))


# ------------------------------------------------------------------ 图 5
def fig5():
    rows = pd.read_csv(os.path.join(Y10, "csv", "y10_perceptual.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, H1))
    ST = {"full": (K0, "o", "-", "LoRA full"), "A": (K1, "s", "--", "LoRA A")}

    for a, key, ylab, sc, title in [
            (ax[0], "sat_hi", "饱和像素率 /%", 100.0,
             "(a) 饱和像素率随 s 显著上升"),
            (ax[1], "lap_var", "拉普拉斯方差（锐度代理，无量纲）", 1.0,
             "(b) 锐度随 s 无单调变化")]:
        for tag, (col, mk, ls, lab) in ST.items():
            sub = rows[(rows.tag == tag) & (np.abs(rows.w - 1.0) < 1e-9)]
            ss = sorted(set(sub.s))
            prof = [sc * np.mean([r[key] for _, r in sub[sub.s == s].iterrows()])
                    for s in ss]
            a.plot(ss, prof, marker=mk, ms=2.8, lw=.9, ls=ls, color=col,
                   label=lab, markerfacecolor=col)
        a.set_xlabel("引导强度 s（w=1.0 固定）")
        a.set_ylabel(ylab)
        a.set_title(title)
        a.legend(loc="upper left", fontsize=6.0)

    ax[0].set_ylim(-.05, 1.30)
    fig.subplots_adjust(wspace=.3)
    return save(fig, os.path.join(Y10, "fig", "fig_y10_sat.png"))


# ------------------------------------------------------------------ 图 6
def fig6():
    E = pd.read_csv(os.path.join(Y11, "csv", "y11_sstar.csv"))
    K = pd.read_csv(os.path.join(Y11, "csv", "y11_kappa.csv"))
    BAND = (0.111, 0.311)          # SD1.5 参考带 κ=0.211±0.10

    fig, ax = plt.subplots(1, 2, figsize=(W_FULL, H1))
    for a, (pid, sd) in zip(ax, [("obj00", 11), ("obj03", 11)]):
        row = K[(K.prompt_id == pid) & (K.seed == sd)].iloc[0]
        g = E[(E.prompt_id == pid) & (E.seed == sd)].sort_values("w")
        lw_ = np.log(g.w.values)
        ls_ = np.log(g.s.values)
        xs = np.linspace(lw_.min(), lw_.max(), 60)
        for kap, ls in [(BAND[0], ":"), (BAND[1], ":")]:
            a.plot(np.exp(xs), np.exp(np.log(S0) - kap * xs), ":",
                   color=K3, lw=.8, zorder=1)
        b, c0 = np.polyfit(lw_, ls_, 1)
        a.plot(np.exp(xs), np.exp(c0 + b * xs), "--", color=K0, lw=1.0,
               zorder=2,
               label=f"拟合 κ={row.kappa:.3f}，R²={row.r2:.3f}")
        a.plot(g.w, g.s, "o", color=K0, ms=3.2, mfc="white", mew=.7,
               zorder=3, label="实测 s*(w)")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_yticks([5, 6, 7, 8, 9, 10, 11])
        a.set_yticklabels(["5", "6", "7", "8", "9", "10", "11"])
        a.yaxis.set_minor_formatter(mticker.NullFormatter())
        a.set_xlabel("LoRA 权重 w")
        a.set_ylabel("最佳 s*(w)")
        a.set_title(f"({'ab'[list(ax).index(a)]}) {pid}/seed{sd}："
                    f"κ={row.kappa:.3f} [{row.kappa_lo:.3f}, {row.kappa_hi:.3f}]")
        a.legend(loc="upper right", fontsize=5.8)
        a.text(.03, .03, f"灰点线：SD1.5 参考带 κ={BAND[0]:.2f}/{BAND[1]:.2f}",
               transform=a.transAxes, fontsize=5.2, color=K2, va="bottom")
    fig.subplots_adjust(wspace=.3)
    return save(fig, os.path.join(Y11, "fig", "fig_y11_kappa.png"))


if __name__ == "__main__":
    setup()
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        print("wrote", f())
