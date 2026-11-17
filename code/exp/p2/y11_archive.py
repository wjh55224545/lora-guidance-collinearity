# -*- coding: utf-8 -*-
"""Y11 归档收尾：画图 + 快照 + 清理 bug 图。"""
import shutil, hashlib, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = r"E:\研究\results\论文二_LoRA共线\Y11_2026-11-08"
EXP = r"E:\研究\exp\p2"
os.makedirs(RES + r"\fig", exist_ok=True)
os.makedirs(RES + r"\exp_snapshot", exist_ok=True)

# ---------- 图 1：s*(w) 轨迹 ----------
E = pd.read_csv(RES + r"\csv\y11_sstar.csv")
fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
for ax, (pid, sd) in zip(axes.ravel(),
                         [("obj00", 11), ("obj00", 22), ("obj03", 11), ("obj03", 22)]):
    g = E[(E.prompt_id == pid) & (E.seed == sd)].sort_values("w")
    ax.plot(g.w, g.s, "o-", color="#2563eb")
    ax.set_title(f"{pid} / seed{sd}", fontsize=10)
    ax.set_ylabel(r"$s^*(w)$")
    ax.grid(alpha=.3)
for ax in axes[1]:
    ax.set_xlabel("LoRA weight w")
fig.suptitle("Y11 (SD2.1-base): $s^*(w)$ = argmin_s ||x(w,s)-ref$)_1$  (64x64 gray, ref = w=1, s=7.5)")
fig.tight_layout()
fig.savefig(RES + r"\fig\fig_y11_sstar.png", dpi=160)
plt.close(fig)

# ---------- 图 2：log s* vs log w 线性拟合（可辨识格 vs 不可辨识格） ----------
K = pd.read_csv(RES + r"\csv\y11_kappa.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (pid, sd) in zip(axes, [("obj00", 11), ("obj03", 11)]):
    row = K[(K.prompt_id == pid) & (K.seed == sd)].iloc[0]
    g = E[(E.prompt_id == pid) & (E.seed == sd)].sort_values("w")
    ax.plot(g.w, g.s, "o", color="#2563eb", label="measured $s^*(w)$")
    import numpy as np
    lw = np.log(g.w.values); ls = np.log(g.s.values)
    b, a = np.polyfit(lw, ls, 1)
    xs = np.linspace(lw.min(), lw.max(), 50)
    ax.plot(np.exp(xs), np.exp(a + b * xs), "--", color="#dc2626",
            label=f"fit: $\\kappa$={row.kappa:.3f}, $R^2$={row.r2:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{pid}/seed{sd}: $\\kappa$={row.kappa:.3f} [{row.kappa_lo:.3f},{row.kappa_hi:.3f}]", fontsize=10)
    ax.set_xlabel("w"); ax.set_ylabel(r"$s^*(w)$"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.suptitle("Y11: SD2.1-base vs SD1.5 reference band $\\kappa$=0.211$\\pm$0.10 (identifiable cells)")
fig.tight_layout()
fig.savefig(RES + r"\fig\fig_y11_kappa.png", dpi=160)
plt.close(fig)
print("figs OK")

# ---------- 快照 ----------
for f in ["y11_sd21.py", "y11_reanalyze.py", "y11_run.log", "y11_run2.log"]:
    shutil.copy2(os.path.join(EXP, f), os.path.join(RES, "exp_snapshot", f))
print("snapshot OK")

# ---------- 删除 bug 图目录（先校验 md5 已在重跑前核对过） ----------
bug = RES + r"\img_buggy_seedpassed-as-s"
n = len(os.listdir(bug))
shutil.rmtree(bug)
print(f"removed buggy img dir ({n} files)")

# ---------- 校验正图数量 ----------
n_img = len([f for f in os.listdir(RES + r"\img") if f.endswith(".png")])
print("final img count:", n_img)
