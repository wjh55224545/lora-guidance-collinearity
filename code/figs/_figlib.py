# -*- coding: utf-8 -*-
"""《计算机工程》插图统一风格（供两篇论文的全部图重绘使用）。

依据期刊官网与投稿指南（ecice06.com）：
  1. 本刊为黑白印刷，图表不用彩色；若必须用彩色须用高对比度。
      -> 本模块只用灰度 + 填充图案(hatch) + 标记(marker) + 线型(linestyle)
         四重编码，保证黑白激光打印后仍可区分。
  2. 图中的中文字体为华文中宋，英文字体为 Times New Roman。
      -> font.family 列表 = [Times New Roman, STZhongsong]，matplotlib 3.6+
         的逐字形回退会让拉丁字符走 Times、汉字走华文中宋。
      -> mathtext 用 custom，斜体与正体都指向 Times New Roman。
  3. 坐标轴标签须含变量名（无单位者在图题注中说明）。
  4. 图表须清晰，不用扫描方式录入 -> 矢量化后端 + 600 dpi 位图双份。
  5. 图宽按正文排版宽度取值：正文宽 15.8 cm，图按 15.2 cm 插入，
     故 figsize 宽度直接取 15.2 cm（=5.984 in），字体按 1:1 呈现。

用法：
    from _figlib import setup, save, GRAY, MK, LS, HATCH, finish
    setup()
    fig, ax = plt.subplots(...)
    ...
    finish(fig)          # 统一留白
    save(fig, r"...\fig_xx.png")   # 同时出 .png(600dpi) 与 .pdf(矢量)
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------- 尺寸 ----------------
W_FULL = 15.2 / 2.54          # 15.2 cm 通栏
W_HALF = 7.4 / 2.54           # 7.4 cm 半栏

# ---------------- 灰度档 ----------------
K0 = "#000000"    # 主（期望曲线 / 首条）
K1 = "#3f3f3f"    # 次
K2 = "#6e6e6e"
K3 = "#9a9a9a"
K4 = "#c4c4c4"    # 最弱（基线 / 背景带）
GRAY = [K0, K1, K2, K3, K4]

# 区分编码：同一图内第 i 条曲线取 MK[i]、LS[i]、HATCH[i]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1))]
HATCH = ["", "///", "\\\\\\", "...", "xxx", "---"]
# 柱/箱填充：白底黑边 + 图案，靠图案而非颜色区分
FILLS = ["white", "#d9d9d9", "#a6a6a6", "#737373", "white", "#bfbfbf"]


def setup(fs=7.0):
    """装载期刊排版参数。fs 为图内基准字号（pt），印刷后即最终字号。"""
    rcParams.update({
        # 字体：拉丁走 Times New Roman，汉字回退到华文中宋
        "font.family": ["Times New Roman", "STZhongsong"],
        "font.size": fs,
        "axes.titlesize": fs,          # 子图标题与基准同号，靠粗体区分
        "axes.labelsize": fs,
        "xtick.labelsize": fs - 0.5,
        "ytick.labelsize": fs - 0.5,
        "legend.fontsize": fs - 1.0,
        "figure.titlesize": fs + 0.5,
        # 数学：变量斜体 Times
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.default": "it",
        # 框线与刻度
        "axes.linewidth": 0.6,
        "axes.edgecolor": K0,
        "axes.labelcolor": K0,
        "axes.titlecolor": K0,
        "axes.titleweight": "bold",
        "axes.titlepad": 3.0,
        "axes.unicode_minus": False,
        "xtick.color": K0,
        "ytick.color": K0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.top": True,
        "ytick.right": True,
        # 网格：浅灰细点线，压在数据之下
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.35,
        "grid.linestyle": (0, (1, 1.2)),
        "axes.axisbelow": True,
        # 线条
        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.55,
        # 图例
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": K3,
        "legend.fancybox": False,
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.25,
        "legend.handletextpad": 0.5,
        "legend.handlelength": 1.9,
        "legend.borderaxespad": 0.3,
        # 输出
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def finish(fig, wspace=0.32, hspace=0.28):
    """统一子图间距（不呼叫 tight_layout，避免中文标题被压缩后字号失真）。"""
    fig.subplots_adjust(wspace=wspace, hspace=hspace)


def save(fig, path, target_cm=None, iters=4):
    """出 600 dpi PNG（Word 用）+ 同目录矢量 PDF（备查）。

    先把「按 tight 边界裁剪后的实际宽度」迭代校准到 target_cm，
    使图在 Word 里按 15.2 cm 插入时与设计尺寸 1:1，图内字号即为印刷字号。

    换版心（如 LNCS 12.2 cm）时不改布局代码，走环境变量：
      FIG_TARGET_CM=12.2  FIG_SCALE=0.803
    FIG_SCALE 会把**所有字号与线宽**同比缩，故 12.2 cm 版与 15.2 cm 版
    逐像素等价（只差物理尺寸），不会因为"字没缩、画布缩"而互相压住。
    """
    import matplotlib.text as mtext
    from PIL import Image

    if target_cm is None:
        target_cm = float(os.environ.get("FIG_TARGET_CM", 15.2))
    k = float(os.environ.get("FIG_SCALE", 1.0))
    if k != 1.0:
        for t in fig.findobj(mtext.Text):
            try:
                t.set_fontsize(t.get_fontsize() * k)
            except Exception:
                pass
        for ln in fig.findobj(matplotlib.lines.Line2D):
            ln.set_linewidth(ln.get_linewidth() * k)
            ln.set_markersize(max(ln.get_markersize() * k, 0.6))

    base = os.path.splitext(path)[0]
    probe = base + "_probe.png"
    for _ in range(iters):
        fig.savefig(probe, dpi=200, bbox_inches="tight", pad_inches=0.02)
        w_cm = Image.open(probe).size[0] / 200 * 2.54
        if target_cm is None or abs(w_cm - target_cm) < 0.12:
            break
        k = target_cm / w_cm
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * k, h * k)
    if os.path.exists(probe):
        os.remove(probe)

    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".pdf")
    im = Image.open(path)
    print(f"-> {path}  {im.size[0]/600*2.54:.2f} cm 宽  {im.size}")
    plt.close(fig)
    return path
