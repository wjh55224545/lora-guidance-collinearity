"""Y6: is kappa one number, or three?

The paper reports a single kappa.  L2's grid covers 24 prompts in three classes
(object / person / style), so the class split is free to test and it is the first
thing a reader will ask about, because style transfer and object insertion are
the two use cases the community actually describes differently.

This also fixes a trap we walked into: Y2's dense grid used only four object
prompts, so comparing its kappa to L2's 91-cell average produced a spurious
"refining the grid lowers kappa" claim.  Comparing like with like (object only)
makes the gap vanish.  The lesson is encoded here as a guard: every comparison
prints the matched class subset alongside the pooled number.

Nothing is generated; this reads L2_2026-11-06/csv/L2_images.csv only.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import y_analyze as Y

ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "results", "论文二_LoRA共线",
                   "L2_2026-11-06", "csv", "L2_images.csv")
OUT = os.path.join(HERE, "out_y", "y6")

plt.rcParams.update({
    "figure.dpi": 190, "savefig.dpi": 190, "font.size": 8.4,
    "axes.edgecolor": "#444", "axes.grid": True, "grid.color": "#e8e8e8",
    "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
})
ORDER = ["object", "person", "style"]
COL = {"object": "#1f6fb4", "person": "#e08a1e", "style": "#7a4fa3"}


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(SRC)
    d = d[d.route == "grid"]
    K = Y.kappa_cells(d, "local", boot=0)
    if isinstance(K, tuple):
        K = K[0]
    K = K.dropna(subset=["kappa"])
    K.to_csv(os.path.join(OUT, "y6_cells.csv"), index=False)

    print(f"cells {len(K)}  prompts {K.prompt_id.nunique()}  "
          f"classes {K.pclass.nunique()}")

    g = (K.groupby("pclass").kappa
         .agg(size="size", mean="mean", median="median", std="std")
         .reindex(ORDER))
    g["se"] = g["std"] / np.sqrt(g["size"])
    g["lo"] = g["mean"] - 1.96 * g["se"]
    g["hi"] = g["mean"] + 1.96 * g["se"]
    g.to_csv(os.path.join(OUT, "y6_by_class.csv"))
    print("\n=== kappa by prompt class ===")
    print(g.round(3).to_string())

    groups = [K[K.pclass == c].kappa.values for c in ORDER]
    kw = stats.kruskal(*groups)
    an = stats.f_oneway(*groups)
    print(f"\n  Kruskal-Wallis  H={kw.statistic:.2f}  p={kw.pvalue:.4f}")
    print(f"  one-way ANOVA   F={an.statistic:.2f}  p={an.pvalue:.4f}")
    for a, b in (("object", "person"), ("person", "style"),
                 ("object", "style")):
        t = stats.ttest_ind(K[K.pclass == a].kappa, K[K.pclass == b].kappa,
                            equal_var=False)
        print(f"  {a:6s} vs {b:6s}  diff {K[K.pclass==a].kappa.mean() - K[K.pclass==b].kappa.mean():+.3f}"
              f"  Welch p={t.pvalue:.3f}")

    # ---- the like-for-like guard ------------------------------------------
    print("\n=== like-for-like check (the trap) ===")
    obj = K[K.pclass == "object"].kappa
    y2p = os.path.join(HERE, "out_y", "an_y2_full_local.csv")
    if os.path.exists(y2p):
        y2 = pd.read_csv(y2p).kappa.dropna()
        t = stats.ttest_ind(y2, obj, equal_var=False)
        print(f"  L2 object only   n={len(obj):3d}  kappa={obj.mean():.3f} "
              f"SE={obj.std(ddof=1)/np.sqrt(len(obj)):.3f}")
        print(f"  Y2 dense (object) n={len(y2):3d}  kappa={y2.mean():.3f} "
              f"SE={y2.std(ddof=1)/np.sqrt(len(y2)):.3f}")
        print(f"  diff {y2.mean() - obj.mean():+.3f}   Welch p={t.pvalue:.3f}")
        print(f"  pooled L2 (3 classes) kappa={K.kappa.mean():.3f}  "
              f"<- do NOT compare Y2 against this")
        with open(os.path.join(OUT, "y6_like_for_like.txt"), "w",
                  encoding="utf-8") as f:
            f.write(f"L2 object n={len(obj)} kappa={obj.mean():.4f}\n"
                    f"Y2 dense object n={len(y2)} kappa={y2.mean():.4f}\n"
                    f"diff={y2.mean()-obj.mean():+.4f} welch_p={t.pvalue:.4f}\n"
                    f"L2 pooled all classes kappa={K.kappa.mean():.4f}\n")

    # ---- per prompt --------------------------------------------------------
    pp = (K.groupby(["pclass", "prompt_id"]).kappa
          .agg(["mean", "size"]).reset_index())
    pp.to_csv(os.path.join(OUT, "y6_per_prompt.csv"), index=False)
    print("\n=== per prompt (mean over seeds) ===")
    print(pp.round(3).to_string(index=False))
    print(f"\n  prompt spread: {pp['mean'].min():.3f} .. {pp['mean'].max():.3f}"
          f"  (ratio {pp['mean'].max() / max(pp['mean'].min(), 1e-6):.1f}x)")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.4),
                           gridspec_kw=dict(width_ratios=[1.0, 1.15]))

    a = ax[0]
    data = [K[K.pclass == c].kappa.values for c in ORDER]
    bp = a.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                   medianprops=dict(color="#333", lw=1.4),
                   whiskerprops=dict(color="#888"),
                   capprops=dict(color="#888"))
    for patch, c in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COL[c])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COL[c])
    rng = np.random.default_rng(3)
    for i, (c, v) in enumerate(zip(ORDER, data), start=1):
        a.scatter(i + rng.uniform(-0.11, 0.11, len(v)), v, s=13,
                  color=COL[c], alpha=0.75, zorder=5, lw=0)
        a.scatter([i], [v.mean()], s=64, marker="D", color=COL[c],
                  edgecolor="white", lw=1.0, zorder=6)
    a.axhline(K.kappa.mean(), ls="--", lw=1.2, color="#999")
    a.text(0.985, 0.97, f"pooled mean {K.kappa.mean():.3f}",
           transform=a.transAxes, fontsize=6.4, color="#888", va="top",
           ha="right")
    a.set_xticks([1, 2, 3])
    a.set_xticklabels([f"{c}\nn={len(v)}" for c, v in zip(ORDER, data)],
                      fontsize=7.6)
    a.set_ylabel(r"$\kappa$  (parabola-refined, per cell)")
    a.set_title(f"(a)  class matters (Kruskal p={kw.pvalue:.3f})",
                fontsize=8.6, loc="left")
    a.set_ylim(-0.02, 0.52)

    b = ax[1]
    for c in ORDER:
        h = pp[pp.pclass == c]
        xs = np.arange(len(h)) if False else h["mean"].values
        b.scatter(np.full(len(h), ORDER.index(c)), h["mean"], s=30,
                  color=COL[c], zorder=5, edgecolor="white", lw=0.6)
    for c in ORDER:
        h = pp[pp.pclass == c]
        b.plot([ORDER.index(c), ORDER.index(c)], [h["mean"].min(), h["mean"].max()],
               lw=2.2, color=COL[c], alpha=0.35, solid_capstyle="round")
    b.axhline(0.211, ls="--", lw=1.2, color="#c0392b",
              label="L2 pooled, 3 classes: 0.211")
    b.axhline(0.159, ls=":", lw=1.2, color="#1f6fb4",
              label="Y2 dense, object only: 0.159")
    b.set_xticks([0, 1, 2])
    b.set_xticklabels(ORDER, fontsize=8.0)
    b.set_xlim(-0.5, 2.5)
    b.set_ylabel(r"per-prompt $\kappa$")
    b.set_title("(b)  prompt spread is 10x, class shifts the mean",
                fontsize=8.6, loc="left")
    b.set_ylim(-0.02, 0.62)
    b.legend(fontsize=6.3, loc="upper right", framealpha=0.95,
             handlelength=1.6, borderpad=0.4)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_y6.png")
    fig.savefig(p, bbox_inches="tight")
    print("\nwrote", p)


if __name__ == "__main__":
    main()
