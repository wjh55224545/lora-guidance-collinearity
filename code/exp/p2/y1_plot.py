"""Y1 figure: did the two LoRAs actually learn anything?

Left: training loss for both concepts, parsed out of the run logs (no retraining).
Right: the size of the edit at w=0/1/2, as pix_l2 against the w=0 image.  The
2.0 threshold is drawn because it is the line that separates "this LoRA does
something" from "this LoRA is inert" -- an inert LoRA must never be reported as
evidence of weak collinearity, so the check has to be visible, not asserted.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "论文二_LoRA共线", "Y1_2026-11-08")
SRC = os.path.join(ROOT, "exp", "p2", "out_y")

INERT = 2.0
plt.rcParams.update({
    "figure.dpi": 190, "savefig.dpi": 190, "font.size": 8.4,
    "axes.edgecolor": "#444", "axes.grid": True, "grid.color": "#e8e8e8",
    "grid.linewidth": 0.6, "figure.facecolor": "white",
    "axes.facecolor": "white",
})

STEP = re.compile(r"step\s+(\d+)/\d+\s+loss\s+([0-9.]+)")


def parse(tag):
    p = os.path.join(RES, "run_logs", f"y1_{tag}_train.log")
    if not os.path.exists(p):
        p = os.path.join(SRC, "logs", f"y1_{tag}_train.log")
    st, ls = [], []
    for line in open(p, encoding="utf-8", errors="ignore"):
        m = STEP.search(line)
        if m:
            st.append(int(m.group(1)))
            ls.append(float(m.group(2)))
    return np.array(st), np.array(ls)


def main():
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3),
                           gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    # ---- (a) training loss ------------------------------------------------
    a = ax[0]
    cols = {"A": "#1f6fb4", "B": "#d1495b"}
    names = {"A": "A: flat vector, teal–coral", "B": "B: chinese ink wash"}
    for tag in ("A", "B"):
        s, l = parse(tag)
        if not len(s):
            continue
        a.plot(s, l, "-", lw=0.7, color=cols[tag], alpha=0.35)
        k = max(5, len(l) // 25)
        sm = np.convolve(l, np.ones(k) / k, mode="valid")
        a.plot(s[k - 1:], sm, "-", lw=1.8, color=cols[tag],
               label=f"{names[tag]}   (final {l[-1]:.3f})")
    a.set_xlabel("step")
    a.set_ylabel("DDPM loss")
    a.set_title("(a)  both concepts train, 1000 steps each", fontsize=8.8,
                loc="left")
    a.legend(fontsize=6.6, loc="upper right", framealpha=0.92)
    a.set_ylim(0, max(0.30, float(np.percentile(np.concatenate(
        [parse("A")[1], parse("B")[1]]), 99)) * 1.1))

    # ---- (b) edit size vs the inert threshold ----------------------------
    b = ax[1]
    xs, labels, colors = [], [], []
    rows = []
    for tag in ("A", "B"):
        p = os.path.join(RES, "csv", f"y1_verify_{tag}.csv")
        if not os.path.exists(p):
            p = os.path.join(SRC, f"y1_verify_{tag}.csv")
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append((tag, r.prompt_id, r.w, r.pix_l2_base))
    for i, (tag, pid, w, v) in enumerate(rows):
        xs.append(v)
        labels.append(f"{tag}·{pid[-2:]}\nw={w:g}")
        colors.append(cols[tag])
    y = np.arange(len(xs))
    b.barh(y, xs, color=colors, height=0.62, alpha=0.9)
    b.axvline(INERT, ls="--", lw=1.4, color="#c0392b")
    b.text(0.985, 0.955, f"red dashed = INERT threshold ({INERT:g})",
           transform=b.transAxes, ha="right", va="top", fontsize=6.4,
           color="#c0392b")
    for yy, v in zip(y, xs):
        b.text(v + 0.4, yy, f"{v:.1f}", va="center", fontsize=6.4, color="#333")
    b.set_yticks(y)
    b.set_yticklabels(labels, fontsize=6.0)
    b.invert_yaxis()
    b.set_xlabel(r"pix\_l2 vs the $w$=0 image (64×64)")
    b.set_title(r"(b)  both LoRAs move the picture, far past inert",
                fontsize=8.8, loc="left")
    b.set_xlim(0, max(xs) * 1.22)

    fig.tight_layout()
    outp = os.path.join(RES, "fig", "fig_Y1.png")
    fig.savefig(outp, bbox_inches="tight")
    print("wrote", outp)
    print(pd.DataFrame(rows, columns=["concept", "prompt", "w", "pix_l2_base"])
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
