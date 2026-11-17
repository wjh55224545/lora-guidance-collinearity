"""Unified kappa table over the *common* 4-prompt panel (paper 2, §5.7).

Why this exists: §5.7 reports a subject-adapter table and a per-prompt sign count,
but the two were not drawn from the same panel -- the sign count folded in the
teapot's 8-prompt run, which the surrounding text says is not comparable. This
script re-derives every number from the raw grids restricted to obj00--obj03, so
the table, the sign count and the strength column all share one panel.

Estimator is the same one as kappa_fit.py (argmin over s of pix_l2, then the slope
of log s* on log w), so numbers stay comparable with the rest of the paper.

Usage:  python kappa_panel.py [--out TABLE.csv]
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # E:\研究
RES = os.path.join(ROOT, "results", "论文二_LoRA共线")

PANEL = ["obj00", "obj01", "obj02", "obj03"]
W_LO, W_HI = 0.8, 1.2

# (label, csv, note).  The reference adapter is deliberately absent: its kappa in
# §5.7 (0.208, high segment 0.943) comes from the original slice grid of §4, not
# from a subject-adapter run, so folding it in here would mix two estimators.
SOURCES = [
    ("teapot 800", os.path.join(HERE, "out_b1_grid_S", "images.csv"), "subject"),
    ("teapot 1600", os.path.join(HERE, "out_b1_grid_S2", "images.csv"), "subject"),
    ("chair 800", os.path.join(RES, "T1_2026-11-14", "images_S3.csv"), "subject"),
    ("vase 800", os.path.join(RES, "T1b_S4", "images_S4.csv"), "subject"),
    ("boot 800", os.path.join(HERE, "out_b1_grid_S5", "images.csv"), "subject"),
    # note is deliberately NOT "subject": the five SD1.5 points above are the subject
    # table in §5.7 and their span 0.045--0.204 is quoted in the text.  Folding a
    # cross-base point into that summary would silently redefine it.
    # `strength_w1` for this row is NOT comparable with the rows above: SDXL renders at
    # 1024 px, so the 64x64 thumbnail keeps more detail and every pixel distance scales
    # up.  kappa is a log-log slope and is comparable; the raw distance is not.
    ("teapot 800 (SDXL)", os.path.join(HERE, "out_xl_xl_S1", "images.csv"), "SDXL"),
]


def _fits(d: pd.DataFrame, w_sel=None) -> list[float]:
    """Per-(prompt,seed) kappa, argmin over s then slope of log s* on log w."""
    kk = []
    d = d if w_sel is None else d[d.w.isin(w_sel)]
    # a segment only has 3-4 w values, so the minimum differs from the full grid
    min_w = 5 if w_sel is None else 3
    min_pts = 4 if w_sel is None else 3
    for _, g in d.groupby(["prompt_id", "seed"]):
        piv = g.pivot_table(index="w", columns="s", values="pix_l2", aggfunc="median")
        if piv.shape[0] < min_w:
            continue
        s_star = piv.idxmin(axis=1)
        x = np.log(piv.index.values.astype(float))
        y = np.log(s_star.values.astype(float))
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_pts:
            continue
        kk.append(-float(np.polyfit(x[ok], y[ok], 1)[0]))
    return kk


def strength(d: pd.DataFrame, w: float = 1.0, s: float = 7.5) -> float:
    """Median pix_l2_base at (w, s=7.5) -- distance to that prompt's w=0 image.

    Held at s = 7.5 (the default guidance) on purpose: at fixed w, changing s also
    changes the picture through CFG, so a median over all s would fold the guidance
    effect into what is supposed to be the adapter's own perturbation.  This is the
    convention §5.7 quotes (chair 13.36, boot 12.68); averaging over s instead gives
    13.33 and 13.31 and would make the table disagree with the text.
    Not comparable across bases: SDXL renders at 1024 px and every distance scales up.
    """
    g = d[(np.isclose(d.w, w)) & (np.isclose(d.s, s))]
    return float(g.pix_l2_base.median()) if len(g) else float("nan")


def main():
    import sys
    out = os.path.join(HERE, "out_kappa_panel.csv")
    if "--out" in sys.argv:                       # 之前不认 --out，箭头总指向 exp/p2
        out = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
    rows = []
    for label, path, note in SOURCES:
        if not os.path.exists(path):
            rows.append(dict(adapter=label, note=note, n_fits=0, status="missing: " + path))
            continue
        raw = pd.read_csv(path)
        d = raw[raw.prompt_id.isin(PANEL)]
        if not len(d):
            rows.append(dict(adapter=label, note=note, n_fits=0, status="no panel prompts"))
            continue
        n_prompts = d.prompt_id.nunique()
        allf = _fits(d)
        lo = _fits(d, sorted(w for w in d.w.unique() if w <= W_LO))
        hi = _fits(d, sorted(w for w in d.w.unique() if w >= W_HI))
        pos = sum(1 for v in allf if v > 0)
        rows.append(dict(
            adapter=label, note=note, n_prompts=n_prompts, n_fits=len(allf),
            kappa=float(np.median(allf)) if allf else float("nan"),
            iqr_lo=float(np.percentile(allf, 25)) if allf else float("nan"),
            iqr_hi=float(np.percentile(allf, 75)) if allf else float("nan"),
            kmin=float(np.min(allf)) if allf else float("nan"),
            kmax=float(np.max(allf)) if allf else float("nan"),
            n_pos=pos, n_neg=len(allf) - pos,
            kappa_lo=float(np.median(lo)) if lo else float("nan"),
            kappa_hi=float(np.median(hi)) if hi else float("nan"),
            strength_w1=strength(d),
            status="ok", source=os.path.relpath(path, ROOT),
        ))
    t = pd.DataFrame(rows)
    cols = ["adapter", "note", "n_prompts", "n_fits", "kappa", "iqr_lo", "iqr_hi",
            "kmin", "kmax", "n_pos", "n_neg", "kappa_lo", "kappa_hi",
            "strength_w1", "status"]
    t = t.reindex(columns=cols + [c for c in t.columns if c not in cols])

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("== kappa over the common 4-prompt panel (obj00--obj03) ==")
    print(t[["adapter", "n_fits", "kappa", "iqr_lo", "iqr_hi", "n_pos", "n_neg",
             "kappa_lo", "kappa_hi", "strength_w1"]].to_string(index=False,
                                                               float_format=lambda v: "%.3f" % v))
    sub = t[(t.note == "subject") & (t.status == "ok")]
    if len(sub):
        print("\nsubject adapters: n=%d fits, %d positive, %d negative, kappa spans %.3f--%.3f"
              % (sub.n_fits.sum(), sub.n_pos.sum(), sub.n_neg.sum(),
                 sub.kappa.min(), sub.kappa.max()))
        print("strength at w=1: " + ", ".join("%s %.2f (k=%.3f)" % (r.adapter, r.strength_w1, r.kappa)
                                              for r in sub.itertuples()))
    non = t[(t.note != "subject") & (t.status == "ok")]
    for r in non.itertuples():
        print("\n[%s] pooled over all four prompts, so an unidentifiable cell enters as a zero "
              "slope; the paper quotes the median over identifiable cells instead "
              "(§5.8: 0.134 for SDXL), not this pooled value (%.3f)." % (r.adapter, r.kappa))

    t.to_csv(out, index=False, float_format="%.6f")
    print("\n-> " + out)


if __name__ == "__main__":
    main()
