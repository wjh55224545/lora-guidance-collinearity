"""G-ablation (paper 2): is g* constant, and what does the slice of it cost?

The calibration of Eq. (calib) is written as "hold the effective guidance at a
target g* and solve for s".  The target has to come from somewhere: in the
paper it is the g_eff of the reference configuration (w=1, s=7.5).  That choice
carries an assumption which has never been tested on its own -- that g* is the
same at every w.  This script tests it with data already on disk and no new
generation.

Four questions:

  Q1  Does g*(w) = s*(w) * (1 + w rho(w)) stay put as w moves?  If yes the
      confusion is purely one of units and the calibration is exact; the
      measured kappa < 1 says it does not, and this puts a number on how far it
      drifts.
  Q2  If we take g* from the reference configuration and keep it constant (what
      the paper does), how far off is the predicted s*, per w?
  Q3  How much does that cost in pixel error against per-cell search, compared
      with not compensating at all?
  Q4  How much does s* vary across prompts at fixed w?  A constant compensation
      cannot follow that, and the spread is the honest limit on any single-s
      rule.

Rule followed here: every number printed is computed from
`L2_2026-11-06/csv/an_isoline.csv` (per-cell argmin of pix_l2 on the 9-point s
grid) and `E3a_2026-11-06/e3a_wrho.csv` (measured w rho per prompt class).  No
model is loaded.

Run:
    python exp/p2/g_ablate.py --out results/论文二_LoRA共线/GABL_2026-11-14
"""
import argparse
import os

import numpy as np
import pandas as pd

L2 = "results/论文二_LoRA共线/L2_2026-11-06/csv"
E3A = "results/论文二_LoRA共线/E3a_2026-11-06"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", default=L2)
    ap.add_argument("--e3a", default=E3A)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    iso = pd.read_csv(os.path.join(a.l2, "an_isoline.csv"))
    img = pd.read_csv(os.path.join(a.l2, "L2_images.csv"))
    wr = pd.read_csv(os.path.join(a.e3a, "e3a_wrho.csv"))
    img = img[img.route == "grid"]

    # ---- Q1: does the target g* drift with w?
    rows = []
    for cls, g in iso.groupby("pclass"):
        med = g.groupby("w").s_star.median()
        w_rho = wr[wr.prompt == cls].set_index("w").wrho
        common = [w for w in med.index if w in w_rho.index]
        if not common:
            continue
        for w in common:
            gstar = med[w] * (1.0 + w_rho[w])
            rows.append(dict(pclass=cls, w=w, s_star_med=med[w],
                             wrho=w_rho[w], g_star=gstar,
                             g_star_rel=gstar / (med[1.0] * (1.0 + w_rho[1.0]))
                             if 1.0 in med.index else np.nan))
    q1 = pd.DataFrame(rows)
    q1.to_csv(os.path.join(a.out, "g1_target_drift.csv"), index=False)

    drift = (q1.groupby("pclass")
             .g_star_rel.agg(["min", "max", "median"])
             .assign(cv=lambda d: q1.groupby("pclass").g_star_rel.std()
                     / q1.groupby("pclass").g_star_rel.mean())
             .round(3))

    # ---- Q2: constant-g* prediction of s*, three variants
    pred = []
    for cls, g in iso.groupby("pclass"):
        med = g.groupby("w").s_star.median()
        w_rho = wr[wr.prompt == cls].set_index("w").wrho
        common = [w for w in med.index if w in w_rho.index]
        if 1.0 not in common:
            continue
        g_ref = med[1.0] * (1.0 + w_rho[1.0])
        for w in common:
            truth = med[w]
            # (i) hold g* at its reference value (the paper's rule)
            s_const_g = g_ref / (1.0 + w_rho[w])
            # (ii) no compensation at all, s left at the reference 7.5
            s_none = med[1.0]
            # (iii) equal w*s -- the community shortcut
            s_ws = med[1.0] * 1.0 / w
            for name, sp in (("const_g", s_const_g),
                             ("no_comp", s_none),
                             ("equal_ws", s_ws)):
                pred.append(dict(pclass=cls, w=w, rule=name,
                                 s_pred=float(sp), s_true=float(truth),
                                 abs_err=abs(sp - truth),
                                 rel_err=abs(sp - truth) / truth))
    q2 = pd.DataFrame(pred)
    q2.to_csv(os.path.join(a.out, "g2_sstar_prediction.csv"), index=False)
    q2s = (q2.groupby("rule")
           .agg(med_rel_err=("rel_err", "median"),
                p90_rel_err=("rel_err", lambda x: np.percentile(x, 90)),
                med_abs_err=("abs_err", "median"))
           .round(3))
    q2w = (q2.pivot_table(index=["pclass", "w"], columns="rule",
                          values="rel_err").round(3))
    q2w.to_csv(os.path.join(a.out, "g2b_rel_err_by_w.csv"))

    # ---- Q3: pixel cost.  For each (prompt, seed, w) compare the pixel error
    # at the rule's s against the error at that cell's own argmin.
    cost = []
    for cls, g in iso.groupby("pclass"):
        med = g.groupby("w").s_star.median()
        w_rho = wr[wr.prompt == cls].set_index("w").wrho
        common = [w for w in med.index if w in w_rho.index]
        if 1.0 not in common:
            continue
        g_ref = med[1.0] * (1.0 + w_rho[1.0])
        grid = sorted(img[img.pclass == cls].s.unique())
        for _, r in g.iterrows():
            w = r.w
            if w not in w_rho.index:
                continue
            cand = {
                "const_g": g_ref / (1.0 + w_rho[w]),
                "no_comp": med[1.0],
                "equal_ws": med[1.0] / w,
                "oracle": r.s_star,
            }
            cell = img[(img.pclass == cls) & (img.prompt_id == r.prompt_id)
                       & (img.seed == r.seed) & (img.w == w)]
            if not len(cell):
                continue
            for name, sp in cand.items():
                snap = min(grid, key=lambda x: abs(x - sp))
                v = cell[cell.s == snap].pix_l2
                if not len(v):
                    continue
                cost.append(dict(pclass=cls, prompt_id=r.prompt_id,
                                 seed=r.seed, w=w, rule=name,
                                 s_used=snap, pix_l2=float(v.iloc[0])))
    q3 = pd.DataFrame(cost)
    q3.to_csv(os.path.join(a.out, "g3_pixel_cost_cells.csv"), index=False)
    # normalise per cell against the oracle so cells do not weigh by scale
    piv = q3.pivot_table(index=["pclass", "prompt_id", "seed", "w"],
                         columns="rule", values="pix_l2")
    rel = piv.div(piv["oracle"], axis=0).reset_index().melt(
        id_vars=["pclass", "prompt_id", "seed", "w"],
        var_name="rule", value_name="ratio")
    rel = rel[rel.rule != "oracle"]
    agg = rel.groupby("rule").ratio.agg(med_ratio="median", mean_ratio="mean",
                                        worse_frac=lambda x: float(
                                            np.mean(x > 1.05)))
    agg["p90_ratio"] = rel.groupby("rule").ratio.quantile(0.90)
    q3s = agg.round(3)[["med_ratio", "mean_ratio", "p90_ratio", "worse_frac"]]
    q3s.to_csv(os.path.join(a.out, "g3b_pixel_cost_summary.csv"))

    # ---- Q4: spread of s* across prompts at fixed w
    q4 = (iso[iso.w != 1.0]
          .groupby(["pclass", "w"]).s_star
          .agg(n="size", med="median", q25=lambda x: x.quantile(.25),
               q75=lambda x: x.quantile(.75), mn="min", mx="max")
          .assign(iqr=lambda d: d.q75 - d.q25,
                  rel_spread=lambda d: (d.mx - d.mn) / d.med)
          .round(3))
    q4.to_csv(os.path.join(a.out, "g4_prompt_spread.csv"))

    print("=== Q1 g*(w) drift, relative to w=1 ===")
    print(drift.to_string())
    print("\n=== Q2 s* prediction error by rule (median over cells) ===")
    print(q2s.to_string())
    print("\n=== Q2b relative error by w ===")
    print(q2w.to_string())
    print("\n=== Q3 pixel cost relative to per-cell oracle ===")
    print(q3s.to_string())
    print("\n=== Q4 spread of s* across prompts ===")
    print(q4.to_string())


if __name__ == "__main__":
    main()
