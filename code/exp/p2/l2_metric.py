"""L2-metric robustness (paper 2): does any of it depend on WHICH distance?

Every L2 readout so far used pix_l2 (raw pixel L2 to the reference image).  A
reviewer can ask whether kappa and the route ranking are an artefact of that one
choice.  This recomputes the two headline readouts with clip_i_ref (CLIP image
embedding distance to the same reference) and with clip_t (text alignment):

    kappa      exponent of the empirical iso-line s*(w), per (prompt, seed)
    routes     mean distance per route at matched intent

Agreement is what makes the paper's numbers quotable; disagreement would mean
the "s compensates w" curve is metric-specific and has to be reported per metric.

Reads L2_images.csv only.  Safe to run while the long run is still appending.

Run:  python l2_metric.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2_analyze as A  # noqa: E402

OUT = os.path.join(HERE, "out_long2")
METRICS = ["pix_l2", "clip_i_ref", "clip_i_base", "clip_t"]


def kappa_table(d):
    rows = []
    for val in METRICS:
        wd = A.wide(d, val)
        # iso-line needs a "distance to reference", so flip the clip
        # similarities into distances (monotone -> same iso-line)
        if val.startswith("clip"):
            wd = -wd
        E, K, imp = A.part2_isoline(wd)
        rows.append(dict(metric=val, n=len(K),
                         kappa_mean=K.kappa.mean(), kappa_med=K.kappa.median(),
                         kappa_p10=K.kappa.quantile(.1),
                         kappa_p90=K.kappa.quantile(.9),
                         r2_med=K.r2.median(),
                         mono=float((K.kappa > 0).mean())))
    return pd.DataFrame(rows)


def route_table(t):
    rows = []
    for val in METRICS:
        m = t.groupby("route")[val].agg(["mean", "median", "max"])
        for r in m.index:
            rows.append(dict(route=r, metric=val, mean=m.loc[r, "mean"],
                             median=m.loc[r, "median"], max=m.loc[r, "max"]))
    return pd.DataFrame(rows)


def main():
    d = pd.read_csv(os.path.join(OUT, "L2_images.csv"))
    g, t = d[d.segment == "G"], d[d.segment == "T"]
    print(f"G {len(g)} rows, T {len(t)} rows\n")

    k = kappa_table(g)
    k.to_csv(os.path.join(OUT, "an_metric_kappa.csv"), index=False)
    print("=== kappa by metric (iso-line s*(w)) ===")
    print(k.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    r = route_table(t)
    r.to_csv(os.path.join(OUT, "an_metric_routes.csv"), index=False)
    print("\n=== route means by metric ===")
    print(r.pivot_table(index="route", columns="metric",
                        values="mean").to_string(
        float_format=lambda v: f"{v:.3f}"))

    # does the route ORDER survive the metric switch?
    print("\n=== route order per metric (best first) ===")
    # pix_l2 is a distance (small = close); the clip columns are similarities
    # (large = close).  Sorting both ascending silently reverses the clip order
    # -- that mistake is why this comment exists.
    for val in METRICS:
        s = r[r.metric == val].set_index("route")["mean"]
        s = s.sort_values(ascending=(val == "pix_l2"))
        print(f"{val:12s} " + " < ".join(s.index))

    print("\n=== paired tests (per prompt, seed, w), sign = closer is better ===")
    for val, sign in [("pix_l2", 1), ("clip_i_ref", -1), ("clip_i_base", -1)]:
        piv = t.pivot_table(index=["prompt_id", "seed", "w"],
                            columns="route", values=val) * sign
        for a_, b_ in [("geff_const", "fixed_s"), ("ws_equal", "fixed_s"),
                       ("geff_const", "ws_equal")]:
            dd = piv[a_] - piv[b_]
            try:
                pv = stats.wilcoxon(dd).pvalue
            except ValueError:
                pv = np.nan
            print(f"{val:12s} {a_:11s} - {b_:11s} "
                  f"mean {dd.mean():+.4f}  wins {float((dd < 0).mean()):.2f}  "
                  f"p={pv:.2e}")

    # per-prompt agreement of the iso-line exponent
    base = A.wide(g, "pix_l2")
    alt = -A.wide(g, "clip_i_ref")
    try:
        K1 = A.part2_isoline(base)[1].kappa
        K2 = A.part2_isoline(alt)[1].kappa
        if len(K1) == len(K2):
            rho = stats.spearmanr(K1, K2).statistic
            print(f"\nper-(prompt,seed) kappa agreement pix_l2 vs clip_i_ref: "
                  f"rho={rho:.3f}  median|diff|={np.median(np.abs(K1-K2)):.3f}")
    except Exception as e:                      # noqa: BLE001
        print("kappa agreement skipped:", e)


if __name__ == "__main__":
    main()
