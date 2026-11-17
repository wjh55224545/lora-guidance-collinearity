"""Y5 (paper 2): re-estimate L2's kappa with the interpolating estimator.

Why this is the first thing to run (costs one minute, no GPU):

  The whole interval kappa in [0.17, 0.27] rests on the *grid* estimator
  s*(w) = argmin over 9 s values, so log s* can only take 9 levels and the
  quantisation step (~0.18 in log space) is the same size as the movement that
  w is supposed to produce.  E2c-2 showed what that costs: on the same rows,
  swapping grid for a parabola through the s-curve moved the paired p-values
  from .0067/.0115 to .0847/.0616, i.e. it flipped a "significant" variant
  effect into nothing.

  If L2's kappa moves by more than 0.05 under the same swap, then the published
  interval is an artefact of the grid and has to be replaced, not defended.

  Reporting rule: this is a *self-falsification* check.  Both estimators are
  always printed; a result where quad agrees with grid is only believable
  because the disagreement case was pre-committed to.

Run:  python y5_l2_quad.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2d_analyze as A  # noqa: E402

L2 = os.path.join(HERE, "out_long2", "L2_images.csv")
ALT = r"E:\研究\results\论文二_LoRA共线\L2_2026-11-06\csv\L2_images.csv"
OUT = os.path.join(HERE, "out_y")


def load():
    p = L2 if os.path.exists(L2) else ALT
    D = pd.read_csv(p)
    G = D[D.segment == "G"].copy()
    G["tag"] = "L2"
    print(f"  source={os.path.basename(p)}  rows={len(G)}  "
          f"prompts={G.prompt_id.nunique()}  seeds={sorted(G.seed.unique())}")
    print(f"  w grid={sorted(G.w.unique())}")
    print(f"  s grid={sorted(G.s.unique())}")
    return G


def per_prompt(G, method):
    K, edge = A.kappa_of(G, method=method)
    return K, edge


def boot_ci(v, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    os.makedirs(OUT, exist_ok=True)
    G = load()

    res = {}
    print("\n=== kappa under the two s* estimators ===")
    print(f"{'estimator':10s} {'n_fit':>6s} {'mean':>7s} {'median':>7s} "
          f"{'sd':>6s} {'95% CI (cell bootstrap)':>26s} {'edge':>5s}")
    for method in ("grid", "quad"):
        K, edge = per_prompt(G, method)
        v = K.kappa.values
        lo, hi = boot_ci(v)
        res[method] = K
        print(f"{method:10s} {len(v):6d} {v.mean():7.3f} {np.median(v):7.3f} "
              f"{v.std(ddof=1):6.3f}   [{lo:+.3f}, {hi:+.3f}]      {edge:5.2f}")

    g = res["grid"].set_index(["prompt_id", "seed"]).kappa
    q = res["quad"].set_index(["prompt_id", "seed"]).kappa
    common = g.index.intersection(q.index)
    d = (q.loc[common] - g.loc[common]).values
    tt = stats.ttest_1samp(d, 0.0)
    print(f"\n  paired grid -> quad over {len(common)} cells: "
          f"diff={d.mean():+.3f}  sd={d.std(ddof=1):.3f}  "
          f"t={tt.statistic:+.2f}  p={tt.pvalue:.4g}")
    rho = stats.spearmanr(g.loc[common], q.loc[common])
    print(f"  rank agreement: spearman rho={rho.statistic:+.3f} (p={rho.pvalue:.4g})")

    # the decision rule, fixed before looking at the numbers
    shift = abs(res["quad"].kappa.mean() - res["grid"].kappa.mean())
    print("\n=== verdict ===")
    if shift > 0.05:
        print(f"  |shift|={shift:.3f} > 0.05  -> the published interval "
              f"[0.17, 0.27] is grid-contaminated; replace it with the quad "
              f"interval above and say so in the text.")
    else:
        print(f"  |shift|={shift:.3f} <= 0.05  -> L2's interval survives the "
              f"estimator swap.  The E2c-2 instability was a *small-sample* "
              f"effect (12 cells per variant), not a property of L2's 91 cells.")

    # how far is either estimator from the two reference hypotheses
    for name, k in (("grid", res["grid"].kappa.mean()),
                    ("quad", res["quad"].kappa.mean())):
        print(f"  {name}: kappa={k:.3f} -> distance to kappa=0 is {k:.3f}, "
              f"to kappa=1 (equal w*s) is {1 - k:.3f}, ratio {1 / k:.1f}x")

    tag = res["grid"].assign(estimator="grid")
    tag = pd.concat([tag, res["quad"].assign(estimator="quad")], ignore_index=True)
    tag.to_csv(os.path.join(OUT, "y5_l2_kappa.csv"), index=False)

    S = pd.DataFrame([dict(estimator=m,
                           kappa=res[m].kappa.mean(),
                           kappa_med=res[m].kappa.median(),
                           kappa_sd=res[m].kappa.std(ddof=1),
                           n_fit=len(res[m])) for m in ("grid", "quad")])
    S.to_csv(os.path.join(OUT, "y5_summary.csv"), index=False)
    print(f"\n  wrote {OUT}/y5_{{l2_kappa,summary}}.csv")


if __name__ == "__main__":
    main()
