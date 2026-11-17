"""E2b companion: is rho(w) a power law or a saturating curve?

Four points spanning 10x cannot tell these apart by eye, so fit both and compare
residuals.  This matters because "alpha_t is a power-law exponent" and "alpha_t
is the local elastic slope of a linear-plus-saturation law" are different claims,
and only the second one survives.

  power:  log rho = alpha * log w + c
  satur:  rho = A * w / (1 + w/w0)

w0 is the half-saturation scale: rho -> A*w0/2 as w -> inf.  For the middle of
the trajectory w0 is unidentifiable (it runs off to 1e6+, i.e. the data are
linear there); at both ends it lands inside or near the measured window, which
is exactly where the local slope drops below 1.

Run:  python e2b_fit.py
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def fit_power(x, y):
    a, b = np.polyfit(np.log(x), np.log(y), 1)
    r = y - np.exp(a * np.log(x) + b)
    return a, np.nan, float((r ** 2).sum())


def fit_sat(x, y):
    def f(x, A, w0):
        return A * x / (1 + x / w0)

    p, _ = curve_fit(f, x, y, p0=[y[-1] / x[-1], 0.1], maxfev=40000)
    r = y - f(x, *p)
    return np.nan, float(p[1]), float((r ** 2).sum())


def main():
    d = pd.read_csv(os.path.join(OUT, "e2b_rel.csv"))
    w = np.array(sorted(d.rel.unique()))
    print(f"w = {w}   span {w[-1]/w[0]:.0f}x")

    rows = []
    for p in d.prompt.unique():
        for t in sorted(d.t_frac.unique(), reverse=True):
            s = d[(d.prompt == p) & (np.abs(d.t_frac - t) < 1e-9)].sort_values("rel")
            x, y = s.rel.values, s.rho.values
            a1, w0_1, ss1 = fit_power(x, y)
            a2, w0_2, ss2 = fit_sat(x, y)
            rows.append(dict(prompt=p, t=t, alpha=a1, w0=w0_2,
                             ss_power=ss1, ss_sat=ss2))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "e2b_fit_compare.csv"), index=False)

    print("\nmodel comparison (a model 'wins' if its SSR is < half the other's)")
    print(f"  saturating better : {(r.ss_sat < r.ss_power * 0.5).sum():2d} / {len(r)}")
    print(f"  power better      : {(r.ss_power < r.ss_sat * 0.5).sum():2d} / {len(r)}")
    print(f"  inconclusive      : {((r.ss_sat >= r.ss_power * 0.5) & (r.ss_power >= r.ss_sat * 0.5)).sum():2d} / {len(r)}")

    print("\nalpha_t and half-saturation point w0(t)")
    print(f"{'prompt':8s} {'t':>6s} {'alpha':>7s} {'w0':>12s} {'regime':>10s}")
    for _, q in r.iterrows():
        regime = "linear" if q.w0 > 1.0 else ("saturating" if q.w0 < 0.6 else "transition")
        print(f"{q.prompt:8s} {q.t:6.3f} {q.alpha:7.3f} {q.w0:12.4g} {regime:>10s}")

    print(f"\nw0 median {r.w0.median():.3g}   "
          f"fraction with w0 < 0.2 (saturated inside the measured window): "
          f"{(r.w0 < 0.2).mean():.2f}")
    print("wrote", os.path.join(OUT, "e2b_fit_compare.csv"))


if __name__ == "__main__":
    main()
