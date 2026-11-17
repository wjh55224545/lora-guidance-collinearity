"""T3: does kappa survive retraining?  Table + pre-registered criteria.

Three teapot adapters trained from the same 24 concept images with the same recipe
and only the training seed changed (S: unseeded/historical, S6: seed 2024, S7: seed
2025), plus lora_C, the third independently trained style adapter, which until now
had a latent-space rho but no full-grid kappa.

Reads the raw grids, restricts every one of them to the common obj00--obj03 panel
(the same estimator kappa_panel.py uses), and prints the seed spread next to the
across-adapter spread.  Writes kappa_seed.csv.

Run: python t3_analyze.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kappa_panel import _fits, PANEL  # noqa: E402  same estimator, on purpose

ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "论文二_LoRA共线")
T3 = os.path.join(RES, "T3_seed_2026-11-17")

TEAPOT = [
    ("S  (seed unrecorded)", os.path.join(HERE, "out_b1_grid_S", "images.csv")),
    ("S6 (seed 2024)", os.path.join(HERE, "out_t3_S6", "images.csv")),
    ("S7 (seed 2025)", os.path.join(HERE, "out_t3_S7", "images.csv")),
]
STYLE_C = ("C  (pencil sketch, third independent style adapter)",
           os.path.join(HERE, "out_t3_C", "images.csv"))


def one(label, path):
    if not os.path.exists(path):
        return dict(adapter=label, kappa=np.nan, n_fits=0, status="missing")
    d = pd.read_csv(path)
    d = d[d.prompt_id.isin(PANEL)]
    k = _fits(d)
    if not k:
        return dict(adapter=label, kappa=np.nan, n_fits=0, status="no fits")
    return dict(adapter=label, n_prompts=int(d.prompt_id.nunique()), n_fits=len(k),
                kappa=float(np.median(k)), kmin=float(np.min(k)),
                kmax=float(np.max(k)),
                iqr_lo=float(np.percentile(k, 25)),
                iqr_hi=float(np.percentile(k, 75)),
                n_pos=int(sum(v > 0 for v in k)), n_neg=int(sum(v <= 0 for v in k)),
                status="ok", source=os.path.relpath(path, ROOT))


def main():
    rows = [one(lab, p) for lab, p in TEAPOT]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(T3, "kappa_seed.csv"), index=False, float_format="%.6f")
    print(t[["adapter", "n_prompts", "n_fits", "kappa", "iqr_lo", "iqr_hi",
             "kmin", "kmax", "n_pos", "n_neg", "status"]].to_string(index=False))

    ks = t.kappa.dropna().values
    if len(ks) >= 2:
        spread = float(ks.max() - ks.min())
        print("\ntraining-seed spread over %d teapot samples: %.3f  (%.3f .. %.3f)"
              % (len(ks), spread, ks.min(), ks.max()))
        # across-adapter reference: the five subject adapters of Sec. 5.7
        p5 = os.path.join(RES, "T1c_S5", "kappa_panel_5.csv")
        if os.path.exists(p5):
            p = pd.read_csv(p5)
            p = p[p.note == "subject"]
            ref = float(p.kappa.max() - p.kappa.min())
            print("across-subject spread (5 adapters):      %.3f  (%.3f .. %.3f)"
                  % (ref, p.kappa.min(), p.kappa.max()))
            print("\nC1 seed spread < across-adapter spread : %s  (%.3f vs %.3f)"
                  % ("PASS" if spread < ref else "FAIL", spread, ref))
        print("C2 all three positive                   : %s  (%d/%d)"
              % ("PASS" if (ks > 0).all() else "FAIL", int((ks > 0).sum()), len(ks)))
        print("C3 spread <= 0.05 (a +-0.025 band)      : %s  (%.3f)"
              % ("PASS" if spread <= 0.05 else "FAIL", spread))

    c = one(*STYLE_C)
    print("\nthird independent style adapter (was latent-space rho only):")
    print(pd.DataFrame([c]).to_string(index=False))


if __name__ == "__main__":
    main()
