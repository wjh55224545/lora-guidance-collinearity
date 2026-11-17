# -*- coding: utf-8 -*-
"""Y10 (paper 2): the perceptual consequences of the g_eff inflation.

Question this answers (the reviewer's "so what?"):
  kappa ~ 0.2 says s_eff inflates by up to +82% (w=2).  Is that inflation
  *harmless* (a number that only matters for cross-paper comparability) or
  does it leave visible traces in the pixels?

Pre-registered criteria (written before running, cf. log discipline):
  C1  If saturation rate and Laplacian energy rise monotonically with s at
      fixed w (and/or with w at fixed s), the inflation has a perceptual
      signature -- "oversharpen / oversaturate" is a real cost, not a bookkeeping
      artefact.
  C2  Two-factor standardized regression  metric ~ z(log w) + z(log s):
      if |beta_w| < 0.15 * |beta_s|, the artefact is driven by the guidance
      channel (w acts through g_eff); otherwise w has an independent
      artefact channel and the protocol must say so.

Data: every image of the Y2 grids is on disk (y2_full 1760, y2_A 1472),
with (prompt_id, seed, w, s) in images.csv.  No generation needed.

Failure mode: if the metrics are flat in both w and s, report "no perceptual
signature at this grid" and drop the claim -- do not fish.
"""
from __future__ import annotations

import csv
import os
import numpy as np
from PIL import Image

HERE = r"E:\研究\exp\p2"
OUT = r"E:\研究\results\论文二_LoRA共线\Y10_2026-11-08"
GRID = [
    ("full", os.path.join(HERE, "out_y", "y2_full")),
    ("A",    os.path.join(HERE, "out_y", "y2_A")),
]


def metrics(path):
    """Laplacian variance (sharpness), saturation rates, mean luminance."""
    im = Image.open(path).convert("L")
    a = np.asarray(im, dtype=np.float64) / 255.0
    lap = (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1)
           + np.roll(a, -1, 1) - 4.0 * a)
    return float(lap[1:-1, 1:-1].var()), float((a > 0.985).mean()), \
        float((a < 0.015).mean()), float(a.mean())


def main():
    os.makedirs(os.path.join(OUT, "csv"), exist_ok=True)
    rows = []
    for tag, root in GRID:
        csv_p = os.path.join(root, "images.csv")
        img_d = os.path.join(root, "img")
        if not os.path.exists(csv_p):
            continue
        for r in csv.DictReader(open(csv_p, encoding="utf-8")):
            p = r.get("img_path") or ""
            if not p or not os.path.exists(p):
                # images.csv records the path but y_gen saved with quality=88
                # as .jpg in some runs; fall back to the img dir listing.
                cand = os.path.join(
                    img_d, f"{r['prompt_id']}_sd{r['seed']}_w{r['w']}_s{r['s']}.png")
                p = cand if os.path.exists(cand) else ""
            if not p:
                continue
            lv, shi, slo, ml = metrics(p)
            rows.append(dict(tag=tag, prompt_id=r["prompt_id"],
                             seed=r["seed"], w=float(r["w"]), s=float(r["s"]),
                             lap_var=lv, sat_hi=shi, sat_lo=slo, lum=ml))
    csv_out = os.path.join(OUT, "csv", "y10_perceptual.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[y10] {len(rows)} images scored -> {csv_out}")

    # ---- analysis -----------------------------------------------------
    def z(x):
        x = np.asarray(x, float)
        return (x - x.mean()) / (x.std() + 1e-12)

    print("\n== per-tag: standardized two-factor regression ==")
    for tag in ("full", "A"):
        sub = [r for r in rows if r["tag"] == tag]
        if len(sub) < 100:
            continue
        lw, ls = z([np.log(r["w"]) for r in sub]), z([np.log(r["s"]) for r in sub])
        X = np.column_stack([lw, ls, np.ones(len(sub))])
        for name in ("lap_var", "sat_hi"):
            y = z([r[name] for r in sub])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            print(f"[{tag}] {name}: beta_w={beta[0]:+.3f} beta_s={beta[1]:+.3f} "
                  f"|b_w/b_s|={abs(beta[0]/beta[1]):.3f} R2={r2:.3f}")

    print("\n== monotonicity: metric vs s at w=1.0 (mean over prompts/seeds) ==")
    for tag in ("full", "A"):
        sub = [r for r in rows if r["tag"] == tag and abs(r["w"] - 1.0) < 1e-9]
        if not sub:
            continue
        ss = sorted(set(r["s"] for r in sub))
        prof = [np.mean([r["sat_hi"] for r in sub if r["s"] == s]) for s in ss]
        prof_l = [np.mean([r["lap_var"] for r in sub if r["s"] == s]) for s in ss]
        mono = all(prof[i] <= prof[i + 1] + 1e-4 for i in range(len(prof) - 1))
        mono_l = all(prof_l[i] <= prof_l[i + 1] + 1e-3 for i in range(len(prof_l) - 1))
        print(f"[{tag}] w=1.0  sat_hi: {prof[0]:.4f} -> {prof[-1]:.4f} "
              f"({prof[-1]/max(prof[0],1e-9):.2f}x) monotone={mono}")
        print(f"[{tag}] w=1.0  lap_var: {prof_l[0]:.1f} -> {prof_l[-1]:.1f} "
              f"({prof_l[-1]/max(prof_l[0],1e-9):.2f}x) monotone={mono_l}")

    print("\n== w sweep at s=7.5 (mean over prompts/seeds) ==")
    for tag in ("full", "A"):
        sub = [r for r in rows if r["tag"] == tag and abs(r["s"] - 7.5) < 1e-9]
        if not sub:
            continue
        ws = sorted(set(r["w"] for r in sub))
        prof = [np.mean([r["sat_hi"] for r in sub if r["w"] == v]) for v in ws]
        prof_l = [np.mean([r["lap_var"] for r in sub if r["w"] == v]) for v in ws]
        print(f"[{tag}] s=7.5  w {ws[0]}->{ws[-1]}  "
              f"sat_hi {prof[0]:.4f}->{prof[-1]:.4f}  "
              f"lap_var {prof_l[0]:.1f}->{prof_l[-1]:.1f}")


if __name__ == "__main__":
    main()
