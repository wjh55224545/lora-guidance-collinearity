"""E-metric (paper 2): which distance decides whether s compensates w?

Why this exists
---------------
L2's headline readouts all used `pix_l2`, which -- checked in longrun_p2.py -- is
the L2 distance between **64x64 thumbnails** of the test image and the reference
(w0=1, s0=7.5).  A thumbnail L2 is dominated by global tone and saturation, not
by structure.  That matters because the `ws_equal` route prescribes s = W0*S0/w,
i.e. s = 37.5 at w = .2 and s = 3.75 at w = 2 -- far outside the 3..14 grid the
iso-line was fitted on.  At s = 37.5 the image is heavily over-saturated, so the
thumbnail L2 explodes while a CLIP embedding barely notices.

Four metrics were then recomputed on the finished L2 table.  The ordering of
`ws_equal` turned out to be **metric-invariant** -- it is the worst route under
pix_l2, clip_i_ref, clip_i_base and clip_t alike (win rate 14-21%, p < 1e-51).
So "equal w.s is clearly wrong" is safe.

What is NOT safe is the smaller claim: `geff_const` beats `fixed_s` by -6.3% on
pix_l2 (p = 9e-18), but the gap is insignificant on clip_i_ref (p = .11) and
**reverses** on clip_i_base (+0.0050, p = 1.7e-10).  Since pix_l2 is a 64x64
thumbnail L2 it may be reading tone rather than structure -- exactly the channel
the `ws_equal` blow-up lives in.  This experiment therefore asks a third and
fourth metric -- **SSIM on full 512x512 grayscale** (structure) and
**full-resolution L2** (tone at native scale) -- on freshly generated images for
all three routes, to decide whether the geff-vs-fixed gain is structural.

Design
------
  prompts : PROMPT_N per class (object/person/style), the same prompt text as L2
  seeds   : two, so the win rates are not one-draw
  images  : for each (prompt, seed): 2 anchors (base w=0, reference w0,s0)
            + 3 routes x 8 w, images written to disk as PNG
  metrics : pix64 (L2's own), pix512, ssim, clip_i_ref, clip_t

Readouts
--------
  1  per-metric, per-w route means
  2  paired win rate per (prompt, seed, w): geff_const vs fixed_s, and
     geff_const vs ws_equal -- the two claims L2 currently makes
  3  rank agreement between metrics on the same images (Spearman)

Run:  python e_metric.py --per-class 2 --seeds 11,22
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longrun_p2 as L  # noqa: E402

OUT = os.path.join(HERE, "out_emetric")


def ssim_gray(a, b):
    """SSIM on 8-bit grayscale, full resolution.  skimage's default
    (7x7 uniform, data_range=255) -- kept explicit so it is reproducible."""
    from skimage.metrics import structural_similarity as S
    from PIL import Image
    ga = np.asarray(a.convert("L"), np.uint8)
    gb = np.asarray(b.convert("L"), np.uint8)
    return float(S(gb, ga, data_range=255))


def l2_full(a, b):
    x = np.asarray(a, np.float32).ravel() / 255.0
    y = np.asarray(b, np.float32).ravel() / 255.0
    return float(np.linalg.norm(x - y))


def pick_prompts(per_class):
    out = []
    for c in ("object", "person", "style"):
        got = [p for p in L.PROMPTS if p[1] == c][:per_class]
        out.extend(got)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--seeds", default="11,22")
    ap.add_argument("--prompts", default="")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")]
    prompts = ([p for p in L.PROMPTS if p[0] in a.prompts.split(",")]
               if a.prompts else pick_prompts(a.per_class))
    os.makedirs(a.outdir, exist_ok=True)
    imgdir = os.path.join(a.outdir, "imgs")
    os.makedirs(imgdir, exist_ok=True)

    R = L.Runner()
    L.Runner.instance = R
    print(f"{len(prompts)} prompts x {len(seeds)} seeds x "
          f"({len(L.W_GRID)} w x 3 routes + 2 anchors)", flush=True)

    rows = []
    for seed in seeds:
        for pid, pclass, text in prompts:
            t0 = time.time()
            a_ = R.anchors(pid, text, seed)
            ref, base = a_["ref"], a_["base"]
            for route in ("fixed_s", "geff_const", "ws_equal"):
                rho = L.RHO.get(pclass, L.RHO_DEFAULT)
                imgs, cfg = [], []
                for w in L.W_GRID:
                    s = L.route_s(route, w, rho)
                    t1 = time.time()
                    im = R.gen(text, w, s, seed)
                    imgs.append(im)
                    cfg.append((w, s, time.time() - t1))
                tf, imf = R.score(imgs, [text] * len(imgs))
                for im, (w, s, dt), t_i, i_i in zip(imgs, cfg, tf, imf):
                    p = os.path.join(imgdir,
                                     f"{pid}_{seed}_{route}_w{w}_s{s:.2f}.png")
                    im.save(p)
                    rows.append(dict(
                        prompt_id=pid, pclass=pclass, seed=seed,
                        route=route, w=w, s=s,
                        pix64=float(np.linalg.norm(
                            R.thumb_np(im) - a_["thumb_ref"])),
                        pix512=l2_full(im, ref),
                        ssim=ssim_gray(im, ref),
                        clip_i_ref=float(i_i @ a_["i_ref"]),
                        clip_t=float(t_i @ i_i)))
            print(f"  {pid}/{seed} {len(rows)} rows "
                  f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.outdir, "emetric_images.csv"), index=False)
    analyse(df, a.outdir)


def analyse(df, outdir):
    from scipy import stats
    DIST = {"pix64": +1, "pix512": +1, "ssim": -1,
            "clip_i_ref": -1, "clip_t": -1}

    print("\n=== 1  route means per metric (lower = closer to reference) ===")
    for m, sign in DIST.items():
        piv = df.pivot_table(index="w", columns="route", values=m,
                             aggfunc="mean") * sign
        print(f"\n-- {m}{' (1-ssim)' if m == 'ssim' else ''}")
        print(piv.round(4).to_string())
        piv.to_csv(os.path.join(outdir, f"emetric_byw_{m}.csv"))

    print("\n=== 2  paired win rates (per prompt, seed, w) ===")
    out = []
    for m, sign in DIST.items():
        piv = df.pivot_table(index=["prompt_id", "seed", "w"], columns="route",
                             values=m) * sign
        for a_, b_ in (("geff_const", "fixed_s"), ("geff_const", "ws_equal"),
                       ("ws_equal", "fixed_s")):
            dd = piv[a_] - piv[b_]
            out.append(dict(metric=m, pair=f"{a_} vs {b_}",
                            n=len(dd),
                            wins_a=float((dd < 0).mean()),
                            mean_delta=float(dd.mean()),
                            med_delta=float(dd.median())))
    W = pd.DataFrame(out)
    W.to_csv(os.path.join(outdir, "emetric_paired.csv"), index=False)
    print(W.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== 3  rank agreement between metrics (same images) ===")
    M = df[list(DIST)].copy()
    for m, s in DIST.items():
        M[m] = M[m] * s
    print(M.corr(method="spearman").round(3).to_string())
    M.corr(method="spearman").round(3).to_csv(
        os.path.join(outdir, "emetric_agreement.csv"))

    print("\n=== 4  where the routes disagree most (w x route, ssim) ===")
    print(df.groupby(["route", "w"]).agg(
        ssim=("ssim", "mean"), pix64=("pix64", "mean"),
        s_max=("s", "max")).round(3).to_string())


if __name__ == "__main__":
    main()
