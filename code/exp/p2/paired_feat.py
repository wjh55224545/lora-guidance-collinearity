# -*- coding: utf-8 -*-
"""Paired feature distance, and why FID is not applicable to this panel.

Two findings, both from this file's own output, decide the design.

1. FID/KID do not work here.  On a 24-prompt panel the images differ far more
   across prompts than across conditions, so a distribution distance measures
   prompt heterogeneity.  The proof is the split-half baseline: two disjoint
   halves of ONE condition are 35.5 apart, while every between-condition
   distance is 18.2--18.9.  A metric whose own sampling noise is twice the
   effect cannot order conditions.  (dist_metrics.py keeps that measurement.)

2. What does work is the paired distance in a semantic feature space, because it
   shares the paired design of the rest of the paper: for the same prompt and
   seed, how far is the (w,s) image from its w=1 reference, in Inception or CLIP
   features?  The paper already reports this in pixel L2 and CLIP; Inception adds
   a third space that neither the paper nor a referee can accuse of being chosen
   after the fact.

Reading.  If compensation helps only in pixels, the pixel gain (6.4% at the
median) should shrink toward zero in both semantic spaces.  That is the honest
outcome to report: it bounds the claim to "closer to the reference image", not
"semantically more faithful".

Run:
    python exp/p2/paired_feat.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dist_metrics import InceptionFeat, TF_CLIP  # noqa: E402
from PIL import Image  # noqa: E402

PANEL = r"E:\研究\exp\p2\out_dist\images.csv"
OUT = r"E:\研究\results\论文二_LoRA共线\DIST_2026-11-14"
REF = "A_w1.0_s7.5"
PAIRS = [("B1_w1.5_s7.5", "C1_w1.5_s6.0", 1.5), ("B2_w2.0_s7.5", "C2_w2.0_s5.0", 2.0)]
BOOT = 4000
SEED = 0


@torch.no_grad()
def clip_feats(paths, model, dev, bs=16):
    fs = []
    for i in range(0, len(paths), bs):
        ims = torch.stack([TF_CLIP(Image.open(p).convert("RGB"))
                           for p in paths[i:i + bs]])
        fs.append(model(pixel_values=ims.to(dev)).pooler_output.float().cpu().numpy())
    return np.concatenate(fs, 0)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(PANEL)
    d["key"] = d.prompt_id + "_" + d.seed.astype(str)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import CLIPModel
    inc = InceptionFeat().to(dev).eval()
    clip = CLIPModel.from_pretrained(r"E:\研究\models\clip_l14",
                                     torch_dtype=torch.float32,
                                     local_files_only=True).to(dev).eval()
    clip = clip.vision_model

    tags = [REF] + [t for p in PAIRS for t in p[:2]]
    F = {}
    for t in tags:
        sub = d[d.tag == t]
        if sub.empty:
            print(f"  {t}: not generated yet, skipped", flush=True)
            continue
        paths = sub.img_path.tolist()
        F[(t, "inception")] = inc_extract(paths, inc, dev)
        F[(t, "clip")] = clip_feats(paths, clip, dev)
        print(f"  {t}: {len(paths)}", flush=True)

    # pixel distances to the w=1 reference, same pairing, streamed
    ref = d[d.tag == REF].reset_index(drop=True)
    P = {}
    for t in tags:
        sub = d[d.tag == t].reset_index(drop=True)
        if sub.empty:
            continue
        P[(t, "pixel-512")] = pixel_dists(sub, ref, full=True)
        P[(t, "pixel-64")] = pixel_dists(sub, ref, full=False)
        print(f"  {t}: pixel done", flush=True)

    rows = []
    for fname in ("pixel-512", "pixel-64", "inception", "clip"):
        ref = d[d.tag == REF].set_index("key")
        for tb, tc, w in PAIRS:
            if (tb, fname) not in F and (tb, fname) not in P:
                print(f"  [{fname}] w={w}: incomplete, skipped", flush=True)
                continue
            gb = d[d.tag == tb].set_index("key")
            gc = d[d.tag == tc].set_index("key")
            keys = sorted(set(ref.index) & set(gb.index) & set(gc.index))
            if not keys:
                continue
            pos = {t: {k: i for i, k in enumerate(d[d.tag == t].key)}
                   for t in (REF, tb, tc)}
            if fname.startswith("pixel"):
                db = P[(tb, fname)][[pos[tb][k] for k in keys]]
                dc = P[(tc, fname)][[pos[tc][k] for k in keys]]
            else:
                fb = F[(tb, fname)][[pos[tb][k] for k in keys]]
                fc = F[(tc, fname)][[pos[tc][k] for k in keys]]
                fr = F[(REF, fname)][[pos[REF][k] for k in keys]]
                db = np.linalg.norm(fb - fr, axis=1)
                dc = np.linalg.norm(fc - fr, axis=1)
            delta = dc - db
            rng = np.random.default_rng(SEED)
            bs = np.array([np.median(delta[rng.integers(0, len(delta), len(delta))])
                           for _ in range(BOOT)])
            lo, hi = np.quantile(bs, [.025, .975])
            rows.append(dict(
                feature=fname, w=w, n=len(keys),
                d_uncomp_med=float(np.median(db)),
                d_comp_med=float(np.median(dc)),
                rel_change=float(np.median(dc) / np.median(db) - 1),
                delta_med=float(np.median(delta)),
                delta_lo=float(lo), delta_hi=float(hi),
                frac_improved=float((delta < 0).mean())))
            r = rows[-1]
            print(f"  [{fname}] w={w}: d_uncomp {r['d_uncomp_med']:.2f} -> "
                  f"d_comp {r['d_comp_med']:.2f} ({100 * r['rel_change']:+.1f}%), "
                  f"paired delta {r['delta_med']:+.3f} "
                  f"[{r['delta_lo']:+.3f},{r['delta_hi']:+.3f}], "
                  f"improved {100 * r['frac_improved']:.0f}%", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "paired_feat.csv"), index=False,
               float_format="%.4f")
    print(out.to_string(index=False))


def inc_extract(paths, model, dev, bs=16):
    from dist_metrics import extract
    return extract(paths, model, dev, bs)


def thumb(a, n=64):
    """Same definition as y_gen.Runner.thumb: the pixel-64 metric."""
    return np.asarray(Image.fromarray(a).resize((n, n)), np.float32).reshape(-1) / 255.0


def pixel_dists(df, ref_df, full=True):
    """Distance of each image to its w=1 reference, same (prompt, seed).

    Streamed one image at a time: holding 1440 512x512 float32 arrays is 4.5 GB.
    full=True gives pixel-512 (flattened L2 over 255-scaled pixels, the paper's
    primary distance); full=False gives the 64x64 thumbnail variant.
    """
    rp = dict(zip(ref_df.key, ref_df.img_path))
    out = []
    for _, row in df.iterrows():
        a = np.asarray(Image.open(row.img_path).convert("RGB"), np.float32)
        b = np.asarray(Image.open(rp[row.key]).convert("RGB"), np.float32)
        if full:
            d = float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)) / 255.0)
        else:
            d = float(np.linalg.norm(thumb(a.astype(np.uint8)) - thumb(b.astype(np.uint8))))
        out.append(d)
    return np.array(out)


if __name__ == "__main__":
    main()
