# -*- coding: utf-8 -*-
"""FID / KID for paper 2's distribution panel.

The question.  Every metric in the paper pairs one image at (w,s) with one
reference image at (1,7.5) for the same prompt and seed.  A paired distance is
silent about whether a condition has moved closer to w=1 *as a distribution*.
FID and KID answer that, and they are the standard currency in generation venues.

What is compared.  Five conditions on the same adapter, 288 samples each over 24
prompts x 12 seeds (see dist_panel.py):
    A  (1.0, 7.5) reference
    B1 (1.5, 7.5) inflated, uncompensated     C1 (1.5, 6.0) compensated
    B2 (2.0, 7.5) inflated, uncompensated     C2 (2.0, 5.0) compensated
The claim under test is dist(A, C) < dist(A, B): compensation moves the whole
distribution toward the unexpanded one, not just the median image.

Two feature spaces, deliberately.  Inception pool3 is what FID means; CLIP's
image tower is reported alongside because a reviewer may object that FID's
ImageNet features are not the right notion of identity for LoRA edits.  If both
spaces agree the conclusion does not rest on the feature choice.

Statistics.  FID is a biased estimator and 288 samples is small, so every FID is
reported with a half-split spread (R random splits, distance each time, std
across splits).  KID is unbiased; its per-split spread is reported the same way.
No p-value is claimed: with 288 samples the honest statement is a gap that is
large against its own split-to-split noise.

Run:
    python exp/p2/dist_metrics.py
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import inception_v3

PANEL = r"E:\研究\exp\p2\out_dist\images.csv"
OUT = r"E:\研究\results\论文二_LoRA共线\DIST_2026-11-14"
IMG = 288
PCA_DIM = 64
R_SPLIT = 40
SEED = 0

PAIRS = [("A_w1.0_s7.5", "B1_w1.5_s7.5"), ("A_w1.0_s7.5", "C1_w1.5_s6.0"),
         ("A_w1.0_s7.5", "B2_w2.0_s7.5"), ("A_w1.0_s7.5", "C2_w2.0_s5.0"),
         ("B2_w2.0_s7.5", "C2_w2.0_s5.0")]

# FID's own preprocessing: 299px, scaled to [-1, 1] (pytorch-fid convention).
TF_INC = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299),
                             transforms.ToTensor()])
# CLIP's own preprocessing.  Handing the image tower 299px in [-1,1] would still
# return features, but they would not be CLIP features -- the patch grid and the
# channel statistics both differ, and the comparison would be meaningless.
TF_CLIP = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])])


class InceptionFeat(nn.Module):
    """Inception-v3 pool3 (2048-d), the feature FID is defined on."""

    def __init__(self):
        super().__init__()
        m = inception_v3(weights="IMAGENET1K_V1", aux_logits=True,
                         transform_input=False, init_weights=False)
        m.fc = nn.Identity()
        m.AuxLogits = None
        self.m = m
        self.buf = None
        m.avgpool.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.buf = torch.flatten(out, 1)

    @torch.no_grad()
    def forward(self, x):
        self.buf = None
        # TF_INC leaves the image in [0,1]; FID's Inception expects [-1,1]
        _ = self.m((x - 0.5) / 0.5)
        return self.buf


@torch.no_grad()
def extract(paths, model, device, bs=16, tf=None):
    tf = tf or TF_INC
    feats = []
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        ims = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk])
        feats.append(model(ims.to(device)).float().cpu().numpy())
    return np.concatenate(feats, 0)


def frechet(x, y):
    mu1, mu2 = x.mean(0), y.mean(0)
    s1 = np.cov(x, rowvar=False)
    s2 = np.cov(y, rowvar=False)
    cs = sqrtm(s1 @ s2)
    if np.iscomplexobj(cs):
        cs = cs.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


def pca_project(feats, k):
    """Project every condition into one shared k-d subspace.

    Why.  288 samples give a covariance of rank at most 287, so a 2048-d FID is
    singular and sqrtm returns garbage (complex or NaN).  Fitting one subspace on
    the pooled features of ALL conditions keeps the projection identical across
    pairs, so a difference between two FIDs is still attributable to the
    conditions and not to different projections.
    """
    allx = np.concatenate(list(feats.values()), 0)
    mu = allx.mean(0)
    _, _, vt = np.linalg.svd(allx - mu, full_matrices=False)
    p = vt[:k].T
    return {t: (v - mu) @ p for t, v in feats.items()}


def kid(x, y):
    """Unbiased MMD^2 with the cubic polynomial kernel used by KID."""
    d = x.shape[1]

    def k(a, b):
        return (a @ b.T / d + 1.0) ** 3

    m, n = len(x), len(y)
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    sx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    sy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(sx + sy - 2 * kxy.mean())


def half_split(x, y, fn, rng, reps=R_SPLIT):
    """Split both sets in half, score, repeat: spread from sampling alone."""
    vals = []
    hx, hy = len(x) // 2, len(y) // 2
    for _ in range(reps):
        ix = rng.permutation(len(x))[:hx]
        iy = rng.permutation(len(y))[:hy]
        vals.append(fn(x[ix], y[iy]))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(PANEL)
    tags = d.tag.unique().tolist()
    print(f"{len(d)} images, {len(tags)} conditions: {tags}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    models["inception"] = InceptionFeat().to(dev).eval()
    # CLIP image tower, same weights the project already validated
    from transformers import CLIPModel
    clip = CLIPModel.from_pretrained(r"E:\研究\models\clip_l14",
                                     torch_dtype=torch.float32,
                                     local_files_only=True).to(dev).eval()
    models["clip"] = clip.vision_model

    feats = {}
    for tag in tags:
        paths = d[d.tag == tag].img_path.tolist()
        for fname, model in models.items():
            if fname == "clip":
                with torch.no_grad():
                    fs = []
                    for i in range(0, len(paths), 16):
                        ims = torch.stack([TF_CLIP(Image.open(p).convert("RGB"))
                                           for p in paths[i:i + 16]])
                        out = model(pixel_values=ims.to(dev)).pooler_output
                        fs.append(out.float().cpu().numpy())
                    feats[(tag, fname)] = np.concatenate(fs, 0)
            else:
                feats[(tag, fname)] = extract(paths, model, dev)
        print(f"  {tag}: {len(paths)} feats", flush=True)

    rng = np.random.default_rng(SEED)
    floors = {}
    rows = []
    for fname in ("inception", "clip"):
        # FID in a shared low-d subspace; KID keeps the full space (a kernel does
        # not need an invertible covariance).
        low = pca_project({t: feats[(t, fname)] for t in tags}, PCA_DIM)
        # The decisive control: two disjoint halves of ONE condition.  This is
        # what the metric reports when there is no condition difference at all,
        # so any between-condition gap smaller than it is not interpretable.
        xa = low["A_w1.0_s7.5"]
        f0_mu, f0_sd = half_split(xa, xa, frechet, rng)
        floors[fname] = (f0_mu, f0_sd)
        print(f"  [{fname}] split-half baseline FID-{PCA_DIM} "
              f"{f0_mu:.1f}+-{f0_sd:.1f}", flush=True)
        for a, b in PAIRS:
            xl, yl = low[a], low[b]
            f_mu, f_sd = half_split(xl, yl, frechet, rng)
            x, y = feats[(a, fname)], feats[(b, fname)]
            k_mu, k_sd = half_split(x, y, kid, rng)
            # noise floor: KID of a condition against itself on the same splits
            k0_mu, k0_sd = half_split(x, x, kid, rng)
            rows.append(dict(feature=fname, a=a, b=b, n=len(x),
                             fid=f_mu, fid_sd=f_sd, fid_floor=f0_mu,
                             kid=k_mu * 1000, kid_sd=k_sd * 1000,
                             kid_floor=k0_mu * 1000, kid_floor_sd=k0_sd * 1000))
            print(f"  [{fname}] {a} vs {b}: FID-{PCA_DIM} {f_mu:.1f}+-{f_sd:.1f}  "
                  f"KID {k_mu * 1000:.2f}+-{k_sd * 1000:.2f} "
                  f"(floor {k0_mu * 1000:.2f})", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "dist_metrics.csv"), index=False,
             float_format="%.4f")

    np.savez(os.path.join(OUT, "features.npz"),
             **{f"{t}|{f}": v for (t, f), v in feats.items()})

    lines = ["== distribution distance to the reference condition ==",
             f"n={IMG} per condition, {R_SPLIT} half-splits",
             f"FID in a shared {PCA_DIM}-d PCA subspace (288 samples cannot "
             "support a 2048-d covariance)", ""]
    for fname in ("inception", "clip"):
        sub = r[r.feature == fname]
        f0, f0sd = floors[fname]
        lines.append(f"-- {fname} features --")
        lines.append(f"split-half baseline of one condition (no difference at all): "
                     f"FID {f0:.1f}+-{f0sd:.1f}")
        for row in sub.itertuples():
            verdict = ("below baseline -> not interpretable"
                       if row.fid < f0 else "above baseline")
            lines.append(f"{row.a:16s} vs {row.b:16s}  FID{PCA_DIM} {row.fid:7.1f} "
                         f"+-{row.fid_sd:5.1f} [{verdict}]   "
                         f"KIDx1e3 {row.kid:6.2f} +-{row.kid_sd:4.2f} "
                         f"(floor {row.kid_floor:5.2f})")
        for w in ("1.5", "2.0"):
            g = sub[sub.a.str.startswith("A")]
            bu = g[g.b.str.startswith("B" + w[0])]
            co = g[g.b.str.startswith("C" + w[0])]
            if len(bu) and len(co):
                lines.append(
                    f"   w={w}: FID {bu.fid.iloc[0]:.1f} (uncomp) -> "
                    f"{co.fid.iloc[0]:.1f} (comp), "
                    f"ratio {co.fid.iloc[0] / bu.fid.iloc[0]:.2f}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "dist_readout.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
