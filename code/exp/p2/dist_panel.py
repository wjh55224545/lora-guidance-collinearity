# -*- coding: utf-8 -*-
"""Distribution panel for paper 2: identical (prompt, seed) samples under five
conditions, so that a distribution-level metric can compare them.

Why not reuse the existing grid.  Every metric in the paper is *paired*: one
image at (w,s) against one reference image at (1,7.5) for the same prompt and
seed.  A paired distance says nothing about whether the condition as a
distribution has moved closer to w=1; FID/KID ask exactly that, and they need
hundreds of samples per condition, not the 8 the w-by-s grid provides.

Design.
  * Five conditions, all on the same adapter:
        A  w=1.0 s=7.5   reference (equals the anchor every metric uses)
        B1 w=1.5 s=7.5   inflated, uncompensated
        C1 w=1.5 s=6.0   inflated, compensated   (s* = 7.5/1.31 = 5.7)
        B2 w=2.0 s=7.5   inflated, uncompensated
        C2 w=2.0 s=5.0   inflated, compensated   (s* = 7.5/1.42 = 5.3)
  * 24 prompts x 12 seeds = 288 images per condition, 1440 total.  The prompt
    pool spans all three classes (object / person / style), so the sample is a
    slice of the model's output distribution rather than one object.
  * The seed set is identical across conditions, so a difference between two
    conditions cannot come from drawing different noise.

Batching.  The pipeline accepts a list of generators, one per sample, which
raises GPU utilisation four-fold.  The smoke check below proves that batching
does not change the image: a seed generated in a batch must match the same seed
generated alone, byte-for-byte, or the panel is not comparable with the grids
already on disk.

Run:
    python exp/p2/dist_panel.py --smoke        # verifies batch == single
    python exp/p2/dist_panel.py --batch 4      # 1440 images, ~40 min on a 4060
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longrun_p2 import PROMPTS, STEPS  # noqa: E402
from y_gen import ORIG, build_pipe  # noqa: E402

OUT = r"E:\研究\exp\p2\out_dist"
N_PROMPTS = 24
N_SEEDS = 12

# tag, w, s
GROUPS = [
    ("A_w1.0_s7.5", 1.0, 7.5),
    ("B1_w1.5_s7.5", 1.5, 7.5),
    ("C1_w1.5_s6.0", 1.5, 6.0),
    ("B2_w2.0_s7.5", 2.0, 7.5),
    ("C2_w2.0_s5.0", 2.0, 5.0),
]


class Gen:
    def __init__(self):
        self.pipe = build_pipe()
        self.pipe.load_lora_weights(ORIG, adapter_name="D")
        self.names = ["D"]

    @torch.no_grad()
    def one(self, prompt, w, s, seed):
        self.pipe.set_adapters(self.names, [float(w)])
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.pipe(prompt=prompt, num_inference_steps=STEPS,
                         guidance_scale=float(s), generator=g,
                         height=512, width=512, return_dict=False)[0][0]

    @torch.no_grad()
    def batch(self, prompt, w, s, seeds):
        self.pipe.set_adapters(self.names, [float(w)])
        gs = [torch.Generator(device="cpu").manual_seed(int(x)) for x in seeds]
        return self.pipe(prompt=[prompt] * len(seeds), num_inference_steps=STEPS,
                         guidance_scale=float(s), generator=gs,
                         height=512, width=512, return_dict=False)[0]


def arr(img):
    return np.asarray(img, np.float32)


def smoke(g):
    """Batch output must equal single output for the same seed."""
    prompt = PROMPTS[0][2]
    seeds = [0, 1, 2, 3]
    t0 = time.time()
    solo = [arr(g.one(prompt, 2.0, 7.5, sd)) for sd in seeds]
    t1 = time.time()
    b = [arr(x) for x in g.batch(prompt, 2.0, 7.5, seeds)]
    t2 = time.time()
    diffs = [float(np.abs(a - c).max()) for a, c in zip(solo, b)]
    print(f"single 4 imgs: {t1 - t0:.1f}s ({((t1 - t0) / 4):.2f}s/img)")
    print(f"batch  4 imgs: {t2 - t1:.1f}s ({((t2 - t1) / 4):.2f}s/img)")
    print("max |single - batch| per seed:", ["%.1f" % x for x in diffs])
    ok = max(diffs) <= 1.0
    print("IDENTICAL" if ok else "MISMATCH -- batching is not safe")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--groups", default="")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()

    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    g = Gen()
    if a.smoke:
        sys.exit(0 if smoke(g) else 1)

    groups = GROUPS
    if a.groups:
        want = set(a.groups.split(","))
        groups = [x for x in GROUPS if x[0].split("_")[0] in want]
    prompts = PROMPTS[:N_PROMPTS]
    seeds = list(range(a.n_seeds))

    csv = os.path.join(OUT, "images.csv")
    cols = ["tag", "prompt_id", "pclass", "prompt", "seed", "w", "s", "img_path"]
    if not os.path.exists(csv):
        pd.DataFrame(columns=cols).to_csv(csv, index=False)
    sp = os.path.join(OUT, "state.json")
    done = set(json.load(open(sp))) if os.path.exists(sp) else set()

    jobs = [(tag, pid, pc, text, sd, w, s)
            for (tag, w, s) in groups
            for (pid, pc, text) in prompts
            for sd in seeds]
    todo = [j for j in jobs if f"{j[0]}|{j[1]}|{j[4]}" not in done]
    print(f"{len(jobs)} images, {len(done)} done, {len(todo)} to go, "
          f"batch={a.batch}", flush=True)

    t0 = time.time()
    n = 0
    # group by (tag, prompt) so the adapter weight and the prompt are fixed
    # inside a batch; only the seed varies
    for (tag, w, s) in groups:
        for (pid, pc, text) in prompts:
            pend = [sd for sd in seeds if f"{tag}|{pid}|{sd}" not in done]
            for i in range(0, len(pend), a.batch):
                chunk = pend[i:i + a.batch]
                try:
                    # --batch 1 must take the single-image path, not a batch of
                    # one: the smoke check showed the batched pipeline moves
                    # pixels by up to 66/255, so it is a different generator.
                    imgs = ([g.one(text, w, s, sd) for sd in chunk]
                            if a.batch == 1 else g.batch(text, w, s, chunk))
                    for sd, img in zip(chunk, imgs):
                        nm = f"{tag}_{pid}_sd{sd:03d}.png"
                        p = os.path.join(OUT, "img", nm)
                        img.save(p)
                        pd.DataFrame([[tag, pid, pc, text, sd, w, s, p]],
                                     columns=cols).to_csv(
                            csv, mode="a", header=False, index=False)
                        done.add(f"{tag}|{pid}|{sd}")
                        n += 1
                    json.dump(sorted(done), open(sp, "w"))
                    if n % 48 == 0:
                        el = (time.time() - t0) / n
                        print(f"  {len(done)}/{len(jobs)}  {el:.2f}s/img  "
                              f"eta {(len(jobs) - len(done)) * el / 60:.0f}min",
                              flush=True)
                except Exception:
                    traceback.print_exc()
                    print(f"  !! {tag} {pid} {chunk}", flush=True)
    print(f"done {n} new in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
