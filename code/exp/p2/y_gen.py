"""Paper 2, phase Y: the (w, s) grid on LoRAs that are *not* slices of the
original `lora_detail_sd15`.

Three consumers share this runner:

  Y2   dense s grid.  The published kappa rests on argmin over 9 s values, so
       log s* is quantised to ~0.18 and the movement w produces is the same
       size.  Y2 re-measures the SAME range at 0.5 instead of 1.5 spacing,
       which separates "kappa is small" from "the grid cannot see kappa".
       Pre-committed rule: if kappa jumps above 0.5, the interval [0.17,0.27]
       is dead and everything gets re-reported.

  Y1   two independently trained LoRAs (see y1_train.py).  Same grid as L2, so
       kappa is directly comparable to the 0.19-0.27 already published.

  Y3   two LoRAs at once.  Here the deliverable is not kappa but *images*:
       with every 512x512 png on disk, the pairwise distance matrix that red
       line 7z demands becomes computable after the fact.  The single-adapter
       anchors (1,0) and (0,1) are generated in the same run so the composed
       grid has a yardstick in the same coordinate system.

Design decisions that are easy to get wrong
  * the reference image for pix_l2 is generated with the *same adapter
    configuration* as the grid, otherwise it carries a different LoRA and every
    distance is off by a constant.  For two adapters the reference is
    (W0, W0, S0).
  * pix_l2 is identically 0 at (W0, S0) -- by construction -- so pix_l2_base
    (distance to the w=0 image, identical across configs) is written alongside
    and is the one that is comparable *across* adapters.
  * every image is saved.  L2 threw 8712 images away and lost the ability to
    answer "does output depend only on a combined quantity" for good.
  * one CSV row per image, appended, plus state.json -> resumable.

Run:
    python y_gen.py --mode single --tag y2_full --lora PATH --w 0.2,... --s 3,3.5,...
    python y_gen.py --mode two --tag y3 --lora A=pathA --lora B=pathB
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch

from longrun_p2 import PROMPTS, STEPS, W0, S0

HERE = os.path.dirname(os.path.abspath(__file__))
SD15 = r"E:\研究\models\sd15.safetensors"
CFG = r"E:\研究\models\sd15_cfg"
CLIP_DIR = r"E:\研究\models\clip_l14"
ORIG = r"E:\研究\models\lora\lora_detail_sd15.safetensors"
TRAINED_DIR = r"E:\研究\models\lora\trained"
OUTROOT = os.path.join(HERE, "out_y")

W_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
S_GRID = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]
# 0.5 spacing over the same 3..14 range: 23 points, so the argmin is quantised
# in log space at ~0.069 instead of ~0.18.  Same endpoints means the dense run
# is a strict refinement of the coarse one, not a different experiment.
S_DENSE = [round(3.0 + 0.5 * i, 1) for i in range(23)]

COLS = ["tag", "mode", "prompt_id", "pclass", "prompt", "seed", "w", "w2", "s",
        "clip_t", "clip_i_ref", "clip_i_base", "pix_l2", "pix_l2_base",
        "img_path", "gen_s", "lora"]


def resolve(name):
    if name in ("full", "orig"):
        return ORIG
    if os.path.isabs(name) or os.path.sep in name:
        return name
    for cand in (f"{name}.safetensors", f"lora_{name}.safetensors"):
        p = os.path.join(TRAINED_DIR, cand)
        if os.path.exists(p):
            return p
    raise SystemExit(f"no such lora: {name} (looked for {name}.safetensors / "
                     f"lora_{name}.safetensors under {TRAINED_DIR})")


def build_pipe():
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_single_file(
        SD15, config=CFG, torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


class Runner:
    """N adapters, one pipe.  Adapter i is applied with weight w_i.

    The guidance *operator* is a knob too, and it is the one every reviewer of
    an image-generation paper asks about: `guidance_scale` is only meaningful
    inside a particular CFG variant.  Two alternatives are supported, both
    applied to the anchors as well so that every comparison stays internal:

      rescale > 0   variance-normalised CFG ("guidance rescale"): the guided
                    prediction's std is rescaled back to the conditional one.
                    This removes the *magnitude* channel by construction.
      neg != ""     a negative prompt, so the unconditional branch is not the
                    empty string.
    """

    def __init__(self, adapters, rescale=0.0, neg=""):
        from transformers import CLIPModel, CLIPProcessor
        self.names = [n for n, _ in adapters]
        self.pipe = build_pipe()
        for n, p in adapters:
            self.pipe.load_lora_weights(p, adapter_name=n)
        self.clip = CLIPModel.from_pretrained(
            CLIP_DIR, torch_dtype=torch.float16, local_files_only=True,
            use_safetensors=True).to("cuda").eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
        self.cache = {}
        self.rescale = float(rescale)
        self.neg = neg or ""

    @torch.no_grad()
    def gen(self, prompt, ws, s, seed):
        self.pipe.set_adapters(self.names, [float(x) for x in ws])
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        kw = {}
        if self.rescale > 0:
            kw["guidance_rescale"] = self.rescale
        if self.neg:
            kw["negative_prompt"] = self.neg
        return self.pipe(prompt=prompt, num_inference_steps=STEPS,
                         guidance_scale=float(s), generator=g,
                         height=512, width=512, return_dict=False, **kw)[0][0]

    @torch.no_grad()
    def score(self, img, text):
        import torch.nn.functional as F
        ti = self.proc(text=[text], return_tensors="pt", padding=True,
                       truncation=True).to("cuda")
        tf = F.normalize(self.clip.text_projection(
            self.clip.text_model(**ti).pooler_output).float(), dim=-1)[0]
        ii = self.proc(images=[img], return_tensors="pt").to("cuda")
        imf = F.normalize(self.clip.visual_projection(
            self.clip.vision_model(**ii).pooler_output).float(), dim=-1)[0]
        return tf, imf

    @staticmethod
    def thumb(img, n=64):
        return np.asarray(img.resize((n, n)), np.float32).reshape(-1) / 255.0

    def anchors(self, text, seed, zero_ws, ref_ws):
        """(w=0 image, reference image at ref_ws + S0), per prompt and seed.

        The cache key MUST include the prompt: keying on seed alone is the bug
        that silently compared every prompt against the first prompt's image in
        the E2c-2 draft (found 2026-11-07 10:30, 316 rows discarded).
        """
        key = (text, seed)
        if key in self.cache:
            return self.cache[key]
        b = self.gen(text, zero_ws, S0, seed)
        r = self.gen(text, ref_ws, S0, seed)
        _, fb = self.score(b, text)
        _, fr = self.score(r, text)
        self.cache[key] = ((b, fb, self.thumb(b)), (r, fr, self.thumb(r)))
        return self.cache[key]


def _mk(**kw):
    row = {c: np.nan for c in COLS}
    row.update(kw)
    return row


def run(mode, tag, adapters, wg, w2g, sg, prompts, seeds, ref_ws, outdir,
        rescale=0.0, neg=""):
    imgdir = os.path.join(outdir, "img")
    os.makedirs(imgdir, exist_ok=True)
    csv = os.path.join(outdir, "images.csv")
    state_p = os.path.join(outdir, "state.json")
    done = set(json.load(open(state_p))) if os.path.exists(state_p) else set()
    if not os.path.exists(csv):
        pd.DataFrame(columns=COLS).to_csv(csv, index=False)

    zero_ws = [0.0] * len(adapters)
    R = Runner(adapters, rescale=rescale, neg=neg)
    if mode == "single":
        jobs = [((pid, pc, text), sd, w, np.nan, s)
                for (pid, pc, text) in prompts for sd in seeds
                for w in wg for s in sg]
    else:
        jobs = [((pid, pc, text), sd, w1, w2, s)
                for (pid, pc, text) in prompts for sd in seeds
                for w1 in wg for w2 in w2g for s in sg]
    print(f"[{tag}] {len(jobs)} images, {len(done)} done, "
          f"{len(adapters)} adapter(s)", flush=True)

    t0, n_new = time.time(), 0
    for (pid, pc, text), sd, w, w2, s in jobs:
        key = f"{pid}|{sd}|{w}|{w2}|{s}"
        if key in done:
            continue
        try:
            (b, fb, tb), (r, fr, tr) = R.anchors(text, sd, zero_ws, ref_ws)
            ws = [w] if mode == "single" else [w, w2]
            ta = time.time()
            img = R.gen(text, ws, s, sd)
            gen_s = time.time() - ta
            ft, fi = R.score(img, text)
            nm = f"{pid}_sd{sd}_w{w:g}" + (f"_w2{w2:g}" if mode == "two" else "") \
                 + f"_s{s:g}.png"
            ipath = os.path.join(imgdir, nm)
            img.save(ipath)
            row = _mk(tag=tag, mode=mode, prompt_id=pid, pclass=pc, prompt=text,
                      seed=sd, w=w, w2=w2, s=s,
                      clip_t=float(ft @ fi), clip_i_ref=float(fi @ fr),
                      clip_i_base=float(fi @ fb),
                      pix_l2=float(np.linalg.norm(R.thumb(img) - tr)),
                      pix_l2_base=float(np.linalg.norm(R.thumb(img) - tb)),
                      img_path=ipath, gen_s=round(gen_s, 3),
                      lora="+".join(f"{n}@{p}" for n, p in adapters))
            pd.DataFrame([row], columns=COLS).to_csv(
                csv, mode="a", header=False, index=False)
            done.add(key)
            json.dump(sorted(done), open(state_p, "w"))
            n_new += 1
            if n_new % 25 == 0:
                el = (time.time() - t0) / n_new
                left = (len(jobs) - len(done)) * el / 60
                print(f"  {len(done)}/{len(jobs)}  {el:.2f}s/img  "
                      f"eta {left:.0f}min", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"  !! {key}", flush=True)
    print(f"[{tag}] done {n_new} new in {(time.time() - t0) / 60:.1f} min",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "two"], default="single")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora", action="append", default=[],
                    help="single: NAME|PATH ; two: NAME|PATH (twice)")
    ap.add_argument("--w", type=str, default="")
    ap.add_argument("--w2", type=str, default="")
    ap.add_argument("--s", type=str, default="")
    ap.add_argument("--dense-s", action="store_true")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--seeds", type=str, default="11,22")
    ap.add_argument("--ref-w", type=float, default=-1.0)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rescale", type=float, default=0.0,
                    help="variance-normalising CFG rescale (0 = plain CFG)")
    ap.add_argument("--neg", type=str, default="",
                    help="negative prompt for the unconditional branch")
    a = ap.parse_args()

    specs = [s.split("=", 1) for s in a.lora] or [["full", "full"]]
    adapters = [(n, resolve(p)) for n, p in specs]
    if a.mode == "two" and len(adapters) != 2:
        raise SystemExit("two-adapter mode needs exactly two --lora NAME=PATH")
    for n, p in adapters:
        if not os.path.exists(p):
            raise SystemExit(f"missing lora file: {p}")

    wg = [float(x) for x in a.w.split(",")] if a.w else W_GRID
    w2g = [float(x) for x in a.w2.split(",")] if a.w2 else [0.0]
    sg = [float(x) for x in a.s.split(",")] if a.s \
        else (S_DENSE if a.dense_s else S_GRID)
    seeds = [int(x) for x in a.seeds.split(",")]
    prompts = PROMPTS[:a.prompts]
    ref_ws = ([a.ref_w] if a.ref_w >= 0 else
              ([W0] if a.mode == "single" else [W0, W0]))
    if a.smoke:
        prompts, wg, sg, seeds = prompts[:1], [0.0, W0], [S0], seeds[:1]
        w2g = w2g[:1]

    outdir = a.outdir or os.path.join(OUTROOT + ("_smoke" if a.smoke else ""),
                                      a.tag)
    os.makedirs(outdir, exist_ok=True)
    run(a.mode, a.tag, adapters, wg, w2g, sg, prompts, seeds, ref_ws, outdir,
        rescale=a.rescale, neg=a.neg)


if __name__ == "__main__":
    main()
