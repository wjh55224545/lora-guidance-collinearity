"""L2 (paper 2): ~10 h long run around I1 (per-step collinearity) and I2 (the
g_eff invariant).

E3a established, in latent space only, that (a) equal w*s is the wrong rule and
(b) the geff correction buys 40% over leaving s alone.  The missing evidence is
on images: does an equal-g_eff (w,s) pair actually look the same, and does it
hold across prompts and seeds?  That is what this runner produces.

Segments
  R  reference          w=0 and (w0,s0) per (prompt, seed)  -> the two anchors
  G  image grid         prompt x seed x w x s                -> is the g_eff level
                                                                 set a curve or a point
  T  route comparison   3 routes at matched g_eff            -> fixed_s / geff_const / ws_equal

Cost model measured here: 512x512, 20 steps, SD1.5 fp16 on a 4060 = 2.4 s/image.
So 10 h is about 15 000 images, and the grid is sized to that rather than to a
round number of prompts.

Metrics (all cheap, one CLIP forward per image):
  clip_t      cos(text, image)          -- absolute prompt alignment, not a
                                          self-comparison (the anchors share the
                                          same text, so cos(t,t) would be 1.0)
  clip_i_ref  cos(image, the (w0,s0) image)  -- is this configuration equivalent
                                          (analyse as 1-cos: it is the only
                                          scale with resolution, cos itself
                                          sits in 0.94-1.00)
  clip_i_base cos(image, the w=0 image)      -- how far it drifted off the base
  pix_l2      L2 on 64x64 thumbnails    -- a CLIP-free cross-check; on the smoke
                                          run this is the one that separates
                                          the routes (0.0 to 22.0) while
                                          clip_i_ref moves only .94-.999

Resumable: one CSV row per image, appended as each job finishes; finished keys
live in out_long2/state.json.

Run:
    python longrun_p2.py --target-hours 9.9
    python longrun_p2.py --max-jobs 1 --seeds 11 --prompts 2   # smoke
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SD15 = r"E:\研究\models\sd15.safetensors"
CFG = r"E:\研究\models\sd15_cfg"
LORA = r"E:\研究\models\lora\lora_detail_sd15.safetensors"
CLIP_DIR = r"E:\研究\models\clip_l14"
ADAPTER = "d"
OUT = os.path.join(HERE, "out_long2")

W0, S0 = 1.0, 7.5
STEPS = 20
W_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
S_GRID = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]
SEEDS = [11, 22, 33, 44]

# rho_t medians measured on this LoRA in E2c (240 rows, t over the whole
# trajectory).  These drive the geff_const route, so the route is a function of a
# measured quantity rather than of a fitted constant.
RHO = {"object": 0.208, "person": 0.295, "style": 0.200}
RHO_DEFAULT = 0.234

_RAW = [
    ("object", "a red ceramic teapot on a wooden table"),
    ("object", "a blue vintage bicycle leaning on a brick wall"),
    ("object", "a glass of iced coffee with condensation, macro shot"),
    ("object", "an antique brass pocket watch on dark velvet"),
    ("object", "a stack of old hardcover books, warm lamp light"),
    ("object", "a pair of leather boots standing in shallow water"),
    ("object", "a white porcelain vase with dried flowers"),
    ("object", "a silver acoustic guitar against a grey wall"),
    ("person", "a portrait photo of sks person, close up, natural light"),
    ("person", "a portrait photo of sks person, side profile, studio light"),
    ("person", "a portrait photo of sks person, outdoors, golden hour"),
    ("person", "a portrait photo of sks person, black and white, film grain"),
    ("person", "a portrait photo of sks person, wearing a wool coat"),
    ("person", "a portrait photo of sks person, low angle, overcast sky"),
    ("person", "a portrait photo of sks person, candid, cafe interior"),
    ("person", "a portrait photo of sks person, high key, soft shadows"),
    ("style", "an oil painting of a village at dusk, thick brush strokes"),
    ("style", "an oil painting of a stormy sea, impasto texture"),
    ("style", "an oil painting of a quiet street after rain"),
    ("style", "an oil painting of autumn trees along a river"),
    ("style", "an oil painting of a still life with fruit and drapery"),
    ("style", "an oil painting of a mountain pass in morning fog"),
    ("style", "an oil painting of a harbour with moored boats"),
    ("style", "an oil painting of a field of sunflowers at noon"),
]
# (prompt_id, pclass, text) -- the id keeps rows unique even when two prompts
# share a prefix, and pclass selects the measured rho median for the geff route.
PROMPTS = [(f"{c[:3]}{i:02d}", c, t) for i, (c, t) in enumerate(_RAW)]

COLS = ["segment", "prompt_id", "pclass", "prompt", "seed", "w", "s", "route",
        "steps", "clip_t", "clip_i_ref", "clip_i_base", "pix_l2", "gen_s",
        "img_path"]


# ---------------------------------------------------------------- environment
def build_pipe():
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_single_file(
        SD15, config=CFG, torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(LORA, adapter_name=ADAPTER)
    return pipe


def build_clip():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_DIR, torch_dtype=torch.float16,
                                      local_files_only=True,
                                      use_safetensors=True).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
    return model, proc


# ------------------------------------------------------------------- pipeline
class Runner:
    def __init__(self):
        self.pipe = build_pipe()
        self.clip, self.proc = build_clip()
        self._anchor_cache = {}

    @torch.no_grad()
    def gen(self, prompt, w, s, seed, steps=STEPS):
        self.pipe.set_adapters([ADAPTER], [float(w)])
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        img = self.pipe(prompt=prompt, num_inference_steps=steps,
                        guidance_scale=float(s), generator=g,
                        height=512, width=512, return_dict=False)[0][0]
        return img

    @torch.no_grad()
    def score(self, imgs, texts):
        """One batched CLIP pass -> normalised text and image features.

        transformers 5.19's get_text_features / get_image_features return output
        objects instead of tensors, and asymmetrically so (the text pooler is
        pre-projection at 768 while the image pooler is post-projection at 768,
        so re-projecting it raises 768-vs-1024).  Going through the towers and
        applying each projection explicitly is the only verifiable form: a
        teapot prompt then scores .31 against its own image and .08/.14 against
        unrelated prompts, which is the expected contrast for ViT-L/14.
        """
        ti = self.proc(text=texts, return_tensors="pt", padding=True,
                       truncation=True).to("cuda")
        tf = F.normalize(self.clip.text_projection(
            self.clip.text_model(**ti).pooler_output).float(), dim=-1)
        ii = self.proc(images=imgs, return_tensors="pt").to("cuda")
        imf = F.normalize(self.clip.visual_projection(
            self.clip.vision_model(**ii).pooler_output).float(), dim=-1)
        return tf, imf

    @staticmethod
    def thumb_np(img, n=64):
        a = np.asarray(img.resize((n, n)), np.float32) / 255.0
        return a.reshape(-1)

    def anchors(self, pid, text, seed):
        """Base (w=0) and reference (w0,s0) images.  Cached: both are reused by
        every configuration of the same (prompt, seed)."""
        key = (pid, seed)
        if key in self._anchor_cache:
            return self._anchor_cache[key]
        t0 = time.time()
        b = self.gen(text, 0.0, S0, seed)
        r = self.gen(text, W0, S0, seed)
        tf, imf = self.score([b, r], [text, text])
        rec = dict(base=b, ref=r, t=tf[0], i_base=imf[0], i_ref=imf[1],
                   thumb_base=self.thumb_np(b), thumb_ref=self.thumb_np(r),
                   gen_s=2 * (time.time() - t0) / 2)
        self._anchor_cache[key] = rec
        return rec


def route_s(route, w, rho):
    """The s a route prescribes for a given w (constant routes only)."""
    if route == "fixed_s":
        return S0
    if route == "geff_const":
        return S0 * (1.0 + W0 * rho) / (1.0 + w * rho)
    if route == "ws_equal":
        return W0 * S0 / w
    raise ValueError(route)


# ----------------------------------------------------------------------- jobs
def job_grid(prompt, seed, imgdir):
    """Segment G: the full (w, s) grid for one (prompt, seed)."""
    pid, pclass, text = prompt
    R = Runner.instance
    a = R.anchors(pid, text, seed)
    rows = []
    for w in W_GRID:
        imgs, cfg = [], []
        for s in S_GRID:
            t0 = time.time()
            im = R.gen(text, w, s, seed)
            imgs.append(im)
            cfg.append((w, s, time.time() - t0))
        tf, imf = R.score(imgs, [text] * len(imgs))
        for im, (w2, s2, dt), t_i, i_i in zip(imgs, cfg, tf, imf):
            rows.append(dict(
                segment="G", prompt_id=pid, pclass=pclass, prompt=text,
                seed=seed, w=w2, s=s2, route="grid", steps=STEPS,
                clip_t=float(t_i @ i_i),
                clip_i_ref=float(i_i @ a["i_ref"]),
                clip_i_base=float(i_i @ a["i_base"]),
                pix_l2=float(np.linalg.norm(R.thumb_np(im) - a["thumb_ref"])),
                gen_s=dt, img_path=""))
        # keep one sample image per (prompt, seed, w) so figures stay possible
        # without writing 10k PNGs (that would be ~5 GB of disk for no gain).
        if w in (0.4, 1.0, 2.0):
            p = os.path.join(imgdir, f"G_{pclass}_{seed}_w{w}.jpg")
            imgs[4].save(p, quality=88)
    return rows, a


def job_route(prompt, seed, imgdir):
    """Segment T: three routes at matched g_eff, one w at a time."""
    pid, pclass, text = prompt
    R = Runner.instance
    a = R.anchors(pid, text, seed)
    rho = RHO.get(pclass, RHO_DEFAULT)
    rows = []
    for route in ("fixed_s", "geff_const", "ws_equal"):
        imgs, cfg = [], []
        for w in W_GRID:
            s = route_s(route, w, rho)
            t0 = time.time()
            im = R.gen(text, w, s, seed)
            imgs.append(im)
            cfg.append((w, s, time.time() - t0))
        tf, imf = R.score(imgs, [text] * len(imgs))
        for im, (w2, s2, dt), t_i, i_i in zip(imgs, cfg, tf, imf):
            rows.append(dict(
                segment="T", prompt_id=pid, pclass=pclass, prompt=text,
                seed=seed, w=w2, s=s2, route=route, steps=STEPS,
                clip_t=float(t_i @ i_i),
                clip_i_ref=float(i_i @ a["i_ref"]),
                clip_i_base=float(i_i @ a["i_base"]),
                pix_l2=float(np.linalg.norm(R.thumb_np(im) - a["thumb_ref"])),
                gen_s=dt, img_path=""))
        if route == "geff_const":
            p = os.path.join(imgdir, f"T_{pclass}_{seed}.jpg")
            imgs[4].save(p, quality=88)
    return rows


def jkey(j):
    return f"{j['segment']}|{j['prompt'][0]}|{j['prompt'][1][:24]}|{j['seed']}"


def build_jobs(prompts, seeds, segments):
    jobs = []
    for seed in seeds:
        for p in prompts:
            if "G" in segments:
                jobs.append(dict(segment="G", prompt=p, seed=seed))
            if "T" in segments:
                jobs.append(dict(segment="T", prompt=p, seed=seed))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-hours", type=float, default=9.9)
    ap.add_argument("--segments", default="GT")
    ap.add_argument("--max-jobs", type=int, default=0)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--prompts", type=int, default=0)
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()

    prompts = PROMPTS[:args.prompts] if args.prompts else PROMPTS
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS)

    os.makedirs(args.outdir, exist_ok=True)
    imgdir = os.path.join(args.outdir, "imgs")
    os.makedirs(imgdir, exist_ok=True)
    state_p = os.path.join(args.outdir, "state.json")
    log_p = os.path.join(args.outdir, "L2.log")
    csv_p = os.path.join(args.outdir, "L2_images.csv")

    def log(m):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(log_p, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    jobs = build_jobs(prompts, seeds, args.segments)
    done = set()
    if os.path.exists(state_p):
        with open(state_p, encoding="utf-8") as f:
            done = set(json.load(f))
    todo = [j for j in jobs if jkey(j) not in done]
    if args.max_jobs:
        todo = todo[:args.max_jobs]

    est_per_job = {"G": len(W_GRID) * len(S_GRID) * 2.6 + 6,
                   "T": 3 * len(W_GRID) * 2.6 + 6}
    cpu_est = sum(est_per_job[j["segment"]] for j in todo)
    log(f"START segments={args.segments} prompts={len(prompts)} seeds={seeds} "
        f"todo={len(todo)}/{len(jobs)} est_hours={cpu_est/3600:.2f}")

    Runner.instance = Runner()
    log("models loaded")

    deadline = time.time() + args.target_hours * 3600
    t_start = time.time()
    n_img = 0
    for j in todo:
        if time.time() >= deadline:
            log(f"DEADLINE reached, {len(todo)}-{todo.index(j)} jobs left")
            break
        try:
            if j["segment"] == "G":
                rows, _ = job_grid(j["prompt"], j["seed"], imgdir)
            else:
                rows = job_route(j["prompt"], j["seed"], imgdir)
        except Exception:
            log(f"FAIL {jkey(j)} :: {traceback.format_exc().splitlines()[-1]}")
            continue
        pd.DataFrame(rows, columns=COLS).to_csv(
            csv_p, mode="a", header=not os.path.exists(csv_p), index=False)
        done.add(jkey(j))
        n_img += len(rows)
        with open(state_p, "w", encoding="utf-8") as f:
            json.dump(sorted(done), f)
        el = time.time() - t_start
        left = len(todo) - todo.index(j) - 1
        rate = n_img / max(el, 1e-6)
        log(f"OK {j['segment']} {j['prompt'][0]:6s} s{j['seed']} "
            f"{len(rows):3d} imgs in {el/3600:.2f}h  imgs={n_img} "
            f"({rate*60:.0f}/min)  jobs_left={left}  "
            f"eta={(cpu_est - n_img*2.6)/3600:.2f}h")
    log(f"STOP imgs={n_img} elapsed={(time.time()-t_start)/3600:.2f}h "
        f"state={len(done)}")


if __name__ == "__main__":
    main()
