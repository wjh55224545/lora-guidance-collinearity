"""E2c-2 (paper 2): the same (w,s) grid, four different LoRAs.

The one claim this experiment can settle that L2 could not:

  I1 says w and s are not two independent knobs, because both act on the same
  text-conditioned residual.  If that mechanism is right, kappa must depend on
  *where the LoRA sits*: a text-encoder-only update and a UNet-only update put
  the identity on opposite sides of the conditioning path, so their
  (w,s) contours cannot be the same curve.

Variants produced by `make_lora_variants.py`:
  full       original  (already measured in L2; re-run here only if asked)
  te_only    lora_te_* only            -- identity enters via the prompt embedding
  unet_only  lora_unet_* only          -- identity enters via the denoiser
  no_attn2   everything except _attn2_ -- cross-attention LoRA removed
  r4         rank-4 truncation         -- same direction, ~51% of the energy

Design notes that cost time if ignored
--------------------------------------
* anchors (w=0 and (w0,s0)) are per (prompt, seed) and are generated *with the
  same variant*, otherwise the reference image carries a different LoRA.
* appending one CSV row per image and keeping state.json makes the run
  resumable; a 3 h job that dies at 90% should not lose the 90%.
* metrics are the L2 four: clip_t, clip_i_ref, clip_i_base, pix_l2.  SSIM was
  only used once, on the metric-adjudication subset, because it needs a second
  full-resolution pass.
* DDIM + fixed seed + 20 steps, matching L2, so w and s are the only things
  that move.

Run:
    python e2d_variants.py --tag te_only --prompts 8 --seeds 11,22
    python e2d_variants.py --tag te_only --smoke
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
VARIANT_DIR = r"E:\研究\models\lora\variants"
ORIG = r"E:\研究\models\lora\lora_detail_sd15.safetensors"

W_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
S_GRID = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]

COLS = ["tag", "prompt_id", "pclass", "prompt", "seed", "w", "s",
        "clip_t", "clip_i_ref", "clip_i_base", "pix_l2", "pix_l2_base",
        "img_path", "gen_s"]


def lora_path(tag):
    if tag == "full":
        return ORIG
    p = os.path.join(VARIANT_DIR, f"{tag}.safetensors")
    if not os.path.exists(p):
        raise SystemExit(f"missing variant file: {p}")
    return p


def build_pipe(lora):
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_single_file(
        SD15, config=CFG, torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(lora, adapter_name="v")
    return pipe


class Runner:
    def __init__(self, lora):
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.pipe = build_pipe(lora)
        self.clip = CLIPModel.from_pretrained(
            CLIP_DIR, torch_dtype=torch.float16, local_files_only=True,
            use_safetensors=True).to("cuda").eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_DIR,
                                                  local_files_only=True)
        self.cache = {}

    @torch.no_grad()
    def gen(self, prompt, w, s, seed):
        self.pipe.set_adapters(["v"], [float(w)])
        g = self.torch.Generator(device="cpu").manual_seed(int(seed))
        return self.pipe(prompt=prompt, num_inference_steps=STEPS,
                         guidance_scale=float(s), generator=g,
                         height=512, width=512, return_dict=False)[0][0]

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

    def anchors(self, text, seed):
        """(base at w=0, reference at (W0,S0)) — per prompt and seed.

        NOTE ON A BUG THAT SHIPPED AND WAS CAUGHT 11-07 10:30: the cache key
        was `seed` alone, so the first prompt's reference image was reused for
        every later prompt.  Symptom: pix_l2 at the anchor (w=1.0, s=7.5) is
        exactly 0 for obj00 but 25-31 for obj01/02, i.e. every non-first prompt
        was being compared against somebody else's image.  longrun_p2.py gets
        this right (key = (pid, seed)); this class did not.  Key must include
        the prompt text.
        """
        key = (text, seed)
        if key in self.cache:
            return self.cache[key]
        b = self.gen(text, 0.0, S0, seed)
        r = self.gen(text, W0, S0, seed)
        _, fb = self.score(b, text)
        _, fr = self.score(r, text)
        self.cache[key] = ((b, fb, self.thumb(b)), (r, fr, self.thumb(r)))
        return self.cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--seeds", type=str, default="11,22")
    ap.add_argument("--w", type=str, default="")
    ap.add_argument("--s", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    a = ap.parse_args()

    wg = [float(x) for x in a.w.split(",")] if a.w else W_GRID
    sg = [float(x) for x in a.s.split(",")] if a.s else S_GRID
    seeds = [int(x) for x in a.seeds.split(",")]
    prompts = PROMPTS[:a.prompts]
    if a.smoke:
        prompts, wg, sg, seeds = prompts[:1], [0.0, 1.0], [S0], seeds[:1]

    out = a.outdir or os.path.join(HERE, "out_e2d", a.tag)
    os.makedirs(out, exist_ok=True)

    # 必须把图存下来。L2 那批 8712 张图跑完就丢了（img_path 整列为空），
    # 直接后果是"输出是否只依赖一个组合量 g(w,s)"这个共线性的定义式问题
    # 永远只能拿"到单点参考的距离"间接回答，而那个量在参考点处恒为 0、
    # 对 s 的灵敏度在低 w 端又接近 0，两个毛病叠起来让 kappa 的估计只能靠
    # 曲线的两端（见 13:00 的诊断）。
    # 存 512 原图，约 400 KB/张，全网格 5760 张约 2.3 GB；有了图，两两距离
    # 矩阵、SSIM、论文插图都能后验做，不用重跑。
    imgdir = os.path.join(out, "img")
    os.makedirs(imgdir, exist_ok=True)
    csv = os.path.join(out, "images.csv")
    state_p = os.path.join(out, "state.json")
    done = set()
    if os.path.exists(state_p):
        done = set(json.load(open(state_p)))
    if not os.path.exists(csv):
        pd.DataFrame(columns=COLS).to_csv(csv, index=False)

    R = Runner(lora_path(a.tag))
    jobs = [(p, sd, w, s) for p in prompts for sd in seeds
            for w in wg for s in sg]
    print(f"[{a.tag}] {len(jobs)} images, {len(done)} already done", flush=True)
    t0 = time.time()
    n_new = 0
    for (pid, pc, text), sd, w, s in jobs:
        key = f"{pid}|{sd}|{w}|{s}"
        if key in done:
            continue
        try:
            (b, fb, tb), (r, fr, tr) = R.anchors(text, sd)
            ta = time.time()
            img = R.gen(text, w, s, sd)
            gen_s = time.time() - ta
            ft, fi = R.score(img, text)
            ipath = os.path.join(imgdir, f"{pid}_sd{sd}_w{w:g}_s{s:g}.png")
            img.save(ipath)
            row = dict(tag=a.tag, prompt_id=pid, pclass=pc, prompt=text,
                       seed=sd, w=w, s=s,
                       clip_t=float(ft @ fi), clip_i_ref=float(fi @ fr),
                       clip_i_base=float(fi @ fb),
                       pix_l2=float(np.linalg.norm(R.thumb(img) - tr)),
                       # pix_l2 is the distance to *this variant's own* output at
                       # (W0,S0), so it is identically 0 at that grid point and
                       # the whole w-dependence of the surface collapses onto
                       # that one column (see the 11-07 13:00 diagnosis).  It is
                       # kept unchanged so E2c-2 stays numerically comparable to
                       # L2, but reported alongside pix_l2_base below.
                       #
                       # pix_l2_base is the distance to the *no-LoRA* image at
                       # w=0, which is the same picture for every variant (at
                       # w=0 the LoRA contribution is exactly zero), so this one
                       # IS comparable across variants, and its zero sits at
                       # w=0 -- outside the measured grid, whose smallest w is
                       # 0.2.  No free GPU cost: the base image and its thumb are
                       # already produced by anchors().
                       pix_l2_base=float(np.linalg.norm(R.thumb(img) - tb)),
                       img_path=ipath,
                       gen_s=round(gen_s, 3))
            pd.DataFrame([row], columns=COLS).to_csv(csv, mode="a",
                                                    header=False, index=False)
            done.add(key)
            json.dump(sorted(done), open(state_p, "w"))
            n_new += 1
            if n_new % 20 == 0:
                el = (time.time() - t0) / n_new
                print(f"  {n_new}/{len(jobs)-len(done)+n_new}  "
                      f"{el:.2f}s/img  eta {(len(jobs)-len(done))*el/60:.0f}min",
                      flush=True)
        except Exception:
            traceback.print_exc()
            print(f"  !! {key}", flush=True)
    print(f"[{a.tag}] done {n_new} images in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
