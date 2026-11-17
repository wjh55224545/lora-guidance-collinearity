"""Paper 2, T2: the (w, s) grid measured on an SDXL base instead of SD1.5.

Same grid, same prompts, same seeds, same pix_l2 definition as the SD1.5 runs
(obj00-03 x w in W_GRID x s in S_GRID x seed 11 = 288 images), so kappa is
directly comparable across bases.  Only the pipeline and the resolution change.

Why a separate file rather than a flag in y_gen.py: every published kappa came
out of y_gen.py, and a base switch that silently altered a shared code path is
exactly the failure mode that produced the panel-mixing bug in T1c.  The
distance code below is a verbatim copy of y_gen.Runner.thumb, and the reference
logic is the same (reference image generated with the same adapter configuration
at (W0, S0)).

Run:
    python xl_grid.py --tag xl_S1 --lora "E:\\研究\\models\\lora\\trained\\lora_xl_S1.safetensors" --prompts 4 --seeds 11
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
from longrun_p2 import PROMPTS, S0, STEPS  # noqa: E402
from y_gen import CLIP_DIR, COLS, S_GRID, W_GRID  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SDXL = r"E:\研究\models\sdxl\sd_xl_base_1.0.safetensors"
RES = 1024


def build_pipe():
    from diffusers import DDIMScheduler
    import xl_train as T
    pipe = T.load_pipe(torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


class Runner:
    def __init__(self, lora_path, lora_name="xl", no_clip=True):
        from transformers import CLIPModel, CLIPProcessor
        self.name = lora_name
        self.no_clip = no_clip
        self.pipe = build_pipe()
        self.pipe.load_lora_weights(lora_path, adapter_name=lora_name)
        # 全量 fp16 pipeline 约 6.9 GB，可见显存 6.94 GB：直接上卡会在 VAE 解码时 OOM。
        # 按模块 offload，峰值约 5.2 GB，实测 25.8 s/张（1024 px, 20 步）。
        self.pipe.enable_model_cpu_offload()
        # CLIP 打分默认关闭（--clip 才开）。两处实测：
        #   在卡上：与 offload 的 UNet 抢显存（6755/8188 MiB），45 s/张；
        #   在 CPU(fp32)：显存降到 5927 MiB，但 CLIP-L14 单图要 ~30 s，比生成本身还久。
        # 而且 SDXL 的 clip_i/clip_t 与 SD1.5 不同基座、不可比，本文对 SDXL 只报 pixel 口径，
        # 所以这几列写 NaN 而不是花 30 s 换一个不能用的数。
        self.clip = None
        self.proc = None
        if not no_clip:
            self.clip = CLIPModel.from_pretrained(
                CLIP_DIR, torch_dtype=torch.float16, local_files_only=True,
                use_safetensors=True).to("cuda").eval()
            self.proc = CLIPProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
        self.cache = {}

    @torch.no_grad()
    def gen(self, prompt, w, s, seed):
        self.pipe.set_adapters([self.name], [float(w)])
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.pipe(prompt=prompt, num_inference_steps=STEPS,
                         guidance_scale=float(s), generator=g,
                         height=RES, width=RES, return_dict=False)[0][0]

    @torch.no_grad()
    def score(self, img, text):
        if self.no_clip:
            return None, None
        import torch.nn.functional as F
        ti = self.proc(text=[text], return_tensors="pt", padding=True,
                       truncation=True)
        tf = F.normalize(self.clip.text_projection(
            self.clip.text_model(**ti).pooler_output).float(), dim=-1)[0]
        ii = self.proc(images=[img], return_tensors="pt")
        imf = F.normalize(self.clip.visual_projection(
            self.clip.vision_model(**ii).pooler_output).float(), dim=-1)[0]
        return tf, imf

    @staticmethod
    def thumb(img, n=64):
        return np.asarray(img.resize((n, n)), np.float32).reshape(-1) / 255.0

    def anchors(self, text, seed):
        """(w=0 image, reference at (W0,S0)) -- cache key includes the prompt."""
        key = (text, seed)
        if key in self.cache:
            return self.cache[key]
        b = self.gen(text, 0.0, S0, seed)
        r = self.gen(text, 1.0, S0, seed)
        _, fb = self.score(b, text)
        _, fr = self.score(r, text)
        self.cache[key] = ((b, fb, self.thumb(b)), (r, fr, self.thumb(r)))
        return self.cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--seeds", type=str, default="11")
    ap.add_argument("--w", type=str, default="")
    ap.add_argument("--s", type=str, default="")
    ap.add_argument("--clip", action="store_true",
                    help="同时算 CLIP 指标；默认关闭，实测每张多 30 s 且跨基座不可比")
    a = ap.parse_args()

    wg = [float(x) for x in a.w.split(",")] if a.w else W_GRID
    sg = [float(x) for x in a.s.split(",")] if a.s else S_GRID
    seeds = [int(x) for x in a.seeds.split(",")]
    prompts = PROMPTS[:a.prompts]
    outdir = a.outdir or os.path.join(HERE, f"out_xl_{a.tag}")

    imgdir = os.path.join(outdir, "img")
    os.makedirs(imgdir, exist_ok=True)
    csv = os.path.join(outdir, "images.csv")
    state_p = os.path.join(outdir, "state.json")
    done = set(json.load(open(state_p))) if os.path.exists(state_p) else set()
    if not os.path.exists(csv):
        pd.DataFrame(columns=COLS).to_csv(csv, index=False)

    jobs = [((pid, pc, text), sd, w, s)
            for (pid, pc, text) in prompts for sd in seeds for w in wg for s in sg]
    print(f"[{a.tag}] {len(jobs)} images, {len(done)} done, SDXL {RES}px", flush=True)

    R = Runner(a.lora, lora_name=a.tag, no_clip=not a.clip)
    t0, n_new = time.time(), 0
    for (pid, pc, text), sd, w, s in jobs:
        key = f"{pid}|{sd}|{w}|{np.nan}|{s}"
        if key in done:
            continue
        try:
            (b, fb, tb), (r, fr, tr) = R.anchors(text, sd)
            ta = time.time()
            img = R.gen(text, w, s, sd)
            gen_s = time.time() - ta
            ft, fi = R.score(img, text)
            nm = f"{pid}_sd{sd}_w{w:g}_s{s:g}.png"
            ipath = os.path.join(imgdir, nm)
            img.save(ipath)
            row = dict(tag=a.tag, mode="single", prompt_id=pid, pclass=pc,
                       prompt=text, seed=sd, w=w, w2=np.nan, s=s,
                       clip_t=np.nan if ft is None else float(ft @ fi),
                       clip_i_ref=np.nan if fi is None else float(fi @ fr),
                       clip_i_base=np.nan if fi is None else float(fi @ fb),
                       pix_l2=float(np.linalg.norm(R.thumb(img) - tr)),
                       pix_l2_base=float(np.linalg.norm(R.thumb(img) - tb)),
                       img_path=ipath, gen_s=round(gen_s, 3),
                       lora=f"{a.tag}@{a.lora}")
            pd.DataFrame([row], columns=COLS).to_csv(
                csv, mode="a", header=False, index=False)
            done.add(key)
            json.dump(sorted(done), open(state_p, "w"))
            n_new += 1
            if n_new % 10 == 0:
                el = (time.time() - t0) / n_new
                left = (len(jobs) - len(done)) * el / 60
                print(f"  {len(done)}/{len(jobs)}  {el:.2f}s/img  eta {left:.0f}min",
                      flush=True)
        except Exception:
            traceback.print_exc()
            print(f"  !! {key}", flush=True)
    print(f"[{a.tag}] done {n_new} new in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
