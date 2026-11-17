# -*- coding: utf-8 -*-
"""Y11 (paper 2): does kappa survive a backbone change?  SD2.1-base.

Closes the last structural objection after Y1 (independent training): every
measurement so far lives on SD1.5.  A reviewer needs one sentence -- "is this
an SD1.5 quirk?" -- to confine the whole paper.  Y11 re-runs the smallest
self-contained loop on SD2.1-base:

  1. build the concept set with SD2.1 itself (same 24-image recipe as Y1, so
     the LoRA learns a style the base model can already produce);
  2. train ONE rank-8 LoRA (concept A, flat-vector look; Y1 already showed
     cross-training robustness on SD1.5, so one concept suffices here);
  3. verify it is not inert (w=0 vs w in {1,2}, threshold pix_l2_base >= 2.0);
  4. run the SAME w x s grid as L2 (w 8 values, s in {3,4,5,6,7.5,9,10.5,12,14},
     2 prompts x 2 seeds) so kappa is directly comparable to the 0.19-0.27
     published interval;
  5. analyse: s*(w) = argmin_s pix_l2(img(w,s), img(1,7.5)); kappa = -slope of
     log(s*/S0) on log(w/W0); piecewise at the same cut (w<=0.8 vs w>=1.2).

Pre-registered criteria (before running):
  C1  if max pix_l2_base < 2.0 -> report "untrained", never "no collinearity".
  C2  if |kappa_21 - 0.211| <= 0.10 -> "same magnitude across backbones";
      if larger -> report the difference as-is and confine the SD1.5 claim.
  C3  quad estimator is NOT used (three prior failures, red line 7x).

Run:  python y11_sd21.py            # full loop, resumable at every stage
      python y11_sd21.py --smoke    # 4-step train, tiny grid
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SD21 = r"E:\研究\models\sd21_base"
CONCEPT_ROOT = r"E:\研究\models\concepts21"
LORA_OUT = r"E:\研究\models\lora\trained\lora_A21.safetensors"
OUT = r"E:\研究\results\论文二_LoRA共线\Y11_2026-11-08"

W_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
S_GRID = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]
SEEDS = [11, 22]
PROMPTS = [("obj00", "object", "a red ceramic teapot on a wooden table"),
           ("obj03", "object", "a wooden chair in a bright room")]
STYLE = ("flat vector illustration, thick black outlines, limited palette "
         "of teal and coral, simple geometric shapes, plain background")
SUBJECTS = ["a teapot", "a wooden chair", "a bicycle", "a lighthouse",
            "a coffee cup", "a mountain range", "a sailing boat", "a mushroom",
            "a hot air balloon", "a street lamp", "a rose", "a small house"]


def load_pipe(dtype=torch.float16):
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        SD21, torch_dtype=dtype, local_files_only=True,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def img_vec(im):
    return np.asarray(im.resize((64, 64)).convert("L"), np.float32).reshape(-1) / 255.0


# ---------------------------------------------------------------- stage 1
def build_concept(steps_img=20):
    meta = os.path.join(CONCEPT_ROOT, "captions.json")
    if os.path.exists(meta):
        return json.load(open(meta))
    os.makedirs(CONCEPT_ROOT, exist_ok=True)
    pipe = load_pipe()
    items, i = [], 0
    for sd in (101, 202):
        for subj in SUBJECTS:
            cap = f"{STYLE}, {subj}"
            g = torch.Generator(device="cpu").manual_seed(sd + i)
            im = pipe(prompt=cap, num_inference_steps=steps_img, guidance_scale=7.5,
                      generator=g, height=512, width=512,
                      return_dict=False)[0][0]
            p = os.path.join(CONCEPT_ROOT, f"{i:02d}.png")
            im.save(p)
            items.append(dict(path=p, caption=cap))
            i += 1
    json.dump(items, open(meta, "w"), indent=1)
    del pipe
    torch.cuda.empty_cache()
    print(f"  built {len(items)} concept images -> {CONCEPT_ROOT}", flush=True)
    return items


# ---------------------------------------------------------------- stage 2
def train_lora(items, steps, rank=8, lr=2e-4, warmup=50):
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers
    from diffusers.loaders import StableDiffusionLoraLoaderMixin
    from PIL import Image

    pipe = StableDiffusionPipeline.from_pretrained(
        SD21, torch_dtype=torch.float32, local_files_only=True,
        safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet.to("cuda").train()
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2, betas=(0.9, 0.999))
    sched = DDPMScheduler.from_config(pipe.scheduler.config)

    vae = pipe.vae.to("cuda").eval()
    te = pipe.text_encoder.to("cuda").eval()
    tok = pipe.tokenizer
    lat, emb = [], []
    with torch.no_grad():
        for it in items:
            im = Image.open(it["path"]).convert("RGB").resize((512, 512))
            x = torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1.0)
            x = x.permute(2, 0, 1)[None].to("cuda", torch.float32)
            lat.append(vae.encode(x).latent_dist.sample() * 0.18215)
            t = tok(it["caption"], padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").to("cuda")
            emb.append(te(**t).last_hidden_state)
    lat, emb = torch.cat(lat).cpu(), torch.cat(emb).cpu()
    del vae, te
    torch.cuda.empty_cache()

    rng, losses, t0 = np.random.default_rng(0), [], time.time()
    n = lat.shape[0]
    print(f"  training: {n} images, {steps} steps, rank {rank}, "
          f"{sum(p.numel() for p in params)/1e6:.2f}M params", flush=True)
    for st in range(1, steps + 1):
        cur = lr * (st / warmup if st < warmup else
                    0.5 * (1 + math.cos(math.pi * (st - warmup) / max(1, steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = cur
        i = int(rng.integers(n))
        x0, e = lat[i:i+1].to("cuda"), emb[i:i+1].to("cuda")
        noise = torch.randn_like(x0)
        tt = torch.randint(0, sched.config.num_train_timesteps, (1,), device="cuda").long()
        pred = unet(sched.add_noise(x0, noise, tt), tt,
                    encoder_hidden_states=e).sample
        loss = F.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if st % 50 == 0 or st == 1:
            el = time.time() - t0
            print(f"    step {st:4d}/{steps}  loss {np.mean(losses[-50:]):.4f}"
                  f"  {el/st*1000:.0f} ms/step  eta {(steps-st)*el/st/60:.1f} min",
                  flush=True)

    sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    os.makedirs(os.path.dirname(LORA_OUT), exist_ok=True)
    StableDiffusionLoraLoaderMixin.save_lora_weights(
        save_directory=LORA_OUT.replace(".safetensors", ""),
        unet_lora_layers=sd, safe_serialization=True)
    made = LORA_OUT.replace(".safetensors", "") + "/pytorch_lora_weights.safetensors"
    if os.path.exists(made):
        os.replace(made, LORA_OUT)
    del unet, pipe
    torch.cuda.empty_cache()
    return float(np.mean(losses[-50:]))


# ---------------------------------------------------------------- stage 3+4
def gen_all(smoke=False):
    """verify inertness + full grid; returns list of dict rows."""
    pipe = load_pipe()
    pipe.load_lora_weights(LORA_OUT, adapter_name="v")

    def gen(pid_text, w, sd, gs=7.5, steps=20):
        pipe.set_adapters(["v"], [float(w)])
        g = torch.Generator(device="cpu").manual_seed(int(sd))
        return pipe(prompt=pid_text, num_inference_steps=steps, guidance_scale=gs,
                    generator=g, height=512, width=512, return_dict=False)[0][0]

    rows = []
    for pid, pcl, text in PROMPTS:
        for sd in SEEDS:
            ref = img_vec(gen(text, 1.0, sd, 7.5))   # reference (w=1, s=7.5)
            b = img_vec(gen(text, 0.0, sd))          # w=0 base for inertness
            for w in W_GRID:
                for s in S_GRID:
                    if smoke and (w, s) not in ((0.2, 3.0), (1.0, 7.5), (2.0, 14.0)):
                        continue
                    p = os.path.join(OUT, "img",
                                     f"{pid}_sd{sd}_w{w}_s{s}.png")
                    if os.path.exists(p):
                        v = img_vec(__import__("PIL.Image", fromlist=["Image"])
                                    .open(p))
                    else:
                        im = gen(text, w, sd, gs=s)
                        im.save(p)
                        v = img_vec(im)
                    rows.append(dict(
                        prompt_id=pid, pclass=pcl, seed=sd, w=w, s=s,
                        pix_l2=float(np.linalg.norm(v - ref)),
                        pix_l2_base=float(np.linalg.norm(v - b)),
                        img_path=p))
            print(f"  grid {pid} seed{sd} done ({len(rows)} rows so far)", flush=True)
    del pipe
    torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- stage 5
def analyse(rows):
    import pandas as pd
    from scipy import stats
    df = pd.DataFrame(rows)
    E = df.loc[df.groupby(["prompt_id", "seed", "w"]).pix_l2.idxmin()]
    fits, pw = [], []
    for (pid, sd), g in E.groupby(["prompt_id", "seed"]):
        g = g[g.w != 1.0].sort_values("w")
        x, y = np.log(g.w.values / 1.0), np.log(g["s"].values / 7.5)
        r = stats.linregress(x, y)
        fits.append(dict(prompt_id=pid, seed=sd, kappa=-r.slope, r2=r.rvalue**2,
                         kappa_lo=-r.slope-1.96*r.stderr,
                         kappa_hi=-r.slope+1.96*r.stderr))
        lo, hi = g[g.w <= 0.8], g[g.w >= 1.2]
        if len(lo) >= 3 and len(hi) >= 3:
            rl = stats.linregress(np.log(lo.w.values), np.log(lo["s"].values))
            rh = stats.linregress(np.log(hi.w.values), np.log(hi["s"].values))
            pw.append(dict(prompt_id=pid, seed=sd,
                           kappa_lo_seg=-rl.slope, kappa_hi_seg=-rh.slope))
    K, P = pd.DataFrame(fits), pd.DataFrame(pw)
    os.makedirs(os.path.join(OUT, "csv"), exist_ok=True)
    df.to_csv(os.path.join(OUT, "csv", "y11_grid.csv"), index=False)
    E.to_csv(os.path.join(OUT, "csv", "y11_sstar.csv"), index=False)
    K.to_csv(os.path.join(OUT, "csv", "y11_kappa.csv"), index=False)
    P.to_csv(os.path.join(OUT, "csv", "y11_piecewise.csv"), index=False)
    print("\n=== Y11 kappa (SD2.1) ===")
    print(K.to_string(index=False))
    print(f"\nmean kappa = {K.kappa.mean():.3f}  "
          f"(SD1.5 reference: 0.211 grid / interval [0.17,0.27])")
    if len(P):
        print("\npiecewise:")
        print(P.to_string(index=False))
        t = stats.wilcoxon(P.kappa_lo_seg, P.kappa_hi_seg)
        print(f"piecewise diff Wilcoxon p={t.pvalue:.3g}")
    inert = df[df.w == 0.0]
    print(f"\ninertness check: (w=0 rows should be absent; "
          f"pix_l2_base at w=0.2 min = "
          f"{df[df.w==0.2].pix_l2_base.min():.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    a = ap.parse_args()

    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    print("=== Y11: kappa on SD2.1-base ===", flush=True)
    items = build_concept()
    if a.smoke:
        items = items[:4]
    final = train_lora(items, 4 if a.smoke else a.steps)
    print(f"  final loss {final:.4f}", flush=True)
    rows = gen_all(smoke=a.smoke)
    if not a.smoke:
        analyse(rows)
        import pandas as pd
        pd.DataFrame([dict(steps=a.steps, final_loss=final)]).to_csv(
            os.path.join(OUT, "csv", "y11_train.csv"), index=False)
    print("=== done ===")


if __name__ == "__main__":
    main()
