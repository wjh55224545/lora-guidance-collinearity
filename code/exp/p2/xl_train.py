"""Paper 2, T2: a subject LoRA trained on SDXL instead of SD1.5.

Why this exists: every kappa in the paper so far is measured on SD1.5 adapters.
A reviewer's first objection to "g* = s(1+w*rho)" is that it may be a property of
one base model rather than of the LoRA-CFG interaction.  T2 is the cross-base
check: same subject recipe, same (w, s) grid, SDXL as the base.

What had to change for SDXL (and nothing else)
  * two text encoders.  prompt_embeds = cat(CLIP-L hidden[-2] (768),
    OpenCLIP-bigG hidden[-2] (1280)) -> 2048, plus pooled text_embeds (1280)
    and time_ids (6) passed through added_cond_kwargs.
  * VAE scaling factor 0.13025, not SD1.5's 0.18215.
  * training in bf16, not fp32.  SD1.5's y1_train.py could afford an fp32 UNet
    (0.86 GB); SDXL's fp32 UNet is 10.3 GB and does not fit 8 GB at all.
    sdxl_vram_probe.py measured fp16/bf16 UNet + rank-8 LoRA + gradient
    checkpointing at 1024 px to peak at 5.57 GB, so bf16 is the only route.
  * 1024 px native resolution (SDXL was trained there; 512 degrades it).

Everything downstream -- pix_l2 on a 64x64 thumbnail, argmin over s, the slope
of log s* on log w -- is base-model agnostic and is reused verbatim, which is
what makes the comparison admissible.

Run:
    python xl_train.py --subject "a photo of a red ceramic teapot" --tag xl_S1 --steps 800
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SDXL = r"E:\研究\models\sdxl\sd_xl_base_1.0.safetensors"
# 单文件 ckpt 只含权重，diffusers 还要一份 config 目录才能构造模型。
# 系统代理会让 HF 返回 502，这份 config 是从 hf-mirror 直连拉的（只有 json/txt，4.2 MB）。
CFG_XL = r"E:\研究\models\sdxl_cfg"
CONCEPT_ROOT = r"E:\研究\models\concepts"
LORA_OUT = r"E:\研究\models\lora\trained"

RES = 1024
VAE_SCALE = 0.13025
STEPS_IMG = 25

SCENES = [
    "on a wooden table", "on a marble counter", "on a windowsill at sunrise",
    "in a sunlit kitchen", "on a white shelf", "next to a stack of books",
    "on a picnic blanket", "on a dark slate surface", "in a glass cabinet",
    "on a garden bench", "against a plain grey wall", "on a cafe table",
]
SEEDS = [707, 808]


def load_pipe(dtype, keep=("unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2")):
    """Load the XL pipeline from the single-file ckpt, keeping only some components.

    The whole pipeline in fp16 is ~6.9 GB against 6.94 GB of visible VRAM, so no
    stage may hold all four networks at once.  `keep` prunes to what the stage
    actually uses: generation needs everything (with CPU offload), precompute
    needs only VAE + the two text encoders, training needs only the UNet.
    """
    from diffusers import StableDiffusionXLPipeline
    if not os.path.exists(SDXL):
        raise SystemExit(f"SDXL 权重不存在：{SDXL}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        SDXL, config=CFG_XL, torch_dtype=dtype, use_safetensors=True,
        local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    for name in ("unet", "vae", "text_encoder", "text_encoder_2",
                 "tokenizer", "tokenizer_2"):
        if name not in keep and hasattr(pipe, name) and getattr(pipe, name) is not None:
            setattr(pipe, name, None)
    return pipe


def build_subject_images(subject, tag):
    """24 (image, caption) pairs at 1024 px, cached on disk."""
    from diffusers import DDIMScheduler
    d = os.path.join(CONCEPT_ROOT, tag)
    meta = os.path.join(d, "captions.json")
    if os.path.exists(meta):
        return json.load(open(meta))
    os.makedirs(d, exist_ok=True)

    pipe = load_pipe(torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # 全量 fp16 pipeline 约 6.9 GB，visible VRAM 只有 6.94 GB：直接 .to("cuda")
    # 会在 VAE 解码那一步 OOM。按模块 offload，峰值只剩 UNet 的 5.1 GB。
    pipe.enable_model_cpu_offload()

    items, i, t0 = [], 0, time.time()
    for sd in SEEDS:
        for sc in SCENES:
            cap = f"{subject} {sc}"
            g = torch.Generator(device="cpu").manual_seed(sd + i)
            img = pipe(prompt=cap, num_inference_steps=STEPS_IMG, guidance_scale=7.5,
                       generator=g, height=RES, width=RES, return_dict=False)[0][0]
            p = os.path.join(d, f"{i:02d}.png")
            img.save(p)
            items.append(dict(path=p, caption=cap))
            i += 1
            if i % 4 == 0:
                print(f"    概念图 {i}/{len(SEEDS)*len(SCENES)}  "
                      f"{(time.time()-t0)/i:.1f} s/张", flush=True)
    json.dump(items, open(meta, "w"), indent=1)
    del pipe
    torch.cuda.empty_cache()
    print(f"  built {len(items)} subject images -> {d}", flush=True)
    return items


def precompute(items):
    """(latents, prompt_embeds, pooled, time_ids) -- training never touches VAE or text encoders."""
    from PIL import Image
    pipe = load_pipe(torch.float32, keep=("vae", "text_encoder", "text_encoder_2",
                                          "tokenizer", "tokenizer_2"))
    vae = pipe.vae.to("cuda").eval()
    te1, te2 = pipe.text_encoder.to("cuda").eval(), pipe.text_encoder_2.to("cuda").eval()
    tk1, tk2 = pipe.tokenizer, pipe.tokenizer_2

    lat, emb, pooled = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for n, it in enumerate(items):
            im = Image.open(it["path"]).convert("RGB").resize((RES, RES))
            x = torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1.0)
            x = x.permute(2, 0, 1)[None].to("cuda", torch.float32)
            lat.append((vae.encode(x).latent_dist.sample() * VAE_SCALE).cpu())

            heads = []
            for te, tk in ((te1, tk1), (te2, tk2)):
                t = tk(it["caption"], padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt").to("cuda")
                out = te(t.input_ids, output_hidden_states=True)
                heads.append(out.hidden_states[-2])          # penultimate
                if te is te2:
                    pooled.append(out[0].cpu())              # text_embeds (1,1280)
            emb.append(torch.cat(heads, dim=-1).cpu())       # (1,77,2048)
            if (n + 1) % 8 == 0:
                print(f"    precompute {n+1}/{len(items)}  {(time.time()-t0)/(n+1):.2f} s/张",
                      flush=True)

    lat = torch.cat(lat).to(torch.float32)
    emb = torch.cat(emb).to(torch.float32)
    pooled = torch.cat(pooled).to(torch.float32)
    # SDXL 的 time_ids = (orig_h, orig_w, crop_y, crop_x, target_h, target_w)，生成时全等于 RES
    tid = torch.tensor([[RES, RES, 0, 0, RES, RES]], dtype=torch.float32).repeat(len(items), 1)
    del vae, te1, te2, pipe
    torch.cuda.empty_cache()
    print(f"  latents {tuple(lat.shape)}  embeds {tuple(emb.shape)}  "
          f"pooled {tuple(pooled.shape)}", flush=True)
    return lat, emb, pooled, tid


def train(lat, emb, pooled, tid, steps, rank, lr, out_path, warmup=50, log_every=25):
    from diffusers import DDPMScheduler
    from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict

    pipe = load_pipe(torch.bfloat16, keep=("unet",))
    unet = pipe.unet.to("cuda").train()
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    unet.enable_gradient_checkpointing()
    # 输入必须是梯度检查点认可的（use_reentrant=False 才支持 added_cond_kwargs 里的张量）
    params = [p for p in unet.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2, betas=(0.9, 0.999))
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    n_img = lat.shape[0]
    print(f"  trainable params {n_par/1e6:.2f} M over {n_img} images, "
          f"{steps} steps, rank {rank}, bf16, {RES}px", flush=True)

    rng = np.random.default_rng(0)
    t0, losses = time.time(), []
    for st in range(1, steps + 1):
        cur_lr = lr * (st / warmup if st < warmup else
                       0.5 * (1 + math.cos(math.pi * (st - warmup) /
                                           max(1, steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        i = int(rng.integers(n_img))
        x0 = lat[i:i + 1].to("cuda", torch.bfloat16)
        e = emb[i:i + 1].to("cuda", torch.bfloat16)
        te = pooled[i:i + 1].to("cuda", torch.bfloat16)
        ti = tid[i:i + 1].to("cuda", torch.bfloat16)
        noise = torch.randn_like(x0)
        tt = torch.randint(0, sched.config.num_train_timesteps, (1,),
                           device="cuda").long()
        xn = sched.add_noise(x0, noise, tt)
        pred = unet(xn, tt, encoder_hidden_states=e, added_cond_kwargs={
            "text_embeds": te, "time_ids": ti}).sample
        loss = F.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if st % log_every == 0 or st == 1:
            el = time.time() - t0
            print(f"    step {st:4d}/{steps}  loss {np.mean(losses[-log_every:]):.4f}"
                  f"  lr {cur_lr:.2e}  {el/st*1000:.0f} ms/step  "
                  f"eta {(steps-st)*el/st/60:.1f} min", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    d = out_path.replace(".safetensors", "")
    os.makedirs(d, exist_ok=True)
    StableDiffusionXLLoraLoaderMixin.save_lora_weights(
        save_directory=d, unet_lora_layers=sd, text_encoder_lora_layers=None,
        safe_serialization=True)
    made = os.path.join(d, "pytorch_lora_weights.safetensors")
    if os.path.exists(made) and made != out_path:
        os.replace(made, out_path)
    print(f"  saved {out_path}  ({os.path.getsize(out_path)/1e6:.2f} MB, "
          f"{len(sd)} tensors)", flush=True)
    del unet, pipe
    torch.cuda.empty_cache()
    return float(np.mean(losses[-50:]))


def verify(lora_path, prompts, seeds=(11,)):
    """Does the LoRA move the picture?  w=0 vs w in {1,2}, same pix_l2_base scale."""
    from diffusers import DDIMScheduler
    pipe = load_pipe(torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # 全量 fp16 pipeline 约 6.9 GB，visible VRAM 只有 6.94 GB：直接 .to("cuda")
    # 会在 VAE 解码那一步 OOM。按模块 offload，峰值只剩 UNet 的 5.1 GB。
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(lora_path, adapter_name="v")

    def gen(p, w, sd):
        pipe.set_adapters(["v"], [float(w)])
        g = torch.Generator(device="cpu").manual_seed(int(sd))
        im = pipe(prompt=p, num_inference_steps=STEPS_IMG, guidance_scale=7.5,
                  generator=g, height=RES, width=RES, return_dict=False)[0][0]
        return np.asarray(im.resize((64, 64)), np.float32).reshape(-1) / 255.0

    out = []
    for p, pc, text in prompts:
        for sd in seeds:
            b = gen(text, 0.0, sd)
            for w in (1.0, 2.0):
                dd = float(np.linalg.norm(gen(text, w, sd) - b))
                out.append(dict(prompt_id=p, seed=sd, w=w, pix_l2_base=dd))
                print(f"    {p} seed{sd} w={w}: pix_l2_base={dd:.3f}", flush=True)
    del pipe
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="a photo of a red ceramic teapot")
    ap.add_argument("--tag", default="xl_S1")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="")
    ap.add_argument("--n-verify", type=int, default=2)
    a = ap.parse_args()

    out = a.out or os.path.join(LORA_OUT, f"lora_{a.tag}.safetensors")
    print(f"=== T2 SDXL: {a.tag} ({a.subject}) ===", flush=True)
    items = build_subject_images(a.subject, a.tag)
    lat, emb, pooled, tid = precompute(items)
    final = train(lat, emb, pooled, tid, a.steps, a.rank, a.lr, out)

    from longrun_p2 import PROMPTS
    rows = verify(out, PROMPTS[:a.n_verify])
    mx = max(r["pix_l2_base"] for r in rows)
    rec = dict(subject=a.subject, tag=a.tag, steps=a.steps, rank=a.rank,
               lr=a.lr, base="SDXL 1.0", res=RES, n_images=len(items),
               final_loss=final, path=out, bytes=os.path.getsize(out),
               max_pix_base=mx)
    os.makedirs(os.path.join(HERE, "out_xl"), exist_ok=True)
    import pandas as pd
    fp = os.path.join(HERE, "out_xl", "xl_train_log.csv")
    pd.DataFrame([rec]).to_csv(fp, mode="a",
                               header=not os.path.exists(fp), index=False)
    print(f"  max pix_l2_base at w in {{1,2}} = {mx:.3f}", flush=True)
    if mx < 2.0:
        print("  !! INERT：LoRA 几乎不改变图像，只能报“未训成”，不能报“无共线”。",
              flush=True)
    print(f"=== done {out} ===", flush=True)


if __name__ == "__main__":
    main()
