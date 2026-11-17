"""Qualitative panel for paper 2: same guidance budget, different dial settings.

Three configurations chosen so that g_eff = s(1 + w*rho) matches the default (w=1, s=7.5)
with rho = 0.208 (the object-prompt value used throughout the paper):

    (w=0.6, s=8.0)   g_eff = 8.0 * (1 + 0.6*0.208) = 9.00
    (w=1.0, s=7.5)   g_eff = 7.5 * (1 + 1.0*0.208) = 9.06   default
    (w=2.0, s=6.4)   g_eff = 6.4 * (1 + 2.0*0.208) = 9.06

plus two controls that keep w but return s to the default, i.e. deliberately wrong g_eff:

    (w=2.0, s=7.5)   g_eff = 11.6   over-guided
    (w=0.6, s=7.5)   g_eff = 8.4    under-guided

If the identity holds, the top row should look nearly identical and the bottom row should not.
"""
import os

import numpy as np
import torch
from PIL import Image

SD15 = r"E:\研究\models\sd15.safetensors"
LORA = r"E:\研究\models\lora\trained\lora_REAL.safetensors"
OUT = r"E:\研究\results\论文二_LoRA共线\Z6_panel_2026-11-14"
PROMPT = "a red ceramic teapot on a wooden table"
SEED = 11

CONFIGS = [
    ("w=0.6, s=8.0", 0.6, 8.0),
    ("w=1.0, s=7.5", 1.0, 7.5),
    ("w=2.0, s=6.4", 2.0, 6.4),
    ("w=2.0, s=7.5", 2.0, 7.5),
    ("w=0.6, s=7.5", 0.6, 7.5),
]


def main():
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    os.makedirs(OUT, exist_ok=True)
    pipe = StableDiffusionPipeline.from_single_file(
        SD15, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(LORA, adapter_name="A")

    paths = []
    for name, w, s in CONFIGS:
        pipe.set_adapters(["A"], [w])
        g = torch.Generator(device="cpu").manual_seed(SEED)
        img = pipe(prompt=PROMPT, num_inference_steps=20, guidance_scale=s,
                   generator=g, height=512, width=512, return_dict=False)[0][0]
        p = os.path.join(OUT, "w%.1f_s%.1f.png" % (w, s))
        img.save(p)
        paths.append((name, p))
        print("saved", p, flush=True)

    # montage
    ims = [Image.open(p) for _, p in paths]
    W = 256
    ims = [im.resize((W, W)) for im in ims]
    canvas = Image.new("RGB", (W * len(ims), W), "white")
    for i, im in enumerate(ims):
        canvas.paste(im, (i * W, 0))
    canvas.save(os.path.join(OUT, "panel.png"))
    print("panel ->", os.path.join(OUT, "panel.png"))


if __name__ == "__main__":
    main()
