"""Subject LoRA for paper 2: a *subject* adapter, not a style adapter.

Why this exists: every adapter measured so far is either a public "real-photo" LoRA
or a style concept (flat vector, ink wash, pencil). A reviewer of a personalization
paper will ask whether the collinearity claim holds for a *subject*, which is what
DreamBooth-style users actually train. This trains one.

Run:  python b1_subject.py --steps 800
Out:  models/lora/trained/lora_S.safetensors  (+ concepts/S/*.png)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import y1_train as Y

SUBJECT = "a photo of a red ceramic teapot"
SCENES = [
    "on a wooden table", "on a marble counter", "on a windowsill at sunrise",
    "in a sunlit kitchen", "on a white shelf", "next to a stack of books",
    "on a picnic blanket", "on a dark slate surface", "in a glass cabinet",
    "on a garden bench", "against a plain grey wall", "on a cafe table",
]
SEEDS = [707, 808]


def build_subject_images(subject, tag):
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    d = os.path.join(Y.CONCEPT_ROOT, tag)
    meta = os.path.join(d, "captions.json")
    if os.path.exists(meta):
        return json.load(open(meta))
    os.makedirs(d, exist_ok=True)
    pipe = StableDiffusionPipeline.from_single_file(
        Y.SD15, config=Y.CFG, torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    items, i = [], 0
    for sd in SEEDS:
        for sc in SCENES:
            cap = f"{subject} {sc}"
            g = torch.Generator(device="cpu").manual_seed(sd + i)
            img = pipe(prompt=cap, num_inference_steps=20, guidance_scale=7.5,
                       generator=g, height=512, width=512, return_dict=False)[0][0]
            p = os.path.join(d, f"{i:02d}.png")
            img.save(p)
            items.append(dict(path=p, caption=cap))
            i += 1
    json.dump(items, open(meta, "w"), indent=1)
    print(f"  built {len(items)} subject images -> {d}", flush=True)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default=os.path.join(Y.LORA_OUT, "lora_S.safetensors"))
    ap.add_argument("--subject", default=SUBJECT,
                    help="subject phrase, e.g. 'a photo of a red ceramic teapot'")
    ap.add_argument("--tag", default="S", help="concept dir name under concepts/")
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed; None keeps the historical unseeded run")
    a = ap.parse_args()
    items = build_subject_images(a.subject, a.tag)
    lat, emb = Y.precompute(items)
    Y.train(lat, emb, a.steps, a.rank, a.lr, a.out, seed=a.seed)
    print("done", a.out)


if __name__ == "__main__":
    main()
