"""E2c (paper 2): repeat E2b's amplitude sweep with a REAL SD1.5 LoRA.

E2b used a synthetic rank-4 delta applied uniformly to all 64 cross-attention
projections.  Two things could make that unrepresentative:

  * a real LoRA is rank-8 and sits on every attention/FFN projection in the
    UNet *and* in the text encoder, not just cross-attention
  * its per-layer magnitudes are heterogeneous

So rerun the whole measurement with `lora_detail_sd15.safetensors`
("Add More Details" rank-8 LoRA, kohya format, 792 tensors) and see whether the
local elastic slope alpha_t keeps the same shape.

A useful accident: the base trajectory is identical to E2b's.  At w=0 the LoRA
contributes nothing, so both experiments roll out the same 20 DDIM steps from
the same seed and evaluate on the same x_t.  The only difference is the
perturbation family, which makes this a clean A/B.

Two self-checks are built in:
  (1) w=0 must reproduce the un-adapted model to <1e-5 (else set_adapters is
      leaking);
  (2) the adapter must actually change the output at w=1 (else we are measuring
      noise).

Run:  python e2c_real_lora.py --steps 20 --ws 0.25,0.5,1.0,1.5
"""
import argparse
import os

import numpy as np
import torch

SD15 = r"E:\研究\models\sd15.safetensors"
CFG = r"E:\研究\models\sd15_cfg"
LORA = r"E:\研究\models\lora\lora_detail_sd15.safetensors"
ADAPTER = "d"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

PROMPTS = {
    "person": "a portrait photo of sks person, close up, natural light",
    "style": "an oil painting of a village at dusk, thick brush strokes",
    "object": "a red ceramic teapot on a wooden table",
}


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


def lora_magnitudes(pipe):
    """Per-layer ||scaling * B A|| / ||W|| on the UNet at w=1."""
    rel = []
    for _, mod in pipe.unet.named_modules():
        la = getattr(mod, "lora_A", None)
        lb = getattr(mod, "lora_B", None)
        if la is None or ADAPTER not in la:
            continue
        A = la[ADAPTER].weight.float()
        B = lb[ADAPTER].weight.float()
        W = mod.base_layer.weight.float()
        sc = mod.scaling[ADAPTER]
        dW = (B.reshape(B.shape[0], -1) @ A.reshape(A.shape[0], -1)) * sc \
            if A.dim() > 2 else (B @ A) * sc
        rel.append((dW.norm() / W.norm()).item())
    return np.array(rel)


@torch.no_grad()
def unet_out(pipe, x, t, emb):
    return pipe.unet(x, t, encoder_hidden_states=emb).sample


@torch.no_grad()
def rollout(pipe, prompt, steps, s_cfg=7.5, seed=1234):
    """Base trajectory with the adapter at w=0 (identical to E2b's)."""
    pipe.set_adapters([ADAPTER], [0.0])
    pe, ne = pipe.encode_prompt(prompt, "cuda", 1, True, negative_prompt="")
    pe, ne = pe.to(torch.float16), ne.to(torch.float16)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 4, 64, 64, generator=g).to("cuda", torch.float16)
    pipe.scheduler.set_timesteps(steps, device="cuda")

    cache = []
    for t in pipe.scheduler.timesteps:
        tt = t.expand(1) if t.dim() == 0 else t
        e_c = unet_out(pipe, x, tt, pe).float()
        e_u = unet_out(pipe, x, tt, ne).float()
        cache.append((int(round(float(t))), x.clone(), pe, ne, e_c, e_u))
        eps = e_u + s_cfg * (e_c - e_u)
        x = pipe.scheduler.step(eps.to(torch.float16), int(round(float(t))), x,
                                return_dict=False)[0]
    return cache


@torch.no_grad()
def measure(pipe, cache, w):
    pipe.set_adapters([ADAPTER], [w])
    rows = []
    for t_int, x, pe, ne, e_c, e_u in cache:
        tt = torch.tensor([t_int], device=x.device)
        l_c = unet_out(pipe, x, tt, pe).float()
        l_u = unet_out(pipe, x, tt, ne).float()
        d_c = (l_c - e_c).flatten()
        d_u = (l_u - e_u).flatten()
        dg = (e_c - e_u).flatten()
        sig = d_c - d_u
        n_dg = dg.norm().item()
        rows.append(dict(
            w=w, step=t_int, t_frac=t_int / 1000.0,
            rho=sig.norm().item() / max(n_dg, 1e-9),
            align=torch.nn.functional.cosine_similarity(sig[None], dg[None]).item(),
            spur=d_u.norm().item() / max(sig.norm().item(), 1e-9),
            rho_e=d_u.norm().item() / max(e_u.flatten().norm().item(), 1e-9),
            n_dg=n_dg, n_sig=sig.norm().item(),
        ))
    pipe.set_adapters([ADAPTER], [0.0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ws", type=str, default="0.25,0.5,1.0,1.5")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    ws = [float(x) for x in a.ws.split(",")]
    os.makedirs(OUT, exist_ok=True)

    print("loading SD1.5 + real LoRA ...")
    pipe = build_pipe()
    rel = lora_magnitudes(pipe)
    print(f"  UNet LoRA layers: n={len(rel)}  ||dW||/||W|| at w=1: "
          f"median {np.median(rel):.4f}  p10 {np.percentile(rel,10):.4f}  "
          f"p90 {np.percentile(rel,90):.4f}")

    # self-check (1): w=0 must equal the un-adapted model
    pe, ne = pipe.encode_prompt("a photo", "cuda", 1, True, negative_prompt="")
    pe, ne = pe.to(torch.float16), ne.to(torch.float16)
    g = torch.Generator(device="cpu").manual_seed(7)
    xchk = torch.randn(1, 4, 64, 64, generator=g).to("cuda", torch.float16)
    tchk = torch.tensor([500], device="cuda")
    pipe.set_adapters([ADAPTER], [0.0])
    o0 = unet_out(pipe, xchk, tchk, pe).float()
    pipe.set_adapters([ADAPTER], [1.0])
    o1 = unet_out(pipe, xchk, tchk, pe).float()
    print(f"  self-check: |eps(w=0)-eps(w=1)|/|eps(w=0)| = "
          f"{(o1-o0).norm().item()/o0.norm().item():.4f}  (must be > 0.005)")

    rows = []
    for pname, ptxt in PROMPTS.items():
        print(f"\n[{pname}] base trajectory ({a.steps} steps, w=0) ...")
        cache = rollout(pipe, ptxt, a.steps, seed=a.seed)
        for w in ws:
            for r in measure(pipe, cache, w):
                r["prompt"] = pname
                rows.append(r)
            print(f"   w={w:g} done")

    import pandas as pd
    df = pd.DataFrame(rows)[["prompt", "w", "step", "t_frac", "rho",
                             "align", "spur", "rho_e", "n_dg", "n_sig"]]
    csv = os.path.join(OUT, "e2c_real_lora.csv")
    df.to_csv(csv, index=False)

    print("\n" + "=" * 76)
    print("per (prompt, w): rho range, peak location")
    print("-" * 76)
    print(f"{'prompt':8s} {'w':>5s} {'min':>8s} {'max':>8s} {'max/min':>8s} {'peak t':>7s}")
    for p in PROMPTS:
        for w in ws:
            v = df[(df.prompt == p) & (df.w == w)].sort_values("t_frac")
            if not len(v):
                continue
            print(f"{p:8s} {w:5.2f} {v.rho.min():8.4f} {v.rho.max():8.4f} "
                  f"{v.rho.max()/max(v.rho.min(),1e-12):8.2f} "
                  f"{v.loc[v.rho.idxmax(),'t_frac']:7.3f}")

    print("\n" + "=" * 76)
    print("local elastic slope alpha_t = dlog(rho)/dlog(w)")
    print("-" * 76)
    print(f"{'t/1000':>7s} " + "".join(f"{p:>10s}" for p in PROMPTS) + "   R2min")
    for t in sorted(df.t_frac.unique(), reverse=True):
        line = f"{t:7.3f} "; r2s = []
        for p in PROMPTS:
            s = df[(df.prompt == p) & (np.abs(df.t_frac - t) < 1e-9)].sort_values("w")
            x = np.log(s.w.values); y = np.log(s.rho.values)
            sl, ic = np.polyfit(x, y, 1)
            pred = sl * x + ic
            ss = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            line += f"{sl:10.3f}"; r2s.append(ss)
        print(line + f"   {min(r2s):.4f}")

    print("\nwrote", csv)


if __name__ == "__main__":
    main()
