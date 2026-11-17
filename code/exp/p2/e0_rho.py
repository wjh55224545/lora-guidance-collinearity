"""E0/E2 (paper 2): measure the guidance-gain decomposition along a real
sampling trajectory, in latent space only.

At every DDIM step t we hold x_t fixed and evaluate four quantities:

    e_u = UNet(x_t, t, "empty")        base unconditional
    e_c = UNet(x_t, t, prompt)         base conditional
    d_u = UNet_LoRA(x_t, t, "empty") - e_u        Delta(empty)
    d_c = UNet_LoRA(x_t, t, prompt)  - e_c        Delta(cond)

Derived per timestep:
    rho   = ||d_c - d_u|| / ||e_c - e_u||     gain inflation (relative)
    align = cos(d_c - d_u, e_c - e_u)         is the LoRA acting as gain?
    spur  = ||d_u|| / ||d_c - d_u||           spurious unconditional shift
    rho_e = ||d_u|| / ||e_u||                 size of Delta(empty) vs base

Then the composite guidance strength actually seen by the sampler:
    g_eff = s * (1 + w * rho * align) + w * <d_u, unit(e_c-e_u)> / ||e_c-e_u||

The question this script answers: does `rho` drift across t by more than 3x?
If yes, a single global scalar cannot describe the w-s coupling, which is the
whole point of the paper.

The LoRA here is a *controlled synthetic rank-4 delta* (5% of each target
weight's Frobenius norm), so this run validates the measurement, not the
magnitude.  Swap in a real LoRA later via --real.

Run:  python e0_rho.py
"""
import argparse
import json
import os

import numpy as np
import torch

SD15 = r"E:\研究\models\sd15.safetensors"
CFG = r"E:\研究\models\sd15_cfg"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

TARGETS = ("attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0")


def build_pipe():
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_single_file(
        SD15, config=CFG, torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def install_delta(pipe, rank=4, rel=0.05, seed=0, device="cuda"):
    """Random rank-`rank` delta on cross-attention, ||dW||_F = rel*||W||_F.

    Returns (handles, modules) -- the global scale lives in SCALE[0].
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    SCALE = [0.0]
    mods = []

    def hook(mod, inp, out):
        if SCALE[0] == 0.0:
            return out
        x = inp[0].to(torch.float32)
        d = (x @ mod._A.T) @ mod._B.T
        return (out.to(torch.float32) + SCALE[0] * d).to(out.dtype)

    n = 0
    for name, mod in pipe.unet.named_modules():
        if not any(name.endswith(t) for t in TARGETS):
            continue
        if not isinstance(mod, torch.nn.Linear):
            continue
        W = mod.weight.detach().float().cpu()
        o, i = W.shape
        A = torch.randn(rank, i, generator=g) / i ** 0.5
        B = torch.randn(o, rank, generator=g)
        dW = B @ A
        B *= (rel * W.norm() / max(dW.norm().item(), 1e-9))
        mod._A = A.to(device)
        mod._B = B.to(device)
        mod._h = mod.register_forward_hook(hook)
        mods.append((name, mod, W.norm().item(), (B @ A).norm().item()))
        n += 1
    print(f"  delta installed on {n} cross-attn projections (rank {rank}, rel {rel})")
    if mods:
        nm, _, wn, dn = mods[0]
        print(f"  check {nm}: ||W||={wn:.3f}  ||dW||={dn:.3f}  ratio={dn/wn:.3f}")
    return SCALE, mods


@torch.no_grad()
def encode(pipe, prompts, device="cuda"):
    emb = {}
    for key, txt in prompts.items():
        pe, ne = pipe.encode_prompt(txt, device, 1, True,
                                    negative_prompt="")
        emb[key] = (pe.to(torch.float16), ne.to(torch.float16))
    return emb


@torch.no_grad()
def unet_out(pipe, x, t, emb):
    return pipe.unet(x, t, encoder_hidden_states=emb).sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--rel", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=float, default=1.0, help="LoRA weight used for g_eff")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    print("loading SD1.5 ...")
    pipe = build_pipe()
    SCALE, mods = install_delta(pipe, a.rank, a.rel, a.seed, dev)

    PROMPTS = {
        "person": "a portrait photo of sks person, close up, natural light",
        "style": "an oil painting of a village at dusk, thick brush strokes",
        "object": "a red ceramic teapot on a wooden table",
    }

    rows = []
    for pname, ptxt in PROMPTS.items():
        emb = encode(pipe, {"c": ptxt})
        ec, eu = emb["c"][0], emb["c"][1]

        g = torch.Generator(device="cpu").manual_seed(1234)
        x = torch.randn(1, 4, 64, 64, generator=g).to(dev, torch.float16)
        pipe.scheduler.set_timesteps(a.steps, device=dev)

        for t in pipe.scheduler.timesteps:
            tt = t.expand(1) if t.dim() == 0 else t
            # base predictions
            SCALE[0] = 0.0
            e_c = unet_out(pipe, x, tt, ec).float()
            e_u = unet_out(pipe, x, tt, eu).float()
            # LoRA predictions (w = 1)
            SCALE[0] = 1.0
            l_c = unet_out(pipe, x, tt, ec).float()
            l_u = unet_out(pipe, x, tt, eu).float()
            SCALE[0] = 0.0

            d_c = (l_c - e_c).flatten()
            d_u = (l_u - e_u).flatten()
            dg = (e_c - e_u).flatten()
            sig = d_c - d_u

            n_dg = dg.norm().item()
            rho = sig.norm().item() / max(n_dg, 1e-9)
            align = torch.nn.functional.cosine_similarity(
                sig[None], dg[None]).item()
            spur = d_u.norm().item() / max(sig.norm().item(), 1e-9)
            rho_e = d_u.norm().item() / max(e_u.flatten().norm().item(), 1e-9)
            proj_u = torch.dot(d_u, dg).item() / max(n_dg, 1e-9)
            g_eff = a.w * (rho * align) + a.w * proj_u  # in units of ||dg||

            rows.append(dict(prompt=pname, step=int(round(float(t))),
                             t_frac=float(t) / 1000.0,
                             rho=rho, align=align, spur=spur, rho_e=rho_e,
                             proj_u=proj_u, g_eff_unit=g_eff,
                             n_dg=n_dg, n_sig=sig.norm().item()))

            # advance the trajectory with BASE CFG (s=7.5) -- x_t shared by all
            s = 7.5
            eps = e_u + s * (e_c - e_u)
            x = pipe.scheduler.step(eps.to(torch.float16),
                                    int(round(float(t))), x,
                                    return_dict=False)[0]

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(OUT, "e0_rho.csv")
    df.to_csv(csv, index=False)

    print("\n" + "=" * 74)
    print("rho(t) = ||Delta(c)-Delta(u)|| / ||e_c - e_u||   (gain inflation)")
    print("-" * 74)
    print(f"{'t':>4s} {'t/1000':>7s} " + "".join(f"{p:>12s}" for p in PROMPTS))
    for t in sorted(df.step.unique(), reverse=True):
        sub = df[df.step == t]
        line = f"{t:4d} {sub.t_frac.iloc[0]:7.3f} "
        for p in PROMPTS:
            r = sub[sub.prompt == p].rho
            line += f"{(r.iloc[0] if len(r) else float('nan')):12.4f}"
        print(line)

    print("\n" + "=" * 74)
    print(f"{'metric':10s} {'min':>9s} {'max':>9s} {'max/min':>9s} {'cv':>7s}")
    print("-" * 74)
    for col in ["rho", "align", "spur", "rho_e"]:
        v = df[col].abs() if col == "align" else df[col]
        mn, mx = v.min(), v.max()
        cv = v.std() / max(abs(v.mean()), 1e-12)
        print(f"{col:10s} {mn:9.4f} {mx:9.4f} {mx/max(mn,1e-12):9.2f} {cv:7.3f}")

    print("\nper-prompt rho range:")
    for p in PROMPTS:
        v = df[df.prompt == p].rho
        print(f"  {p:8s} min {v.min():.4f}  max {v.max():.4f}  "
              f"ratio {v.max()/max(v.min(),1e-12):.2f}  first {v.iloc[0]:.4f}  "
              f"last {v.iloc[-1]:.4f}")
    print("\nwrote", csv)


if __name__ == "__main__":
    main()
