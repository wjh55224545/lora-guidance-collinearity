"""E2b (paper 2): is the rho_t *shape* a property of the architecture, or an
artifact of how big we made the synthetic LoRA?

E0/E2 measured rho_t at a single perturbation size (rel = 5% of ||W||_F) and
found a 2.8-3.4x swing, with person/style peaking mid-trajectory and object
monotone.  If that shape is real it must survive a 5x change in rel.
If the shape moves with rel, then what we measured was numerical noise and
the paper's I1 has to be re-derived.

Design note -- the base trajectory is shared.  We first roll out 20 DDIM steps
with the *base* model only (SCALE=0) and cache every x_t together with the
base predictions e_c / e_u.  All rel settings are then evaluated on exactly
the same x_t, so the only thing that changes between curves is the LoRA size.
This removes trajectory drift as a confound, which a naive "rerun per rel"
would not.

The delta is installed *normalised* once (||dW||_F = ||W||_F) and the actual
size is applied through the forward hook scale, so switching rel costs nothing
and does not re-randomise the direction.  Same seed -> same subspace, only the
amplitude differs.  That is what makes the three curves comparable.

Run:  python e2b_rel.py
"""
import argparse
import os

import numpy as np
import torch

SD15 = r"E:\研究\models\sd15.safetensors"
CFG = r"E:\研究\models\sd15_cfg"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

TARGETS = ("attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0")

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
    return pipe.to("cuda")


def install_unit_delta(pipe, rank=4, seed=0, device="cuda"):
    """Install a rank-`rank` delta normalised so ||dW||_F == ||W||_F.

    The applied size is then exactly SCALE[0] * ||W||_F, i.e. SCALE[0] is rel.
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
        B *= (W.norm() / max(dW.norm().item(), 1e-9))   # rel == 1
        mod._A = A.to(device)
        mod._B = B.to(device)
        mod._h = mod.register_forward_hook(hook)
        mods.append(name)
        n += 1
    print(f"  delta on {n} cross-attn projections (rank {rank}, unit norm)")
    return SCALE, mods


@torch.no_grad()
def encode(pipe, prompts, device="cuda"):
    emb = {}
    for key, txt in prompts.items():
        pe, ne = pipe.encode_prompt(txt, device, 1, True, negative_prompt="")
        emb[key] = (pe.to(torch.float16), ne.to(torch.float16))
    return emb


@torch.no_grad()
def unet_out(pipe, x, t, emb):
    return pipe.unet(x, t, encoder_hidden_states=emb).sample


@torch.no_grad()
def rollout(pipe, prompt, steps, s_cfg=7.5, seed=1234, device="cuda"):
    """Roll out the BASE trajectory, cache x_t and base e_c / e_u at each step."""
    emb = encode(pipe, {"c": prompt}, device)
    ec, eu = emb["c"][0], emb["c"][1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 4, 64, 64, generator=g).to(device, torch.float16)
    pipe.scheduler.set_timesteps(steps, device=device)

    cache = []
    for t in pipe.scheduler.timesteps:
        tt = t.expand(1) if t.dim() == 0 else t
        e_c = unet_out(pipe, x, tt, ec).float()
        e_u = unet_out(pipe, x, tt, eu).float()
        cache.append((int(round(float(t))), x.clone(), ec, eu, e_c, e_u))
        eps = e_u + s_cfg * (e_c - e_u)
        x = pipe.scheduler.step(eps.to(torch.float16), int(round(float(t))), x,
                                return_dict=False)[0]
    return cache


@torch.no_grad()
def measure(pipe, SCALE, cache, rel):
    """Evaluate the rho family at one rel, on the cached base trajectory."""
    SCALE[0] = rel
    out = []
    for t_int, x, ec, eu, e_c, e_u in cache:
        tt = torch.tensor([t_int], device=x.device)
        l_c = unet_out(pipe, x, tt, ec).float()
        l_u = unet_out(pipe, x, tt, eu).float()
        d_c = (l_c - e_c).flatten()
        d_u = (l_u - e_u).flatten()
        dg = (e_c - e_u).flatten()
        sig = d_c - d_u
        n_dg = dg.norm().item()
        n_sig = sig.norm().item()
        out.append(dict(
            rel=rel, step=t_int, t_frac=t_int / 1000.0,
            rho=n_sig / max(n_dg, 1e-9),
            align=torch.nn.functional.cosine_similarity(sig[None], dg[None]).item(),
            spur=d_u.norm().item() / max(n_sig, 1e-9),
            rho_e=d_u.norm().item() / max(e_u.flatten().norm().item(), 1e-9),
            n_dg=n_dg, n_sig=n_sig,
        ))
    SCALE[0] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--rels", type=str, default="0.02,0.05,0.10")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rels = [float(x) for x in a.rels.split(",")]
    os.makedirs(OUT, exist_ok=True)

    print("loading SD1.5 ...")
    pipe = build_pipe()
    SCALE, _ = install_unit_delta(pipe, a.rank, a.seed)

    rows = []
    for pname, ptxt in PROMPTS.items():
        print(f"\n[{pname}] rolling out base trajectory ({a.steps} steps) ...")
        cache = rollout(pipe, ptxt, a.steps)
        for rel in rels:
            for r in measure(pipe, SCALE, cache, rel):
                r["prompt"] = pname
                rows.append(r)
            print(f"   rel={rel:.2f} done")

    import pandas as pd
    df = pd.DataFrame(rows)[["prompt", "rel", "step", "t_frac", "rho",
                             "align", "spur", "rho_e", "n_dg", "n_sig"]]
    csv = os.path.join(OUT, "e2b_rel.csv")
    df.to_csv(csv, index=False)

    print("\n" + "=" * 78)
    print("per (prompt, rel):  rho range and peak location")
    print("-" * 78)
    print(f"{'prompt':8s} {'rel':>5s} {'min':>8s} {'max':>8s} {'max/min':>8s} "
          f"{'peak t':>7s} {'cv':>6s}")
    for p in PROMPTS:
        for rel in rels:
            v = df[(df.prompt == p) & (df.rel == rel)].sort_values("t_frac")
            if not len(v):
                continue
            peak = v.loc[v.rho.idxmax(), "t_frac"]
            print(f"{p:8s} {rel:5.2f} {v.rho.min():8.4f} {v.rho.max():8.4f} "
                  f"{v.rho.max()/max(v.rho.min(),1e-12):8.2f} {peak:7.3f} "
                  f"{v.rho.std()/v.rho.mean():6.3f}")

    print("\n" + "=" * 78)
    print("shape stability across rel  (is the curve the same after normalising?)")
    print("-" * 78)
    from scipy import stats
    for p in PROMPTS:
        sub = df[df.prompt == p]
        curves = {}
        for rel in rels:
            v = sub[sub.rel == rel].sort_values("t_frac").rho.values
            curves[rel] = v / v.mean()          # scale-free shape
        print(f"\n[{p}]")
        base = rels[0]
        for rel in rels[1:]:
            sh = curves[rel]
            r_pear = np.corrcoef(curves[base], sh)[0, 1]
            r_spear = stats.spearmanr(curves[base], sh).statistic
            print(f"   rel {base:.2f} vs {rel:.2f}:  pearson {r_pear:.4f}   "
                  f"spearman {r_spear:.4f}   max|diff| {np.abs(curves[base]-sh).max():.4f}")
        allc = np.vstack([curves[r] for r in rels])
        print(f"   across-rel spread of normalised curve: max {allc.max(axis=0).max()-allc.min(axis=0).min():.4f}, "
              f"mean half-range {np.mean((allc.max(axis=0)-allc.min(axis=0))/2):.4f}")

    print("\nwrote", csv)


if __name__ == "__main__":
    main()
