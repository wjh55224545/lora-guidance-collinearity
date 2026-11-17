"""E3a (paper 2): is g_eff the invariant, or is w*s?

The pivot identity is algebra, not an empirical claim:

    eps_tilde = eps_u + s*(eps_c - eps_u) + w*[d_u + s*(d_c - d_u)]
              = eps_u + s*dg + w*d_u + w*s*sig

with dg = eps_c - eps_u, d_u = LoRA output on the empty prompt, sig = the part
of the LoRA delta that differs between the two prompts.  Because it is algebra,
verifying it proves nothing.  What is NOT algebraic, and is the whole paper, is
the following question:

    does the SCALAR

        g_eff(t) = ||eps_tilde - eps_u|| / ||dg||

    determine eps_tilde well enough that two different (w, s) with the same
    g_eff produce the same image?

If yes, the two knobs are one knob and the controller in I2 is justified.  If
no, g_eff is only a direction and the paper must retreat to "w and s are
coupled" without claiming a sufficient statistic.

Design.  For each prompt we walk the base trajectory once (w=0) and cache
(eps_c, eps_u, x_t).  Then for each w we run two extra UNet calls per step to
get the LoRA outputs, which gives dg / d_u / sig.  From those four vectors
eps_tilde is available in closed form for ANY s, so a full (w,s) grid costs no
further forward passes.  The final latent x_0 is obtained by replaying the
scheduler on the constructed eps, which is arithmetic only.

Three rival routes to "spend the same effort as the reference (w0, s0)":

    geff   pick s so that g_eff(t) matches the reference, step by step
    ws     pick s so that w*s matches the reference      (constant-product rule)
    s      pick s = s0                                   (leave guidance alone)

For each we report the relative L2 and cosine against the reference eps_tilde,
per step, and the relative L2 of the resulting x_0.  The prediction of the
paper is geff << ws; if that fails the honest outcome is to say so.

Self-checks
    (1) d_u must scale linearly in w (this is what E2b/E2c's alpha_t ~ 1 says,
        but it is re-checked here because g_eff's closed form relies on it);
    (2) eps_tilde at (w0, s0) must equal eps_u + s0*dg + w0*d_u + w0*s0*sig
        built from the independently measured d_u, sig -- i.e. the two ways of
        assembling the same quantity must agree to fp16 tolerance.

Run:  python e3a_geff.py --steps 20 --prompts 3
"""
import argparse
import os

import numpy as np
import pandas as pd
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

W0, S0 = 1.0, 7.5           # the reference operating point


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


@torch.no_grad()
def unet_out(pipe, x, t, emb):
    return pipe.unet(x, t, encoder_hidden_states=emb).sample


@torch.no_grad()
def rollout_base(pipe, prompt, steps, s_cfg, seed):
    """w=0 trajectory: the same one E2b/E2c used, so results are comparable."""
    pipe.set_adapters([ADAPTER], [0.0])
    pe, ne = pipe.encode_prompt(prompt, "cuda", 1, True, negative_prompt="")
    pe, ne = pe.to(torch.float16), ne.to(torch.float16)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0 = torch.randn(1, 4, 64, 64, generator=g).to("cuda", torch.float16)
    pipe.scheduler.set_timesteps(steps, device="cuda")
    x = x0.clone()
    cache = []
    for t in pipe.scheduler.timesteps:
        ti = int(round(float(t)))
        tt = t.expand(1) if t.dim() == 0 else t
        e_c = unet_out(pipe, x, tt, pe).float()
        e_u = unet_out(pipe, x, tt, ne).float()
        cache.append(dict(t=ti, x=x.clone(), pe=pe, ne=ne, e_c=e_c, e_u=e_u))
        eps = e_u + s_cfg * (e_c - e_u)
        x = pipe.scheduler.step(eps.to(torch.float16), ti, x,
                                return_dict=False)[0]
    return cache, x0


@torch.no_grad()
def lora_deltas(pipe, cache, w):
    """d_u, dg, sig for every step at adapter scale w."""
    pipe.set_adapters([ADAPTER], [w])
    out = []
    for c in cache:
        tt = torch.tensor([c["t"]], device="cuda")
        l_c = unet_out(pipe, c["x"], tt, c["pe"]).float()
        l_u = unet_out(pipe, c["x"], tt, c["ne"]).float()
        dg = c["e_c"] - c["e_u"]
        d_u = l_u - c["e_u"]
        sig = (l_c - c["e_c"]) - d_u
        out.append(dict(t=c["t"], dg=dg, d_u=d_u, sig=sig))
    return out


def eps_of(c, d, w, s):
    """Pivot identity, assembled from measured pieces."""
    return c["e_u"] + s * d["dg"] + w * d["d_u"] + w * s * d["sig"]


def g_eff(c, d, w, s):
    """Relative guided amplitude: how far eps_tilde moved from eps_u, in units
    of the CFG gap.  This is a definition, not a claim; the claim being tested
    is that it is a sufficient statistic."""
    num = (s * d["dg"] + w * d["d_u"] + w * s * d["sig"]).flatten().norm()
    den = d["dg"].flatten().norm()
    return (num / torch.clamp(den, min=1e-9)).item()


def solve_s_for_geff(c, d, w, target, lo=0.0, hi=40.0, it=60):
    """Monotone in s for fixed w, so bisection is exact enough."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if g_eff(c, d, w, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@torch.no_grad()
def replay_x0(pipe, cache, eps_list):
    """Run the scheduler on a constructed eps sequence -- arithmetic only."""
    x = cache[0]["x"].clone()
    for c, eps in zip(cache, eps_list):
        x = pipe.scheduler.step(eps.to(torch.float16), c["t"], x,
                                return_dict=False)[0]
    return x


def rel_l2(a, b):
    return ((a - b).flatten().norm() / torch.clamp(
        b.flatten().norm(), min=1e-9)).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten()[None], b.flatten()[None]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ws", type=str, default="0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0")
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--grid", type=int, default=1,
                    help="also dump the (w,s) g_eff surface for the contour fig")
    a = ap.parse_args()
    ws = [float(v) for v in a.ws.split(",")]
    if W0 not in ws:
        ws.append(W0)
        ws.sort()
    os.makedirs(OUT, exist_ok=True)

    print("loading SD1.5 + real LoRA ...")
    pipe = build_pipe()

    rows, grid_rows = [], []
    for pname, ptxt in list(PROMPTS.items())[:a.prompts]:
        print(f"\n[{pname}] base rollout ({a.steps} steps, w=0)")
        cache, x0 = rollout_base(pipe, ptxt, a.steps, S0, a.seed)
        # reference: w0, s0
        dref = lora_deltas(pipe, cache, W0)
        ref_eps = [eps_of(c, d, W0, S0) for c, d in zip(cache, dref)]
        ref_g = [g_eff(c, d, W0, S0) for c, d in zip(cache, dref)]
        x0_ref = replay_x0(pipe, cache, ref_eps)
        print(f"   g_eff at ref: min {min(ref_g):.3f} med "
              f"{np.median(ref_g):.3f} max {max(ref_g):.3f}")

        prev_d, prev_w = None, None
        for w in ws:
            d = lora_deltas(pipe, cache, w)
            # self-check (1): the LoRA delta must be linear in w
            if prev_d is not None:
                pw = prev_w
                rat = [(dd["d_u"].flatten().norm()
                        / torch.clamp(dp["d_u"].flatten().norm(),
                                      min=1e-9)).item()
                       for dd, dp in zip(d, prev_d)]
                print(f"   linearity |d_u(w={w})|/|d_u(w={pw})| = "
                      f"median {np.median(rat):.4f} (w ratio {w/pw:.3f})")
            prev_d, prev_w = d, w

            # self-check (2): the two assemblies of the same eps_tilde agree
            direct = [eps_of(c, dd, w, S0) for c, dd in zip(cache, d)]
            alt = []
            for c, dd in zip(cache, d):
                l_u = c["e_u"] + dd["d_u"]
                l_c = c["e_c"] + dd["d_u"] + dd["sig"]
                alt.append(l_u + S0 * (l_c - l_u))
            sc2 = max(rel_l2(x, y) for x, y in zip(direct, alt))
            if abs(w - W0) < 1e-9:
                print(f"   self-check (2) assembly agreement: {sc2:.2e}")

            # the three rival ways to match the reference
            plans = []
            s_geff = [solve_s_for_geff(c, dd, w, tgt)
                      for c, dd, tgt in zip(cache, d, ref_g)]
            plans.append(("geff", s_geff))
            # A fourth route is needed to justify the WORD "per-step" in the
            # title: same g_eff target but a single constant s (the mean of the
            # per-step solution).  If geff_const is close to geff, the per-step
            # part of I2 buys nothing and must be dropped from the claim.
            plans.append(("geff_const", [float(np.mean(s_geff))] * len(cache)))
            plans.append(("ws", [W0 * S0 / w] * len(cache)))
            plans.append(("s", [S0] * len(cache)))

            for name, s_list in plans:
                eps = [eps_of(c, dd, w, s) for c, dd, s in zip(cache, d, s_list)]
                ge = [g_eff(c, dd, w, s) for c, dd, s in zip(cache, d, s_list)]
                x0w = replay_x0(pipe, cache, eps)
                per_step = [rel_l2(e, r) for e, r in zip(eps, ref_eps)]
                per_cos = [cos(e, r) for e, r in zip(eps, ref_eps)]
                rows.append(dict(
                    prompt=pname, w=w, route=name,
                    s_mean=float(np.mean(s_list)),
                    s_min=float(np.min(s_list)), s_max=float(np.max(s_list)),
                    geff_mean=float(np.mean(ge)),
                    geff_relerr=float(np.mean(np.abs(np.array(ge) - np.array(ref_g))
                                              / np.maximum(ref_g, 1e-9))),
                    eps_rel_l2_mean=float(np.mean(per_step)),
                    eps_rel_l2_last=float(per_step[-1]),
                    eps_cos_mean=float(np.mean(per_cos)),
                    x0_rel_l2=rel_l2(x0w, x0_ref),
                ))
                print(f"   w={w:4.2f} {name:5s} s~{np.mean(s_list):6.3f} "
                      f"eps_rel {np.mean(per_step):.4f}  x0_rel "
                      f"{rows[-1]['x0_rel_l2']:.4f}")

            if a.grid:
                for c, dd in zip(cache, d):
                    for s in np.arange(2.0, 13.01, 0.5):
                        grid_rows.append(dict(prompt=pname, w=w, s=float(s),
                                              t=c["t"], t_frac=c["t"] / 1000.0,
                                              g_eff=g_eff(c, dd, w, s)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e3a_routes.csv"), index=False)
    if grid_rows:
        pd.DataFrame(grid_rows).to_csv(
            os.path.join(OUT, "e3a_grid.csv"), index=False)

    ROUTES = ["geff", "geff_const", "ws", "s"]
    print("\n" + "=" * 88)
    print("mean relative L2 vs the reference (w0=%.1f, s0=%.1f), lower = better"
          % (W0, S0))
    print("-" * 88)
    print(f"{'route':11s} {'eps_rel_l2':>11s} {'eps_cos':>9s} {'x0_rel_l2':>11s} "
          f"{'geff_relerr':>12s}   (n)")
    for r in ROUTES:
        v = df[df.route == r]
        print(f"{r:11s} {v.eps_rel_l2_mean.mean():11.4f} "
              f"{v.eps_cos_mean.mean():9.5f} {v.x0_rel_l2.mean():11.4f} "
              f"{v.geff_relerr.mean():12.4f}   ({len(v)})")

    print("\nper-w x0 relative L2 (the number a figure-level claim would rest on)")
    print(f"{'w':>5s} " + "".join(f"{r:>12s}" for r in ROUTES))
    for w in sorted(df.w.unique()):
        v = df[df.w == w]
        print(f"{w:5.2f} " + "".join(
            f"{v[v.route==r].x0_rel_l2.mean():12.4f}" for r in ROUTES))

    print("\nwrote", os.path.join(OUT, "e3a_routes.csv"))


if __name__ == "__main__":
    main()
