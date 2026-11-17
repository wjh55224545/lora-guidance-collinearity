"""S2 of the B-track probe (paper 2): is there any heterogeneity to allocate?

The probe asks whether splitting a fixed guidance budget across several LoRAs by
`||Delta_i(c)|| / rho_i` beats splitting it equally.  That question is moot if
every adapter has the same `rho_i`: with rho_i identical, `g_eff = s(1 + sum_i
w_i rho_i)` depends only on `sum_i w_i`, so every allocation with the same total
w is the *same* configuration.  Decision gate #1 therefore has a hard
precondition -- the spread of rho_i must be >= 2x -- and this script measures it
before any image is generated.

Definitions deliberately copied from `e0_rho.py` so the numbers land on the same
scale as Table 3 of the manuscript (kappa ~ 0.211, rho reported per prompt):

    dg_i   = e_c - e_u                         base CFG direction at x_t
    D_i(c) = UNet_i(x_t, t, c) - UNet_0(x_t, t, c)     adapter i, conditional
    D_i(u) = UNet_i(x_t, t, "") - UNet_0(x_t, t, "")   adapter i, unconditional
    rho_i  = ||D_i(c) - D_i(u)|| / ||dg_i||    gain inflation of adapter i
    align  = cos(D_i(c) - D_i(u), dg_i)        is it acting as gain
    mag_i  = ||D_i(c)||                        raw displacement size at w=1
    eff_i  = mag_i / rho_i                     displacement per unit of g_eff

`eff_i` is the greedy sort key.  A request that buys more displacement per unit
of the shared budget should get more of it -- that is the entire hypothesis.

The trajectory is the *base* model's (s=7.5), not each adapter's, so every
adapter is measured at identical x_t.  Measuring each on its own trajectory would
confound adapter strength with the states it happens to visit.

What would kill the probe
  * rho spread < 2x  -> allocation is a no-op, report and stop (gate #1 item 3)
  * rho drifting with w -> the linear form g_eff = s(1 + sum w_i rho_i) is wrong
    and every route comparison built on it is measuring something else

Run:
    python s2_rho.py --lora A B C --prompts 2 --steps 20
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SD15 = r"E:\研究\models\sd15.safetensors"
CFG = r"E:\研究\models\sd15_cfg"
LORA_DIR = r"E:\研究\models\lora\trained"


def build_pipe():
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_single_file(
        SD15, config=CFG, torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


@torch.no_grad()
def unet_out(pipe, x, t, emb):
    return pipe.unet(x, t, encoder_hidden_states=emb).sample


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--prompts", type=int, default=2,
                    help="how many of longrun_p2.PROMPTS to measure on")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ws", type=float, nargs="+", default=[0.6, 1.0, 1.5])
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "out_probe"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from longrun_p2 import PROMPTS

    prompts = PROMPTS[:a.prompts]
    print(f"=== S2: rho heterogeneity over {a.lora} ===", flush=True)
    print(f"    prompts: {[p[0] for p in prompts]}", flush=True)

    pipe = build_pipe()
    avail = []
    for name in a.lora:
        p = os.path.join(LORA_DIR, f"lora_{name}.safetensors")
        if not os.path.exists(p):
            print(f"  !! missing {p}, skipped", flush=True)
            continue
        pipe.load_lora_weights(p, adapter_name=name)
        avail.append(name)
    print(f"  loaded adapters: {avail}", flush=True)

    rows = []
    t00 = time.time()
    for pid, pclass, text in prompts:
        pe, ne = pipe.encode_prompt(text, "cuda", 1, True, negative_prompt="")
        pe, ne = pe.to(torch.float16), ne.to(torch.float16)

        g = torch.Generator(device="cpu").manual_seed(1234)
        x = torch.randn(1, 4, 64, 64, generator=g).to("cuda", torch.float16)
        pipe.scheduler.set_timesteps(a.steps, device="cuda")

        for t in pipe.scheduler.timesteps:
            tt = t.expand(1) if t.dim() == 0 else t
            # base predictions: all adapters off
            pipe.set_adapters([avail[0]], [0.0])
            e_c = unet_out(pipe, x, tt, pe).float()
            e_u = unet_out(pipe, x, tt, ne).float()
            dg = (e_c - e_u).flatten()
            n_dg = dg.norm().item()

            for name in avail:
                for w in a.ws:
                    pipe.set_adapters([name], [float(w)])
                    l_c = unet_out(pipe, x, tt, pe).float()
                    l_u = unet_out(pipe, x, tt, ne).float()
                    D_c = (l_c - e_c).flatten() / w      # per unit w
                    D_u = (l_u - e_u).flatten() / w
                    sig = D_c - D_u
                    rho = sig.norm().item() / max(n_dg, 1e-9)
                    mag = D_c.norm().item()
                    align = torch.nn.functional.cosine_similarity(
                        sig[None], dg[None]).item()
                    rows.append(dict(
                        prompt_id=pid, pclass=pclass, adapter=name,
                        w=w, step=int(round(float(t))),
                        t_frac=float(t) / 1000.0, rho=rho, align=align,
                        mag=mag, eff=mag / max(rho, 1e-9),
                        spur=D_u.norm().item() / max(sig.norm().item(), 1e-9)))

            # advance on the BASE trajectory (s=7.5), shared by all adapters
            pipe.set_adapters([avail[0]], [0.0])
            eps = e_u + 7.5 * (e_c - e_u)
            x = pipe.scheduler.step(eps.to(torch.float16),
                                    int(round(float(t))), x,
                                    return_dict=False)[0].to(torch.float16)
        print(f"  {pid} done ({time.time()-t00:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    fp = os.path.join(a.out, "s2_rho.csv")
    df.to_csv(fp, index=False)

    # ---- linearity check: rho must not move with w (D was already divided by w)
    print("\n=== rho vs w (should be flat; D divided by w) ===")
    piv = df.pivot_table(index="adapter", columns="w", values="rho", aggfunc="median")
    print(piv.round(4).to_string())

    # ---- the gate
    print("\n=== per-adapter summary (median over prompts x timesteps x w) ===")
    sm = df.groupby("adapter").agg(
        rho=("rho", "median"), rho_lo=("rho", lambda v: v.quantile(.25)),
        rho_hi=("rho", lambda v: v.quantile(.75)),
        align=("align", "median"), mag=("mag", "median"),
        eff=("eff", "median")).round(4)
    print(sm.to_string())

    rho_med = sm["rho"]
    spread = float(rho_med.max() / max(rho_med.min(), 1e-9))
    eff_spread = float(sm["eff"].max() / max(sm["eff"].min(), 1e-9))
    ok = spread >= 2.0

    print("\n=== decision gate #1, item 3 ===")
    print(f"  rho spread       = {spread:.3f}x   (need >= 2.0)  -> {'PASS' if ok else 'FAIL'}")
    print(f"  eff spread       = {eff_spread:.3f}x  (sort key, not a gate)")
    for ad in sm.index:
        print(f"    {ad}: rho={rho_med[ad]:.4f}  eff={sm.loc[ad,'eff']:.4f}  "
              f"rank={int(sm['eff'].rank(ascending=False)[ad])}")

    json.dump(dict(spread=spread, eff_spread=eff_spread, gate_pass=ok,
                   rho=rho_med.to_dict(), eff=sm["eff"].to_dict(),
                   n_lora=len(avail), prompts=[p[0] for p in prompts]),
              open(os.path.join(a.out, "s2_gate.json"), "w"), indent=1)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
