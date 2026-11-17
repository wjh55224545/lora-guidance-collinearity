# -*- coding: utf-8 -*-
"""Z4: APG vs constant compensation on the SAME (w, s) grid -- reviewer P0-4.

Question: when w pushes g_eff beyond the safe band, does APG (Sadat et al.,
ICLR 2025, arXiv:2410.02416 -- down-weight the parallel component of the CFG
update in D_theta space) and does our constant compensation (Eq.(6), s_eff =
S0*(1+W0*rho)/(1+w*rho)) each restore the reference, and are they
complementary?

PRE-REGISTERED criteria (written before running, apply on cells with
w*rho > 0.15 only; below that both are inside noise per 5.3):
  C1 inflation restoration: median pix_l2 (vs ref anchor) and median
     saturated-pixel fraction are LOWER for apg and for comp than for plain.
  C2 division of labour: identity restoration |clip_i_ref - 1| reduction --
     comp >= apg;  saturation suppression -- apg > comp.
  C3 complementarity: comp_apg is not worse than plain on BOTH pix_l2 and
     saturation, and beats min(comp, apg) on at least one.
Any failing criterion is reported as failing, verbatim.

APG config: eta = 0 (kill parallel component), beta = -0.5 (reverse momentum),
norm threshold r = 0 (off).  Projection in D_theta (x0-prediction) space, as
the paper requires; eps<->D conversion uses the DDPM closed form with the
scheduler's alphas_cumprod:
    D = (x - sqrt(1-a)*eps) / sqrt(a)      eps = (x - sqrt(a)*D) / sqrt(1-a)
Sanity gate (part of the protocol): the custom plain-CFG loop must reproduce
the diffusers pipeline within 8/255 RMSE at one (prompt, w, s, seed) before
any APG number is trusted.

Run:  /d/python/python.exe exp/p2/z4_apg_compare.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"E:\研究")
import longrun_p2 as L

OUT = r"E:\研究\results\论文二_LoRA共线\Z4_2026-11-08"
WS = [1.0, 1.5, 2.0]
SS = [5.0, 7.5, 10.0, 15.0]
SEEDS = [11, 22]
NPROMPT = 4
MODES = ["plain", "apg", "comp", "comp_apg"]
ETA, BETA, RNORM = 0.0, -0.5, 0.0


def route_s_comp(w, rho):
    """Eq.(6) at the default-cell effective guidance (same as Z3 cal route)."""
    return L.S0 * (1.0 + L.W0 * rho) / (1.0 + w * rho)


@torch.no_grad()
def gen_custom(pipe, prompt, w, s, seed, mode, steps=L.STEPS):
    pipe.set_adapters(["d"], [float(w)])
    do_apg = mode in ("apg", "comp_apg")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn((1, 4, 64, 64), generator=g, dtype=torch.float16).to("cuda")
    sched = pipe.scheduler
    sched.set_timesteps(steps, device="cuda")
    pe = pipe.encode_prompt(prompt, "cuda", 1, False, None)[0]
    ne = pipe.encode_prompt("", "cuda", 1, False, None)[0]
    mb_avg = None
    alphas = sched.alphas_cumprod.to("cuda")
    for t in sched.timesteps:
        ti = t
        a = alphas[ti].float()
        sa, s1ma = a.sqrt(), (1 - a).sqrt()
        e_c = pipe.unet(x, t, encoder_hidden_states=pe).sample.float()
        e_u = pipe.unet(x, t, encoder_hidden_states=ne).sample.float()
        D_c = (x.float() - s1ma * e_c) / sa
        D_u = (x.float() - s1ma * e_u) / sa
        if not do_apg:
            Dg = D_u + s * (D_c - D_u)          # == plain CFG, in D space
        else:
            diff = D_c - D_u
            if BETA != 0.0:
                mb_avg = BETA * mb_avg + diff if mb_avg is not None else diff
                diff = mb_avg
            if RNORM > 0:
                n = diff.norm(dim=[1, 2, 3], keepdim=True)
                diff = diff * torch.clamp(RNORM / n, max=1.0)
            denom = (D_c * D_c).sum(dim=[1, 2, 3], keepdim=True) + 1e-12
            par = (diff * D_c).sum(dim=[1, 2, 3], keepdim=True) / denom * D_c
            orth = diff - par
            Dg = D_c + (s - 1.0) * (orth + ETA * par)
        eps = (x.float() - sa * Dg) / s1ma       # inverse of the DDPM x0 form
        x = sched.step(eps.half(), ti, x).prev_sample
    img = pipe.vae.decode(x / pipe.vae.config.scaling_factor).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    arr = (img[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def score_batch(clip, proc, imgs, texts):
    ti = proc(text=texts, return_tensors="pt", padding=True,
              truncation=True).to("cuda")
    tf = F.normalize(clip.text_projection(
        clip.text_model(**ti).pooler_output).float(), dim=-1)
    ii = proc(images=imgs, return_tensors="pt").to("cuda")
    imf = F.normalize(clip.visual_projection(
        clip.vision_model(**ii).pooler_output).float(), dim=-1)
    return tf, imf


def thumb64(img):
    a = np.asarray(img.resize((64, 64)), np.float32) / 255.0
    return a.reshape(-1)


def saturated_np(img, thr=0.95):
    a = np.asarray(img.resize((64, 64)), np.float32) / 255.0
    return float((a.max(axis=2) > thr).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    pipe = L.build_pipe()
    clip, proc = L.build_clip()
    prompts = L.PROMPTS[:NPROMPT]

    # ---- sanity gate: custom plain loop vs diffusers pipeline ----
    pid, pclass, text = prompts[0]
    pipe.set_adapters(["d"], [1.0])
    g = torch.Generator(device="cpu").manual_seed(11)
    ref_img = pipe(prompt=text, num_inference_steps=L.STEPS, guidance_scale=7.5,
                   generator=g, height=512, width=512, return_dict=False)[0][0]
    mine = gen_custom(pipe, text, 1.0, 7.5, 11, "plain")
    rmse = float(np.sqrt(np.mean((np.asarray(ref_img, np.float32) -
                                  np.asarray(mine, np.float32)) ** 2)))
    print(f"[sanity] plain-loop vs pipeline RMSE = {rmse:.2f}/255")
    assert rmse < 8.0, "custom loop does not match pipeline"
    json.dump({"sanity_rmse": rmse}, open(os.path.join(OUT, "sanity.json"), "w"))
    torch.cuda.empty_cache()

    rows = []
    csv_path = os.path.join(OUT, "z4_raw.csv")
    for pid, pclass, text in prompts:
        rho = L.RHO.get(pclass, L.RHO_DEFAULT)
        for seed in L.SEEDS if hasattr(L, "SEEDS") else SEEDS:
            b = gen_custom(pipe, text, 0.0, L.S0, seed, "plain")
            r = gen_custom(pipe, text, L.W0, L.S0, seed, "plain")
            tf, imf = score_batch(clip, proc, [b, r], [text, text])
            t_vec, i_ref = tf[0], imf[1]
            thumb_ref = thumb64(r)
            for w in WS:
                for s in SS:
                    for mode in MODES:
                        s_use = {"plain": s, "apg": s,
                                 "comp": route_s_comp(w, rho),
                                 "comp_apg": route_s_comp(w, rho)}[mode]
                        img = gen_custom(pipe, text, w, s_use, seed, mode)
                        tf2, imf2 = score_batch(clip, proc, [img], [text])
                        rows.append(dict(
                            prompt_id=pid, pclass=pclass, seed=seed, w=w, s=s,
                            mode=mode, s_use=s_use, rho=rho, wr=w * rho,
                            clip_t=float(F.cosine_similarity(tf2[0], t_vec, dim=0)),
                            clip_i_ref=float(F.cosine_similarity(imf2[0], i_ref, dim=0)),
                            pix_l2=float(np.mean((thumb64(img) - thumb_ref) ** 2)),
                            sat=saturated_np(img),
                        ))
                        print(f"{pid} sd{seed} w{w} s{s} {mode}: "
                              f"ct={rows[-1]['clip_t']:.3f} ci={rows[-1]['clip_i_ref']:.3f} "
                              f"pl2={rows[-1]['pix_l2']:.4f} sat={rows[-1]['sat']:.4f}",
                              flush=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        report(df, f)


def report(df, f):
    hi = df[df.wr > 0.15].copy()
    print("\n===== Z4 report (pre-registered, cells with w*rho>0.15) =====")
    print("cells:", len(hi))
    g = hi.groupby("mode")[["pix_l2", "sat", "clip_i_ref", "clip_t"]].median()
    print(g.round(4))
    f.write("cells: %d\n%s\n" % (len(hi), g.round(4).to_string()))
    c1 = (g.loc["apg", "pix_l2"] < g.loc["plain", "pix_l2"] and
          g.loc["comp", "pix_l2"] < g.loc["plain", "pix_l2"] and
          g.loc["apg", "sat"] < g.loc["plain", "sat"] and
          g.loc["comp", "sat"] < g.loc["plain", "sat"])
    line1 = "C1 inflation restoration (apg & comp < plain on pix_l2 and sat): " + \
        ("PASS" if c1 else "FAIL")
    idrest = abs(1 - g.loc["comp", "clip_i_ref"]) <= abs(1 - g.loc["apg", "clip_i_ref"])
    sat_sup = g.loc["apg", "sat"] < g.loc["comp", "sat"]
    line2 = ("C2 identity restoration comp>=apg: %s (%.4f vs %.4f); "
             "saturation suppression apg<comp: %s"
             % ("PASS" if idrest else "FAIL", g.loc["comp", "clip_i_ref"],
                g.loc["apg", "clip_i_ref"], "PASS" if sat_sup else "FAIL"))
    c3 = (g.loc["comp_apg", "pix_l2"] <= g.loc["plain", "pix_l2"] and
          g.loc["comp_apg", "sat"] <= g.loc["plain", "sat"])
    line3 = "C3 comp_apg not worse than plain on both: " + ("PASS" if c3 else "FAIL")
    for ln in (line1, line2, line3):
        print(ln); f.write(ln + "\n")


if __name__ == "__main__":
    main()
