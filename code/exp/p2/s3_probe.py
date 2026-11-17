"""S3/S4 of the B-track probe (paper 2): does the budget split by
`||Delta_i(c)|| / rho_i` beat splitting it equally?

The claim under test (execution plan §3.1):

    at a fixed effective guidance g_eff = s(1 + sum_i w_i rho_i), ranking the
    adapters by eff_i = ||Delta_i(c)|| / rho_i and giving the budget in
    proportion to that ranking beats an equal split.

Four routes, all at the SAME g_eff and the same total w:

    equal    w_i = W0 for all i                            (the incumbent)
    greedy   w_i proportional to eff_i, rescaled so sum_i w_i = n*W0
    random   same total and same g_eff, shares drawn at random  (the control that
             decides whether *the ranking* carries information, or whether any
             unequal split wins by breaking symmetry -- without this route a win
             for greedy cannot be attributed to the sort key)
    single   the whole budget to the adapter with the largest eff_i

A `random` route that ties `greedy` is the honest read: allocation would then be
a lottery, not a measurement.

Every route keeps `sum_i w_i = n*W0`, so all four spend the same "w budget", and
s is solved so that g_eff matches the equal-baseline.  s is the free variable --
that is what "fixed g_eff" means operationally.  s is recorded per row; a route
that needs s outside [3, 16] is flagged, because CFG there is out of the range
this manuscript has ever validated.

Readouts
  clip_t          text alignment (the generator actually following the prompt)
  clip_i_<name>   similarity to that adapter's own concept image prototype
  clip_i_mean     mean of the per-adapter similarities -- the "are both concepts
                  present" summary the gate uses as its DINO-I stand-in
  pix_l2          pixel distance to the equal-baseline image (diagnostic)

Gate (written before the run, execution plan §3.3):
  1. clip_t up >= 2% OR clip_i_mean up >= 3% versus equal, median over cells
  2. the same direction in >= 2/3 of adapter pairings
  3. rho spread >= 2x (measured separately in s2_rho.py)

Run:
    python s3_probe.py --smoke          # 2 cells, ~2 min
    python s3_probe.py --prompts 3 --seeds 3
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SD15 = r"E:\研究\models\sd15.safetensors"
CFG = r"E:\研究\models\sd15_cfg"
LORA_DIR = r"E:\研究\models\lora\trained"
CONCEPT_ROOT = r"E:\研究\models\concepts"
CLIP_DIR = r"E:\研究\models\clip_l14"

S0 = 7.5          # base CFG, the same anchor every L2/Y-series number uses
STEPS = 20


def build_pipe(adapters):
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_single_file(
        SD15, config=CFG, torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    for name in adapters:
        p = os.path.join(LORA_DIR, f"lora_{name}.safetensors")
        pipe.load_lora_weights(p, adapter_name=name)
    return pipe


def build_clip():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_DIR, torch_dtype=torch.float16,
                                      local_files_only=True,
                                      use_safetensors=True).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
    return model, proc


class Probe:
    def __init__(self, adapters):
        self.adapters = adapters
        self.pipe = build_pipe(adapters)
        self.clip, self.proc = build_clip()
        self._proto = {}

    @torch.no_grad()
    def gen(self, prompt, weights, s, seed):
        """weights: dict adapter -> w.  Only listed adapters are active."""
        names = [a for a in self.adapters if a in weights]
        self.pipe.set_adapters(names, [float(weights[a]) for a in names])
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.pipe(prompt=prompt, num_inference_steps=STEPS,
                         guidance_scale=float(s), generator=g,
                         height=512, width=512, return_dict=False)[0][0]

    @torch.no_grad()
    def img_feat(self, imgs):
        ii = self.proc(images=imgs, return_tensors="pt").to("cuda")
        return F.normalize(self.clip.visual_projection(
            self.clip.vision_model(**ii).pooler_output).float(), dim=-1)

    @torch.no_grad()
    def text_feat(self, texts):
        ti = self.proc(text=texts, return_tensors="pt", padding=True,
                       truncation=True).to("cuda")
        return F.normalize(self.clip.text_projection(
            self.clip.text_model(**ti).pooler_output).float(), dim=-1)

    @torch.no_grad()
    def proto(self, name):
        """Mean CLIP image embedding of that adapter's own concept set."""
        if name in self._proto:
            return self._proto[name]
        from PIL import Image
        fs = sorted(glob.glob(os.path.join(CONCEPT_ROOT, name, "*.png")))
        embs = []
        for k in range(0, len(fs), 12):
            batch = [Image.open(f).convert("RGB").resize((224, 224))
                     for f in fs[k:k + 12]]
            embs.append(self.img_feat(batch).mean(0))
        v = torch.stack(embs).mean(0)
        v = F.normalize(v, dim=-1)
        self._proto[name] = v
        return v

    @staticmethod
    def thumb(img, n=64):
        return np.asarray(img.resize((n, n)), np.float32).reshape(-1) / 255.0


def allocate(route, names, W0, rho, eff, rng):
    """w per adapter for a route, plus the s that pins g_eff to the equal baseline.

    Total w is held at n*W0 for every route; s absorbs the difference in
    sum_i w_i rho_i.  Returns (weights, s, flagged).
    """
    n = len(names)
    g_star = S0 * (1.0 + sum(W0 * rho[a] for a in names))

    if route == "equal":
        w = {a: W0 for a in names}
    elif route == "greedy":
        tot = sum(eff[a] for a in names)
        w = {a: n * W0 * eff[a] / tot for a in names}
    elif route == "random":
        p = rng.dirichlet(np.ones(n))
        w = {a: float(n * W0 * pi) for a, pi in zip(names, p)}
    elif route == "single":
        best = max(names, key=lambda a: eff[a])
        w = {best: n * W0}
    else:
        raise ValueError(route)

    denom = 1.0 + sum(w[a] * rho[a] for a in w)
    s = g_star / max(denom, 1e-9)
    flagged = not (3.0 <= s <= 16.0)
    # the "single" route can push one adapter to n*W0; above ~3 the LoRA is past
    # anything this manuscript has validated, so those cells are reported apart
    w_over = max(w.values()) > 3.0
    return w, s, flagged, w_over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--w0", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--rho", type=str, default=os.path.join(HERE, "out_probe"),
                    help="dir holding s2_rho.csv")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "out_probe"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from longrun_p2 import PROMPTS

    # ---- rho and eff per adapter, from S2
    fp = os.path.join(a.rho, "s2_rho.csv")
    if not os.path.exists(fp):
        raise SystemExit(f"run s2_rho.py first; {fp} missing")
    s2 = pd.read_csv(fp)
    rho = s2.groupby("adapter").rho.median().to_dict()
    eff = s2.groupby("adapter").eff.median().to_dict()
    names_all = [x for x in a.lora if x in rho]
    print(f"=== S3/S4 probe: adapters {names_all} ===")
    for k in names_all:
        print(f"    {k}: rho={rho[k]:.4f}  eff={eff[k]:.4f}")

    # ---- cells: every pairing of size 2, plus one of size 3
    cells = [c for c in itertools.combinations(names_all, 2)]
    if len(names_all) >= 3:
        cells.append(tuple(names_all[:3]))
    if a.smoke:
        cells, proms, seeds, w0s = cells[:1], PROMPTS[:1], [11], [1.0]
    else:
        proms = PROMPTS[:a.prompts]
        seeds = [11, 22, 33][:a.seeds]
        w0s = a.w0

    R = Probe(names_all)
    rows, t00 = [], time.time()
    for cell in cells:
        for pid, pclass, text in proms:
            tf = R.text_feat([text])[0]
            protos = {n: R.proto(n) for n in cell}
            for seed in seeds:
                base_img = R.gen(text, {n: w0s[0] for n in cell}, S0, seed)
                tb = R.thumb(base_img)
                for W0 in w0s:
                    for route in ("equal", "greedy", "random", "single"):
                        rng = np.random.default_rng(
                            abs(hash((cell, pid, seed, W0, route))) % (2 ** 31))
                        w, s, flagged, w_over = allocate(route, list(cell), W0, rho, eff,
                                                        rng)
                        t0 = time.time()
                        img = R.gen(text, w, s, seed)
                        dt = time.time() - t0
                        fi = R.img_feat([img])[0]
                        per = {n: float(fi @ protos[n]) for n in cell}
                        rows.append(dict(
                            cell="+".join(cell), n=len(cell), prompt_id=pid,
                            pclass=pclass, seed=seed, W0=W0, route=route,
                            w=json.dumps({k: round(v, 4) for k, v in w.items()}),
                            sum_w=float(sum(w.values())), s=float(s),
                            s_flagged=bool(flagged), w_over=bool(w_over),
                            g_eff=float(s * (1.0 + sum(w[k] * rho[k] for k in w))),
                            clip_t=float(fi @ tf),
                            clip_i_mean=float(np.mean(list(per.values()))),
                            pix_l2=float(np.linalg.norm(R.thumb(img) - tb)),
                            gen_s=dt, **{f"clip_i_{k}": v for k, v in per.items()}))
            print(f"  {('+'.join(cell)):8s} {pid} {len(seeds)} seeds "
                  f"done ({time.time()-t00:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    fp = os.path.join(a.out, "s3_probe.csv")
    df.to_csv(fp, index=False)

    # ---------------------------------------------------------------- readout
    ok = df[~df.s_flagged & ~df.w_over].copy()
    print(f"\ncells {df.groupby(['cell','prompt_id','seed','W0']).ngroups}, "
          f"rows {len(df)}, dropped: s out of range {int(df.s_flagged.sum())}, "
          f"w>3 {int(df.w_over.sum())}")

    print("\n=== per-route means (all cells) ===")
    agg = ok.groupby("route").agg(
        clip_t=("clip_t", "mean"), clip_i=("clip_i_mean", "mean"),
        pix_l2=("pix_l2", "mean"), s=("s", "mean"), g_eff=("g_eff", "mean"),
        gen_s=("gen_s", "mean")).round(4)
    print(agg.to_string())

    eq = ok[ok.route == "equal"].set_index(
        ["cell", "prompt_id", "seed", "W0"])
    rel = []
    for route in ("greedy", "random", "single"):
        sub = ok[ok.route == route].set_index(["cell", "prompt_id", "seed", "W0"])
        j = sub.join(eq, rsuffix="_eq", how="inner").reset_index()
        if not len(j):
            continue
        rel.append(dict(
            route=route, n=len(j),
            d_clip_t=float((j.clip_t / j.clip_t_eq - 1).median() * 100),
            d_clip_i=float((j.clip_i_mean / j.clip_i_mean_eq - 1).median() * 100),
            win_t=float((j.clip_t > j.clip_t_eq).mean()),
            win_i=float((j.clip_i_mean > j.clip_i_mean_eq).mean()),
            per_cell_t=float(j.groupby("cell").apply(
                lambda g: (g.clip_t / g.clip_t_eq - 1).median()).mean() * 100),
            per_cell_i=float(j.groupby("cell").apply(
                lambda g: (g.clip_i_mean / g.clip_i_mean_eq - 1).median()).mean() * 100),
            n_cell_up_t=int((j.groupby("cell").apply(
                lambda g: (g.clip_t / g.clip_t_eq).median() > 1)).sum()),
            n_cell_up_i=int((j.groupby("cell").apply(
                lambda g: (g.clip_i_mean / g.clip_i_mean_eq).median() > 1)).sum()),
            n_cell=int(j.cell.nunique())))
    rl = pd.DataFrame(rel)
    print("\n=== versus equal (median relative change, %) ===")
    print(rl.round(3).to_string(index=False))
    rl.to_csv(os.path.join(a.out, "s3_vs_equal.csv"), index=False)

    # ---------------------------------------------------------------- the gate
    g = rl[rl.route == "greedy"]
    gate = {}
    if len(g):
        g = g.iloc[0]
        c1 = (g.d_clip_t >= 2.0) or (g.d_clip_i >= 3.0)
        c1_metric = "clip_t" if g.d_clip_t >= 2.0 else (
            "clip_i" if g.d_clip_i >= 3.0 else "none")
        c2 = (g.n_cell_up_t / g.n_cell >= 2 / 3) or (g.n_cell_up_i / g.n_cell >= 2 / 3)
        gate = dict(c1=bool(c1), c1_metric=c1_metric, c2=bool(c2),
                    d_clip_t=g.d_clip_t, d_clip_i=g.d_clip_i,
                    n_cell_up_t=g.n_cell_up_t, n_cell_up_i=g.n_cell_up_i,
                    n_cell=g.n_cell)
        print("\n=== decision gate #1, items 1-2 (greedy vs equal) ===")
        print(f"  item1 effect size : clip_t {g.d_clip_t:+.2f}%  "
              f"clip_i {g.d_clip_i:+.2f}%   -> {'PASS' if c1 else 'FAIL'} ({c1_metric})")
        print(f"  item2 consistency : clip_t {g.n_cell_up_t}/{g.n_cell} cells up, "
              f"clip_i {g.n_cell_up_i}/{g.n_cell}   -> {'PASS' if c2 else 'FAIL'}")
        if len(rl) > 1:
            r0 = rl[rl.route == "random"].iloc[0]
            print(f"\n  [control] random split: clip_t {r0.d_clip_t:+.2f}%  "
                  f"clip_i {r0.d_clip_i:+.2f}%  -> "
                  f"{'ranking adds nothing' if abs(r0.d_clip_i) >= abs(g.d_clip_i) * 0.7 else 'ranking carries information'}")

    json.dump(dict(gate=gate, rho=rho, eff=eff, agg=agg.to_dict()),
              open(os.path.join(a.out, "s3_gate.json"), "w"), indent=1, default=float)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
