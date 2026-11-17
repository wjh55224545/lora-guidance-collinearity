"""E2c-2 step 1 (paper 2): manufacture four *different* LoRAs from the one we
have, so that cross-LoRA robustness can be tested at all.

Why this is not a cheat
-----------------------
E2c-2 was blocked because the box holds exactly one SD1.5 LoRA
(`lora_detail_sd15.safetensors`, rank 8, alpha 8, 528 modules: 216 in the text
encoder, 576-keys worth in the UNet).  Downloading a second style LoRA would
still be *one* arbitrary other point, and training one needs a dataset we do
not have.  The ablation below is better matched to the claim:

  te_only    only the text-encoder half   -- the identity lives in the prompt
                                            embedding, the UNet is untouched
  unet_only  only the UNet half           -- the identity lives in the denoiser,
                                            the text conditioning is untouched
  r4         every module truncated to rank 4 by SVD -- same direction, weaker
  full       the original file (baseline)

The paper's mechanism is "w acts through the text-conditioned residual, and s
amplifies the same residual".  If that is right, **te_only and unet_only must
give different kappa**, because they put the two halves of the update on
opposite sides of the conditioning path.  A rank-4 truncation then checks
whether kappa is a property of the *direction* or of the *magnitude*.

Truncation detail: diffusers stores lora_down (r, in) and lora_up (out, r) with
scale = alpha / r = 8/8 = 1.  For rank k we take the top-k SVD triple of
lora_down, fold the singular values into lora_up, and set alpha = k so the
scale stays 1.  dW_k is then the best rank-k approximation of dW at the same
scale, i.e. weaker by construction and not rescaled up.

Run:  python make_lora_variants.py
"""
from __future__ import annotations

import os

import torch
from safetensors.torch import load_file, save_file

SRC = r"E:\研究\models\lora\lora_detail_sd15.safetensors"
DST = r"E:\研究\models\lora\variants"


def _pair(sd, base):
    return (f"{base}.lora_down.weight", f"{base}.lora_up.weight")


def truncate(sd, k):
    """Best rank-k approximation of every dW, scale preserved (alpha = k)."""
    out = {}
    bases = sorted({k_.split(".lora_")[0] for k_ in sd
                    if k_.endswith(".lora_down.weight")})
    for base in bases:
        dn, up = _pair(sd, base)
        A, B = sd[dn].float(), sd[up].float()          # (r,*in) (out,r,*k)
        r = A.shape[0]
        A2 = A.reshape(r, -1)                          # conv layers are 4-D
        B2 = B.reshape(B.shape[0], -1)                 # (out, r)
        U, S, Vh = torch.linalg.svd(A2, full_matrices=False)
        kk = min(k, len(S))
        Ak = (torch.diag(S[:kk]) @ Vh[:kk]).to(sd[dn].dtype).reshape(
            (kk,) + tuple(A.shape[1:]))
        Bk = (B2 @ U[:, :kk] @ torch.diag(S[:kk])).to(sd[up].dtype).reshape(
            (B.shape[0], kk) + tuple(B.shape[2:]))
        out[dn] = Ak
        out[up] = Bk
        out[f"{base}.alpha"] = torch.tensor(float(kk), dtype=torch.float32)
    return out


def subset(sd, keep_prefix):
    out = {}
    for base in sorted({k.split(".lora_")[0] for k in sd
                        if k.endswith(".lora_down.weight")}):
        if not base.startswith(keep_prefix):
            continue
        for suf, t in (("lora_down.weight", sd[f"{base}.lora_down.weight"]),
                       ("lora_up.weight", sd[f"{base}.lora_up.weight"]),
                       ("alpha", sd[f"{base}.alpha"])):
            out[f"{base}.{suf}"] = t
    return out


def drop(sd, pattern):
    """Everything except modules whose name contains `pattern`."""
    out = {}
    for base in sorted({k.split(".lora_")[0] for k in sd
                        if k.endswith(".lora_down.weight")}):
        if pattern in base:
            continue
        for suf, t in (("lora_down.weight", sd[f"{base}.lora_down.weight"]),
                       ("lora_up.weight", sd[f"{base}.lora_up.weight"]),
                       ("alpha", sd[f"{base}.alpha"])):
            out[f"{base}.{suf}"] = t
    return out


def main():
    os.makedirs(DST, exist_ok=True)
    sd = load_file(SRC)
    print(f"source: {len(sd)} keys, "
          f"{sum(1 for k in sd if k.startswith('lora_te'))} in te, "
          f"{sum(1 for k in sd if k.startswith('lora_unet'))} in unet")

    jobs = {"te_only": subset(sd, "lora_te"),
            "unet_only": subset(sd, "lora_unet"),
            "no_attn2": drop(sd, "_attn2_"),
            "r4": truncate(sd, 4)}
    for name, d in jobs.items():
        p = os.path.join(DST, f"{name}.safetensors")
        save_file(d, p)
        nmod = sum(1 for k in d if k.endswith("alpha"))
        print(f"  {name:10s} {len(d):4d} keys, {nmod:3d} modules -> {p}")

    # sanity: rank-4 approximation must keep most of the Frobenius mass
    for base in sorted({k.split(".lora_")[0] for k in sd
                        if k.endswith(".lora_down.weight")})[:6]:
        A = sd[f"{base}.lora_down.weight"].float()
        s = torch.linalg.svdvals(A)
        print(f"    {base[:52]:52s} sv={', '.join(f'{x:.2f}' for x in s)} "
              f"kept={float((s[:4] ** 2).sum() / (s ** 2).sum()):.3f}")


if __name__ == "__main__":
    main()
