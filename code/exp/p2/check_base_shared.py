"""验证 pix_l2_base 的前提：w=0 时不同 LoRA 变体是否生成同一张图。

为什么必须先验证：
  pix_l2_base 之所以能跨变体比较，唯一的理由就是"w=0 时 LoRA 贡献为零，
  于是所有人共用同一张 base 图"。这一步如果成立，五个变体的 pix_l2_base
  就有共同零点；如果不成立（比如某个变体在 w=0 时仍有残余影响），那
  pix_l2_base 就和 pix_l2 一样不可比，这批实验就又白跑了。

判据：同一 (prompt, seed) 下两个变体的 w=0 图，pix L2 ≈ 0 且 CLIP 相似度 ≈ 1。
"""

import numpy as np
import torch

from e2d_variants import Runner, lora_path
from longrun_p2 import PROMPTS, S0

PID, PC, TEXT = PROMPTS[0]
SEED = 11


def one(tag):
    R = Runner(lora_path(tag))
    img = R.gen(TEXT, 0.0, S0, SEED)
    t = R.thumb(img)
    _, f = R.score(img, TEXT)
    del R
    torch.cuda.empty_cache()
    return t, f


def main():
    tags = ["te_only", "full", "no_attn2"]
    out = {t: one(t) for t in tags}
    print()
    print("=== w=0 images across variants (prompt obj00, seed 11, s=7.5) ===")
    base = "te_only"
    for t in tags:
        l2 = float(np.linalg.norm(out[t][0] - out[base][0]))
        cos = float(out[t][1] @ out[base][1])
        print(f"  {t:10s} vs {base}: pix_l2={l2:.6f}   clip={cos:.6f}")
    print()
    d = max(float(np.linalg.norm(out[t][0] - out[base][0])) for t in tags)
    c = min(float(out[t][1] @ out[base][1]) for t in tags)
    ok = d < 0.5 and c > 0.999
    print(f"max pix_l2 = {d:.6f}  (want < 0.5)")
    print(f"min clip   = {c:.6f}  (want > 0.999)")
    print("VERDICT:", "SHARED BASE OK -> pix_l2_base is comparable"
          if ok else "NOT SHARED -> pix_l2_base is NOT comparable, stop and rethink")


if __name__ == "__main__":
    main()
