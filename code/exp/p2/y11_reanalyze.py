# -*- coding: utf-8 -*-
"""Re-run Y11 analysis from the 288 saved images (analysis bugfix, no GPU)."""
import os, sys
import numpy as np
from PIL import Image

OUT = r"E:/研究/results/论文二_LoRA共线/Y11_2026-11-08"
W_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
S_GRID = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]
SEEDS = [11, 22]
PROMPTS = [("obj00", "a red ceramic teapot on a wooden table"),
           ("obj03", "a wooden chair in a bright room")]

def vec(p):
    return np.asarray(Image.open(p).convert("L").resize((64, 64)),
                      np.float32).reshape(-1) / 255.0

rows = []
for pid, text in PROMPTS:
    for sd in SEEDS:
        ref = vec(os.path.join(OUT, "img", f"{pid}_sd{sd}_w1.0_s7.5.png"))
        b = None  # w=0 was not saved in the grid; inertness via w=0.2 spread
        for w in W_GRID:
            for s in S_GRID:
                p = os.path.join(OUT, "img", f"{pid}_sd{sd}_w{w}_s{s}.png")
                if not os.path.exists(p):
                    continue
                v = vec(p)
                rows.append(dict(prompt_id=pid, pclass="object", seed=sd,
                                 w=w, s=s, pix_l2=float(np.linalg.norm(v - ref)),
                                 img_path=p))
print(f"{len(rows)} images scored")

sys.path.insert(0, r"E:/研究/exp/p2")
import y11_sd21 as Y
Y.OUT = OUT
Y.analyse(rows)
