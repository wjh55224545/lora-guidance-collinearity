"""Score generated panels with ImageReward (paper 2, extra metric).

Why. Every image metric in the paper is either a distance to a reference image
(pixel L2, DINO) or a text-image similarity (CLIP-T).  A referee for a
generation venue will ask for a preference model: does the compensated output
look *better*, not merely *closer*?  CLIP-T saturates here, so the question is
open.

What this measures. ImageReward is trained on human preferences, so it answers
"which of two images a person would pick", not "which is closer to the w=1
reference".  Both answers are useful and they need not agree: if compensation
lowers the preference score, that is a real limitation and gets reported.

Run (needs the weights at models/ImageReward.pt):
    python exp/p2/ir_score.py --images-csv exp/p2/out_ir/images.csv \
        --out results/论文二_LoRA共线/IR_2026-11-14
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from _tool import ir_compat  # noqa: E402

WEIGHTS = r"E:\研究\models\ImageReward.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ir_compat.patch()
    import ImageReward as RM  # noqa: E402
    import torch  # noqa: E402

    model = RM.load(a.weights, device="cuda")
    d = pd.read_csv(a.images_csv)
    if a.limit:
        d = d.head(a.limit)
    print(f"{len(d)} images", flush=True)

    scores = []
    for i, r in d.iterrows():
        if not isinstance(r.img_path, str) or not os.path.exists(r.img_path):
            scores.append(float("nan"))
            continue
        s = model.score(r.prompt, [r.img_path])
        scores.append(float(s[0]) if hasattr(s, "__len__") else float(s))
        if (len(scores)) % 20 == 0:
            print(f"  {len(scores)}/{len(d)}", flush=True)
    d["image_reward"] = scores
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "ir_scores.csv")
    d.to_csv(p, index=False)

    ok = d.dropna(subset=["image_reward"])
    print("\n=== ImageReward by (w, s) ===")
    print(ok.pivot_table(index="w", columns="s", values="image_reward",
                         aggfunc="mean").round(3).to_string())
    print("\nwrote", p)


if __name__ == "__main__":
    main()
