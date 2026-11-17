"""Second subject run: same teapot, twice the steps (1600).

Why: the subject adapter of b1_subject.py gives kappa=0.036, five times below the
reference adapter. The obvious objection is that the adapter is simply undertrained
(24 images, 800 steps), so its guidance channel is weak rather than uncoupled. This run
isolates strength from adapter type: same images, same everything, 1600 steps.

Run: python b1_subject_long.py
Out: models/lora/trained/lora_S2.safetensors
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import y1_train as Y
import b1_subject as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--out", default=os.path.join(Y.LORA_OUT, "lora_S2.safetensors"))
    a = ap.parse_args()
    items = B.build_subject_images()          # same 24 images, cached
    lat, emb = Y.precompute(items)
    Y.train(lat, emb, a.steps, 8, 1e-4, a.out)
    print("done", a.out)


if __name__ == "__main__":
    main()
