"""T3: does kappa survive retraining?

The manuscript's softest limitation is that only two independently trained LoRAs
have a full-grid kappa (they differ by 0.014), and a third was measured in latent
space only. This script closes both gaps:

  * same subject, same 24 concept images, only the training seed changes
    (seed 2024 -> lora_S6, seed 2025 -> lora_S7), so the spread across S/S6/S7
    is pure training randomness;
  * lora_C (pencil sketch, third independent style adapter) gets its full grid,
    so the "two complete samples" becomes three.

Run:  python t3_seed.py --stage all
Stages: train6, grid6, train7, grid7, gridC, fit
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable

T3 = os.path.join(ROOT, "results", "论文二_LoRA共线", "T3_seed_2026-11-17")
LORA = os.path.join(ROOT, "models", "lora", "trained")
CONCEPTS = os.path.join(ROOT, "models", "concepts")

W = "0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0"
S = "3,4,5,6,7.5,9,10.5,12,14"


def sh(cmd, log):
    print("+ " + " ".join(cmd), flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def train(tag, seed):
    out = os.path.join(LORA, "lora_%s.safetensors" % tag)
    if os.path.exists(out):
        print("skip train %s (exists)" % tag, flush=True)
        return
    sh([PY, os.path.join(HERE, "b1_subject.py"), "--steps", "800",
        "--seed", str(seed), "--tag", "S", "--out", out],
       os.path.join(T3, "train_%s.log" % tag))


def fit(tag):
    csv = os.path.join(HERE, "out_t3_%s" % tag, "images.csv")
    subprocess.run([PY, os.path.join(HERE, "kappa_fit.py"), csv, tag], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "train6", "grid6", "train7", "grid7",
                             "gridC", "fit"])
    a = ap.parse_args()
    os.makedirs(T3, exist_ok=True)

    # grid() derives the out dir from the tag; keep one name per adapter
    def run_grid(tag, lora_name):
        out = os.path.join(HERE, "out_t3_%s" % tag)
        if os.path.exists(os.path.join(out, "images.csv")):
            rows = sum(1 for _ in open(os.path.join(out, "images.csv"))) - 1
            if rows >= 288:
                print("skip grid %s (%d rows)" % (tag, rows), flush=True)
                return
        sh([PY, os.path.join(HERE, "y_gen.py"), "--mode", "single",
            "--tag", "t3_%s" % tag,
            "--lora", "%s=%s" % (lora_name, os.path.join(LORA, "lora_%s.safetensors" % lora_name)),
            "--w", W, "--s", S, "--prompts", "4", "--seeds", "11",
            "--outdir", out],
           os.path.join(T3, "grid_%s.log" % tag))

    if a.stage in ("all", "train6"):
        train("S6", 2024)
    if a.stage in ("all", "grid6"):
        run_grid("S6", "S6")
    if a.stage in ("all", "train7"):
        train("S7", 2025)
    if a.stage in ("all", "grid7"):
        run_grid("S7", "S7")
    if a.stage in ("all", "gridC"):
        run_grid("C", "C")
    if a.stage in ("all", "fit"):
        for t in ("S6", "S7", "C"):
            try:
                fit(t)
            except subprocess.CalledProcessError:
                print("fit %s failed (grid missing?)" % t, flush=True)

    # collect artifacts next to the summary
    for t in ("S6", "S7", "C"):
        src = os.path.join(HERE, "out_t3_%s" % t, "images.csv")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(T3, "images_%s.csv" % t))
        src = os.path.join(HERE, "out_t3_%s" % t, "images_kappa_fits.csv")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(T3, "kappa_fits_%s.csv" % t))
    print("done", flush=True)


if __name__ == "__main__":
    main()
