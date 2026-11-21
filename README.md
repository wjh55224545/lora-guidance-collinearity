# lora-guidance-collinearity

Reproducibility package for **"Collinearity Between the LoRA Weight Scale and the
Guidance Strength"** (under review, Neurocomputing).

The paper's claim is narrow and testable: in personalized text-to-image generation the
LoRA weight scale $w$ and the classifier-free guidance strength $s$ are **not** two
independent knobs. Substituting the LoRA residual into CFG gives

    g_eff = s (1 + w rho) + w rho_relax

so the whole adapter contribution is multiplied once by $s$. The paper measures a
collinearity index $kappa$ (slope of log s* on log w), shows it is piecewise, and turns
it into a calibration rule that costs zero searches.

## What is and is not in this repository

| Included | Excluded |
|---|---|
| All analysis code (`code/`) | Model weights (SD1.5, SDXL, ImageReward) - licence forbids redistribution |
| All result CSVs (`data/`) | Generated image PNGs (~9 GB) |
| Per-experiment summaries (`experiments/`) | Anything under a `models/` path |
| The three figures (`figures/`) | |
| Manuscript PDF + highlights (`docs/`) | |

Every number in the manuscript can be traced to a CSV in `data/` via the table below.

## Data dictionary: manuscript claim -> file -> script

| Manuscript | Quantity | File | Script |
|---|---|---|---|
| Sec. 5.2 | kappa = 0.211 mean, 0.23 median, 91 fits, 13 unidentifiable | `data/L2_2026-11-06/csv/an_kappa.csv` | `code/exp/p2/l2_analyze.py` |
| Sec. 5.2 | piecewise 0.043 / close to one, p = 8.7e-13 | `data/L2_2026-11-06/csv/an_kappa_piecewise.csv` | `code/exp/p2/l2_analyze.py` |
| Sec. 5.6 | two-adapter additivity, seeds 11/22 | `data/Y3_2026-11-08/csv/` | `code/exp/p2/y3_analyze.py`, `y3_seed_cmp.py` |
| Sec. 5.7 | retraining two LoRAs moves kappa by 0.014 | `data/L2b_kappa_robust_2026-11-07/csv/` | `code/exp/p2/kappa_robust.py` |
| Sec. 5.7 | five subject adapters: 0.105 / 0.079 / 0.204 / 0.045 / 0.048 | `data/T1c_S5/kappa_panel_5.csv` | `code/exp/p2/kappa_panel.py` |
| Sec. 5.7 | training-seed reproducibility, same teapot reseeded | `data/T3_seed_2026-11-17/kappa_seed.csv` | `code/exp/p2/t3_analyze.py` |
| Sec. 5.8 | SDXL 1.0: 0.092-0.159 in 3 of 4 cells | `data/T2_SDXL_2026-11-15/kappa_SDXL.csv` | `code/exp/p2/xl_kappa.py` |
| Sec. 5.9 | calibration vs exhaustive: 93.3% of 95%, 2.33% reversal | `data/Z5_2026-11-11/z5_pairs.csv` | `code/exp/p2/z5_fine.py` |
| Sec. 5.10 | goal-aware band: s* error 4.9% median, 14.3% p90 | `data/GABL_2026-11-14/g2_sstar_prediction.csv` | `code/exp/p2/g_ablate.py` |
| Sec. 4.2 | ImageReward: no preference gain measured | `data/IR_2026-11-14/ir_pairs.csv` | `code/exp/p2/ir_analyze.py` |
| Sec. 5.4 | FID/KID/DIST ladder | `data/DIST_2026-11-14/dist_metrics.csv` | `code/exp/p2/dist_metrics.py` |
| Sec. 5.3 | latent-side kappa and alpha_t | `data/rhoT_2026-11-05/` | `code/exp/p2/e0_rho.py` |

## Reproducing a single number (no GPU needed)

The CSVs are the fitted results, so the headline numbers re-fit in seconds on CPU:

    python code/exp/p2/kappa_fit.py data/T1c_S5/images_S5.csv S5
    python code/exp/p2/kappa_panel.py            # rebuilds the 5-point table

`kappa_panel.py` takes `--out` and reads the per-adapter grids; strength is taken at a
single point (w=1, s=7.5) so that the s axis does not leak into it.

## Regenerating the grids (GPU needed)

An 8 GB consumer GPU is enough. The recipe used throughout:

    # 1. train a subject LoRA: rank-8, 800 steps, 24 self-generated concept images
    python code/exp/p2/b1_subject.py --steps 800 --seed 2024 --tag S \
        --out models/lora/trained/lora_S6.safetensors

    # 2. one 4-prompt x 8w x 9s grid = 288 images (~13 min on a 4060)
    python code/exp/p2/y_gen.py --mode single --tag b1_S6 \
        --lora "S6=models/lora/trained/lora_S6.safetensors" \
        --w 0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0 --s 3,4,5,6,7.5,9,10.5,12,14 \
        --prompts 4 --seeds 11 --outdir out_grid_S6

    # 3. fit kappa
    python code/exp/p2/kappa_fit.py out_grid_S6/images.csv S6

Weights are downloaded by the scripts on first use (SD1.5 base, optionally SDXL 1.0).
SDXL needs the four changes documented in `code/exp/p2/xl_train.py`: dual text encoders,
VAE scaling factor 0.13025, `added_cond_kwargs` (1280-d text embeds + 6 time ids), and
bf16 with gradient checkpointing to fit in 8 GB.

## Numbers we deliberately do not claim

- Calibration is a zero-cost analytic estimate, **not** a pixel optimum.
- kappa is a diagnostic of **one training run on one adapter class**. Retraining one
  subject with a different seed spreads it by 0.177, more than the 0.159 spanned by five
  different subject adapters, so kappa must not be used to compare adapters across
  classes. What is stable across training is rho (three independent adapters within 15%).
- Two metric families disagree: compensation moves pixels and barely moves CLIP. We do
  not write "both metrics agree".
- The multi-LoRA allocation route is closed: three isomorphic LoRAs gave rho within a
  1.15x range, below the 2x criterion.

## Environment

Python 3.10, torch 2.x with CUDA, diffusers, transformers, peft, pandas, scipy,
matplotlib. See `requirements.txt`. Figures follow a black-and-white house style
(`code/figs/_figlib.py`): Times New Roman + STZhongsong, 15.2 cm canvas, 600 dpi.

## Licence

Code MIT. Data and figures CC BY 4.0. The manuscript is (c) the authors.
