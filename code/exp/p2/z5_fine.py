"""Z5 (paper 2): does calibration still win when the grid is fine?

The soft spot this closes: in the pixel-objective check (§5.9 of v2) not a single
calibrated s fell on the 9-point grid (0/564).  A reviewer will ask the obvious
question -- "you win on a 9-point grid because the optimum sits between grid
points; refine the grid and the win disappears".  No existing data can answer
that, because off-grid s was never generated.

Design frozen in the manuscript log §8.37, before any image of this run exists:

  cells     24 (prompt, seed): obj00..obj07 x seed 11/22/33
  w         {0.5, 1.5, 2.0}   -- underestimate / overestimate / strong inflation
  s         5.0 .. 15.0 step 0.5  (21 points, 2.3x denser than the 9-point grid)
  scale     24 * 3 * 21 = 1512 images
  metrics   pix_l2, clip_i_ref, clip_t ; gate tau = this cell's clip_t at (w=1, s=7.5)
  compare   fine-grid pixel oracle / fine-grid clip champion / calibrated exact
            (off-grid, s from eq. 6) / fixed s

Pre-registered criteria, NOT to be edited after the run:
  C1  |median pairwise relative difference| between the fine-grid pixel oracle and
      calibrated exact <= 3%  -> "calibration == the best any finite grid can do,
      the advantage was resolution"
      > 3%                  -> "calibration beats any finite grid"
  C2  median |s_champion - s_cal| <= 0.5 (one new grid step) -> eq.(6) points at
      the right place
  C3  median clip_i_ref of the fine-grid champion >= that of the 9-point champion
      -> refining must not cost fidelity

Run:
    python z5_fine.py --smoke      # 1 cell, 3 w, 5 s -> ~1 min
    python z5_fine.py              # full, ~1.8 h, resumable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from longrun_p2 import PROMPTS, RHO, RHO_DEFAULT, S0, W0  # noqa: E402

OUT = r"E:\研究\results\论文二_LoRA共线\Z5_2026-11-11"
W_LIST = [0.5, 1.5, 2.0]
S_FINE = [round(5.0 + 0.5 * i, 1) for i in range(21)]   # 5.0 .. 15.0
SEEDS = [11, 22, 33]


def s_cal(w, rho):
    """Eq. (6) of the manuscript: the constant-compensation branch."""
    return S0 * (1.0 + W0 * rho) / (1.0 + w * rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=str, default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from longrun_p2 import Runner

    proms = PROMPTS[:a.prompts]
    seeds = SEEDS[:a.seeds]
    w_list = [1.5] if a.smoke else W_LIST
    s_list = [5.0, 7.5, 10.0] if a.smoke else S_FINE
    if a.smoke:
        # smoke must exercise the SUMMARY path too -- a bug in the merge against
        # L2_images.csv would otherwise surface only after the full 1.8 h run.
        proms, seeds = PROMPTS[:1], SEEDS[:1]
    imgdir = os.path.join(a.out, "img")
    os.makedirs(imgdir, exist_ok=True)

    # ---- work list
    cells = [(pid, pclass, text, seed) for pid, pclass, text in proms
             for seed in seeds]
    jobs = [(c, w, s) for c in cells for w in w_list for s in s_list]
    gate_jobs = [(c, 1.0, 7.5) for c in cells]          # the tau anchor
    state_p = os.path.join(a.out, "z5_state.json")
    done = set(json.load(open(state_p))) if os.path.exists(state_p) else set()
    todo = [j for j in jobs + gate_jobs
            if f"{j[0][0]}|{j[0][3]}|{j[1]}|{j[2]}" not in done]
    print(f"=== Z5 fine grid: {len(cells)} cells, {len(w_list)} w, "
          f"{len(s_list)} s -> {len(jobs)} images ({len(todo)} to do) ===",
          flush=True)

    R = Runner()
    rows, t00, nproc = [], time.time(), 0
    anchor = {}
    for (pid, pclass, text, seed), w, s in todo:
        key = f"{pid}|{seed}|{w}|{s}"
        anchor.setdefault((pid, seed), R.anchors(pid, text, seed))
        an = anchor[(pid, seed)]
        t0 = time.time()
        im = R.gen(text, w, s, seed)
        dt = time.time() - t0
        tf, imf = R.score([im], [text])
        rows.append(dict(prompt_id=pid, pclass=pclass, seed=seed,
                         w=float(w), s=float(s),
                         clip_t=float(tf[0] @ imf[0]),
                         clip_i_ref=float(imf[0] @ an["i_ref"]),
                         pix_l2=float(np.linalg.norm(
                             R.thumb_np(im) - an["thumb_ref"])),
                         gen_s=dt))
        done.add(key)
        nproc += 1
        # NOTE: the flush condition must count jobs processed, NOT len(rows) --
        # rows is reset to [] on every flush, so "len(rows) == len(todo)" is only
        # ever true when no flush has happened yet.  With the old condition the
        # tail (1536 % 100 = 36 jobs, i.e. every tau anchor) was silently dropped
        # and only reached the CSV through the rows still sitting in memory.
        if nproc % 100 == 0 or nproc == len(todo):
            json.dump(sorted(done), open(state_p, "w"))
            fp = os.path.join(a.out, "z5_raw.csv")
            old = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()
            pd.concat([old, pd.DataFrame(rows)], ignore_index=True) \
                .drop_duplicates(subset=["prompt_id", "seed", "w", "s"],
                                 keep="last").to_csv(fp, index=False)
            rows = []
            el = time.time() - t00
            print(f"  {len(done)}/{len(jobs)} done  {el/60:.1f} min  "
                  f"eta {(len(jobs)-len(done))*dt/60:.1f} min", flush=True)

    fp = os.path.join(a.out, "z5_raw.csv")
    df = pd.read_csv(fp)
    if a.smoke:
        print("\n[SMOKE] 1 cell, 1 w, 3 s -- this only proves the code paths run;"
              " the three criteria are meaningless at this size.")

    # ------------------------------------------------------------- tau anchors
    tau = df[np.isclose(df.w, 1.0) & np.isclose(df.s, 7.5)] \
        .set_index(["prompt_id", "seed"]).clip_t
    df = df[~np.isclose(df.w, 1.0)].copy()
    df["tau"] = tau.reindex(
        pd.MultiIndex.from_frame(df[["prompt_id", "seed"]])).to_numpy()
    miss = df.tau.isna()
    if miss.any():
        cells_miss = df.loc[miss, ["prompt_id", "seed"]].drop_duplicates()
        print(f"  [warn] {len(cells_miss)} cells have no tau anchor "
              f"(w=1,s=7.5); dropping them:\n{cells_miss.to_string(index=False)}",
              flush=True)
        df = df[~miss].copy()
    if df.empty:
        print("nothing left to analyse -- rerun to generate the missing images")
        return

    # ------------------------------------------- per (cell, w): champions + cal
    recs = []
    d9 = pd.read_csv(os.path.join(HERE, "out_long2", "L2_images.csv"))
    d9 = d9[(d9.segment == "G") & (d9.route == "grid")]
    for (pid, seed), sub in df.groupby(["prompt_id", "seed"]):
        pclass = sub.pclass.iloc[0]
        rho = RHO.get(pclass, RHO_DEFAULT)
        for w, sw in sub.groupby("w"):
            gated = sw[sw.clip_t >= sw.tau.iloc[0]]
            pool = gated if len(gated) else sw
            pix = pool.loc[pool.pix_l2.idxmin()]
            fid = pool.loc[pool.clip_i_ref.idxmax()]
            sc = s_cal(w, rho)
            hit = sw.iloc[(sw.s - sc).abs().argmin()]
            # 9-point grid champion from the L2 run, same cell and w
            old = d9[(d9.prompt_id == pid) & (d9.seed == seed)
                     & np.isclose(d9.w, w)]
            recs.append(dict(
                prompt_id=pid, pclass=pclass, seed=int(seed), w=float(w),
                rho=rho, tau=float(sw.tau.iloc[0]), n_s=len(sw),
                gate_ok=bool(len(gated)),
                fine_pix=float(pix.pix_l2), fine_pix_s=float(pix.s),
                fine_fid=float(fid.clip_i_ref), fine_fid_s=float(fid.s),
                cal_exact_pix=float(hit.pix_l2), cal_exact_fid=float(hit.clip_i_ref),
                cal_exact_s=float(hit.s), cal_s=float(sc),
                g9_pix=float(old.pix_l2.min()) if len(old) else np.nan,
                g9_fid=float(old.clip_i_ref.max()) if len(old) else np.nan,
                g9_n=int(len(old))))
    r = pd.DataFrame(recs)
    r["d_pix"] = (r.fine_pix / r.cal_exact_pix - 1) * 100
    r["d_s"] = (r.fine_pix_s - r.cal_s).abs()
    r.to_csv(os.path.join(a.out, "z5_pairs.csv"), index=False)

    print("\n=== C1: fine-grid pixel oracle vs calibrated exact ===")
    print(f"  median fine_pix      {r.fine_pix.median():.4f}")
    print(f"  median cal_exact_pix {r.cal_exact_pix.median():.4f}")
    print(f"  median relative diff {r.d_pix.median():+.2f}%   "
          f"(cells {int((r.d_pix>0).sum())}/{len(r)} where the oracle is worse)")
    c1 = bool(r.d_pix.median() <= 3.0)

    print("\n=== C2: does eq.(6) point at the champion? ===")
    print(f"  median |s_champion - s_cal| = {r.d_s.median():.3f} (limit 0.5)")
    c2 = bool(r.d_s.median() <= 0.5)

    print("\n=== C3: refining must not cost fidelity ===")
    rr = r.dropna(subset=["g9_fid"])
    print(f"  fine-grid champion clip_i_ref {rr.fine_fid.median():.4f} vs "
          f"9-point {rr.g9_fid.median():.4f}")
    c3 = bool(rr.fine_fid.median() >= rr.g9_fid.median())

    print(f"\n=== verdict === C1 {'PASS' if c1 else 'FAIL'} | "
          f"C2 {'PASS' if c2 else 'FAIL'} | C3 {'PASS' if c3 else 'FAIL'}")
    if c1:
        print("  -> calibration is no better than a fine grid: the §5.9 advantage"
              " was grid resolution.  Must be written that way.")
    else:
        print("  -> calibration beats any finite grid: the advantage survives "
              "refinement.")

    json.dump(dict(C1=c1, C2=c2, C3=c3,
                   d_pix_med=float(r.d_pix.median()),
                   d_s_med=float(r.d_s.median()),
                   fine_pix=float(r.fine_pix.median()),
                   cal_pix=float(r.cal_exact_pix.median()),
                   n_pairs=len(r)), open(os.path.join(a.out, "z5_gate.json"), "w"),
              indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
