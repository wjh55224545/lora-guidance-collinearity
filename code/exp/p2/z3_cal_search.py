"""Z3 (paper 2): calibration saves search -- P0-3 of the journal-fit review.

Deployment question: the user fixes a weight scale w (identity strength) and
must pick the guidance s.  The folklore procedure is a per-w s-grid search
(9 candidates); Eq.(6) prescribes s directly from ONE rho measurement per
adapter:

    s_cal(w) = S0 * (1 + W0 * rho) / (1 + w * rho)

Cost accounting:
  exhaustive per w   9 generations (the 9-point s grid)
  fixed_s            1 generation (s=7.5), no calibration
  ws_equal           1 generation (s=7.5/w snapped), folklore rule
  cal_snap           1 generation (Eq.(6) snapped to the grid), calibration free
  cal_exact          1 generation at the exact off-grid s_cal (regenerated)

Metric: subject fidelity clip_i_ref (primary, pre-declared) and pix_l2
(secondary) UNDER the text gate clip_t >= tau, where tau = clip_t of the
default config (w=1, s=7.5) of the same cell.  w=1.0 is excluded (calibration
prescribes the default itself; nothing to compare).

Cells: all 91 complete (prompt_id, seed) cells of the L2 G segment.

PRE-REGISTERED judgment (written before running):
  C1 (primary): cal_exact reaches clip_i_ref >= 0.95 x exhaustive-winner
     (same gate) on >= 90% of (cell, w) pairs.
  C2: median per-pair clip_i_ref of cal_exact > medians of fixed_s and
     ws_equal (same gate).
  C3: |median(cal_snap) - median(cal_exact)| <= 0.01 (snapping is enough).
Failures are reported as failures.

Run:
    python z3_cal_search.py --offline
    python z3_cal_search.py --gpu
    python z3_cal_search.py --report
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from longrun_p2 import PROMPTS, RHO, RHO_DEFAULT, S0, W0

L2CSV = os.path.join(HERE, "out_long2", "L2_images.csv")
OUT = r"E:\研究\results\论文二_LoRA共线\Z3_2026-11-08"
CAL_W = [0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 2.0]   # w=1.0 excluded by design
S_GRID = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0]


def s_cal(w, rho):
    return S0 * (1.0 + W0 * rho) / (1.0 + w * rho)


def gated_pick(cand):
    """Among candidates, prefer clip_t >= tau then max clip_i_ref.
    If none passes the gate, that (cell, w) is a MISS for this path."""
    ok = cand[cand.clip_t >= cand.tau.iloc[0]] if len(cand) else cand
    if len(ok) == 0:
        return None
    return ok.loc[ok.clip_i_ref.idxmax()]


def offline():
    d = pd.read_csv(L2CSV)
    g = d[(d.segment == "G") & (d.route == "grid")]
    pmap = {pid: (pclass, text) for pid, pclass, text in PROMPTS}
    rows = []
    for (pid, seed), sub in g.groupby(["prompt_id", "seed"]):
        if len(sub) != 72:
            continue
        pclass, _ = pmap[pid]
        rho = RHO.get(pclass, RHO_DEFAULT)
        tau = float(sub[(sub.w == 1.0) & (sub.s == 7.5)].iloc[0].clip_t)
        sub = sub.assign(tau=tau)
        for w in CAL_W:
            ww = sub[np.isclose(sub.w, w)]
            # exhaustive: 9 s candidates at this w
            ex = gated_pick(ww)
            # prescribed / baseline s values (snap to grid when off-grid)
            cands = dict(fixed=S0, ws_equal=S0 / w, cal_snap=s_cal(w, rho))
            rows_pick = {}
            for k, s5 in cands.items():
                r = ww.iloc[(ww.s - s5).abs().argmin()]
                rows_pick[k] = r
            rec = dict(prompt_id=pid, pclass=pclass, seed=int(seed), w=w,
                       rho=rho, tau=tau,
                       ex_fid=np.nan if ex is None else ex.clip_i_ref,
                       ex_pix=np.nan if ex is None else ex.pix_l2,
                       ex_s=np.nan if ex is None else ex.s)
            for k, r in rows_pick.items():
                rec[f"{k}_fid"] = r.clip_i_ref
                rec[f"{k}_pix"] = r.pix_l2
                rec[f"{k}_gate"] = bool(r.clip_t >= tau)
                rec[f"{k}_s"] = float(r.s)
            rows.append(rec)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "z3_offline.csv"), index=False)
    print(f"offline: {len(df)} (cell, w) pairs")
    for k in ["fixed", "ws_equal", "cal_snap"]:
        gate = df[f"{k}_gate"].mean()
        ok = df.dropna(subset=["ex_fid"])
        ratio = (ok[f"{k}_fid"] / ok.ex_fid).median()
        print(f"  {k:9s} gate {gate:.3f}  median fid ratio vs exhaustive "
              f"{ratio:.4f}  median fid {ok[f'{k}_fid'].median():.4f} "
              f"(exhaustive {ok.ex_fid.median():.4f})")
    print(f"  cal_snap s values e.g. {df[df.w==2.0].cal_snap_s.iloc[0]:.3f} "
          f"at w=2 (grid) vs exact {s_cal(2.0, RHO_DEFAULT):.3f}")
    return df


def gpu():
    from longrun_p2 import Runner, STEPS
    d = pd.read_csv(os.path.join(OUT, "z3_offline.csv"))
    state_p = os.path.join(OUT, "z3_state.json")
    done = set(json.load(open(state_p))) if os.path.exists(state_p) else set()
    R = Runner()
    out_rows = []
    for _, r in d.iterrows():
        key = f"{r.prompt_id}|{int(r.seed)}|{r.w}"
        if key in done:
            continue
        text = [t for pid, _, t in PROMPTS if pid == r.prompt_id][0]
        a = R.anchors(r.prompt_id, text, int(r.seed))
        im = R.gen(text, float(r.w), float(s_cal(r.w, r.rho)), int(r.seed))
        tf, imf = R.score([im], [text])
        out_rows.append(dict(
            prompt_id=r.prompt_id, seed=int(r.seed), w=float(r.w),
            s_exact=float(s_cal(r.w, r.rho)),
            clip_t=float(tf[0] @ imf[0]),
            clip_i_ref=float(imf[0] @ a["i_ref"]),
            pix_l2=float(np.linalg.norm(R.thumb_np(im) - a["thumb_ref"])),
            tau=float(r.tau)))
        done.add(key)
        if len(out_rows) % 20 == 0:
            json.dump(sorted(done), open(state_p, "w"))
            pd.DataFrame(out_rows).to_csv(
                os.path.join(OUT, "z3_exact_raw.csv"), index=False)
            print(f"  {len(done)}/{len(d)} done", flush=True)
    json.dump(sorted(done), open(state_p, "w"))
    pd.DataFrame(out_rows).to_csv(os.path.join(OUT, "z3_exact_raw.csv"),
                                  index=False)
    print("gpu pass done:", len(out_rows), "rows")


def report():
    df = pd.read_csv(os.path.join(OUT, "z3_offline.csv"))
    ex = pd.read_csv(os.path.join(OUT, "z3_exact_raw.csv"))
    df = df.merge(ex[["prompt_id", "seed", "w", "clip_t", "clip_i_ref",
                      "pix_l2"]].rename(columns=dict(
                          clip_t="ce_clip_t", clip_i_ref="ce_fid",
                          pix_l2="ce_pix")), on=["prompt_id", "seed", "w"])
    df["ce_gate"] = df.ce_clip_t >= df.tau
    ok = df[df.ex_fid.notna() & df.ce_gate].copy()
    ok["ce_ratio"] = ok.ce_fid / ok.ex_fid
    ok["cs_ratio"] = ok.cal_snap_fid / ok.ex_fid
    ok["fx_ratio"] = ok.fixed_fid / ok.ex_fid
    ok["we_ratio"] = ok.ws_equal_fid / ok.ex_fid
    print("===== Z3 report (pre-registered) =====")
    print(f"pairs: {len(df)}  exhaustive gated: "
          f"{df.ex_fid.notna().mean():.3f}  cal_exact gated: {df.ce_gate.mean():.3f}")
    c1 = (ok.ce_ratio >= 0.95).mean() >= 0.90
    print(f"C1  cal_exact fid >= 0.95*exhaustive on {(ok.ce_ratio >= 0.95).mean():.3f}"
          f" of gated pairs -> {'PASS' if c1 else 'FAIL'}")
    c2 = (ok.ce_fid.median() > ok.fixed_fid.median() and
          ok.ce_fid.median() > ok.ws_equal_fid.median())
    print(f"C2  medians: exact={ok.ce_fid.median():.4f} snap={ok.cal_snap_fid.median():.4f} "
          f"fixed={ok.fixed_fid.median():.4f} ws_eq={ok.ws_equal_fid.median():.4f} "
          f"exhaustive={ok.ex_fid.median():.4f} -> {'PASS' if c2 else 'FAIL'}")
    c3 = abs(ok.cal_snap_fid.median() - ok.ce_fid.median()) <= 0.01
    print(f"C3  |snap-exact| median gap = "
          f"{abs(ok.cal_snap_fid.median() - ok.ce_fid.median()):.4f} "
          f"-> {'PASS' if c3 else 'FAIL'}")
    print("\nper-w median fid ratio vs exhaustive (gated pairs):")
    for w, gsub in ok.groupby("w"):
        print(f"  w={w:4.1f}  exact={gsub.ce_ratio.median():.3f}  "
              f"snap={gsub.cs_ratio.median():.3f}  fixed={gsub.fx_ratio.median():.3f}  "
              f"ws_eq={gsub.we_ratio.median():.3f}")
    df.to_csv(os.path.join(OUT, "z3_merged.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.offline:
        offline()
    if a.gpu:
        gpu()
    if a.report:
        report()
