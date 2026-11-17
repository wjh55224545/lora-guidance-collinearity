"""Paper 2, phase Y orchestrator: one background task that runs the whole night
in the right order, and survives being interrupted.

Why an orchestrator instead of six manual launches: the GPU is a single 8 GB
4060, so these segments cannot overlap, and every one of them is 20-70 minutes
long.  A single process that walks the list means a crash costs one segment,
not the night.

Order and why
  1  y2_full    dense-s grid on the *original* LoRA.  Runs first because it is
                self-falsification: if kappa jumps above 0.5 the published
                interval is dead, and that has to be known before anything is
                built on top of it.  It also only touches a path that is already
                proven, so it cannot be lost to a training bug.
  2  y1_A / y1_B   train the two independent LoRAs.
  3  y1_A_grid / y1_B_grid   the L2-shaped grid on each, coarse s, so kappa is
                directly comparable to 0.19-0.27.
  4  y2_A      dense-s grid on A (the same refinement as step 1, on a LoRA that
                was actually trained rather than surgically derived).
  5  y3        two adapters at once, with w=0 rows/columns so the composition
                grid carries its own no-LoRA reference.

Each segment is itself resumable (y_gen/y1_train append one row per image and
keep state.json), so re-running this script after any interruption picks up
mid-segment.  A segment is only marked done when its subprocess exits 0.

A dependent segment whose LoRA file is missing is skipped and logged as such --
never silently treated as "0 collinearity".

Run:  python run_y.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_y")
LOGS = os.path.join(OUT, "logs")
PY = sys.executable
TRAINED = r"E:\研究\models\lora\trained"

SEGMENTS = [
    # training first, not last: it is the only segment that can fail for a
    # reason worth reacting to (an inert LoRA changes what the paper can claim),
    # and its verification prints within ~40 min instead of ~3 h.
    ("y1_A_train", ["y1_train.py", "--concept", "A", "--steps", "1000"], []),
    ("y1_B_train", ["y1_train.py", "--concept", "B", "--steps", "1000"], []),
    ("y2_full", ["y_gen.py", "--mode", "single", "--tag", "y2_full",
                 "--lora", "full=full", "--dense-s",
                 "--prompts", "4", "--seeds", "11,22"], []),
    ("y1_A_grid", ["y_gen.py", "--mode", "single", "--tag", "y1_A",
                   "--lora", "A=A", "--prompts", "4", "--seeds", "11,22"],
     [os.path.join(TRAINED, "lora_A.safetensors")]),
    ("y1_B_grid", ["y_gen.py", "--mode", "single", "--tag", "y1_B",
                   "--lora", "B=B", "--prompts", "4", "--seeds", "11,22"],
     [os.path.join(TRAINED, "lora_B.safetensors")]),
    ("y2_A", ["y_gen.py", "--mode", "single", "--tag", "y2_A",
              "--lora", "A=A", "--dense-s",
              "--prompts", "4", "--seeds", "11,22"],
     [os.path.join(TRAINED, "lora_A.safetensors")]),
    ("y3", ["y_gen.py", "--mode", "two", "--tag", "y3",
            "--lora", "A=A", "--lora", "B=B",
            "--w", "0,0.3,0.6,1.0,1.5", "--w2", "0,0.3,0.6,1.0,1.5",
            "--s", "5,7.5,10", "--prompts", "4", "--seeds", "11"],
     [os.path.join(TRAINED, "lora_A.safetensors"),
      os.path.join(TRAINED, "lora_B.safetensors")]),
    # ---- Y7 dose: is kappa monotone in "how much negative prompt", or is the
    # only real distinction empty vs non-empty?  y7_neg (a 4-term string) already
    # halved kappa; these two bracket it with a minimal one-term string and a long
    # 10-term one.  Smaller grid (5w x 5s x 4 prompts = 100 images each) because
    # the question is a level comparison, not a shape estimate.
    ("y7_neg_short", ["y_gen.py", "--mode", "single", "--tag", "y7_neg_short",
                      "--lora", "full=full", "--neg", "blurry",
                      "--w", "0.2,0.6,1.0,1.5,2.0", "--s", "5,6.5,7.5,8.5,10",
                      "--prompts", "4", "--seeds", "11"], []),
    ("y7_neg_long", ["y_gen.py", "--mode", "single", "--tag", "y7_neg_long",
                     "--lora", "full=full",
                     "--neg", "blurry, low quality, watermark, text, extra fingers, "
                              "bad anatomy, worst quality, jpeg artifacts, "
                              "oversaturated, cartoon",
                     "--w", "0.2,0.6,1.0,1.5,2.0", "--s", "5,6.5,7.5,8.5,10",
                     "--prompts", "4", "--seeds", "11"], []),
    # ---- Y7 dose rung 2: the SAME question at the s resolution the other
    # operators were scored at.  The 100-image 5x5 runs above cannot be compared
    # with y7_neg (default 8w x 9s grid): on a 5-point s grid the local estimator
    # returns sd .67 for the naive baseline -- not identifiable there -- and the
    # grid estimator is quantised.  Rebuilt as 5w x 9s x 4 prompts x 1 seed = 180
    # images per rung, which is exactly the subset y2_full and y7_neg can be cut
    # down to: one paired ladder, no grid confound.
    ("y7_neg_short_9s", ["y_gen.py", "--mode", "single", "--tag", "y7_neg_short_9s",
                         "--lora", "full=full", "--neg", "blurry",
                         "--w", "0.2,0.6,1.0,1.5,2.0",
                         "--s", "3,4,5,6,7.5,9,10.5,12,14",
                         "--prompts", "4", "--seeds", "11"], []),
    # ---- n=8 prompts: the dose ladder above has only 4 paired prompts, and the
    # operator difference (neg vs empty) is -0.10 to -0.13 with p=.13 at n=4 --
    # under-powered, not absent.  Rerunning the two rungs that carry the headline
    # claim over prompts obj00-obj07 (all "object" class, so pclass stays fixed)
    # doubles n without touching anything else.  Same tags on purpose: y_gen
    # skips images that already exist, so this only adds obj04-obj07.
    # 9-point s and seed 11 only -- the pairwise test is within-seed, and dense-s
    # here would cost 1472 new images instead of 288 for no extra power.
    ("y7_naive_8p", ["y_gen.py", "--mode", "single", "--tag", "y2_full",
                     "--lora", "full=full", "--s", "3,4,5,6,7.5,9,10.5,12,14",
                     "--prompts", "8", "--seeds", "11"], []),
    ("y7_neg_8p", ["y_gen.py", "--mode", "single", "--tag", "y7_neg",
                   "--lora", "full=full",
                   "--neg", "blurry, low quality, watermark, text",
                   "--s", "3,4,5,6,7.5,9,10.5,12,14",
                   "--prompts", "8", "--seeds", "11"], []),
    ("y7_neg_long_9s", ["y_gen.py", "--mode", "single", "--tag", "y7_neg_long_9s",
                        "--lora", "full=full",
                        "--neg", "blurry, low quality, watermark, text, extra fingers, "
                                 "bad anatomy, worst quality, jpeg artifacts, "
                                 "oversaturated, cartoon",
                        "--w", "0.2,0.6,1.0,1.5,2.0",
                        "--s", "3,4,5,6,7.5,9,10.5,12,14",
                        "--prompts", "4", "--seeds", "11"], []),
    # ---- Y3 follow-ups: the A/B asymmetry (wA shrinks to .09, wB to .51).
    # Two cheap controls, 14 min each.  swap = load B first, so the grid's first
    # axis is B; if the asymmetry follows the *slot* the shrink numbers repeat,
    # if it follows the *adapter* they transpose.  s22 = does the shrink number
    # survive a different seed at all (Y3 itself is a single seed).
    ("y3_swap", ["y_gen.py", "--mode", "two", "--tag", "y3_swap",
                 "--lora", "B=B", "--lora", "A=A",
                 "--w", "0,0.3,0.6,1.0,1.5", "--w2", "0,0.3,0.6,1.0,1.5",
                 "--s", "5,7.5,10", "--prompts", "4", "--seeds", "11"],
     [os.path.join(TRAINED, "lora_B.safetensors"),
      os.path.join(TRAINED, "lora_A.safetensors")]),
    ("y3_s22", ["y_gen.py", "--mode", "two", "--tag", "y3_s22",
                "--lora", "A=A", "--lora", "B=B",
                "--w", "0,0.3,0.6,1.0,1.5", "--w2", "0,0.3,0.6,1.0,1.5",
                "--s", "5,7.5,10", "--prompts", "4", "--seeds", "22"],
     [os.path.join(TRAINED, "lora_A.safetensors"),
      os.path.join(TRAINED, "lora_B.safetensors")]),
    # ---- Y7: is the collinearity about w and s, or about *this* CFG variant?
    # Every kappa so far lives inside one guidance operator (plain CFG, empty
    # negative prompt).  Two alternative operators, same grid, same anchors:
    #   rescale  variance-normalised CFG -> removes the magnitude channel by
    #            construction; if kappa -> 0 here, the degeneracy is a property
    #            of unnormalised CFG, not of the w-s pair
    #   neg      a real negative prompt -> the unconditional branch is no longer
    #            the empty string, so the LoRA also acts on the "what not to
    #            draw" side; if kappa barely moves, the collinearity is intrinsic
    ("y7_rescale", ["y_gen.py", "--mode", "single", "--tag", "y7_rescale",
                    "--lora", "full=full", "--rescale", "0.7",
                    "--prompts", "4", "--seeds", "11,22"], []),
    ("y7_neg", ["y_gen.py", "--mode", "single", "--tag", "y7_neg",
                "--lora", "full=full",
                "--neg", "blurry, low quality, watermark, text",
                "--prompts", "4", "--seeds", "11,22"], []),
    # ---- Y8: what actually carries the negative-prompt effect?  The dose ladder
    # (y7 dose, four rungs 0/1/4/10 words) came out non-monotone -- kappa
    # .206/.101/.074/.158 -- so "more negative words, weaker collinearity" is
    # dead.  What survives is the single contrast empty vs 4 words (-.10 to -.13).
    # That contrast has two candidate causes and they are separable by fixing the
    # word count at 4 and changing only the vocabulary:
    #   content4   four concrete nouns.  Same length, no quality semantics, no
    #              instruction to avoid anything.  If kappa drops the same way,
    #              the cause is "the unconditional branch is no longer empty"
    #              (the LoRA acts on both CFG branches), not negativity.
    #   quality4b  four *other* quality words.  If this one drops and content4
    #              does not, the cause is the quality vocabulary itself, and the
    #              honest claim is narrow: it is about that vocabulary, not about
    #              negative prompts in general.
    # Same grid as the ladder (5w x 9s x 4 prompts x seed 11 = 180 images each)
    # so all rungs are one paired ladder and nothing has to be re-cut.
    ("y8_neg_content4", ["y_gen.py", "--mode", "single", "--tag", "y8_neg_content4",
                         "--lora", "full=full", "--neg", "cat, dog, tree, car",
                         "--w", "0.2,0.6,1.0,1.5,2.0",
                         "--s", "3,4,5,6,7.5,9,10.5,12,14",
                         "--prompts", "4", "--seeds", "11"], []),
    # ---- the one arm from Y8 that needs a fresh-prompt replication: content4
    # dropped kappa exactly as much as the quality string did on obj00-03, which
    # is what killed the "quality vocabulary" hypothesis.  But obj00-03 are also
    # the four prompts where the neg effect lives at all (Y7p8: the effect is
    # zero on obj04-07).  If the content arm also shows nothing on obj04-07, the
    # "non-empty branch" story is prompt-specific rather than a mechanism, and
    # that is the version the paper has to print.  Same tag on purpose so y_gen
    # only adds obj04-obj07 (4 prompts x 45 = 180 images, ~10 min).
    ("y8_neg_content4_8p", ["y_gen.py", "--mode", "single",
                            "--tag", "y8_neg_content4",
                            "--lora", "full=full", "--neg", "cat, dog, tree, car",
                            "--w", "0.2,0.6,1.0,1.5,2.0",
                            "--s", "3,4,5,6,7.5,9,10.5,12,14",
                            "--prompts", "8", "--seeds", "11"], []),
    ("y8_neg_quality4b", ["y_gen.py", "--mode", "single", "--tag", "y8_neg_quality4b",
                          "--lora", "full=full",
                          "--neg", "grainy, jpeg artifacts, oversaturated, cartoon",
                          "--w", "0.2,0.6,1.0,1.5,2.0",
                          "--s", "3,4,5,6,7.5,9,10.5,12,14",
                          "--prompts", "4", "--seeds", "11"], []),
]


def main():
    os.makedirs(LOGS, exist_ok=True)
    state_p = os.path.join(OUT, "seg_state.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {}

    t0 = time.time()
    print(f"=== run_y: {len(SEGMENTS)} segments ===", flush=True)
    for name, argv, needs in SEGMENTS:
        if state.get(name, {}).get("status") == "ok":
            print(f"[skip] {name} already ok", flush=True)
            continue
        missing = [p for p in needs if not os.path.exists(p)]
        if missing:
            print(f"[skip] {name}: missing {missing}", flush=True)
            state[name] = dict(status="skipped_missing_input",
                               missing=missing, at=time.strftime("%F %T"))
            json.dump(state, open(state_p, "w"), indent=1)
            continue

        print(f"\n>>> {name}  ({time.strftime('%F %T')})", flush=True)
        logp = os.path.join(LOGS, f"{name}.log")
        ts = time.time()
        with open(logp, "a", encoding="utf-8") as lf:
            lf.write(f"\n\n===== {name} start {time.strftime('%F %T')} =====\n")
            lf.flush()
            pr = subprocess.Popen([PY, "-u"] + argv, cwd=HERE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace")
            tail = []
            for line in pr.stdout:
                lf.write(line)
                lf.flush()
                # the first version of this file wrote without flushing and the
                # launcher ended in `| tail -200`, so for four minutes there was
                # literally no way to tell a slow step from a hang.  Flush every
                # line; the file is the monitor.
                tail.append(line.rstrip())
                if len(tail) > 60:
                    tail.pop(0)
                if any(k in line for k in ("s/img", "step ", "saved", "done ",
                                           "Traceback", "Error", "!!", "INERT",
                                           "max pix")):
                    print("   " + line.rstrip()[:160], flush=True)
            pr.wait()
        ok = pr.returncode == 0
        state[name] = dict(status="ok" if ok else f"exit{pr.returncode}",
                           minutes=round((time.time() - ts) / 60, 1),
                           at=time.strftime("%F %T"))
        json.dump(state, open(state_p, "w"), indent=1)
        print(f"<<< {name} {'ok' if ok else 'FAILED rc=' + str(pr.returncode)} "
              f"in {state[name]['minutes']} min", flush=True)
        if not ok:
            print("    last lines:", flush=True)
            for l in tail[-12:]:
                print("      " + l[:150], flush=True)

    print(f"\n=== run_y finished in {(time.time() - t0) / 60:.1f} min ===",
          flush=True)
    print(json.dumps(state, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
