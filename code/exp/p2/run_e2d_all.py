"""E2c-2 总控：串行跑完五个 LoRA 变体，可断点续跑。

为什么要一个总控脚本：
  * GPU 只有 8 GB，一个 fp16 SD1.5 pipeline 就占去大半，五个变体**不能并行**，
    只能串行。逐个手敲命令会让"跑到一半被中断"变成常态。
  * 每个变体自己有 `out_e2d/<tag>/state.json`，所以中断后重跑本脚本即从断点继续，
    已完成的变体只会打印 "0 images to do" 后立刻跳过。

运行方式（必须脱离应用进程，见 launcher_e2d.ps1）：
    python run_e2d_all.py                     # 跑全部五个变体
    python run_e2d_all.py --tags te_only r4   # 只跑指定变体

日志：
    out_e2d/_run/all.log        总进度（每个变体起止 + 各变体最后一行）
    out_e2d/_run/<tag>.log      该变体的完整 stdout/stderr
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RUN = os.path.join(HERE, "out_e2d", "_run")

# Redirect at the fd level, right at import time, not inside main().  This
# script is started detached (launcher_e2d.py) with stdout=DEVNULL and its
# launching shell is gone within seconds, so anything relying on an inherited
# stream would be lost -- including a traceback from a failed import, which is
# exactly the case we would most need to see.  dup2 onto a real file means the
# detached process keeps writing no matter who started it.
os.makedirs(RUN, exist_ok=True)
try:
    _LOGF = open(os.path.join(RUN, "launcher.out"), "a", buffering=1,
                 encoding="utf-8", errors="replace")
    os.dup2(_LOGF.fileno(), 1)
    os.dup2(_LOGF.fileno(), 2)
    print(f"\n===== run_e2d_all.py start {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"pid={os.getpid()} =====", flush=True)
except OSError:
    pass

# 顺序有讲究：te_only 已有 144 行断点数据，先把它补完，这样中途万一停掉，
# 也有一整格（1 个 prompt 全覆盖）可用；full 放最后，因为它的数字已经由
# L2 长跑给出，重复跑只是为了在完全相同的口径下对齐。
TAGS = ["te_only", "unet_only", "no_attn2", "r4", "full"]

N_PROMPTS = 8
SEEDS = "11,22"


def log(msg, path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_rows(tag):
    p = os.path.join(HERE, "out_e2d", tag, "images.csv")
    if not os.path.exists(p):
        return 0
    with open(p, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def acquire_lock():
    """独占锁：同一时刻只允许一个总控实例。

    为什么需要：这批实验可能被两条路启动——编辑器会话里的后台任务，和用户
    自己在外部终端里跑的那一份（后者才能扛住"关掉应用"）。两条路都会写
    out_e2d/<tag>/images.csv，同时跑会互相覆盖。

    用 msvcrt 的记录锁而不是 pid 文件：进程死掉时由 OS 释放，不需要清理，
    也不会因为上次被强杀留下"假在跑"的锁。
    """
    import msvcrt
    lockp = os.path.join(RUN, "e2d.lock")
    f = open(lockp, "w+")
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("ANOTHER INSTANCE IS ALREADY RUNNING -- exiting", flush=True)
        print(f"(lock held on {lockp})", flush=True)
        sys.exit(0)
    f.write(str(os.getpid()))
    f.flush()
    return f          # 必须持有引用，否则被 GC 后锁释放


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=TAGS)
    a = ap.parse_args()

    os.makedirs(RUN, exist_ok=True)
    _lock = acquire_lock()
    alllog = os.path.join(RUN, "all.log")
    log("=" * 68, alllog)
    log(f"pid {os.getpid()}  E2c-2 start  tags={a.tags}", alllog)
    log(f"prompts={N_PROMPTS}  seeds={SEEDS}  expect "
        f"{N_PROMPTS * len(SEEDS.split(',')) * 8 * 9} images per tag", alllog)

    t_all = time.time()
    summary = []
    for tag in a.tags:
        tlog = os.path.join(RUN, f"{tag}.log")
        before = count_rows(tag)
        log(f"--- {tag}: {before} rows already on disk, starting", alllog)
        t0 = time.time()
        cmd = [PY, os.path.join(HERE, "e2d_variants.py"),
               "--tag", tag, "--prompts", str(N_PROMPTS), "--seeds", SEEDS]
        with open(tlog, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {tag} =====\n")
            f.flush()
            # 每个变体单独一个子进程：某个变体崩了不会带塌整批，
            # 而 state.json 保证它下次从断点继续。
            rc = subprocess.call(cmd, cwd=HERE, stdout=f,
                                 stderr=subprocess.STDOUT)
        after = count_rows(tag)
        dt = (time.time() - t0) / 60
        log(f"--- {tag}: rc={rc}  {before} -> {after} rows  ({dt:.1f} min)", alllog)
        summary.append((tag, rc, before, after, dt))

    log("=" * 68, alllog)
    log(f"ALL DONE in {(time.time() - t_all) / 60:.1f} min", alllog)
    for tag, rc, b, af, dt in summary:
        log(f"  {tag:10s} rc={rc}  {af} rows  {dt:5.1f} min", alllog)
    bad = [s for s in summary if s[1] != 0]
    if bad:
        log(f"  !! non-zero exit: {[s[0] for s in bad]}", alllog)
    log("DONE_MARKER", alllog)


if __name__ == "__main__":
    main()
