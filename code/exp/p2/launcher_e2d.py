"""把 run_e2d_all.py 以真正脱离的方式拉起（Windows）。

为什么需要这一层：
    Start-Process 启的进程仍留在发起者的作业对象里，发起者（编辑器/终端/沙箱
    回收器）一退出，子进程会被连带清掉——实测过一次：脚本刚建好 CSV 表头就
    没了，没有 traceback、没有 stderr，纯粹是外部回收。

两个关键标志：
    DETACHED_PROCESS (0x8)         不继承控制台，父进程关掉控制台也无影响
    CREATE_BREAKAWAY_FROM_JOB (0x1000000)
                                   显式脱离作业对象。这是能否在父进程退出后
                                   存活的决定性标志。

输出不靠继承的流：run_e2d_all.py 自己 dup2 到
    out_e2d/_run/launcher.out    （含 traceback）
    out_e2d/_run/all.log         （总进度）
    out_e2d/_run/<tag>.log       （各变体完整输出）

用法：
    python launcher_e2d.py             # 后台跑全部五个变体
    python launcher_e2d.py --tags r4   # 只跑指定变体（参数透传）
"""

import os
import subprocess
import sys

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

HERE = os.path.dirname(os.path.abspath(__file__))


def spawn(flags, args):
    return subprocess.Popen(
        [sys.executable, os.path.join(HERE, "run_e2d_all.py")] + args,
        cwd=HERE,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=flags,
    )


def main():
    args = sys.argv[1:]
    base = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    try:
        p = spawn(base | CREATE_BREAKAWAY_FROM_JOB, args)
        note = "breakaway ok"
    except OSError as e:
        # 有些作业对象不允许 breakaway，此时退到不带该标志；若连这个也失败，
        # 说明环境不允许脱离，必须让用户改用计划任务。
        print(f"breakaway failed ({e}); retrying without it", flush=True)
        p = spawn(base, args)
        note = "breakaway NOT available"
    print(f"detached pid={p.pid}  ({note})")
    print("watch: out_e2d/_run/all.log")


if __name__ == "__main__":
    main()
