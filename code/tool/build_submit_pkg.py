# -*- coding: utf-8 -*-
"""重建两份投稿 source zip，内容与当前 paper.tex / paper.pdf 同版本。

用法：D:\\python\\python.exe _tool/build_submit_pkg.py
说明：
- 论文二按 Elsevier 习惯扁平化：manuscript.tex 里 figs/figN_*.pdf -> figN.pdf，图与 cls 同目录。
- 论文一保留 figures_lncs_en/ 子目录（llncs 原样提交，路径与 tex 一致）。
- 只打包源文件，不带 aux/log/out。
"""
import os
import re
import shutil
import zipfile

R = r"E:\研究\results"


def p2():
    d = os.path.join(R, "论文二_LoRA共线", "投稿_D_Neurocomputing")
    tex = open(os.path.join(d, "paper.tex"), encoding="utf-8").read()
    flat = os.path.join(d, "_latex_flat")
    os.makedirs(flat, exist_ok=True)
    # figs/figN_name.pdf -> figN.pdf
    tex_flat = re.sub(r"\{figs/(fig\d+)[^{}]*\.pdf\}", r"{\1.pdf}", tex)
    n = len(re.findall(r"\{figs/fig\d+[^{}]*\.pdf\}", tex))
    open(os.path.join(flat, "manuscript.tex"), "w", encoding="utf-8").write(tex_flat)
    cls_dst = os.path.join(flat, "elsarticle.cls")
    if not os.path.exists(cls_dst):
        shutil.copy2(os.path.join(d, "elsarticle.cls"), cls_dst)
    figs = []
    for f in sorted(os.listdir(os.path.join(d, "figs"))):
        m = re.match(r"(fig\d+).*\.pdf$", f)
        if m:
            dst = os.path.join(flat, m.group(1) + ".pdf")
            shutil.copy2(os.path.join(d, "figs", f), dst)
            figs.append(m.group(1) + ".pdf")
    zp = os.path.join(d, "Neurocomputing_LaTeX.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["manuscript.tex", "elsarticle.cls"] + figs:
            z.write(os.path.join(flat, f), f)
    print("[P2] 图路径替换 %d 处 -> %s" % (n, ", ".join(figs)))
    print("[P2] zip: %s  (%d 项, %.1f KB)" % (zp, 2 + len(figs), os.path.getsize(zp) / 1024))


def p1():
    d = os.path.join(R, "论文一_GBDT遗忘", "投稿_B_DASFAA")
    cls_src = os.path.join(R, "论文一_GBDT遗忘", "_submit_B_pkg", "llncs.cls")
    items = [("paper.tex", os.path.join(d, "paper.tex"))]
    if os.path.exists(cls_src):
        items.append(("llncs.cls", cls_src))
    else:
        print("[P1] 警告：目录内无 llncs.cls，跳过")
    figdir = os.path.join(d, "figures_lncs_en")
    figs = [f for f in sorted(os.listdir(figdir)) if f.endswith(".pdf")]
    for f in figs:
        items.append(("figures_lncs_en/" + f, os.path.join(figdir, f)))
    zp = os.path.join(d, "DASFAA2027_source.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in items:
            z.write(src, arc)
    print("[P1] zip: %s  (%d 项, %.1f KB)" % (zp, len(items), os.path.getsize(zp) / 1024))
    for arc, _ in items:
        print("     -", arc)


if __name__ == "__main__":
    p1()
    p2()
