import io
from pypdf import PdfReader

p = r"E:\研究\results\论文一_GBDT遗忘\投稿_B_DASFAA\paper.pdf"
r = PdfReader(p)
out = ["pages=%d" % len(r.pages)]
for i, pg in enumerate(r.pages):
    t = pg.extract_text() or ""
    t = " ".join(t.split())
    out.append("--- p%d chars=%d ---" % (i + 1, len(t)))
    if i >= len(r.pages) - 2:
        out.append(t[:800])
io.open(r"E:\研究\_tool\pdfpages.txt", "w", encoding="utf-8").write("\n".join(out))
print("ok")
