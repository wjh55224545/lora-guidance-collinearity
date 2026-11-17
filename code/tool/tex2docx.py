# -*- coding: utf-8 -*-
r"""paper.tex -> .docx（供导师批注用，非投稿格式）

用法：
    python tex2docx.py <paper.tex> <out.docx> --fig-root <图根目录> --flavor llncs|elsarticle

设计取舍：
  * 公式不追求排版级还原，用 Unicode + Word 原生上下标，保证"可读、可改"；
  * 图表编号、\ref、\cite 全部按 LaTeX 的顺序重新编号，与 PDF 一致；
  * 段落结构、标题层级、列表、文献表完整保留。
"""
import os
import re
import sys
import argparse

import fitz  # PyMuPDF
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import texmath  # noqa: E402

BODY_FONT = 'Times New Roman'


# ============================== 基础工具 ==============================

def strip_comments(s):
    return re.sub(r'(?<!\\)%.*$', '', s, flags=re.M)


def find_group(s, i):
    """s[i] 应为 '{'，返回 (内容, 结束位置后一位)。"""
    if i >= len(s) or s[i] != '{':
        return '', i
    depth = 1
    j = i + 1
    while j < len(s) and depth:
        if s[j] == '\\':
            j += 2
            continue
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
        j += 1
    return s[i + 1:j - 1], j


def grab_command(text, name, start=0):
    """找 \\name{...}，返回 (内容, 结束位置)。"""
    m = re.search(r'\\' + name + r'\s*(?:\[[^\]]*\])?\s*\{', text[start:])
    if not m:
        return None, start
    b = start + m.end() - 1
    content, end = find_group(text, b)
    return content, end


def unify_whitespace(s):
    return re.sub(r'[ \t\n]+', ' ', s).strip()


MACRO_PAT = re.compile(r'\\(?:new|renew|provide)command\*?\s*\{')


def expand_macros(text):
    """展开 \\newcommand 定义（支持无参宏，含嵌套多轮展开）。"""
    defs = {}
    i = 0
    while True:
        m = MACRO_PAT.search(text, i)
        if not m:
            break
        name, j = find_group(text, m.end() - 1)
        nm = name.strip().lstrip('\\')
        while j < len(text) and text[j] in ' \t\n':
            j += 1
        if j < len(text) and text[j] == '[':
            k = text.find(']', j)
            if k == -1:
                break
            j = k + 1
        while j < len(text) and text[j] in ' \t\n':
            j += 1
        if j < len(text) and text[j] == '{':
            body, j = find_group(text, j)
        else:
            mm = re.match(r'\S+', text[j:])
            if not mm:
                break
            body = mm.group(0)
            j += len(body)
        if nm:
            defs[nm] = body
        i = j

    for _ in range(6):
        changed = False
        for nm, body in defs.items():
            pat = re.compile(r'\\' + re.escape(nm) + r'(?![A-Za-z])')
            if pat.search(text):
                text = pat.sub(lambda _m, b=body: b, text)
                changed = True
        if not changed:
            break
    return text


# ============================== 文档样式 ==============================

def setup_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

    st = doc.styles['Normal']
    st.font.name = BODY_FONT
    st.font.size = Pt(11)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), '华文中宋')
    pf = st.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15
    return doc


def add_runs(par, runs, size=None, base_bold=False):
    for r in runs:
        if not r['t']:
            continue
        run = par.add_run(r['t'])
        run.font.name = BODY_FONT
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '华文中宋')
        if size:
            run.font.size = Pt(size)
        if r.get('b') or base_bold:
            run.bold = True
        if r.get('i'):
            run.italic = True
        if r.get('sup'):
            run.font.superscript = True
        if r.get('sub'):
            run.font.subscript = True
    return par


def add_heading(doc, text, level):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    if level == 1:
        pf.space_before = Pt(14)
        pf.space_after = Pt(6)
        size = 13
    elif level == 3:
        # llncs 的 \subsubsection 是 run-in 斜体小标题，不另起字号
        pf.space_before = Pt(8)
        pf.space_after = Pt(2)
        size = 11
    else:
        pf.space_before = Pt(10)
        pf.space_after = Pt(4)
        size = 11.5
    pf.keep_with_next = True
    runs = texmath.render(text)
    if level == 3:
        for r in runs:
            r['i'] = True
            r['b'] = False
    add_runs(p, runs, size=size)
    for r in p.runs:
        r.bold = True
    return p


def add_body(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    add_runs(p, texmath.render(text))
    return p


def add_caption(doc, text, before=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    if before:
        p.paragraph_format.keep_with_next = True
    add_runs(p, texmath.render(text), size=9.5)
    return p


# ============================== 图 ==============================

def pdf_to_png(pdf_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out = os.path.join(out_dir, base + '.png')
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(pdf_path):
        return out
    d = fitz.open(pdf_path)
    pix = d[0].get_pixmap(dpi=300)
    pix.save(out)
    d.close()
    return out


def add_figure(doc, tex_block, fig_root, tmp_dir, num, labels, bibidx):
    m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex_block)
    if not m:
        return
    rel = m.group(1)
    pdf = rel if os.path.isabs(rel) else os.path.join(fig_root, rel)
    if not os.path.exists(pdf):
        add_body(doc, '[Figure missing: %s]' % rel)
        return
    png = pdf_to_png(pdf, tmp_dir) if pdf.lower().endswith('.pdf') else pdf
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(png, width=Cm(14.5))
    cap, _ = grab_command(tex_block, 'caption')
    if cap:
        txt = clean_inline(unify_whitespace(cap), labels, bibidx)
        add_caption(doc, 'Figure %d. %s' % (num, txt), before=True)


# ============================== 表 ==============================

def parse_colspec(spec):
    spec = re.sub(r'@\{[^}]*\}', '', spec)
    spec = re.sub(r'[|]', '', spec)
    aligns = []
    i = 0
    while i < len(spec):
        c = spec[i]
        if c in 'lcr':
            aligns.append({'l': 'left', 'c': 'center', 'r': 'right'}[c])
            i += 1
        elif c in 'pmbX':
            aligns.append('left')
            i += 1
        else:
            i += 1
    return aligns


def clean_cell(s):
    s = re.sub(r'\\(toprule|midrule|bottomrule|hline|addlinespace|'
               r'cmidrule|endhead|endfirsthead|endfoot|endlastfoot)'
               r'(\([^)]*\))?(\{[^}]*\})?', '', s)
    s = re.sub(r'\\multicolumn\{(\d+)\}\{[^}]*\}(?=\{)', '', s)
    s = s.replace('\\textbf', '\\textbf')
    return s.strip()


def parse_tabular(block):
    m = re.search(r'\\begin\{tabular\}(?:\{[^}]*\})?\s*\{', block)
    if not m:
        return [], []
    spec, after = find_group(block, m.end() - 1)
    end = block.find('\\end{tabular}', after)
    body = block[after:end if end != -1 else len(block)]
    aligns = parse_colspec(spec)

    rows = []
    for raw in re.split(r'\\\\', body):
        raw = raw.strip()
        if not raw or not raw.replace('&', '').strip():
            continue
        # 跳过纯规则线
        if re.fullmatch(r'(?:\\(?:toprule|midrule|bottomrule|hline|'
                        r'addlinespace|cmidrule(?:\([^)]*\))?(?:\{[^}]*\})?)'
                        r'\s*)+', raw):
            continue
        cells = []
        depth = 0
        cur = []
        for ch in raw:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            if ch == '&' and depth == 0:
                cells.append(''.join(cur))
                cur = []
            else:
                cur.append(ch)
        cells.append(''.join(cur))
        rows.append([clean_cell(c) for c in cells])
    # 去掉因规则线产生的全空行
    rows = [r for r in rows if any(c.strip() for c in r)]
    return aligns, rows


def set_three_line(table):
    """把表格改成学术三线表：上下粗线 + 表头下细线，无竖线。"""
    tbl = table._tbl
    tblPr = tbl.tblPr
    for old in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(old)
    borders = OxmlElement('w:tblBorders')
    for edge, sz in (('top', '10'), ('bottom', '10')):
        el = OxmlElement('w:%s' % edge)
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), sz)
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), '000000')
        borders.append(el)
    for edge in ('left', 'right', 'insideH', 'insideV'):
        el = OxmlElement('w:%s' % edge)
        el.set(qn('w:val'), 'none')
        el.set(qn('w:sz'), '0')
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), 'auto')
        borders.append(el)
    tblPr.append(borders)
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement('w:tcBorders')
        bot = OxmlElement('w:bottom')
        bot.set(qn('w:val'), 'single')
        bot.set(qn('w:sz'), '6')
        bot.set(qn('w:space'), '0')
        bot.set(qn('w:color'), '000000')
        b.append(bot)
        tcPr.append(b)


def add_table(doc, block, num, labels, bibidx):
    cap, _ = grab_command(block, 'caption')
    if cap:
        txt = clean_inline(unify_whitespace(cap), labels, bibidx)
        add_caption(doc, 'Table %d. %s' % (num, txt), before=True)
    aligns, rows = parse_tabular(block)
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_three_line(t)
    # 三线表观感：去掉内部竖线
    for ri, row in enumerate(rows):
        for ci in range(ncol):
            cell = t.cell(ri, ci)
            txt = row[ci] if ci < len(row) else ''
            cell.text = ''
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            al = aligns[ci] if ci < len(aligns) else 'left'
            p.alignment = {'left': WD_ALIGN_PARAGRAPH.LEFT,
                           'center': WD_ALIGN_PARAGRAPH.CENTER,
                           'right': WD_ALIGN_PARAGRAPH.RIGHT}[al]
            add_runs(p, texmath.render(txt), size=9)
            if ri == 0:
                for r in p.runs:
                    r.bold = True
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0


# ============================== 正文解析 ==============================

ENV_RE = re.compile(r'\\begin\{(figure|table|equation|align|'
                    r'itemize|enumerate|thebibliography|center|'
                    r'abstract|keyword|quote)\*?\}')


def split_blocks(body):
    """把正文切成 ('text', str) / ('h1', str) / ('h2', str) / ('env', name, str)。"""
    blocks = []
    lines = body.split('\n')
    i = 0
    buf = []

    def flush_text():
        if buf:
            t = '\n'.join(buf).strip()
            if t:
                blocks.append(('text', t))
            buf.clear()

    while i < len(lines):
        line = lines[i]
        m = ENV_RE.search(line)
        if m:
            name = m.group(1)
            # 累积到对应的 end
            chunk = [line]
            depth = 1
            j = i
            end_pat = '\\end{%s}' % name
            while depth > 0:
                if j > i:
                    chunk.append(lines[j])
                seg = lines[j][m.end():] if j == i else lines[j]
                depth += seg.count('\\begin{%s}' % name)
                if '\\end{%s}' % name in seg:
                    depth -= 1
                j += 1
                if j >= len(lines):
                    break
            flush_text()
            blocks.append(('env', name, '\n'.join(chunk)))
            i = j
            continue

        s = line.strip()
        # 单独成行的 \label{...} 归到上一个 section，供 \ref 解析
        lm = re.match(r'^\\label\{([^}]+)\}$', s)
        if lm and blocks and blocks[-1][0] in ('h1', 'h2'):
            prev = blocks[-1]
            blocks[-1] = (prev[0], prev[1], prev[2] + '\\label{%s}' % lm.group(1))
            i += 1
            continue
        mm = re.match(r'\\section\*?\s*\{', s)
        if mm:
            content, end = find_group(s, mm.end() - 1)
            flush_text()
            blocks.append(('h1', content, s[end:]))
            i += 1
            continue
        mm = re.match(r'\\subsection\*?\s*\{', s)
        if mm:
            content, end = find_group(s, mm.end() - 1)
            flush_text()
            blocks.append(('h2', content, s[end:]))
            i += 1
            continue
        mm = re.match(r'\\subsubsection\*?\s*\{', s)
        if mm:
            content, end = find_group(s, mm.end() - 1)
            flush_text()
            blocks.append(('h3', content, s[end:]))
            i += 1
            continue
        mm = re.match(r'\\paragraph\{(.*)\}\s*(.*)$', s)
        if mm:
            flush_text()
            blocks.append(('text', '\\textbf{%s} %s' % (mm.group(1), mm.group(2))))
            i += 1
            continue
        if not s:
            flush_text()
            i += 1
            continue
        buf.append(s)
        i += 1
    flush_text()
    return blocks


def collect_labels(blocks):
    """建立 label -> 显示编号 的映射，并与 LaTeX 的编号顺序一致。"""
    lab = {}
    sec = sub = fig = tab = eq = 0
    for b in blocks:
        if b[0] == 'h1':
            sec += 1
            sub = 0
            for k in re.findall(r'\\label\{([^}]+)\}', b[2]):
                lab[k] = str(sec)
        elif b[0] == 'h2':
            sub += 1
            for k in re.findall(r'\\label\{([^}]+)\}', b[2]):
                lab[k] = '%d.%d' % (sec, sub)
        elif b[0] == 'h3':
            # llncs 的 \subsubsection 是 run-in 无独立编号，\thesubsubsection 等于父节编号
            for k in re.findall(r'\\label\{([^}]+)\}', b[2]):
                lab[k] = '%d.%d' % (sec, sub)
        elif b[0] == 'env':
            name, chunk = b[1], b[2]
            if name == 'figure':
                fig += 1
                for k in re.findall(r'\\label\{([^}]+)\}', chunk):
                    lab[k] = str(fig)
            elif name == 'table':
                tab += 1
                for k in re.findall(r'\\label\{([^}]+)\}', chunk):
                    lab[k] = str(tab)
            elif name in ('equation', 'align'):
                eq += 1
                for k in re.findall(r'\\label\{([^}]+)\}', chunk):
                    lab[k] = str(eq)
    return lab


def collect_bib(body):
    """bibitem key -> 序号（1 起），以及条目文本。"""
    order = []
    entries = []
    for m in re.finditer(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})',
                         body, re.S):
        order.append(m.group(1))
        txt = unify_whitespace(m.group(2))
        entries.append(txt)
    idx = {k: i + 1 for i, k in enumerate(order)}
    return idx, entries


def resolve_refs(text, labels):
    def rep(m):
        k = m.group(1)
        return labels.get(k, '??')
    text = re.sub(r'\\ref\{([^}]+)\}', rep, text)
    text = re.sub(r'\\eqref\{([^}]+)\}', lambda m: '(%s)' % labels.get(m.group(1), '??'), text)
    return text


def do_cites(text, bibidx):
    def rep(m):
        keys = [k.strip() for k in m.group(1).split(',')]
        nums = [bibidx[k] for k in keys if k in bibidx]
        if not nums:
            return '[?]'
        return '[%s]' % ', '.join(str(n) for n in nums)
    text = re.sub(r'\\(?:cite|citep|citet|parencite)'
                  r'(?:\s*\[[^\]]*\])*\s*\{([^}]+)\}', rep, text)
    return text


def clean_inline(text, labels, bibidx):
    text = text.replace('\\\\', ' ')
    text = resolve_refs(text, labels)
    text = do_cites(text, bibidx)
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = re.sub(r'\\(?:noindent|indent|centering|small|footnotesize|'
                  r'normalsize|large|Large|hfill|vfill|par|medskip|'
                  r'smallskip|bigskip|newpage|clearpage|maketitle|'
                  r'begingroup|endgroup|selectfont|sffamily|rmfamily|'
                  r'ttfamily|bfseries|itshape|upshape)\b\s*', '', text)
    text = re.sub(r'\\vspace\*?\{[^}]*\}', '', text)
    text = re.sub(r'\\hspace\*?\{[^}]*\}', ' ', text)
    return unify_whitespace(text)


# ============================== 主流程 ==============================

def convert(tex_path, out_docx, fig_root, tmp_dir, title_override=None):
    raw = open(tex_path, encoding='utf-8').read()
    raw = strip_comments(raw)
    # \iffalse ... \fi 是 TeX 的条件排除块：PDF 里不出现，Word 里也不许出现。
    # 之前没处理，导致 paper.tex 里一段被排除的 "Theorem 2" 段落漏进了给老师批注的 Word。
    while re.search(r'\\iffalse\b', raw):
        new = re.sub(r'\\iffalse\b.*?\\fi\b', '', raw, flags=re.S)
        if new == raw:
            break
        raw = new
    raw = expand_macros(raw)
    body = raw.split('\\begin{document}', 1)[-1]
    body = body.split('\\end{document}')[0]

    # ---- 摘要 / 关键词 ----
    abs_m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', body, re.S)
    abstract = unify_whitespace(abs_m.group(1)) if abs_m else ''
    # llncs 把 \keywords 写在 abstract 环境内部，需要从摘要正文里剔掉
    abstract = re.sub(r'\\keywords\s*\{[^}]*\}', '', abstract)
    abstract = re.sub(r'\\begin\{keywords?\}.*?\\end\{keywords?\}', '', abstract, flags=re.S)
    abstract = unify_whitespace(abstract)

    kw_m = re.search(r'\\begin\{keyword\}(.*?)\\end\{keyword\}', body, re.S)
    if kw_m:
        kw_raw = kw_m.group(1)
        kws = [unify_whitespace(x) for x in re.split(r'\\sep|\\and', kw_raw)]
    else:
        kws_m = re.search(r'\\keywords\{(.*?)\}\s*\n', body, re.S)
        kw_raw = kws_m.group(1) if kws_m else ''
        kws = [unify_whitespace(x) for x in re.split(r'\\and', kw_raw)]
    kws = [k for k in kws if k]

    # ---- 正文起点：去掉 frontmatter ----
    idx_after = 0
    for marker in ('\\end{frontmatter}', '\\end{keyword}', '\\end{abstract}'):
        k = body.rfind(marker)
        if k != -1:
            idx_after = max(idx_after, k + len(marker))
    main = body[idx_after:]

    bibidx, bib_entries = collect_bib(body)
    blocks = split_blocks(main)
    labels = collect_labels(blocks)

    # ---- 建文档 ----
    doc = setup_document()

    if title_override:
        title = title_override
    else:
        t_m = re.search(r'\\title\{(.*?)\}\s*\n', body, re.S)
        title = unify_whitespace(t_m.group(1)) if t_m else 'Untitled'
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.space_after = Pt(10)
    add_runs(tp, texmath.render(title), size=16)
    for r in tp.runs:
        r.bold = True

    for line in ('Jianhao Wang  ·  Shouqiang Liu (corresponding author)',
                 'School of Artificial Intelligence, South China Normal University, '
                 'Foshan 528225, Guangdong, China'):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(line)
        r.font.name = BODY_FONT
        r.font.size = Pt(10.5)

    if abstract:
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        add_runs(h, [{'t': 'Abstract', 'b': True, 'i': False,
                      'sup': False, 'sub': False}], size=11.5)
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        add_runs(p, texmath.render(clean_inline(abstract, labels, bibidx)))

    if kws:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(12)
        add_runs(p, [{'t': 'Keywords: ', 'b': True, 'i': False,
                      'sup': False, 'sub': False}] + texmath.render('; '.join(kws)))

    # ---- 正文 ----
    fig_n = tab_n = 0
    for b in blocks:
        kind = b[0]
        if kind == 'h1':
            add_heading(doc, clean_inline(b[1], labels, bibidx), 1)
        elif kind == 'h2':
            add_heading(doc, clean_inline(b[1], labels, bibidx), 2)
        elif kind == 'h3':
            add_heading(doc, clean_inline(b[1], labels, bibidx), 3)
        elif kind == 'text':
            add_body(doc, clean_inline(b[1], labels, bibidx))
        else:
            name, chunk = b[1], b[2]
            if name == 'figure':
                fig_n += 1
                add_figure(doc, chunk, fig_root, tmp_dir, fig_n, labels, bibidx)
            elif name == 'table':
                tab_n += 1
                add_table(doc, chunk, tab_n, labels, bibidx)
            elif name in ('equation', 'align'):
                eqm = re.search(r'\\label\{([^}]+)\}', chunk)
                num = labels.get(eqm.group(1), '') if eqm else ''
                content = re.sub(r'\\begin\{(equation|align)\*?\}', '', chunk)
                content = re.sub(r'\\end\{(equation|align)\*?\}', '', content)
                content = re.sub(r'\\label\{[^}]*\}', '', content)
                content = unify_whitespace(content)
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(6)
                add_runs(p, texmath.parse(content))
                if num:
                    r = p.add_run('    (%s)' % num)
                    r.font.name = BODY_FONT
            elif name in ('itemize', 'enumerate'):
                items = re.split(r'\\item\s*', chunk)
                k = 0
                for it in items[1:]:
                    k += 1
                    it = re.sub(r'\\end\{(itemize|enumerate)\}', '', it)
                    bullet = '•  ' if name == 'itemize' else '%d.  ' % k
                    p = doc.add_paragraph()
                    p.paragraph_format.left_indent = Cm(0.75)
                    p.paragraph_format.first_line_indent = Cm(-0.4)
                    p.paragraph_format.space_after = Pt(3)
                    r = p.add_run(bullet)
                    r.font.name = BODY_FONT
                    add_runs(p, texmath.render(clean_inline(unify_whitespace(it), labels, bibidx)))
            elif name == 'thebibliography':
                h = doc.add_paragraph()
                h.paragraph_format.space_before = Pt(14)
                h.paragraph_format.space_after = Pt(6)
                add_runs(h, [{'t': 'References', 'b': True, 'i': False,
                              'sup': False, 'sub': False}], size=13)
                for n, e in enumerate(bib_entries, 1):
                    p = doc.add_paragraph()
                    p.paragraph_format.left_indent = Cm(0.9)
                    p.paragraph_format.first_line_indent = Cm(-0.9)
                    p.paragraph_format.space_after = Pt(3)
                    r = p.add_run('[%d]  ' % n)
                    r.font.name = BODY_FONT
                    r.font.size = Pt(9.5)
                    add_runs(p, texmath.render(clean_inline(e, labels, bibidx)), size=9.5)

    doc.save(out_docx)
    return len(blocks), len(bib_entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tex')
    ap.add_argument('out')
    ap.add_argument('--fig-root', default=None)
    ap.add_argument('--tmp', default=None)
    ap.add_argument('--title', default=None)
    a = ap.parse_args()

    fig_root = a.fig_root or os.path.dirname(os.path.abspath(a.tex))
    tmp = a.tmp or os.path.join(os.path.dirname(os.path.abspath(a.out)), '_figcache')
    nb, nref = convert(a.tex, a.out, fig_root, tmp, a.title)
    print('blocks=%d refs=%d -> %s' % (nb, nref, a.out))


if __name__ == '__main__':
    main()
