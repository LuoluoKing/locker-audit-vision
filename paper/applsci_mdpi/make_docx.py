#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 main.tex 生成 MDPI Applied Sciences 官方 Word 模板格式的论文 Word 版。

流水线
  1. 解析 main.tex：前置信息、文末声明、章节结构、图表/公式/算法编号、引用顺序
  2. 生成 pandoc 友好的中间 tex -> pandoc -> raw.docx
  3. 用 pandoc 渲染 LaTeX 片段（前置信息 / 公式表 / 算法表 / 文末声明 / 缩写表 / 参考文献）
  4. 组装最终 docx：样式重映射为 MDPI 样式、注入片段、表格三线边框
运行： python make_docx.py
"""

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from xml.etree import ElementTree as ET

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(SRC_DIR, "main.tex")
BIB = os.path.join(SRC_DIR, "refs.bib")
PDF = os.path.join(SRC_DIR, "Main_Manuscript_AppliedSciences.pdf")
TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(SRC_DIR)), "EXAMPLE", "applsci-template.dot")
OUT_DOCX = os.path.join(SRC_DIR, "Main_Manuscript_AppliedSciences.docx")

TMP = os.path.join(os.environ.get("TEMP", "/tmp"), "wordgen")
os.makedirs(TMP, exist_ok=True)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
Q = "{" + W + "}"


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 小工具

def balanced(s, i):
    """s[i] == '{'；返回 (内容, 闭合括号后的下标)。"""
    assert s[i] == "{", s[max(0, i - 20):i + 5]
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces")


def cmd_arg(tex, name, start=0):
    """取 \\name{...} 的参数；找不到返回 None。"""
    m = re.compile(r"\\" + name + r"\s*\{").search(tex, start)
    if not m:
        return None
    inner, _ = balanced(tex, m.end() - 1)
    return inner


def strip_labels(s):
    return re.sub(r"\\label\{[^}]*\}", "", s)


def strip_comments(s):
    """去掉 LaTeX 的 % 注释（保留 \\%）。"""
    return re.sub(r"(?<!\\)%.*", "", s)


def grab(tex, name):
    """取 \\name{...} 的内容并清掉 label 与注释。"""
    return strip_comments(strip_labels(cmd_arg(tex, name) or ""))


def env_blocks(text, env):
    return list(re.finditer(r"\\begin\{" + env + r"\}(\[[^\]]*\])?(.*?)\\end\{" + env + r"\}", text, re.S))


def tex_clean(s):
    """把 LaTeX 片段里的常见命令转成纯文本（仅用于抽取/比对，不用于输出）。"""
    s = re.sub(r"\\(texttt|url|emph|textit|textbf)\{([^{}]*)\}", r"\2", s)
    s = s.replace("~", " ").replace("\\%", "%").replace("\\&", "&")
    s = s.replace("\\_", "_").replace("---", "—").replace("--", "–")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------------------------------------------------------- 阶段 1：解析

class Doc:
    pass


def parse():
    tex = open(TEX, encoding="utf-8").read()
    d = Doc()

    # --- 前置信息
    d.title = grab(tex, "Title")
    d.address = grab(tex, "address")
    d.corres = grab(tex, "corres")
    d.featured = grab(tex, "featuredapplication")
    d.abstract = grab(tex, "abstract")
    d.keywords = grab(tex, "keyword")

    # --- 文末声明
    d.back = {}
    for name, key in [("authorcontributions", "Author Contributions"),
                      ("funding", "Funding"),
                      ("institutionalreview", "Institutional Review Board Statement"),
                      ("informedconsent", "Informed Consent Statement"),
                      ("dataavailability", "Data Availability Statement"),
                      ("acknowledgments", "Acknowledgments"),
                      ("conflictsofinterest", "Conflicts of Interest")]:
        d.back[key] = grab(tex, name)
    am = re.search(r"\\abbreviations\s*\{", tex)
    d.abbrev_text = ""
    if am:
        _, e1 = balanced(tex, am.end() - 1)
        m2 = re.match(r"\s*\{", tex[e1:])
        if m2:
            d.abbrev_text, _ = balanced(tex, e1 + m2.end() - 1)
    d.abbrev_text = strip_comments(strip_labels(d.abbrev_text))

    # --- 区域切分
    body_start = tex.index(r"\begin{document}") + len(r"\begin{document}")
    back_start = tex.index(r"\authorcontributions")
    body = tex[body_start:back_start]

    # --- 章节结构（含编号）
    d.sec_nums = {}
    n1 = n2 = n3 = 0
    for m in re.finditer(r"\\(section|subsection|subsubsection)\*?\s*\{", body):
        kind = m.group(1)
        title, end = balanced(body, m.end() - 1)
        lm = re.search(r"\\label\{([^}]*)\}", body[end:end + 220])
        label = lm.group(1) if lm else None
        if kind == "section":
            n1 += 1
            n2 = n3 = 0
            num = f"{n1}"
        elif kind == "subsection":
            n2 += 1
            n3 = 0
            num = f"{n1}.{n2}"
        else:
            n3 += 1
            num = f"{n1}.{n2}.{n3}"
        if label:
            d.sec_nums[label] = num

    # --- 图 / 表 / 公式 / 算法编号
    d.fig_nums, d.tab_nums, d.alg_nums = {}, {}, {}
    for i, blk in enumerate(env_blocks(body, "figure"), 1):
        inner = blk.group(2)
        m = re.search(r"\\caption\s*\{", inner)
        if m:
            cap, _ = balanced(inner, m.end() - 1)
            lm = re.search(r"\\label\{([^}]*)\}", cap)
            if lm:
                d.fig_nums[lm.group(1)] = i
    for i, blk in enumerate(env_blocks(body, "table"), 1):
        inner = blk.group(2)
        m = re.search(r"\\caption\s*\{", inner)
        if m:
            cap, _ = balanced(inner, m.end() - 1)
            lm = re.search(r"\\label\{([^}]*)\}", cap)
            if lm:
                d.tab_nums[lm.group(1)] = i

    # 公式：equation / align，按出现顺序编号
    d.eq_nums = {}
    d.eq_blocks = []          # [(kind, inner)] 供后面生成公式表
    counter = 0
    for m in re.finditer(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", body, re.S):
        kind, inner = m.group(1), m.group(2)
        if kind == "equation":
            rows = [inner]
        else:
            rows = [r for r in re.split(r"\\\\\s*(?:\[[^\]]*\])?", inner) if r.strip()]
        rnums = []
        for r in rows:
            counter += 1
            rnums.append(counter)
            lm = re.search(r"\\label\{([^}]*)\}", r)
            if lm:
                d.eq_nums[lm.group(1)] = counter
        d.eq_blocks.append((kind, rows, rnums))

    d.alg_blocks = []
    for i, blk in enumerate(env_blocks(body, "algorithm"), 1):
        inner = blk.group(2)
        cm = re.search(r"\\caption\s*\{", inner)
        cap = ""
        if cm:
            cap, end = balanced(inner, cm.end() - 1)
            inner = inner[:cm.start()] + inner[end:]
        lm = re.search(r"\\label\{([^}]*)\}", inner)
        if lm:
            d.alg_nums[lm.group(1)] = i
        am = re.search(r"\\begin\{algorithmic\}(\[[^\]]*\])?(.*?)\\end\{algorithmic\}", inner, re.S)
        body_alg = am.group(2) if am else ""
        d.alg_blocks.append({"num": i, "caption": strip_labels(cap), "code": body_alg})

    # --- 引用顺序
    d.cite_order = []
    for m in re.finditer(r"\\cite\{([^}]*)\}", tex[body_start:tex.index(r"\end{document}")]):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in d.cite_order:
                d.cite_order.append(k)
    d.cite_nums = {k: i + 1 for i, k in enumerate(d.cite_order)}

    # --- refs.bib 标题（用于引用表顺序核对）
    bib = open(BIB, encoding="utf-8").read()
    d.bib_titles = {}
    for m in re.finditer(r"@\w+\s*\{\s*([^,]+),(.*?)\n\}", bib, re.S):
        key, blk = m.group(1).strip(), m.group(2)
        t = re.search(r"title\s*=\s*[{\"](.+?)[}\"]\s*,", blk, re.S)
        if t:
            d.bib_titles[key] = re.sub(r"\s+", " ", t.group(1)).strip("{} ")

    d.tex = tex
    d.body = body
    d.body_start = body_start
    d.back_start = back_start
    json.dump({"sec": d.sec_nums, "fig": d.fig_nums, "tab": d.tab_nums,
               "eq": d.eq_nums, "alg": d.alg_nums, "cites": d.cite_order},
              open(os.path.join(TMP, "maps.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("[1] 解析完成：章节 %d，图 %d，表 %d，公式 %d，算法 %d，引用 %d 条"
        % (len(d.sec_nums), len(d.fig_nums), len(d.tab_nums), len(d.eq_nums),
           len(d.alg_blocks), len(d.cite_order)))
    return d


# ---------------------------------------------------------------- 阶段 2：中间 tex

def resolve_text(s, d):
    """把 \\ref / \\eqref / \\S\\ref / \\cite 就地解析成编号。"""
    s = re.sub(r"\\S\\ref\{([^}]*)\}", lambda m: "§" + (d.sec_nums.get(m.group(1)) or "?"), s)
    s = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + str(d.eq_nums.get(m.group(1), "?")) + ")", s)

    def ref(m):
        lab = m.group(1)
        for mp in (d.sec_nums, d.fig_nums, d.tab_nums, d.alg_nums):
            if lab in mp:
                return str(mp[lab])
        return "?"

    s = re.sub(r"\\ref\{([^}]*)\}", ref, s)

    def cite(m):
        nums = [d.cite_nums.get(k.strip(), 0) for k in m.group(1).split(",")]
        return "[" + ",".join(str(n) for n in nums) + "]"

    s = re.sub(r"\\cite\{([^}]*)\}", cite, s)
    return s


def _mark_envs(body, pattern, token):
    """把匹配的环境换成 @@TOKENn@@ 标记，返回 (新文本, 数量)。"""
    out, pos, i = [], 0, 0
    for m in re.finditer(pattern, body, re.S):
        out.append(body[pos:m.start()])
        out.append("\n\n@@%s%d@@\n\n" % (token, i))
        i += 1
        pos = m.end()
    out.append(body[pos:])
    return "".join(out), i


def _prefix_captions(body, env, prefix):
    """给某个浮空环境的 caption 加上 'Figure 1. ' / 'Table 1. ' 前缀。"""
    out, pos, i = [], 0, 0
    for m in env_blocks(body, env):
        cm = re.search(r"\\caption\s*\{", m.group(2))
        if not cm:
            continue
        i += 1
        cap_start = m.start(2) + cm.end()
        out.append(body[pos:cap_start])
        out.append("%s %d. " % (prefix, i))
        pos = cap_start
    out.append(body[pos:])
    return "".join(out)


def build_tex(d):
    """产出 pandoc 友好的中间 tex。"""
    body = d.body

    # 公式块内容规整（供后面渲染成「公式 | 编号」表格）
    for i, (kind, rows, labs) in enumerate(d.eq_blocks):
        rows_out = []
        for r in rows:
            r = re.sub(r"\s*\n\s*", " ", strip_labels(r)).strip().replace("\\mathbbm", "\\mathbb")
            rows_out.append("$" + r + "$" if kind == "equation"
                            else "$\\begin{aligned}" + r + "\\end{aligned}$")
        d.eq_blocks[i] = (kind, rows_out, [str(n) for n in labs])

    body, n_eq = _mark_envs(body, r"\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}", "EQ")
    body, n_alg = _mark_envs(body, r"\\begin\{algorithm\}(?:\[[^\]]*\])?.*?\\end\{algorithm\}", "ALG")

    # 章节编号写进标题
    n1 = n2 = n3 = 0
    chunks, last = [], 0
    for m in re.finditer(r"\\(section|subsection|subsubsection)\*?\s*\{", body):
        title, end = balanced(body, m.end() - 1)
        if m.group(1) == "section":
            n1 += 1
            n2 = n3 = 0
            num = str(n1)
        elif m.group(1) == "subsection":
            n2 += 1
            n3 = 0
            num = "%d.%d" % (n1, n2)
        else:
            n3 += 1
            num = "%d.%d.%d" % (n1, n2, n3)
        chunks.append(body[last:m.start()])
        chunks.append("\\%s{%s. %s}" % (m.group(1), num, title))
        last = end
    chunks.append(body[last:])
    body = "".join(chunks)

    # 图 / 表题注编号前缀
    body = _prefix_captions(body, "figure", "Figure")
    body = _prefix_captions(body, "table", "Table")

    # 图片改 300dpi PNG（Word 无法嵌入 PDF 矢量图）
    body = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{[^}]*?)\.pdf\}", r"\1.png}", body)
    # tabularx -> tabular，自定义列类型归一化
    body = re.sub(r"\\begin\{tabularx\}\{[^}]*\}\{([^}]*)\}",
                  lambda m: "\\begin{tabular}{" + re.sub(r"[CXLR]",
                                                          lambda x: {"C": "c", "X": "l", "L": "l", "R": "r"}[x.group(0)],
                                                          m.group(1)) + "}", body)
    body = body.replace("\\end{tabularx}", "\\end{tabular}")

    # 数学修正 + 就地解析 ref/cite
    body = body.replace("\\mathbbm", "\\mathbb")
    # pandoc 会把 \cmidrule(lr){2-4} 当普通文本漏进单元格，这里清掉
    body = re.sub(r"\\cmidrule\s*(\([^)]*\))?\s*(\{[^}]*\})?", "", body)
    body = re.sub(r"\\(toprule|midrule|bottomrule|addlinespace)\b", "", body)
    body = resolve_text(body, d)
    body = strip_labels(body)
    body = re.sub(r"^\s*\\(small|footnotesize|scriptsize|centering|noindent)\s*$", "", body, flags=re.M)

    tex = ("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n"
           + body + "\n\\end{document}\n")
    path = os.path.join(TMP, "main.word.tex")
    open(path, "w", encoding="utf-8").write(tex)
    log("[2] 中间 tex 已生成：%s（%d 字符，公式块 %d，算法块 %d）" % (path, len(tex), n_eq, n_alg))
    return path


# ---------------------------------------------------------------- 阶段 3：pandoc

def run_pandoc(src, dst, ref=None):
    cmd = ["pandoc", src, "-o", dst]
    if ref:
        cmd.append("--reference-doc=" + ref)
    r = subprocess.run(cmd, cwd=SRC_DIR, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        log("  pandoc 失败:", r.stdout[-1500:], r.stderr[-2500:])
        raise SystemExit(1)
    warns = [l for l in (r.stderr or "").splitlines() if "WARNING" in l]
    if warns:
        log("  pandoc 警告 %d 条" % len(warns))
        for l in warns[:8]:
            log("    ", l[:160])
    return dst


def setup_ref():
    ref = os.path.join(TMP, "ref.docx")
    shutil.copyfile(TEMPLATE, ref)
    return ref


# ---------------------------------------------------------------- 阶段 4：片段渲染

XMLSP = "{http://www.w3.org/XML/1998/namespace}space"


def collect_ns(xml):
    head = xml[:xml.index(">") + 1]
    ns = dict(re.findall(r'xmlns:([A-Za-z0-9]+)="([^"]+)"', head))
    m = re.search(r'\sxmlns="([^"]+)"', head)
    if m:
        ns[""] = m.group(1)
    return ns


def register_all(*xmls):
    seen = {}
    for x in xmls:
        seen.update(collect_ns(x))
    for p, u in seen.items():
        ET.register_namespace(p, u)


def load_docx(path):
    z = zipfile.ZipFile(path)
    parts = {n: z.read(n) for n in z.namelist()}
    return parts, parts["word/document.xml"].decode("utf-8")


def render_fragments(fragments, tag, ref):
    """把 LaTeX 片段逐个渲染成 docx 元素列表，返回 {序号: [元素]}。"""
    lines = ["\\documentclass{article}", "\\usepackage{amsmath}", "\\begin{document}", ""]
    for i, f in enumerate(fragments):
        lines += ["\\noindent @@%s%d@@" % (tag, i), "", f, ""]
    lines.append("\\end{document}")
    src = os.path.join(TMP, "frag_%s.tex" % tag)
    dst = os.path.join(TMP, "frag_%s.docx" % tag)
    open(src, "w", encoding="utf-8").write("\n".join(lines))
    run_pandoc(src, dst, ref)
    parts, xml = load_docx(dst)
    register_all(xml)
    _, body = body_of(xml)
    out, cur = {}, None
    for el in list(body):
        if el.tag == Q + "p":
            t = "".join(x.text or "" for x in el.iter(Q + "t")).strip()
            m = re.fullmatch(r"@@%s(\d+)@@" % tag, t)
            if m:
                cur = int(m.group(1))
                out[cur] = []
                continue
        if cur is not None and el.tag != Q + "sectPr":
            out[cur].append(el)
    return out


# ---------------------------------------------------------------- XML 构造

def body_of(xml):
    root = ET.fromstring(xml)
    return root, root.find(Q + "body")


def pstyle(p):
    pPr = p.find(Q + "pPr")
    if pPr is None:
        return None
    ps = pPr.find(Q + "pStyle")
    return ps.get(Q + "val") if ps is not None else None


def set_pstyle(p, val):
    pPr = p.find(Q + "pPr")
    if pPr is None:
        pPr = ET.Element(Q + "pPr")
        p.insert(0, pPr)
    ps = pPr.find(Q + "pStyle")
    if ps is None:
        ps = ET.Element(Q + "pStyle")
        pPr.insert(0, ps)
    ps.set(Q + "val", val)


def set_jc(p, val):
    pPr = p.find(Q + "pPr")
    if pPr is None:
        pPr = ET.Element(Q + "pPr")
        p.insert(0, pPr)
    jc = pPr.find(Q + "jc")
    if jc is None:
        jc = ET.Element(Q + "jc")
        ppr_insert(pPr, "jc", jc)
    jc.set(Q + "val", val)


def zero_indent(p):
    """把 pPr 的 indent 归零（MDPI_3.1_text 自带 left=2608 缩进，放进表格会溢出）。"""
    pPr = p.find(Q + "pPr")
    if pPr is None:
        pPr = ET.Element(Q + "pPr")
        p.insert(0, pPr)
    ind = pPr.find(Q + "ind")
    if ind is None:
        ind = ET.Element(Q + "ind")
        ppr_insert(pPr, "ind", ind)
    for a in ("hanging", "firstLine"):
        if ind.get(Q + a) is not None:
            del ind.attrib[Q + a]
    ind.set(Q + "left", "0")
    ind.set(Q + "firstLine", "0")
    ind.set(Q + "right", "0")


def make_para(text, style=None, jc=None, bold=False):
    p = ET.Element(Q + "p")
    pPr = ET.SubElement(p, Q + "pPr")
    if style:
        ET.SubElement(pPr, Q + "pStyle").set(Q + "val", style)
    if jc:
        ET.SubElement(pPr, Q + "jc").set(Q + "val", jc)
    r = ET.SubElement(p, Q + "r")
    if bold:
        ET.SubElement(ET.SubElement(r, Q + "rPr"), Q + "b")
    t = ET.SubElement(r, Q + "t")
    t.set(XMLSP, "preserve")
    t.text = text
    return p


def make_cell(children, width, borders=None):
    tc = ET.Element(Q + "tc")
    tcPr = ET.SubElement(tc, Q + "tcPr")
    tcW = ET.SubElement(tcPr, Q + "tcW")
    tcW.set(Q + "w", str(width))
    tcW.set(Q + "type", "dxa")
    if borders:
        tcB = ET.SubElement(tcPr, Q + "tcBorders")
        for side in ("top", "bottom"):
            e = ET.SubElement(tcB, Q + side)
            e.set(Q + "val", "single")
            e.set(Q + "sz", "8")
            e.set(Q + "space", "0")
            e.set(Q + "color", "000000")
    for c in children:
        tc.append(c)
    return tc


def make_table(rows, widths, three_line=True, header_rows=0):
    tbl = ET.Element(Q + "tbl")
    tblPr = ET.SubElement(tbl, Q + "tblPr")
    tw = ET.SubElement(tblPr, Q + "tblW")
    tw.set(Q + "w", str(sum(widths)))
    tw.set(Q + "type", "dxa")
    ET.SubElement(tblPr, Q + "jc").set(Q + "val", "center")
    bd = ET.SubElement(tblPr, Q + "tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = ET.SubElement(bd, Q + side)
        on = three_line and side in ("top", "bottom")
        e.set(Q + "val", "single" if on else "none")
        e.set(Q + "sz", "8" if on else "0")
        e.set(Q + "space", "0")
        e.set(Q + "color", "000000" if on else "auto")
    ET.SubElement(tblPr, Q + "tblLayout").set(Q + "type", "fixed")
    cm = ET.SubElement(tblPr, Q + "tblCellMar")
    for s in ("left", "right"):
        e = ET.SubElement(cm, Q + s)
        e.set(Q + "w", "57")
        e.set(Q + "type", "dxa")
    grid = ET.SubElement(tbl, Q + "tblGrid")
    for w in widths:
        ET.SubElement(grid, Q + "gridCol").set(Q + "w", str(w))
    for ri, row in enumerate(rows):
        tr = ET.SubElement(tbl, Q + "tr")
        for ci, cells in enumerate(row):
            hdr = ri < header_rows
            tc = make_cell(cells, widths[ci],
                           borders=("bottom",) if (hdr and three_line) else None)
            tr.append(tc)
    return tbl


def apply_three_line(tbl):
    """给已有表格加三线边框（去掉 pandoc 的 tblStyle）。"""
    tblPr = tbl.find(Q + "tblPr")
    if tblPr is None:
        tblPr = ET.Element(Q + "tblPr")
        tbl.insert(0, tblPr)
    st = tblPr.find(Q + "tblStyle")
    if st is not None:
        tblPr.remove(st)
    old = tblPr.find(Q + "tblBorders")
    if old is not None:
        tblPr.remove(old)
    bd = ET.Element(Q + "tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = ET.SubElement(bd, Q + side)
        on = side in ("top", "bottom")
        e.set(Q + "val", "single" if on else "none")
        e.set(Q + "sz", "8" if on else "0")
        e.set(Q + "space", "0")
        e.set(Q + "color", "000000" if on else "auto")
    after = ("shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription")
    idx = len(tblPr)
    for i, ch in enumerate(tblPr):
        if ch.tag.replace(Q, "") in after:
            idx = i
            break
    tblPr.insert(idx, bd)
    # 表头行加下边框
    rows = tbl.findall(Q + "tr")
    if rows:
        for tc in rows[0].findall(Q + "tc"):
            tcPr = tc.find(Q + "tcPr")
            if tcPr is None:
                tcPr = ET.Element(Q + "tcPr")
                tc.insert(0, tcPr)
            tb = ET.SubElement(tcPr, Q + "tcBorders")
            e = ET.SubElement(tb, Q + "bottom")
            e.set(Q + "val", "single")
            e.set(Q + "sz", "8")
            e.set(Q + "space", "0")
            e.set(Q + "color", "000000")
    for tc in tbl.iter(Q + "tc"):
        for p in tc.findall(Q + "p"):
            if pstyle(p) in (None, "Compact", "BodyText", "FirstParagraph", "SourceCode"):
                set_pstyle(p, "MDPI42tablebody")
    AFTER_B = ("bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
               "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish",
               "webHidden", "color", "spacing", "w", "kern", "position", "sz", "szCs",
               "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl",
               "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath")
    for tc in (rows[0].findall(Q + "tc") if rows else []):
        for r in tc.iter(Q + "r"):
            rPr = r.find(Q + "rPr")
            if rPr is None:
                rPr = ET.Element(Q + "rPr")
                r.insert(0, rPr)
            if rPr.find(Q + "b") is not None:
                continue
            idx = len(rPr)
            for k, ch in enumerate(rPr):
                if ch.tag.replace(Q, "") in AFTER_B:
                    idx = k
                    break
            rPr.insert(idx, ET.Element(Q + "b"))


# ---------------------------------------------------------------- 阶段 4b：参考文献

def extract_references():
    import pymupdf
    doc = pymupdf.open(PDF)
    txt = "\n".join(pg.get_text() for pg in doc)
    idx = txt.rfind("\nReferences\n")
    if idx < 0:
        idx = txt.rfind("References")
    lines = [l.strip() for l in txt[idx:].split("\n")]
    lines = [l for l in lines if l and not re.fullmatch(r"\d+", l)
             and not re.fullmatch(r"\d+ of \d+", l)]
    entries, cur = [], None
    for l in lines:
        m = re.fullmatch(r"(\d+)\.", l)
        if m and int(m.group(1)) == len(entries) + 1:
            cur = []
            entries.append(cur)
            continue
        if cur is None:
            continue
        if re.match(r"^(©|Disclaimer|Publisher|Conflicts of Interest|References)", l):
            break
        cur.append(l)
    out = []
    for e in entries:
        s = ""
        for l in e:
            if s.endswith("-") and l[:1].islower():
                s = s[:-1] + l
            else:
                s = (s + " " + l) if s else l
        out.append(re.sub(r"\s+", " ", s).strip())
    return out


def match_references(d, entries):
    """把 PDF 参考文献条目匹配到 refs.bib 的 key。"""
    pairs = {}
    used = set()
    for i, e in enumerate(entries):
        best, score = None, 0.0
        for k in d.cite_order:
            if k in used:
                continue
            t = d.bib_titles.get(k)
            if not t:
                continue
            m = difflib.SequenceMatcher(None, t.lower(), e.lower()).find_longest_match()
            r = m.size / max(1, len(t))
            if r > score:
                best, score = k, r
        if best:
            used.add(best)
        pairs[i] = (best, score)
    return pairs


# ---------------------------------------------------------------- 阶段 5：组装

STYLE_MAP = {
    "Heading1": "MDPI21heading1",
    "Heading2": "MDPI22heading2",
    "Heading3": "MDPI23heading3",
    "FirstParagraph": "MDPI31text",
    "BodyText": "MDPI31text",
    "Compact": "MDPI31text",
    "CaptionedFigure": "MDPI52figure",
    "ImageCaption": "MDPI51figurecaption",
    "TableCaption": "MDPI41tablecaption",
}


def remap_styles(body):
    n = 0
    for child in body:
        if child.tag == Q + "p":
            s = pstyle(child)
            if s in STYLE_MAP:
                set_pstyle(child, STYLE_MAP[s])
                n += 1
    return n


MNS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
QM = "{" + MNS + "}"

PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl",
             "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens",
             "kinsoku", "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE",
             "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind",
             "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc", "textDirection",
             "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
             "sectPr"]


def ppr_insert(pPr, name, elem):
    """按 OOXML 规定顺序把 elem 插入到 pPr 中。"""
    pos = PPR_ORDER.index(name)
    idx = len(pPr)
    for i, ch in enumerate(pPr):
        nm = ch.tag.replace(Q, "")
        if nm in PPR_ORDER and PPR_ORDER.index(nm) > pos:
            idx = i
            break
    pPr.insert(idx, elem)


def tab_run():
    r = ET.Element(Q + "r")
    ET.SubElement(r, Q + "tab")
    return r


def make_eq_para(p, num):
    """把公式段落改成：居中制表位放公式，右制表位放编号。"""
    set_pstyle(p, "MDPI31text")
    zero_indent(p)
    pPr = p.find(Q + "pPr")
    if pPr is None:
        pPr = ET.Element(Q + "pPr")
        p.insert(0, pPr)
    jc = pPr.find(Q + "jc")
    if jc is None:
        jc = ET.Element(Q + "jc")
        ppr_insert(pPr, "jc", jc)
    jc.set(Q + "val", "left")
    tabs = ET.Element(Q + "tabs")
    for val, pos in (("center", "5233"), ("right", "10466")):
        t = ET.SubElement(tabs, Q + "tab")
        t.set(Q + "val", val)
        t.set(Q + "pos", pos)
    ppr_insert(pPr, "tabs", tabs)
    mi = next((k for k, c in enumerate(p) if c.tag == QM + "oMath"
               or c.tag == QM + "oMathPara"), len(p))
    p.insert(mi, tab_run())
    r = ET.SubElement(p, Q + "r")
    ET.SubElement(r, Q + "tab")
    t = ET.SubElement(r, Q + "t")
    t.set(XMLSP, "preserve")
    t.text = "(%s)" % num
    return p


TEXT_W = 10466          # body 正文可用宽度（twips）= 页宽 11906 − 左右页边距 720×2
MIN_COL = 260           # 单列最小宽度（twips）≈ 0.18 英寸

TCPR_ORDER = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd",
              "noWrap", "tcMar", "textDirection", "tcFitText", "vAlign", "hideMark",
              "headers"]


def tcpr_insert(tcPr, name, elem):
    pos = TCPR_ORDER.index(name)
    idx = len(tcPr)
    for i, ch in enumerate(tcPr):
        nm = ch.tag.replace(Q, "")
        if nm in TCPR_ORDER and TCPR_ORDER.index(nm) > pos:
            idx = i
            break
    tcPr.insert(idx, elem)


def _cell_span(tc):
    gs = tc.find(Q + "tcPr/" + Q + "gridSpan")
    return int(gs.get(Q + "val")) if gs is not None else 1


def normalize_tables(body):
    """给每张表写死列宽：显式 tblGrid + 单元格 tcW + 通栏宽度 + 居中，杜绝出界。"""
    n = 0
    for tbl in body.findall(Q + "tbl"):
        rows = tbl.findall(Q + "tr")
        if not rows:
            continue
        ncols = 0
        for tr in rows:
            ncols = max(ncols, sum(_cell_span(tc) for tc in tr.findall(Q + "tc")))
        if ncols == 0:
            continue
        lens = [0] * ncols
        for tr in rows:
            ci = 0
            for tc in tr.findall(Q + "tc"):
                span = _cell_span(tc)
                if span == 1 and 0 <= ci < ncols:
                    txt = "".join(t.text or "" for t in tc.iter(Q + "t"))
                    lens[ci] = max(lens[ci], len(txt))
                ci += span
        weights = [max(float(MIN_COL), v * 58.0) for v in lens]
        total = sum(weights)
        widths = [max(MIN_COL, int(TEXT_W * w / total)) for w in weights]
        diff = TEXT_W - sum(widths)
        if diff > 0:
            widths[widths.index(max(widths))] += diff
        elif diff < 0:
            k = widths.index(max(widths))
            widths[k] = max(MIN_COL, widths[k] + diff)

        tblPr = tbl.find(Q + "tblPr")
        if tblPr is None:
            tblPr = ET.Element(Q + "tblPr")
            tbl.insert(0, tblPr)
        for tag in ("tblW", "tblInd", "tblLayout", "jc"):
            e = tblPr.find(Q + tag)
            if e is not None:
                tblPr.remove(e)
        tw = ET.Element(Q + "tblW")
        tw.set(Q + "w", str(TEXT_W))
        tw.set(Q + "type", "dxa")
        tblPr.insert(0, tw)
        lay = ET.Element(Q + "tblLayout")
        lay.set(Q + "type", "fixed")
        tblPr.append(lay)
        jc = ET.Element(Q + "jc")
        jc.set(Q + "val", "center")
        tblPr.append(jc)

        grid = tbl.find(Q + "tblGrid")
        if grid is not None:
            tbl.remove(grid)
        grid = ET.Element(Q + "tblGrid")
        for w in widths:
            ET.SubElement(grid, Q + "gridCol").set(Q + "w", str(w))
        tbl.insert(list(tbl).index(tblPr) + 1, grid)

        for tr in rows:
            ci = 0
            for tc in tr.findall(Q + "tc"):
                span = _cell_span(tc)
                tcPr = tc.find(Q + "tcPr")
                if tcPr is None:
                    tcPr = ET.Element(Q + "tcPr")
                    tc.insert(0, tcPr)
                old = tcPr.find(Q + "tcW")
                if old is not None:
                    tcPr.remove(old)
                tcw = ET.Element(Q + "tcW")
                tcw.set(Q + "w", str(sum(widths[ci:ci + span]) or MIN_COL))
                tcw.set(Q + "type", "dxa")
                tcpr_insert(tcPr, "tcW", tcw)
                ci += span
        n += 1
    return n


def center_and_flatten(body):
    """图片段与图表题注：显式居中 + 去掉 MDPI_3.1_text 自带的 2608 缩进。"""
    n = 0
    for p in body.findall(Q + "p"):
        sty = pstyle(p)
        if sty in ("MDPI52figure", "MDPI51figurecaption", "MDPI41tablecaption",
                   "MDPI511onefigurecaption", "MDPI411onetablecaption"):
            set_jc(p, "center")
            zero_indent(p)
            n += 1
    return n


def marker_index(body, token):
    out = {}
    for i, ch in enumerate(list(body)):
        if ch.tag == Q + "p":
            t = "".join(x.text or "" for x in ch.iter(Q + "t")).strip()
            m = re.fullmatch(r"@@%s(\d+)@@" % token, t)
            if m:
                out[int(m.group(1))] = i
    return out


def add_before_sect(body, el):
    """插入到 w:body 末尾的 w:sectPr 之前（sectPr 必须是最后一个子元素）。"""
    for i, ch in enumerate(body):
        if ch.tag == Q + "sectPr":
            body.insert(i, el)
            return
    body.append(el)


def build_front(d):
    a = tex_clean(re.sub(r"\$[^$]*\$", "", d.address))
    frags = ["Article",
             tex_clean(d.title),
             "Jinhui Huang $^{1,}$*",
             "1 " + a,
             "* " + tex_clean(d.corres),
             "\\textbf{Featured Application}",
             d.featured,
             "\\textbf{Abstract}",
             d.abstract,
             "\\textbf{Keywords:} " + d.keywords]
    styles = ["MDPI11articletype", "MDPI12title", "MDPI13authornames",
              "MDPI16affiliation", "MDPI16affiliation",
              "MDPI17abstract", "MDPI17abstract", "MDPI17abstract",
              "MDPI17abstract", "MDPI18keywords"]
    return frags, styles


def build_eq_fragments(d):
    """每个公式块 -> 一个片段；块内每行公式各自成段（行内已带 $ 定界符）。"""
    return ["\n\n".join(rows) for kind, rows, nums in d.eq_blocks]


def alg_lines(code):
    """把 algpseudocode 源码转成逐行 LaTeX（返回 [(缩进, 文本)]）。"""
    ctrl = ("State", "Require", "Ensure", "If", "ElsIf", "Else", "EndIf", "ForAll", "For",
            "EndFor", "While", "EndWhile", "Return", "Procedure", "EndProcedure", "Statex")
    out = []
    level = 0
    # 先按控制命令切分语句
    stmts = []
    cur = None
    for raw in code.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        m = re.match(r"\s*\\([A-Za-z]+)", line)
        name = m.group(1) if m else None
        if name in ctrl and not (cur and name == "Comment"):
            if cur:
                stmts.append(cur)
            cur = line.strip()
        else:
            cur = (cur + " " + line.strip()) if cur else line.strip()
    if cur:
        stmts.append(cur)

    def take(s, name):
        m = re.match(r"\\" + name + r"\s*\{", s)
        if not m:
            return s
        inner, end = balanced(s, m.end() - 1)
        return inner + s[end:]

    for s in stmts:
        name = re.match(r"\\([A-Za-z]+)", s).group(1)
        rest = s[len(name) + 1:].strip()
        if name == "State":
            out.append((level, rest))
        elif name == "Statex":
            out.append((level, ""))
        elif name == "Require":
            out.append((level, "\\textbf{Require:} " + rest))
        elif name == "Ensure":
            out.append((level, "\\textbf{Ensure:} " + rest))
        elif name in ("If",):
            out.append((level, "\\textbf{if} " + take(rest, "If") + " \\textbf{then}"))
            level += 1
        elif name == "ElsIf":
            level = max(0, level - 1)
            out.append((level, "\\textbf{else if} " + take(rest, "ElsIf")))
            level += 1
        elif name == "Else":
            level = max(0, level - 1)
            out.append((level, "\\textbf{else}"))
            level += 1
        elif name in ("EndIf", "EndFor", "EndWhile", "EndProcedure"):
            level = max(0, level - 1)
            out.append((level, "\\textbf{end " + name[3:].lower() + "}"))
        elif name == "ForAll":
            out.append((level, "\\textbf{for all} " + take(rest, "ForAll") + " \\textbf{do}"))
            level += 1
        elif name == "For":
            out.append((level, "\\textbf{for} " + take(rest, "For") + " \\textbf{do}"))
            level += 1
        elif name == "While":
            out.append((level, "\\textbf{while} " + take(rest, "While") + " \\textbf{do}"))
            level += 1
        elif name == "Return":
            out.append((level, "\\textbf{return} " + rest))
        elif name == "Procedure":
            out.append((level, "\\textbf{procedure} " + rest))
            level += 1
        else:
            out.append((level, rest))
    return out


def build_alg_fragments(d):
    frags = []
    for a in d.alg_blocks:
        ls = alg_lines(resolve_text(a["code"], d))
        frags.append("\n\n".join("\\noindent " + ("\\quad " * lv) + (t if t else "~") for lv, t in ls))
    return frags


def build_back(d):
    order = ["Author Contributions", "Funding", "Institutional Review Board Statement",
             "Informed Consent Statement", "Data Availability Statement",
             "Acknowledgments", "Conflicts of Interest"]
    frags, labels = [], []
    for k in order:
        txt = d.back.get(k, "")
        if not txt:
            continue
        txt = resolve_text(txt, d)
        txt = txt.replace("\\\\", " ").replace("\\url", "\\texttt")
        frags.append("\\textbf{" + k + ":} " + txt)
        labels.append(k)
    return frags, labels


def build_abbrev(d):
    lines = [l.strip() for l in d.abbrev_text.split("\n") if l.strip()]
    pairs = []
    for l in lines:
        if l.startswith("The following"):
            continue
        m = re.match(r"([^:]+):\s*(.*?)\\\\$", l)
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


def build_ref_frags(d):
    entries = extract_references()
    pairs = match_references(d, entries)
    by_key = {}
    for i, (k, sc) in pairs.items():
        if k and (k not in by_key or sc > by_key[k][1]):
            by_key[k] = (i, sc)
    frags = []
    for n, k in enumerate(d.cite_order, 1):
        if k in by_key:
            frags.append(tex_clean(entries[by_key[k][0]]))
        else:
            frags.append("")
    return frags, entries, pairs


def write_docx(path, parts, root):
    """把当前 root / parts 写成 docx。"""
    newxml = ET.tostring(root, encoding="unicode")
    newxml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n' + newxml
    parts["word/document.xml"] = newxml.encode("utf-8")
    st = parts["word/styles.xml"].decode("utf-8")
    if 'w:styleId="VerbatimChar"' not in st:
        add = ('<w:style w:type="character" w:styleId="VerbatimChar">'
               '<w:name w:val="Verbatim Char"/>'
               '<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
               '<w:sz w:val="18"/></w:rPr></w:style>')
        st = st.replace("</w:styles>", add + "</w:styles>")
        parts["word/styles.xml"] = st.encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in parts.items():
            z.writestr(n, data)


def snap(n, parts, root):
    """分步快照，便于用 Word 逐步骤定位损坏点。"""
    p = os.path.join(TMP, "step%d.docx" % n)
    write_docx(p, parts, root)
    return p


if __name__ == "__main__":
    doc = parse()
    mid = build_tex(doc)
    ref = setup_ref()
    log("[3] pandoc 转换主文档 ...")
    raw = os.path.join(TMP, "raw.docx")
    run_pandoc(mid, raw, ref)
    log("[3] 完成 ->", raw)

    log("[4] 渲染片段 ...")
    ffrags, fstyles = build_front(doc)
    front = render_fragments(ffrags, "FRONT", ref)
    eqs = render_fragments(build_eq_fragments(doc), "EQF", ref)
    algs = render_fragments(build_alg_fragments(doc), "ALGF", ref)
    bfrags, blabels = build_back(doc)
    back = render_fragments(bfrags, "BACK", ref)
    abbrev = build_abbrev(doc)
    rfrags, ref_entries, ref_pairs = build_ref_frags(doc)
    refs = render_fragments([r for r in rfrags if r], "REF", ref)
    log("    片段：前置 %d，公式 %d，算法 %d，文末 %d，缩写 %d，参考文献 %d"
        % (len(front), len(eqs), len(algs), len(back), len(abbrev), len(refs)))

    log("[5] 组装 docx ...")
    parts, rawxml = load_docx(raw)
    register_all(rawxml)
    root, body = body_of(rawxml)

    remap_styles(body)
    for ch in list(body):
        if ch.tag == Q + "tbl":
            apply_three_line(ch)
    snap(1, parts, root)

    # 前置信息
    for i in reversed(range(len(fstyles))):
        for el in front.get(i, [])[:1]:
            set_pstyle(el, fstyles[i])
        for el in front.get(i, [])[1:]:
            set_pstyle(el, "MDPI31text")
        first = body[0] if len(body) else None
        for el in reversed(front.get(i, [])):
            body.insert(0, el)
    snap(2, parts, root)

    # 公式块（公式段落：居中制表位＋右制表位编号）
    slots = marker_index(body, "EQ")
    miss = 0
    for i, idx in sorted(slots.items(), reverse=True):
        groups = eqs.get(i, [])
        nums = doc.eq_blocks[i][2]
        if len(groups) != len(nums):
            miss += 1
            log("    ! 公式块 %d 渲染段落 %d 个，编号 %d 个" % (i, len(groups), len(nums)))
        els = [make_eq_para(p, n) for p, n in zip(groups, nums)]
        if not els:
            continue
        body[idx] = els[0]
        for k, e in enumerate(els[1:], 1):
            body.insert(idx + k, e)
    if miss:
        log("    公式块数量不匹配：%d 块" % miss)
    snap(3, parts, root)

    # 算法块
    slots = marker_index(body, "ALG")
    for i, idx in sorted(slots.items(), reverse=True):
        a = doc.alg_blocks[i]
        groups = algs.get(i, [])
        rows = []
        for j, p in enumerate(groups, 1):
            set_pstyle(p, "MDPI31text")
            zero_indent(p)
            set_jc(p, "left")
            nump = make_para(str(j), style="MDPI42tablebody")
            rows.append([[nump], [p]])
        cap = make_para("Algorithm %d. %s" % (a["num"], tex_clean(a["caption"])),
                        style="MDPI51figurecaption")
        tbl = make_table(rows, [900, 9500], three_line=True)
        body[idx] = cap
        body.insert(idx + 1, tbl)
    snap(4, parts, root)

    # 文末声明
    for i, key in enumerate(blabels):
        for el in back.get(i, []):
            set_pstyle(el, "MDPI62backmatter")
            add_before_sect(body, el)
    snap(5, parts, root)

    # 缩写表
    add_before_sect(body, make_para("Abbreviations", style="MDPI21heading1"))
    add_before_sect(body, make_para("The following abbreviations are used in this manuscript:",
                                    style="MDPI32textnoindent"))
    arows = [[[make_para(k, style="MDPI42tablebody")], [make_para(v, style="MDPI42tablebody")]]
             for k, v in abbrev]
    add_before_sect(body, make_table(arows, [1600, 8800], three_line=True))
    snap(6, parts, root)

    # 参考文献
    add_before_sect(body, make_para("References", style="MDPI21heading1"))
    for n, k in enumerate(doc.cite_order, 1):
        pars = refs.get(n - 1, [])
        if not pars:
            continue
        p = pars[0]
        set_pstyle(p, "MDPI81references")
        # 去掉样式可能自带的项目符号/编号
        pPr = p.find(Q + "pPr")
        if pPr is not None:
            for tag in ("numPr", "ind"):
                e = pPr.find(Q + tag)
                if e is not None:
                    pPr.remove(e)
        txt = "".join(x.text or "" for x in p.iter(Q + "t"))
        add_before_sect(body, make_para("%d. %s" % (n, txt), style="MDPI81references"))
    snap(7, parts, root)

    # 版式收尾：表格列宽归一化（防出界）+ 图片/题注居中
    n_tbl = normalize_tables(body)
    n_cap = center_and_flatten(body)
    log("    版式：表格列宽归一化 %d 张，图片/题注居中 %d 段" % (n_tbl, n_cap))
    snap(8, parts, root)

    # 写出
    write_docx(OUT_DOCX, parts, root)
    log("[5] 已写出：", OUT_DOCX, "（%.1f KB）" % (os.path.getsize(OUT_DOCX) / 1024.0))

