# -*- coding: utf-8 -*-
"""tools/make_latex_flat.py — 生成符合 Applied Intelligence 投稿要求的
「平铺 + 无自制宏 + 纯 ASCII」LaTeX 源文件包。

期刊 Tips 要求：
  1. All files must be on the same folder level; do not use sub folders
  2. Do not use own macros or fonts
  3. Ensure special characters have been converted to appropriate Tex code
     （本稿正文无特殊字符；但文件头的中文注释需清除，否则出版方编译可能报错）

本脚本：
  · 去掉全部含非 ASCII 字符的注释行（正文内容不含中文，故只影响注释）
  · 展开 \\ie / \\eg / \\etal 三个自制宏为字面文本
  · 把 figures/xxx.pdf 的引用改为 xxx.pdf，并把图片复制到同级目录
  · 保留官方原版 sn-jnl.cls 与 .bst
输出：paper/submission_package/latex_source_flat/  +  latex_source_flat.zip
"""
import io
import os
import re
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper", "submission_springer")
DST = os.path.join(ROOT, "paper", "submission_package", "latex_source_flat")


def is_ascii(s):
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)

    tex = io.open(os.path.join(SRC, "main.tex"), encoding="utf-8").read()
    lines = tex.split("\n")
    kept, dropped, trimmed = [], 0, 0
    for ln in lines:
        if is_ascii(ln):
            kept.append(ln)
            continue
        # 含非 ASCII：**只剥离注释部分**（% 到行尾），保留其前的英文正文。
        # 早期版本按「整行删除」处理，会把
        #   \item Ethics approval ... % V005 中文备注
        # 这类「英文正文 + 中文行内注释」的行整体删掉，导致声明项丢失。
        code = ln.split("%", 1)[0] if "%" in ln else ln
        if code.strip():
            kept.append(code)
            trimmed += 1
        else:
            dropped += 1
    tex = "\n".join(kept)

    # 展开自制宏（Tips: do not use own macros）
    # 内联为 \emph{...}，从而保留斜体与原有标点（不可替换为 "i.e.,"，
    # 否则原文的 \ie, 会变成 "i.e.,," 重复逗号）
    tex = tex.replace("\\newcommand{\\ie}{\\emph{i.e.}}\n", "")
    tex = tex.replace("\\newcommand{\\eg}{\\emph{e.g.}}\n", "")
    tex = tex.replace("\\newcommand{\\etal}{\\emph{et al.}}\n", "")
    tex = re.sub(r"\\ie\b", r"\\emph{i.e.}", tex)
    tex = re.sub(r"\\eg\b", r"\\emph{e.g.}", tex)
    tex = re.sub(r"\\etal\b", r"\\emph{et al.}", tex)

    # 图片平铺：figures/x.pdf -> x.pdf
    tex = tex.replace("{figures/", "{")

    io.open(os.path.join(DST, "main.tex"), "w", encoding="ascii",
            errors="strict").write(tex)

    shutil.copy(os.path.join(SRC, "sn-jnl.cls.orig"), os.path.join(DST, "sn-jnl.cls"))
    shutil.copy(os.path.join(SRC, "sn-mathphys-num.bst"), os.path.join(DST, "sn-mathphys-num.bst"))
    shutil.copy(os.path.join(SRC, "cvmbib.bib"), os.path.join(DST, "cvmbib.bib"))
    nfig = 0
    for f in os.listdir(os.path.join(SRC, "figures")):
        if f.endswith(".pdf"):
            shutil.copy(os.path.join(SRC, "figures", f), os.path.join(DST, f))
            nfig += 1

    # 自检：确认无非 ASCII、无 \ie/\eg/\etal、无 figures/ 子目录引用
    out = io.open(os.path.join(DST, "main.tex"), encoding="ascii").read()
    problems = []
    for k in ["Competing interests", "Ethics approval", "Consent for publication",
              "Data availability", "Materials availability", "Code availability",
              "Author contribution"]:
        if k not in out:
            problems.append("Declarations 缺少 %s" % k)
    if re.search(r"\\ie\b|\\eg\b|\\etal\b", out):
        problems.append("仍存在自制宏")
    if "figures/" in out:
        problems.append("仍存在 figures/ 子目录引用")
    if not is_ascii(out):
        problems.append("仍含非 ASCII 字符")
    if "\\newcommand" in out:
        problems.append("仍含 newcommand")

    print("已生成 %s" % DST)
    print("  移除中文注释行：%d" % dropped)
    print("  图片平铺：%d 张" % nfig)
    print("  自检：%s" % ("通过" if not problems else "未通过 -> " + "; ".join(problems)))

    zpath = os.path.join(ROOT, "paper", "submission_package", "latex_source_flat.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(DST)):
            z.write(os.path.join(DST, f), "latex_source_flat/" + f)
    print("  已打包：%s (%.2f MB)" % (zpath, os.path.getsize(zpath) / 1048576))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())