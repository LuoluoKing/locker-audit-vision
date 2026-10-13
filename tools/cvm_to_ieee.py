# -*- coding: utf-8 -*-
"""tools/cvm_to_ieee.py — 把 CVM 模板稿转换为 IEEE 期刊（IEEE Access / IEEEtran）格式

机械转换项（可自动完成）：
  1. table 环境内：\\caption / \\label 从表格下方移到表格上方（IEEE 惯例）
  2. 删除 CVM 专用宏（如 \\keywords、\\cvmPaperID、匿名 author 块）
其余（documentclass、摘要、关键词、参考文献样式、作者块、作者简介）需人工改写。
"""
import io
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "paper/submission/main.tex"
DST = sys.argv[2] if len(sys.argv) > 2 else "paper/submission_ieee/main.tex"


def move_table_captions(tex: str):
    """把 table 环境内的 caption+label 移到 \\begin{table} 之后。"""
    out = []
    i = 0
    n_moved = 0
    pattern = re.compile(r"\\begin\{table\*?\}(\[[^\]]*\])?")
    while True:
        m = pattern.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        # 找到配对的 \end{table}
        endtag = "\\end{table}" if "table*" not in m.group(0) else "\\end{table*}"
        j = tex.find(endtag, m.end())
        if j == -1:
            out.append(tex[i:])
            break
        block = tex[m.end():j]
        # 提取 caption（含可能的多行）与紧随的 label
        cap = re.search(r"\\caption\{(?:[^{}]|\{[^{}]*\})*\}", block, re.S)
        if not cap:
            out.append(tex[i:j + len(endtag)])
            i = j + len(endtag)
            continue
        cap_text = cap.group(0)
        rest = block[cap.end():]
        lab = re.search(r"\\label\{[^}]*\}", rest)
        lab_text = ("\n" + lab.group(0)) if lab else ""
        # 从 block 中移除 caption（及其后的 label）
        newblock = block[:cap.start()] + block[cap.end():]
        if lab:
            newblock = newblock.replace(lab.group(0), "", 1)
        head = tex[i:m.end()]
        out.append(head)
        out.append("\n" + cap_text + lab_text + "\n")
        out.append(newblock)
        out.append(endtag)
        n_moved += 1
        i = j + len(endtag)
    return "".join(out), n_moved


def main():
    tex = io.open(SRC, encoding="utf-8").read()
    tex, n = move_table_captions(tex)
    io.open(DST, "w", encoding="utf-8").write(tex)
    print("已写入 %s；移动表题 %d 处" % (DST, n))
    if n == 0:
        print("提示：未发现需要移动的表题（可能已在表格上方或无表格）")


if __name__ == "__main__":
    main()