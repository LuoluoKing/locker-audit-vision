# -*- coding: utf-8 -*-
"""tools/latex_fix_missing.py — 自动补齐 LaTeX 缺失宏包（tlnet 镜像）

用法：python tools/latex_fix_missing.py <编译目录> [最大轮次]

循环：读编译日志中的「File `X.sty' not found」→ 从 tlnet archive 下载对应
texlive 包 → 安装到用户 texmf → initexmf --update-fndb → 重新编译。
仅从官方 TeX Live 镜像取包，不修改稿件内容。
"""
import io
import os
import re
import sys
import socket
import subprocess
import tarfile
import urllib.request

for k in list(os.environ):
    if "proxy" in k.lower():
        del os.environ[k]
socket.setdefaulttimeout(120)

TEXMF = os.path.expandvars(r"%APPDATA%/MiKTeX/tex/latex")
# 用户级 texmf 根（TDS 结构：tex/ fonts/ bibtex/ ...）
TEXMFROOT = os.path.expandvars(r"%APPDATA%/MiKTeX")
BASE = "https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/tlnet/archive/"
# sty 名 ≠ tlnet 包名时的别名表（多为历史遗留命名）
ALIASES = {
    "tabto": ["tabto-ltx"],          # tlnet 包名为 tabto-ltx
    "mathpazo": ["psnfss"],          # mathpazo.sty 在 psnfss 包里
    "cuted": ["sttools"],            # cuted 属于 sttools 集合
    "attrib": ["frankenstein"],      # attrib.sty 属于 frankenstein 集合
    "upgreek": ["was"],              # upgreek.sty 属于 was 集合
    "pifont": ["psnfss"],            # pifont.sty 在 psnfss 包里
    "bbm": ["bbm-macros"],           # bbm.sty 在 bbm-macros 包里（bbm 包只有字体）
    "scrextend": ["koma-script"],    # scrextend 属于 koma-script 集合
    "icomma": ["was"],               # icomma.sty 属于 was 集合
    "amsthm": ["amscls"],            # amsthm.sty 属于 amscls 集合
}
PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   ".venv", "Scripts", "python.exe")


def missing_styles(log_path):
    txt = io.open(log_path, encoding="utf-8", errors="ignore").read()
    return sorted(set(re.findall(r"File `([A-Za-z0-9_\-]+)\.sty' not found", txt)))


_TRIED = set()


def fetch_and_install(styles):
    ok = []
    for st in styles:
        pkgs = st.replace(".sty", "")
        # sty 名通常就是包名；少数属于集合包，见 ALIASES
        cands = []
        for c in ALIASES.get(pkgs, []) + [pkgs]:
            cands.append(c + ".tar.xz")
            cands.append(c + ".zip")
        for cand in cands:
            if cand in _TRIED:          # 本轮之前已试过且未产出该文件，不再重试
                continue
            _TRIED.add(cand)
            try:
                with urllib.request.urlopen(BASE + cand, timeout=120) as r:
                    data = r.read()
            except Exception:
                continue
            pkg = cand.rsplit(".", 2)[0].rsplit(".", 1)[0]
            if pkg.endswith(".tar"):
                pkg = pkg[:-4]
            if cand.endswith(".zip"):
                import zipfile
                z = zipfile.ZipFile(io.BytesIO(data))
                entries = [(n, z.read(n)) for n in z.namelist() if not n.endswith("/")]
            else:
                tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:xz")
                entries = [(m.name, tf.extractfile(m).read())
                           for m in tf.getmembers() if m.isfile()]
            ext2dir = {
                ".sty": "tex/latex", ".cls": "tex/latex", ".def": "tex/latex",
                ".cfg": "tex/latex", ".fd": "tex/latex", ".clo": "tex/latex",
                ".ldf": "tex/latex", ".bbx": "tex/latex", ".cbx": "tex/latex",
                ".tex": "tex/latex", ".bbx": "tex/latex",
                ".bst": "bibtex/bst",
                ".tfm": "fonts/tfm", ".vf": "fonts/vf", ".pfb": "fonts/type1",
                ".afm": "fonts/afm", ".enc": "fonts/enc",
                ".otf": "fonts/opentype", ".ttf": "fonts/truetype",
            }
            for n, blob in entries:
                base = os.path.basename(n)
                ext = os.path.splitext(base)[1].lower()
                if ext in (".map",):
                    dest = os.path.join(TEXMFROOT, "fonts/map/dvips", pkg)
                elif ext in ext2dir:
                    dest = os.path.join(TEXMFROOT, ext2dir[ext], pkg)
                else:
                    continue
                os.makedirs(dest, exist_ok=True)
                open(os.path.join(dest, base), "wb").write(blob)
                ok.append(base)
            break
        else:
            print("    未找到包：%s" % st)
    if ok:
        subprocess.run(["initexmf", "--update-fndb"], capture_output=True)
    return ok


def main():
    wd = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    for i in range(rounds):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                           cwd=wd, capture_output=True, text=True, errors="ignore")
        log = os.path.join(wd, "main.log")
        if not os.path.exists(log):
            print("无日志，编译未开始")
            return 1
        miss = missing_styles(log)
        txt = io.open(log, encoding="utf-8", errors="ignore").read()
        fatal = "!  ==> Fatal error" in txt or "Emergency stop" in txt
        if not miss:
            print("第 %d 轮：无缺失宏包，致命错误=%s" % (i + 1, fatal))
            m = re.search(r"Output written on main\.pdf \((\d+) pages", txt)
            print("页数：", m.group(1) if m else "（未生成 PDF）")
            return 0 if m else 2
        print("第 %d 轮：缺失 %s" % (i + 1, ", ".join(miss)))
        installed = fetch_and_install(miss)
        print("    已安装：", installed if installed else "（无）")
        if not installed:
            return 3
    print("达到最大轮次仍未解决")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())