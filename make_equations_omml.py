#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 12 个公式从 PNG 图片转换为 Word 原生公式 (OMML)
====================================================
背景
--------------------------------------------------------------------------
原实现 (make_equations.py) 用 matplotlib 把公式渲染成 PNG 再插入文档，
结果是**图片**：在 Word 里无法编辑、放大会模糊、审稿人要求改符号时无法修改。
多数期刊明确要求公式使用可编辑的公式对象。

做法
--------------------------------------------------------------------------
OMML (Office Math Markup Language) 是 Word 公式的原生 XML 表示。
借助 pandoc（其内置 texmath 可将 LaTeX 数学式转换为 OMML）：

  1. 把 12 个 LaTeX 公式写成 markdown（行间公式 $$...$$）
  2. `pandoc -f markdown -t docx` 生成一个临时 docx（其中的公式已是原生 OMML）
  3. 解压 docx，从 word/document.xml 中按顺序抽取 <m:oMathPara> 元素
  4. 补上命名空间声明后写入 equations_omml.py，供 build_manuscript_zh.py 导入使用

产物
--------------------------------------------------------------------------
  equations_omml.py      EQ_OMML = {编号: OMML XML 字符串}
  （若 pandoc 不可用则报错退出，不会破坏原有 PNG 流程）

用法
--------------------------------------------------------------------------
    python make_equations_omml.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_PY = os.path.join(BASE, "equations_omml.py")

# 与 make_equations.py 完全一致的 12 个公式（LaTeX）
EQS = [
    (1, r"X^{(l+1)} = \sigma\!\left(\hat{D}^{-\frac{1}{2}}\,\hat{A}\,\hat{D}^{-\frac{1}{2}}\,X^{(l)}\,W^{(l)}\right)"),
    (2, r"\hat{A} = A + I , \qquad \hat{D}_{ii} = \sum_{j}\hat{A}_{ij}"),
    (3, r"z_{v} = h_{v}^{(L)} , \qquad v \in \mathcal{V}_{H}\cup\mathcal{V}_{D}"),
    (4, r"\phi(h,r,d) = z_{h}^{\top}\,R\,z_{d} = \sum_{i=1}^{K} R_{ii}\,z_{h,i}\,z_{d,i}"),
    (5, r"\hat{y}_{hd} = \sigma\!\left(\phi(h,r,d)\right) = \frac{1}{1+e^{-\phi(h,r,d)}}"),
    (6, r"\mathcal{L}_{\mathrm{BCE}} = -\frac{1}{|\Omega|}\sum_{(h,d)\in\Omega}\left[\,y_{hd}\log\hat{y}_{hd} + (1-y_{hd})\log\left(1-\hat{y}_{hd}\right)\,\right]"),
    (7, r"\mathcal{L}_{\mathrm{ctr}} = -\log \frac{\exp\!\left(\phi(h,r,d^{+})/\tau\right)}{\exp\!\left(\phi(h,r,d^{+})/\tau\right) + \sum_{j=1}^{K}\exp\!\left(\phi(h,r,d_{j}^{-})/\tau\right)}"),
    (8, r"\mathcal{L} = \mathcal{L}_{\mathrm{BCE}} + \lambda\,\mathcal{L}_{\mathrm{ctr}}"),
    (9, r"\Omega^{-} = \left\{\,(h,d') \;:\; d'\in\mathcal{D},\; (h,d')\notin\Omega^{+}\,\right\}"),
    (10, r"\mathrm{AUROC} = \frac{1}{|\mathcal{P}||\mathcal{N}|}\sum_{p\in\mathcal{P}}\sum_{n\in\mathcal{N}} \mathbb{1}\left[\,s(p) > s(n)\,\right]"),
    (11, r"\mathrm{MRR} = \frac{1}{|\mathcal{Q}|}\sum_{i=1}^{|\mathcal{Q}|}\frac{1}{\mathrm{rank}_{i}}"),
    (12, r"s(h,d) = \phi(h,r,d) - \frac{1}{|\mathcal{D}|}\sum_{d'\in\mathcal{D}}\phi(h,r,d')"),
]

NS = ('xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
      'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"')


def main():
    if shutil.which("pandoc") is None:
        print("[错误] 未找到 pandoc，无法转换为 OMML。")
        print("       安装：https://pandoc.org/installing.html（或 conda install pandoc）")
        sys.exit(1)

    tmp = tempfile.mkdtemp(prefix="omml_")
    md = os.path.join(tmp, "eq.md")
    dx = os.path.join(tmp, "eq.docx")

    with open(md, "w", encoding="utf-8") as f:
        for _, tex in EQS:
            f.write(f"$${tex}$$\n\n")

    print("[1/3] pandoc: LaTeX -> docx(原生公式) ...")
    r = subprocess.run(["pandoc", "-f", "markdown", "-t", "docx",
                        "-o", dx, md],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0 or not os.path.exists(dx):
        print("[错误] pandoc 转换失败：")
        print(r.stderr)
        sys.exit(1)
    print(f"      生成 {dx} ({os.path.getsize(dx)} bytes)")
    if r.stderr.strip():
        print("      pandoc 提示：", r.stderr.strip()[:300])

    print("[2/3] 抽取 OMML ...")
    with zipfile.ZipFile(dx) as z:
        doc_xml = z.read("word/document.xml").decode("utf-8")

    blocks = re.findall(r"<m:oMathPara>.*?</m:oMathPara>", doc_xml, flags=re.S)
    if not blocks:
        blocks = ["<m:oMathPara><m:oMath>" + b + "</m:oMath></m:oMathPara>"
                  for b in re.findall(r"<m:oMath>.*?</m:oMath>", doc_xml, flags=re.S)]
    print(f"      抽到 {len(blocks)} 个公式块（期望 {len(EQS)}）")

    if len(blocks) != len(EQS):
        print("[警告] 数量不一致，将按顺序尽量匹配。")

    # 补命名空间，便于 python-docx 的 parse_xml 直接解析
    fixed = []
    for b in blocks:
        b = b.replace("<m:oMathPara>", f"<m:oMathPara {NS}>", 1)
        fixed.append(b)

    print("[3/3] 写出 equations_omml.py ...")
    with open(OUT_PY, "w", encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n")
        f.write('"""自动生成：Word 原生公式 (OMML)。'
                '请勿手改，重新生成请运行 make_equations_omml.py。"""\n')
        f.write("EQ_OMML = {\n")
        for (num, tex), xml in zip(EQS, fixed):
            esc = xml.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            f.write(f'    {num}: """{esc}""",\n')
        f.write("}\n")
    print(f"      [写入] {OUT_PY} ({os.path.getsize(OUT_PY)} bytes)")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n[完成] 下一步：python build_manuscript_zh.py")


if __name__ == "__main__":
    main()
