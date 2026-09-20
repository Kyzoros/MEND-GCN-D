#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面检查 GEO 差异表达文件
================================================================================
识别: 基因符号列 / logFC 列 / P 值列 / 平台类型 / 是否可用
支持三种常见格式:
  1. GEO2R(limma)  : ID, adj.P.Val, P.Value, t, B, logFC, Gene.symbol, Gene.title
  2. DESeq2(RNA-seq): GeneID, padj, pvalue, lfcSE, stat, log2FoldChange, baseMean, Symbol
  3. 其他(需映射)   : 只有探针 ID, 无基因符号 (Agilent / Illumina 等)
"""
import glob
import os

import pandas as pd

D = r"C:\Users\alikex\Desktop\Work\差异表达基因"

# 可能的列名
SYMBOL_COLS = ["Gene.symbol", "Symbol", "gene_symbol", "GENE_SYMBOL", "GeneSymbol"]
LOGFC_COLS = ["logFC", "log2FoldChange", "log2FC", "Log2FC"]
PVAL_COLS = ["P.Value", "pvalue", "Pvalue", "P.Value", "Pval"]


def detect_platform(df):
    """从 ID 列推断平台"""
    id_col = df.columns[0]
    sample = str(df[id_col].iloc[0]) if len(df) else ""
    if sample.replace(".", "").replace("_", "").isdigit():
        return "Affymetrix(数字探针)"
    if sample.startswith("A_") and "_P" in sample:
        return "Agilent(A_xx_Pxxxx)"
    if "ILMN" in sample.upper():
        return "Illumina(ILMN_)"
    if sample.startswith("ENSG"):
        return "RNA-seq(ENSG基因ID)"
    return f"未知({sample[:22]})"


def main():
    files = sorted(glob.glob(os.path.join(D, "*.top.table.csv")))
    print(f"目录: {D}")
    print(f"找到 {len(files)} 个 top.table 文件\n")

    ok, need_fix, bad = [], [], []

    for f in files:
        name = os.path.basename(f)
        # 从文件名解析 disease_id
        try:
            dis = name.split("_")[1].split(".")[0]
        except Exception:
            dis = "?"

        try:
            df = pd.read_csv(f, sep="\t", low_memory=False)
        except Exception as e:
            print(f"=== {name} ===\n  读取失败: {e}\n")
            bad.append((name, dis, "读取失败"))
            continue

        n = len(df)
        cols = list(df.columns)

        sym_col = next((c for c in SYMBOL_COLS if c in cols), None)
        lfc_col = next((c for c in LOGFC_COLS if c in cols), None)
        pval_col = next((c for c in PVAL_COLS if c in cols), None)

        sym_n = 0
        if sym_col:
            sym_n = int((df[sym_col].astype(str).str.strip() != "").sum())

        platform = detect_platform(df)

        print(f"=== {name} ===")
        print(f"  疾病: {dis}   行数: {n}")
        print(f"  平台: {platform}")
        print(f"  基因符号列: {sym_col or '❌ 无'}   非空: {sym_n}")
        print(f"  logFC列: {lfc_col or '❌ 无'}   P值列: {pval_col or '❌ 无'}")
        print(f"  列: {cols}")

        # 判定
        if sym_col and sym_n > n * 0.5 and lfc_col and pval_col:
            if lfc_col in ("log2FoldChange",):
                print("  ✅ 可用 (RNA-seq/DESeq2 格式, 需列名转换)")
                need_fix.append((name, dis, "列名转换"))
            else:
                print("  ✅ 可用 (标准 GEO2R/limma 格式)")
                ok.append((name, dis, n))
        elif not sym_col:
            print("  ❌ 不可用: 缺少基因符号列, 需换数据集或做探针注释映射")
            bad.append((name, dis, "缺基因符号"))
        else:
            print(f"  ⚠️ 基因符号覆盖率低 ({sym_n}/{n}), 需检查")
            bad.append((name, dis, f"symbol覆盖低{sym_n}/{n}"))
        print()

    # 汇总
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"✅ 直接可用 ({len(ok)}):")
    for name, dis, n in ok:
        print(f"   {dis:<8} {name}  ({n} 行)")
    print(f"\n⚠️ 需列名转换 ({len(need_fix)}):")
    for name, dis, why in need_fix:
        print(f"   {dis:<8} {name}  ({why})")
    print(f"\n❌ 不可用 ({len(bad)}):")
    for name, dis, why in bad:
        print(f"   {dis:<8} {name}  ({why})")


if __name__ == "__main__":
    main()
