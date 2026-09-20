#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 TCM-suite disease-target.csv 构建多病疾病基因集 (扩病种用)。
==========================================================
TCM-suite 的 disease-target 是 DisGeNET 风格的疾病-基因关联
(diseaseId, geneId, gene, ...)。我们的 enrichment 评分只需要
"疾病基因集合" (见 regenerate_enrichment_scores.py 中
`degs[did] = set(gene_symbol)`), 不需要 logFC/P-value。

因此本脚本直接把 disease-target 转成项目兼容的疾病基因集文件,
供 run_multidisease_enrichment.py 使用, 实现几十个疾病的 DATE 评分。

输出: data/processed/multidisease_genesets.csv  (列: disease_id, gene_symbol)
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw", "tcmsuite_disease_target.csv")
OUT = os.path.join(BASE, "data", "processed", "multidisease_genesets.csv")


def main():
    df = pd.read_csv(RAW)
    sub = df[["diseaseId", "gene"]].dropna().drop_duplicates()
    sub.columns = ["disease_id", "gene_symbol"]
    # 过滤掉非基因符号 (空/nan 已在 dropna 处理)
    sub = sub[sub["gene_symbol"].astype(str).str.strip() != ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sub.to_csv(OUT, index=False)
    print(f"疾病数: {sub['disease_id'].nunique()}  关联数: {len(sub)}")
    print(f"每病平均基因数: {sub.groupby('disease_id').size().mean():.1f}")
    print(f"[写入] {OUT}")


if __name__ == "__main__":
    main()
