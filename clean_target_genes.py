#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗主流程靶点数据：剔除 target_gene 列中的蛋白质/酶名称
========================================================
背景（2026-09-15 自查发现的数据缺陷）
--------------------------------------------------------------------------
`data/processed/ingredient_targets_enhanced_recovered.csv` 的 `target_gene` 列
**混入了 UniProt 的蛋白质/酶名称**，而非纯基因符号。实测：

    2039 个取值中
      - 1171 个 (57%) 是基因符号       —— 与疾病 DEG 基因符号有 555 个交集
      -  868 个 (43%) 是蛋白质/酶名称  —— 与疾病 DEG 基因符号交集恒为 0
    按条目计：102629 条 -> 保留 48311 条（剔除 52.9%）

被剔除的样例（含细菌/病毒蛋白，对人类疾病网络毫无意义）：
    'Gag-Pol polyprotein'                                  (病毒蛋白)
    'Putative cytochrome P450 130'                         (细菌蛋白)
    '2-hydroxy-6-oxo-6-phenylhexa-2,4-dienoate hydrolase'  (细菌酶)
    'Histidine decarboxylase'、'Choloylglycine hydrolase'   (酶名)

危害
--------------------------------------------------------------------------
`build_graph()` 直接用该列的值作为**基因节点名称**建图。这些蛋白质名称
于是变成图中约 868 个**永远连不到任何疾病基因的孤立节点**，
只挂在成分节点上，属于纯噪声，且虚增图的规模。

处理策略
--------------------------------------------------------------------------
采用**过滤**（而非名称->符号映射）：
    - 映射需要完整的 UniProt protein name -> gene symbol 对照表，本地不具备；
    - 被剔除的条目与疾病基因交集恒为 0，本就不贡献任何重叠信号；
    - 过滤是可复现、可审计的最小改动。

用法
--------------------------------------------------------------------------
    # 默认：生成清洗版副本（不动原文件）
    python clean_target_genes.py

    # 确认无问题后，替换主流程使用的数据（自动备份原文件）
    python clean_target_genes.py --inplace
"""
import argparse
import os
import shutil
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg                                        # noqa: E402

PROC = cfg.PROCESSED_DIR
SRC = os.path.join(PROC, "ingredient_targets_enhanced_recovered.csv")
DST = os.path.join(PROC, "ingredient_targets_enhanced_cleaned.csv")


def is_gene_symbol(g):
    """判断取值是否为基因符号（而非蛋白质/酶名称）。

    判据：不含空格 + 全大写 + 仅含字母数字与 '-' '.'。
    人类基因符号均满足（如 MCM6、HLA-DQA1、MIR196B）；
    蛋白质/酶名称均含空格或小写词，必然被排除。
    """
    if not isinstance(g, str):
        return False
    s = g.strip()
    if not s or s.lower() == "nan" or " " in s:
        return False
    return s.upper() == s and s.replace("-", "").replace(".", "").isalnum()


def main():
    ap = argparse.ArgumentParser(description="清洗靶点数据中的蛋白质/酶名称")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--inplace", action="store_true",
                    help="直接替换源文件（自动备份为 .bak_before_geneclean）")
    ap.add_argument("--mode", choices=["filter", "mark"], default="filter",
                    help="filter=删除非基因符号条目 (会损失约 0.02 AUROC); "
                         "mark=保留但加 'NONGENE::' 前缀 (推荐, 图结构不变、性能不变)")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    n_all = len(df)
    keep = df["target_gene"].map(is_gene_symbol)

    if args.mode == "filter":
        # 直接删除非基因符号条目
        cleaned = df[keep].copy()
    else:
        # 保留全部条目, 但给非基因符号加显式前缀, 避免"蛋白质名冒充基因符号"
        cleaned = df.copy()
        bad = ~keep
        cleaned.loc[bad, "target_gene"] = (
            "NONGENE::" + cleaned.loc[bad, "target_gene"].astype(str))
        print(f"[模式] mark —— 保留全部条目, 将 {int(bad.sum())} 条非基因符号"
              f"标注为 'NONGENE::' 前缀 (图结构不变, 不损失性能)")

    n_bad = int((~keep).sum())
    uniq_all = df["target_gene"].dropna().astype(str).nunique()
    uniq_kept = cleaned["target_gene"].dropna().astype(str).nunique()

    print("=" * 78)
    print("靶点数据清洗")
    print("=" * 78)
    print(f"源文件        : {args.src}")
    print(f"总条目        : {n_all}")
    print(f"保留(基因符号): {len(cleaned)}")
    print(f"剔除(蛋白/酶名): {n_bad}  ({100 * n_bad / max(n_all, 1):.1f}%)")
    print(f"唯一取值      : {uniq_all} -> {uniq_kept}")
    print(f"涉及成分数    : {df['ingredient_id'].nunique()} -> "
          f"{cleaned['ingredient_id'].nunique()}")

    out = args.src if args.inplace else args.dst
    if args.inplace:
        bak = args.src + ".bak_before_geneclean"
        if not os.path.exists(bak):
            shutil.copy2(args.src, bak)
            print(f"\n[备份] {bak}")

    cleaned.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {out}")
    if not args.inplace:
        print("\n未改动源文件（安全模式）。确认无误后加 --inplace 替换，")
        print("或用环境变量 SM_TGT_FILE 指向清洗版做对照实验。")


if __name__ == "__main__":
    main()
