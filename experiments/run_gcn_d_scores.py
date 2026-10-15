#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 GCN-D 的草药-疾病分数矩阵 (供独立外部验证使用)
====================================================
目的
--------------------------------------------------------------------------
论文需要一个 **完全独立于我们自建金标准** 的外部验证: 用 HERB 数据库 CMap
成分级连接性打分作为外部金标准, 检验 GCN-D 给出的草药-疾病分数是否与之相关。

本脚本只做一件事: 按论文最终配置训练 GCN-D, 把 **全草药 x 全疾病** 的分数矩阵
落盘为 CSV (列 = herb_id + 各疾病), 供 run_herb_external_validation.py 读取。

与论文口径的一致性
--------------------------------------------------------------------------
- 图: build_graph() (H-I-G-D 骨架 + KEGG 基因-基因边), 与 run_fair_comparison 完全一致
- 训练: 全部正例对参与训练 (transductive FULL 协议), 负采样排除该病**全部**已知正例
        (exclude_pos=EXCL, filtered setting, 与 MC1 修复一致)
- 超参: BEST (lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2), seed=42

为什么 FULL 训练不构成循环论证
--------------------------------------------------------------------------
外部金标准 (HERB CMap) 与我们的自建金标准**数据来源与构建方法完全独立**:
  - 自建金标准: 文献/指南 curated + TCMBank 挖掘
  - HERB CMap: 成分扰动后的转录组连接性打分
GCN-D 在训练中从未接触任何 CMap 数据, 因此用 CMap 检验其分数是**真外部验证**。

用法
--------------------------------------------------------------------------
    python run_gcn_d_scores.py --epochs 3000        # 正式 (论文配置)
    python run_gcn_d_scores.py --epochs 300         # 冒烟
    python run_gcn_d_scores.py --epochs 3000 --out my_scores.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg                                    # noqa: E402
from run_fair_comparison import (build_graph, build_exclude, train_gcn,  # noqa: E402
                                 GOLD_PATH, OUT_DIR)

# 论文最终超参 (与 run_cv_final.BEST / sanity 记录一致)
BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)


def main():
    ap = argparse.ArgumentParser(description="导出 GCN-D 分数矩阵")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="输出 CSV 路径 (默认 output/results/paper_deliverables/"
                         "gcn_d_scores.csv)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)
    gold["disease_id"] = gold["disease_id"].astype(str)

    print("=" * 78)
    print("导出 GCN-D 分数矩阵 (供 HERB CMap 独立外部验证)")
    print(f"金标准: {os.path.basename(GOLD_PATH)}  ({len(gold)} 行)")
    print("=" * 78)

    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, g2i = build_graph()
    print(f"[图] 节点={N}  草药={nH}  疾病={nD}")
    print(f"[病] {disease_ids}\n")

    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}

    # 全部正例对 (去重, 且必须在图内)
    pos_pairs = sorted({
        (hidx[str(r.herb_id)], didx[str(r.disease_id)])
        for r in gold[gold["label"] == 1].itertuples()
        if str(r.herb_id) in hidx and str(r.disease_id) in didx})
    print(f"[正例对] {len(pos_pairs)}")

    # filtered setting: 负采样时排除该病全部已知正例 (防泄漏)
    EXCL = build_exclude(gold, herbs, disease_ids)

    print(f"[训练] epochs={args.epochs}  {BEST}")
    S, last_loss = train_gcn(A, X, off_h, off_d, nH, nD, pos_pairs, nH,
                             epochs=args.epochs, seed=args.seed,
                             exclude_pos=EXCL, **BEST)
    print(f"[训练] 末轮 loss={last_loss:.4f}   分数矩阵 S={S.shape}")

    df = pd.DataFrame(S, columns=list(disease_ids))
    df.insert(0, "herb_id", list(herbs))

    out = args.out or os.path.join(OUT_DIR, "gcn_d_scores.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {out}")
    print(f"  行={len(df)} (草药)   列={list(df.columns)}")
    print("\n下一步: python run_herb_external_validation.py "
          f"--scores {os.path.basename(out)} --tag gcn_d")


if __name__ == "__main__":
    main()
