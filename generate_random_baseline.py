#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random 基线 (杠杆③): 固定种子的随机打分, 作为 trivial baseline。
每个方法学论文都应报告"方法 vs 随机", 以证明信号非偶然。
输出: output/results/paper_deliverables/herb_legacy_summary.csv
       (index=herb_id, columns=AD/PAH/RA/T2D, 值=标准正态随机)
注: 原 baseline_comparison.py 此槽位名 HERB-legacy, 但项目无 HERB 草药-疾病关联数据,
    且 TCM-suite 关联映射率仅 24% 且与金标准可能同源(循环论证), 故以 Random 替代, 更严谨。
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = "c:/Users/alikex/Desktop/Work/systems_mapping/"
PROC = os.path.join(BASE, "data", "processed")
PAPER = os.path.join(BASE, "output", "results", "paper_deliverables")
OUT = os.path.join(PAPER, "herb_legacy_summary.csv")


def main():
    # 统一草药集合到 DATE 主矩阵 (491 味), 保证各方法评估集合一致
    date_mat = pd.read_csv(os.path.join(BASE, "output", "results", "weighted_kegg", "hdmap_s_scores.csv"), index_col=0)
    herbs = list(date_mat.index)
    dis = ["AD", "PAH", "RA", "T2D"]
    rng = np.random.default_rng(20260829)
    df = pd.DataFrame(rng.standard_normal((len(herbs), len(dis))),
                      index=herbs, columns=dis)
    df.index.name = "herb_id"
    os.makedirs(PAPER, exist_ok=True)
    df.to_csv(OUT)
    print("Random 矩阵:", df.shape)
    print("[写入]", OUT)


if __name__ == "__main__":
    main()
