#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证模型核心假设 —— 从逻辑出发, 不为论文挑叙事
======================================================================
模型原始逻辑 (见 hdmap_s.compute_tms_weighted 文档):
    TMS = cos(herb_signature, disease_logFC_vector)
    sig × logFC < 0  =>  草药逆转疾病异常表达  =>  治疗
    即: TMS 越负 = 逆转越强 = 越能治病

本脚本检验这个核心假设在金标准上是否成立:
    score_reverse = -cos   (你的逻辑: 逆转=治疗, 越高越好)
    score_engage  = +cos   (对照:     顺势=治疗, 越高越好)

判定:
    score_reverse AUROC 明显 > 0.5  => 假设成立, 模型逻辑没问题
    ≈0.5 或 < 0.5                   => 假设不成立, 或签名噪声淹没信号
                                       (此时应改进签名质量, 而不是换掉逻辑)

说明: 本脚本不做任何"择向"(在测试集上挑更优方向 = 循环论证), 两个方向都报告。

输入: weighted_kegg/hdmap_s_scores_dual_cos.csv (现成的 cos 矩阵)
输出: paper_deliverables/reversal_hypothesis_check.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg  # noqa: E402

COS_MAT = os.path.join(cfg.RESULTS_DIR, "weighted_kegg", "hdmap_s_scores_dual_cos.csv")
OUT = os.path.join(cfg.RESULTS_DIR, "paper_deliverables", "reversal_hypothesis_check.csv")


def main():
    cos = pd.read_csv(COS_MAT, index_col=0)
    cos.index = cos.index.astype(str)
    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    herbs = list(cos.index)
    print(f"cos 矩阵: {cos.shape} | 草药 {len(herbs)} 味")

    rows = []
    for d in cos.columns:
        pos = set(gold[(gold["disease_id"] == d) & (gold["label"] == 1)]["herb_id"].astype(str))
        pos = {h for h in pos if h in cos.index}
        if len(pos) < 3:
            print(f"  {d}: 正例不足({len(pos)}), 跳过")
            continue
        y = np.array([1 if h in pos else 0 for h in herbs], dtype=float)
        c = cos.loc[herbs, d].fillna(0).values.astype(float)
        if len(set(y)) < 2:
            continue
        a_rev = float(roc_auc_score(y, -c))   # 逆转逻辑(你的逻辑): -cos 越高越治疗
        a_eng = float(roc_auc_score(y, c))    # 顺势逻辑(对照):     +cos 越高越治疗
        rows.append({
            "disease": d,
            "n_pos": len(pos),
            "AUROC_reverse(-cos,你的逻辑)": round(a_rev, 4),
            "AUROC_engage(+cos,对照)": round(a_eng, 4),
        })

    df = pd.DataFrame(rows)
    mean_row = {"disease": "MEAN", "n_pos": "-"}
    for col in df.columns[2:]:
        mean_row[col] = round(df[col].mean(), 4)
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 74)
    print("核心假设检验: TMS = cos(草药签名, 疾病logFC), 负值(逆转) = 治疗 ?")
    print("=" * 74)
    print(df.to_string(index=False))
    print("\n注: 两列之和应≈1.0 (取反符号即反转排序), 差值即方向带来的差异")
    print(f"\n[写入] {OUT}")


if __name__ == "__main__":
    main()
