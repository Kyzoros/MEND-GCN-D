#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双通道TMS 验证 (standalone)
================================
基于已有的 TMS 矩阵 (hdmap_s_scores.csv, 即 cos 矩阵) 立即验证双通道方案,
无需重跑昂贵的 main.py 管线。

原理:
  - 逆转通道  TMS_reverse = -cos  (越高 = 拮抗/逆转越强, 适合 antagonist 机制病)
  - 嵌网通道  TMS_engage  = +cos  (越高 = 嵌网/增益越强, 适合 engagement 机制病)
  - 单符号 TMS 被迫用同一方向, 顾此失彼; 双通道按疾病机制先验选通道, 每病取其优。

输出:
  - 每病: AUROC_reverse / AUROC_engage / AUROC_combined(机制先验) / AUROC_oracle(两路最优)
  - 平均 AUROC 对比原始单符号基线 (负TMS=治疗)
  - 机制-符号一致性计数 (每病 combined 是否等于其机制预测的符号)

用法:
    python evaluate_dual_channel.py
    python evaluate_dual_channel.py --tms output/results/weighted_kegg/hdmap_s_scores.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, RESULTS_DIR, GOLD_STANDARD_PATH, DISEASE_TMS_MECHANISM
from data_loader import load_herb_ingredients
from evaluation_metrics import load_gold_standard_mechanism

try:
    from sklearn.metrics import roc_auc_score
    def auroc(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
except Exception:
    def auroc(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        hits = sum(np.searchsorted(np.sort(neg), pos, side="right"))
        return hits / (len(pos) * len(neg))


def build_default_tms_path() -> str:
    cand = [
        os.path.join(RESULTS_DIR, "weighted_kegg", "hdmap_s_scores.csv"),
        os.path.join(RESULTS_DIR, "hdmap_s_scores.csv"),
    ]
    for c in cand:
        if os.path.exists(c):
            return c
    return cand[0]


def main():
    ap = argparse.ArgumentParser(description="双通道TMS 验证 (standalone)")
    ap.add_argument("--tms", default=build_default_tms_path(),
                    help="TMS 得分矩阵 CSV (cos 矩阵) 路径")
    ap.add_argument("--gold", default=GOLD_STANDARD_PATH,
                    help="金标准表 CSV 路径")
    args = ap.parse_args()

    if not os.path.exists(args.tms):
        print(f"[错误] 找不到 TMS 矩阵: {args.tms}")
        sys.exit(1)

    cos_df = pd.read_csv(args.tms)
    if "herb_id" in cos_df.columns:
        cos_df = cos_df.set_index("herb_id")

    herb_ing = load_herb_ingredients()
    gold, mech_map, _, warns = load_gold_standard_mechanism(args.gold, herb_ing)
    for w in warns:
        print(f"[警告] {w}")

    if not gold:
        print("[错误] 未加载任何金标准条目, 请检查 data/gold_standard.csv")
        sys.exit(1)

    print(f"\n双通道TMS 验证: {args.tms}")
    print("=" * 78)

    rows = []
    consistent = 0
    for disease, herb_list in gold.items():
        if disease not in cos_df.columns:
            continue
        col = cos_df[disease]
        all_herbs = list(col.index)
        present = [h for h in herb_list if h in col.index]
        if len(present) < 2:
            continue
        y = np.array([1 if h in set(present) else 0 for h in all_herbs], dtype=float)
        tms = col.loc[all_herbs].values.astype(float)

        # 逆转通道: 负cos=好 → 评分 = -tms
        auc_rev = auroc(y, -tms)
        # 嵌网通道: 正cos=好 → 评分 = +tms
        auc_eng = auroc(y, +tms)
        # 原始单符号基线: 负TMS=治疗 (即逆转通道)
        auc_baseline = auc_rev

        # 机制先验选择
        mech = str(DISEASE_TMS_MECHANISM.get(disease, "engagement")).strip().lower()
        if mech == "antagonist":
            auc_combined = auc_rev
            predicted_sign = "neg(逆转)"
        else:
            auc_combined = auc_eng
            predicted_sign = "pos(嵌网)"
        auc_oracle = max(auc_rev, auc_eng)

        # 机制-符号一致性: combined 选的符号 == 该机制预期符号 ?
        if (mech == "antagonist" and auc_rev >= auc_eng) or \
           (mech == "engagement" and auc_eng >= auc_rev):
            consistent += 1

        rows.append({
            "disease_id": disease,
            "mechanism_prior": mech,
            "AUROC_baseline": round(auc_baseline, 4),
            "AUROC_reverse": round(auc_rev, 4),
            "AUROC_engage": round(auc_eng, 4),
            "AUROC_combined": round(auc_combined, 4),
            "AUROC_oracle": round(auc_oracle, 4),
            "predicted_sign": predicted_sign,
            "n_gold": len(present),
        })
        print("%-8s | mech=%-11s | base %6.4f | rev %6.4f | eng %6.4f | comb %6.4f | oracle %6.4f"
              % (disease, mech, auc_baseline, auc_rev, auc_eng, auc_combined, auc_oracle))

    res = pd.DataFrame(rows)
    if res.empty:
        print("无可用评估 (金标准疾病列与 TMS 矩阵无交集)")
        sys.exit(0)

    print("=" * 78)
    print(f"平均 AUROC:")
    print(f"  原始单符号基线 (负TMS=治) : {res['AUROC_baseline'].mean():.4f}")
    print(f"  双通道-逆转通道           : {res['AUROC_reverse'].mean():.4f}")
    print(f"  双通道-嵌网通道           : {res['AUROC_engage'].mean():.4f}")
    print(f"  双通道-机制先验合并       : {res['AUROC_combined'].mean():.4f}")
    print(f"  双通道-每病取最优(oracle) : {res['AUROC_oracle'].mean():.4f}")
    print(f"  (随机基线 0.5)")
    print(f"机制-符号一致性: {consistent}/{len(res)} 病, combined 选的符号符合其机制预期")

    out = os.path.join(os.path.dirname(args.tms), "evaluate_dual_channel.csv")
    res.to_csv(out, index=False)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
