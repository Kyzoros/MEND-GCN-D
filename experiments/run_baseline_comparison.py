#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线对比 (杠杆③): 统一 FULL 协议评估多个 herb-disease 打分方法, 输出 AUROC 表。

支持方法 (结果文件存在才评估, 缺失自动跳过并提示):
  - DATE        : weighted_kegg/hdmap_s_scores.csv  (我们的方法, regenerate 产出)
  - GCN         : processed/hdcti_cache/gcn_predictions.csv (gcn_target_expansion 产出)
  - HERB-legacy : paper_deliverables/herb_legacy_summary.csv (run_herb --compare-legacy 产出)
  - Cosine-TMS  : weighted_kegg/hdmap_s_scores_cosine.csv (待 generate, 后续补)
  - NetProx     : weighted_kegg/hdmap_s_scores_netprox.csv (待生成, 后续补)

评估: FULL 协议 (阳性=已知治疗草药, 阴性=其余全部), bootstrap CI 见 verify 脚本同款逻辑。
输出: paper_deliverables/baseline_comparison.csv + 打印均值。
"""
import os
import sys
import argparse
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


def load_gold():
    g = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    out = {}
    for d in g["disease_id"].unique():
        pos = set(g[(g["disease_id"] == d) & (g["label"] == 1)]["herb_id"].astype(str))
        out[d] = pos
    return out


def full_auroc(mat, gold):
    herbs = list(mat.index)
    for d in mat.columns:
        p = {h for h in gold.get(d, set()) if h in mat.index}
        if len(p) < 3:
            continue
        y = np.array([1 if h in p else 0 for h in herbs], dtype=float)
        s = mat.loc[herbs, d].fillna(0).values.astype(float)
        if len(set(y)) < 2:
            continue
        yield d, float(roc_auc_score(y, s))


def load_matrix(path, index="herb_id"):
    if not os.path.exists(path):
        return None
    m = pd.read_csv(path)
    if index not in m.columns:
        return None
    m[index] = m[index].astype(str)
    return m.set_index(index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        cfg.RESULTS_DIR, "paper_deliverables", "baseline_comparison.csv"))
    args = ap.parse_args()

    methods = {
        "DATE": os.path.join(cfg.RESULTS_DIR, "weighted_kegg", "hdmap_s_scores.csv"),
        "GCN": os.path.join(cfg.PROCESSED_DIR, "hdcti_cache", "gcn_herb_disease_matrix.csv"),
        "Cosine-TMS": os.path.join(cfg.RESULTS_DIR, "weighted_kegg",
                                   "hdmap_s_scores_cosine.csv"),
        "NetProx": os.path.join(cfg.RESULTS_DIR, "weighted_kegg",
                                "hdmap_s_scores_netprox.csv"),
        # 原 HERB-legacy 槽位: 项目无 HERB 草药-疾病关联, TCM-suite 关联映射率仅24%且与金标准可能同源,
        # 故以 Random 随机基线替代(更严谨, 且无循环论证)。输出由 generate_random_baseline.py 写入。
        "Random": os.path.join(cfg.RESULTS_DIR, "paper_deliverables",
                               "herb_legacy_summary.csv"),
    }

    gold = load_gold()
    rows = []
    for name, path in methods.items():
        m = load_matrix(path)
        if m is None:
            print(f"[跳过] {name}: 未找到 {path}")
            continue
        print(f"[评估] {name}: 矩阵 {m.shape}")
        for d, a in full_auroc(m, gold):
            rows.append({"method": name, "disease": d, "AUROC": round(a, 4)})

    if not rows:
        print("无可用方法结果, 请先生成对应分数矩阵 (DATE 用 regenerate_enrichment_scores.py)")
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="disease", columns="method", values="AUROC")
    pivot.loc["MEAN"] = pivot.mean(numeric_only=True)
    pivot.to_csv(args.out)
    print("\n" + "=" * 70)
    print("基线对比 AUROC (FULL 协议)")
    print("=" * 70)
    print(pivot.to_string())
    print(f"\n[写入] {args.out}")


if __name__ == "__main__":
    main()
