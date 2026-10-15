#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCN 基线 (杠杆③): 用 GCN 增强靶点 (gcn_predictions.csv) 在同样的富集框架下
重算 herb-disease AUROC, 与 DATE 方法对比。
======================================================================
复刻 regenerate_enrichment_scores.py 的 [2/3][3/3] 逻辑, 仅把靶点来源换成 GCN。
输出: paper_deliverables/baseline_gcn.csv + 打印均值。
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防 OMP Error#15
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
from hdmap_s import pair_single_herb_all_diseases_enrichment  # noqa: E402

TOP_K = 1500
GCN_TARGETS = os.path.join(cfg.PROCESSED_DIR, "gcn_targets_for_eval.csv")


def main():
    # 1) GCN 增强靶点 -> 与 ingredient_targets_enhanced.csv 同格式
    gcn = pd.read_csv(os.path.join(cfg.PROCESSED_DIR, "hdcti_cache", "gcn_predictions.csv"))
    gcn = gcn[["ingredient_id", "target_gene", "confidence"]].dropna()
    gcn["ingredient_id"] = gcn["ingredient_id"].astype(int)  # 对齐金标准(int)避免索引错位
    gcn.to_csv(GCN_TARGETS, index=False)
    print(f"[GCN 靶点] {len(gcn)} 条")

    # 2) 4 病 DEG (top-K by |logFC|), 与 regenerate 一致
    prof = pd.read_csv(os.path.join(BASE, "data", "processed", "disease_profiles_processed.csv"))
    degs = {}
    for did, g in prof.groupby("disease_id"):
        sub = g.reindex(g["logFC"].abs().sort_values(ascending=False).index).head(TOP_K)
        degs[did] = set(sub["gene_symbol"].astype(str))
    print(f"[疾病] {list(degs.keys())}")

    # 3) herb ids
    herb_ids = (pd.read_csv(os.path.join(BASE, "data", "processed", "herb_ingredients_processed.csv"))["ingredient_id"]
                .dropna().unique().tolist())

    # 4) 富集 (GCN 靶点)
    mat = pair_single_herb_all_diseases_enrichment(
        degs, herb_ids, GCN_TARGETS,
        os.path.join(BASE, "data", "processed", "herb_ingredients_processed.csv"))
    mat.index = mat.index.astype(str).str.replace(r"\.0$", "", regex=True)
    # 将 ingredient_id 索引映射回 herb_id (Herb_N) 以对齐金标准
    hi = pd.read_csv(os.path.join(BASE, "data", "processed", "herb_ingredients_processed.csv"))
    id2herb = dict(zip(hi["ingredient_id"].astype(int).astype(str),
                       hi["herb_id"].astype(str)))
    mat.index = [id2herb.get(str(h), str(h)) for h in mat.index]
    mat = mat.groupby(level=0).mean()  # 同一草药多个成分 -> 均值去重

    # 统一草药集合到 DATE 主矩阵 (491 味), 保证各方法评估集合一致
    date_mat = pd.read_csv(os.path.join(BASE, "output", "results", "weighted_kegg", "hdmap_s_scores.csv"), index_col=0)
    mat = mat.reindex(date_mat.index).fillna(0)
    # 保存 herb×disease 矩阵, 供 run_baseline_comparison.py 读取 (替代原 gcn_predictions.csv 的成分-靶点格式)
    mat.to_csv(os.path.join(cfg.PROCESSED_DIR, "hdcti_cache", "gcn_herb_disease_matrix.csv"))

    # 5) 评估 (FULL 协议)
    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    print("DEBUG mat idx:", list(mat.index[:3]),
          "| gold herb_id:", list(gold["herb_id"].astype(str).head(3)),
          "| gold diseases:", list(gold["disease_id"].unique()[:5]))
    rows = []
    for d in mat.columns:
        pos = set(gold[(gold["disease_id"] == d) & (gold["label"] == 1)]["herb_id"].astype(str))
        pos &= set(mat.index)
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in mat.index], dtype=float)
        s = mat[d].values.astype(float)
        if len(set(y)) < 2:
            continue
        rows.append((d, round(float(roc_auc_score(y, s)), 4)))
    df = pd.DataFrame(rows, columns=["disease", "AUROC"])
    df.loc["MEAN"] = df["AUROC"].mean()
    out = os.path.join(cfg.RESULTS_DIR, "paper_deliverables", "baseline_gcn.csv")
    df.to_csv(out, index=False)
    print("\n" + "=" * 50)
    print("GCN 基线 AUROC")
    print("=" * 50)
    print(df.to_string())
    print(f"[写入] {out}")


if __name__ == "__main__":
    main()
