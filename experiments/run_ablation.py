#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验 (杠杆④): 在 4 病金标准上, 消融 DATE 的关键设计选择。
  A. DEG top-K 阈值: 300/500/1000/1500/2000/3000
  B. 是否使用 DEG (direction-aware 核心): 用全靶点 vs 仅 DEG
评估: FULL 协议 AUROC (正例=已知草药, 负例=全部项目草药)
"""
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
PAPER = os.path.join(BASE, "output", "results", "paper_deliverables")
sys.path.insert(0, BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg  # noqa: E402
from hdmap_s import pair_single_herb_all_diseases_enrichment  # noqa: E402


def eval_mat(mat):
    mat.index = mat.index.astype(str).str.replace(r"\.0$", "", regex=True)
    hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
    id2herb = dict(zip(hi["ingredient_id"].astype(int).astype(str),
                       hi["herb_id"].astype(str)))
    mat.index = [id2herb.get(str(h), str(h)) for h in mat.index]
    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    rows = []
    for d in mat.columns:
        pos = set(gold[(gold["disease_id"] == d) & (gold["label"] == 1)]
                  ["herb_id"].astype(str)) & set(mat.index)
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in mat.index], dtype=float)
        s = mat[d].values.astype(float)
        if len(set(y)) < 2:
            continue
        rows.append((d, round(float(roc_auc_score(y, s)), 4)))
    return pd.DataFrame(rows, columns=["disease", "AUROC"])


def build_degs(K, use_deg=True):
    prof = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    degs = {}
    for did, g in prof.groupby("disease_id"):
        if use_deg:
            sub = g.reindex(g["logFC"].abs().sort_values(ascending=False).index).head(K)
        else:
            sub = g  # 全基因 (无 DEG 过滤)
        degs[did] = set(sub["gene_symbol"].astype(str))
    return degs


def main():
    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)

    # (1) DATE 完整版 = 已生成的方向校准矩阵 (main.py HDmap-S)
    cal_path = os.path.join(cfg.RESULTS_DIR, "weighted_kegg", "hdmap_s_scores.csv")
    cal = pd.read_csv(cal_path, index_col=0)
    cal.index = cal.index.astype(str)
    rows = []
    for d in cal.columns:
        pos = set(gold[(gold["disease_id"] == d) & (gold["label"] == 1)]
                  ["herb_id"].astype(str)) & set(cal.index)
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in cal.index], dtype=float)
        s = cal[d].values.astype(float)
        if len(set(y)) < 2:
            continue
        rows.append((d, round(float(roc_auc_score(y, s)), 4)))
    df_cal = pd.DataFrame(rows, columns=["disease", "AUROC"])

    # (2) 无方向校准 = 直接 pair 富集 (信号正负抵消 -> 0.50)
    prof = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    degs = {did: set(g.reindex(g["logFC"].abs().sort_values(ascending=False).index)
                     .head(1500)["gene_symbol"].astype(str))
            for did, g in prof.groupby("disease_id")}
    herb_ids = (pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
                ["ingredient_id"].dropna().unique().tolist())
    raw = pair_single_herb_all_diseases_enrichment(
        degs, herb_ids,
        os.path.join(PROC, "ingredient_targets_enhanced.csv"),
        os.path.join(PROC, "herb_ingredients_processed.csv"))
    df_raw = eval_mat(raw)

    out = pd.DataFrame([
        ("DATE (方向校准)", round(df_cal["AUROC"].mean(), 4),
         df_cal["AUROC"].round(4).tolist()),
        ("无方向校准 (原始富集)", round(df_raw["AUROC"].mean(), 4),
         df_raw["AUROC"].round(4).tolist()),
    ], columns=["config", "MEAN_AUROC", "per_disease"])
    out_path = os.path.join(PAPER, "ablation.csv")
    out.to_csv(out_path, index=False)
    print("=" * 70)
    print("消融实验: 方向校准的必要性 (4 病 FULL 协议 AUROC)")
    print("=" * 70)
    print(out.to_string())
    print(f"[写入] {out_path}")


if __name__ == "__main__":
    main()
