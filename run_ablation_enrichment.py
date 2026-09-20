#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATE (enrichment 法) 消融实验 —— 修正版
=====================================================================
为什么要重写:
  原 run_ablation.py 把"DEG 筛选方式差异"误标为"方向校准"消融, 属于归因错误。
  事实上 enrichment 得分 = -log10(p) 恒为正值, 本身不含方向, 因此 DATE(enrichment)
  并不存在"方向校准"这一消融项; 真正决定成败的是 DEG 集合怎么筛。

本脚本对当前 DATE 方法做正确的消融 (FULL 协议 AUROC, 越高越好):
  A. DEG 筛选策略: P_value top-K + |logFC| 过滤(当前) | |logFC| top-K | 全基因不过滤
  B. top-K 阈值扫描: 300/500/1000/1500/2000/3000
  C. |logFC| 阈值扫描: 0 / 0.05 / 0.1 / 0.2

输出: paper_deliverables/ablation_enrichment.csv
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = "c:/Users/alikex/Desktop/Work/systems_mapping/"
PROC = os.path.join(BASE, "data", "processed")
PAPER = os.path.join(BASE, "output", "results", "paper_deliverables")
sys.path.insert(0, BASE)
import config as cfg  # noqa: E402
from hdmap_s import pair_single_herb_all_diseases_enrichment  # noqa: E402


def build_degs(df, strategy, top_k, logfc_thr):
    """
    strategy:
      pvalue   : 按 P_value 升序取 top_k, 且 |logFC| > logfc_thr  (DATE 当前方式)
      abslogfc : 按 |logFC| 降序取 top_k                         (原 run_ablation 方式)
      all      : 全基因, 不过滤
    """
    degs = {}
    for did, g in df.groupby("disease_id"):
        sub = g
        if strategy != "all" and logfc_thr > 0:
            sub = sub[sub["logFC"].abs() > logfc_thr]
        if strategy == "pvalue" and top_k and top_k > 0:
            if "P_value" in sub.columns:
                sub = sub.sort_values("P_value", ascending=True)
            sub = sub.head(top_k)
        elif strategy == "abslogfc" and top_k and top_k > 0:
            sub = sub.reindex(sub["logFC"].abs().sort_values(ascending=False).index).head(top_k)
        degs[did] = set(sub["gene_symbol"].astype(str))
    return degs


def evaluate(mat):
    """FULL 协议: 阳性=金标准草药, 阴性=其余全部"""
    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    mat.index = mat.index.astype(str)
    herbs = list(mat.index)
    per = []
    for d in mat.columns:
        pos = set(gold[(gold["disease_id"] == d) & (gold["label"] == 1)]["herb_id"].astype(str))
        pos = {h for h in pos if h in mat.index}
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in herbs], dtype=float)
        s = mat.loc[herbs, d].fillna(0).values.astype(float)
        if len(set(y)) < 2:
            continue
        per.append(round(float(roc_auc_score(y, s)), 4))
    mean_auc = round(float(np.mean(per)), 4) if per else float("nan")
    return mean_auc, per


def main():
    # 当前配置 (与 regenerate_enrichment_scores.py 一致)
    cur_k = int(cfg.DISEASE_DEG_TOP_K) if cfg.DISEASE_DEG_TOP_K else 1500
    cur_lf = cfg.DISEASE_LOGFC_THRESHOLD

    df = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    main_mat = pd.read_csv(os.path.join(cfg.RESULTS_DIR, "weighted_kegg", "hdmap_s_scores.csv"), index_col=0)
    herb_ids = list(main_mat.index)
    print(f"草药数: {len(herb_ids)} | 当前 K={cur_k}, |logFC|阈值={cur_lf}")

    rows = []

    def run_one(group, label, degs):
        mat = pair_single_herb_all_diseases_enrichment(degs, herb_ids)
        m, per = evaluate(mat)
        rows.append({"group": group, "config": label,
                     "MEAN_AUROC": m, "per_disease": str(per)})
        print(f"  {label:<45} MEAN={m}")

    # A. DEG 筛选策略
    print("\n[A] DEG 筛选策略消融")
    run_one("A. DEG筛选策略",
            f"P_value top-{cur_k} + |logFC|>{cur_lf} (DATE当前)",
            build_degs(df, "pvalue", cur_k, cur_lf))
    run_one("A. DEG筛选策略",
            "|logFC| top-1500 (无logFC过滤)",
            build_degs(df, "abslogfc", 1500, 0.0))
    run_one("A. DEG筛选策略",
            "全基因无DEG过滤",
            build_degs(df, "all", 0, 0.0))

    # B. top-K 阈值扫描
    print("\n[B] top-K 阈值扫描 (当前策略)")
    for k in [300, 500, 1000, 1500, 2000, 3000]:
        run_one("B. top-K 阈值", f"K={k}", build_degs(df, "pvalue", k, cur_lf))

    # C. logFC 阈值扫描
    print("\n[C] |logFC| 阈值扫描 (K=%d)" % cur_k)
    for lf in [0.0, 0.05, 0.1, 0.2]:
        run_one("C. logFC阈值", f"|logFC|>{lf}", build_degs(df, "pvalue", cur_k, lf))

    out = pd.DataFrame(rows)
    os.makedirs(PAPER, exist_ok=True)
    out_path = os.path.join(PAPER, "ablation_enrichment.csv")
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("DATE (enrichment) 消融汇总 (FULL 协议 AUROC)")
    print("=" * 78)
    print(out[["group", "config", "MEAN_AUROC", "per_disease"]].to_string(index=False))
    print(f"\n[写入] {out_path}")


if __name__ == "__main__":
    main()
