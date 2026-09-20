#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从疾病 DEG 直接重算 enrichment TMS 矩阵 (无需重跑整条 main.py 管线)
====================================================================
用途:
  修改 config.yaml 的 disease_deg_top_k / logfc_threshold / pvalue_threshold 后,
  用本脚本秒级重算 hdmap_s_scores.csv, 免去昂贵的扩散签名计算 (Step1/Step2)。

适用条件:
  仅 tms_method = "enrichment"。
  enrichment 只依赖 (疾病基因集, 草药靶点), 不依赖 herb_signatures,
  因此可以独立于扩散步骤重算。若 tms_method 为 weighted/dual/ndcg 请跑 main.py。

一致性:
  基因过滤逻辑与 main.py 的 _filter_disease_genes() 严格一致。

用法:
    python regenerate_enrichment_scores.py
    python regenerate_enrichment_scores.py --top-k 1500     # 临时覆盖 config
    python regenerate_enrichment_scores.py --no-eval        # 只出矩阵不评估
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import config as cfg
from hdmap_s import pair_single_herb_all_diseases_enrichment


def filter_genes(sub, top_k=None):
    """与 main.py 的 _filter_disease_genes() 保持一致"""
    k = cfg.DISEASE_DEG_TOP_K if top_k is None else top_k
    if cfg.DISEASE_LOGFC_THRESHOLD > 0:
        sub = sub[sub["logFC"].abs() > cfg.DISEASE_LOGFC_THRESHOLD]
    if k and k > 0:
        if "P_value" in sub.columns:
            sub = sub.sort_values("P_value", ascending=True)
        else:
            sub = sub.sort_values("logFC", key=lambda s: s.abs(),
                                  ascending=False)
        sub = sub.head(k)
    elif cfg.DISEASE_PVALUE_THRESHOLD > 0 and "P_value" in sub.columns:
        sub = sub[sub["P_value"] < cfg.DISEASE_PVALUE_THRESHOLD]
    return sub


def bootstrap_ci(y, s, n_boot=2000, seed=42):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    if len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    auc = float(roc_auc_score(y, s))
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        bs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return auc, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description="重算 enrichment TMS 矩阵")
    ap.add_argument("--top-k", type=int, default=None,
                    help="临时覆盖 config 的 disease_deg_top_k")
    ap.add_argument("--out", default=None, help="输出路径")
    ap.add_argument("--no-eval", action="store_true", help="只出矩阵不评估")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        cfg.RESULTS_DIR, "weighted_kegg", "hdmap_s_scores.csv")

    # --- 疾病 DEG ---
    dpath = os.path.join(cfg.PROCESSED_DIR, "disease_profiles_processed.csv")
    if not os.path.exists(dpath):
        dpath = os.path.join(cfg.DATA_DIR, "disease_profiles_processed.csv")
    df = pd.read_csv(dpath)

    degs = {}
    print("[1/3] 构建疾病基因集 ...")
    for did in df["disease_id"].unique():
        sub = df[df["disease_id"] == did]
        before = len(sub)
        sub = filter_genes(sub, args.top_k)
        print(f"  {did:<5} {before:>6} → {len(sub):>5} 基因")
        sub = sub.sort_values("logFC", ascending=False)
        degs[did] = set(sub["gene_symbol"])

    # --- herb_ids (与现有主矩阵保持一致) ---
    print("[2/3] 载入草药列表 ...")
    if os.path.exists(out_path):
        herb_ids = list(pd.read_csv(out_path)["herb_id"].astype(str))
    else:
        hi = pd.read_csv(os.path.join(cfg.PROCESSED_DIR,
                                      "herb_ingredients_processed.csv"))
        herb_ids = list(hi["herb_id"].astype(str).unique())
    print(f"  草药数: {len(herb_ids)}")

    # --- 计算 ---
    print("[3/3] 计算 enrichment TMS ...")
    mat = pair_single_herb_all_diseases_enrichment(degs, herb_ids)
    mat.index = mat.index.astype(str)
    mat.index.name = "herb_id"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mat.to_csv(out_path)
    print(f"\n[写入] {out_path}   shape={mat.shape}")

    if args.no_eval:
        return

    # --- 评估 (FULL 协议) ---
    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    herbs = list(mat.index)
    print("\n" + "=" * 76)
    print("评估 (FULL 协议: 阳性 vs 全部草药, 越高越治疗)")
    print("=" * 76)
    print(f"{'disease':<8}{'n_pos':>7}{'AUROC':>9}{'CI_lo':>9}{'CI_hi':>9}"
          f"{'显著性':>14}")
    print("-" * 76)
    aucs, nsig = [], 0
    for d in mat.columns:
        pos = set(gold[(gold["disease_id"] == d) &
                       (gold["label"] == 1)]["herb_id"].astype(str))
        pos = {h for h in pos if h in mat.index}
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in herbs], dtype=float)
        s = mat.loc[herbs, d].values.astype(float)
        a, lo, hi = bootstrap_ci(y, s)
        aucs.append(a)
        if lo > 0.5:
            nsig += 1
        sig = "显著" if lo > 0.5 else ("反向" if hi < 0.5 else "不显著")
        print(f"{d:<8}{len(pos):>7}{a:>9.4f}{lo:>9.4f}{hi:>9.4f}{sig:>14}")
    print("-" * 76)
    print(f"平均 AUROC = {np.mean(aucs):.4f}   显著疾病数: {nsig}/{len(aucs)}")


if __name__ == "__main__":
    main()
