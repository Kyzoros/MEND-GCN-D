#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决定性对比: 原论文网络(PPI, 28896基因) vs 当前改动(KEGG, 8077基因)
================================================================================
背景:
  config.NETWORK_TYPE = "kegg" 把原论文的 PPI 传播网络换成了 KEGG 通路共成员网络。
  PPI 签名基因空间 28896, KEGG 仅 8077 (相差 3.6 倍)。
  此前诊断发现"每病 1500 个 DEG 只有 ~40% 落在 KEGG 签名空间" —— 这很可能是
  KEGG 替换人为造成的, 而非数据固有缺陷。

本脚本:
  直接评估两套评分矩阵在同一金标准下的 FULL 协议 AUROC:
    PPI  : output/results/AD_ppi_results/hdmap_s_scores.csv
    KEGG : output/results/weighted_kegg/hdmap_s_scores.csv (当前主结果)
  并报告每病 DEG 在两种网络基因空间中的覆盖度(用签名 gene_names 计算)。
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "output", "results")
sys.path.insert(0, BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg  # noqa: E402

OUT = os.path.join(RES, "paper_deliverables", "ppi_vs_kegg_check.csv")

PPI_MAT = os.path.join(RES, "AD_ppi_results", "hdmap_s_scores.csv")
KEGG_MAT = os.path.join(RES, "weighted_kegg", "hdmap_s_scores.csv")


def auroc(mat, gold, d):
    mat.index = mat.index.astype(str)
    herbs = list(mat.index)
    pos = set(gold[(gold["disease_id"] == d) & (gold["label"] == 1)]["herb_id"].astype(str))
    pos = {h for h in pos if h in mat.index}
    if len(pos) < 3:
        return None, len(pos)
    y = np.array([1 if h in pos else 0 for h in herbs], dtype=float)
    s = mat.loc[herbs, d].fillna(0).values.astype(float)
    if len(set(y)) < 2:
        return None, len(pos)
    return round(float(roc_auc_score(y, s)), 4), len(pos)


def deg_coverage(network_type, degs_by_dis):
    """DEG 在该网络签名基因空间中的覆盖度"""
    meta_p = os.path.join(PROC, f"herb_signature_meta_{network_type}.json")
    cache_p = os.path.join(PROC, f"herb_signature_cache_{network_type}.npz")
    if not (os.path.exists(meta_p) and os.path.exists(cache_p)):
        return {}
    d = np.load(cache_p)
    genes = set(str(g) for g in d["gene_names"])
    out = {}
    for dis, gs in degs_by_dis.items():
        out[dis] = round(len(gs & genes) / max(len(gs), 1), 3)
    return out


def load_degs(top_k=1500, logfc_thr=0.05):
    prof = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    degs = {}
    for did, g in prof.groupby("disease_id"):
        sub = g
        if logfc_thr > 0:
            sub = sub[sub["logFC"].abs() > logfc_thr]
        sub = sub.sort_values("P_value", ascending=True).head(top_k)
        degs[did] = set(sub["gene_symbol"].astype(str))
    return degs


def main():
    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    degs = load_degs()

    mats = {}
    for name, path in [("PPI(原论文)", PPI_MAT), ("KEGG(当前)", KEGG_MAT)]:
        if os.path.exists(path):
            mats[name] = pd.read_csv(path, index_col=0)
            print(f"载入 {name}: {mats[name].shape} <- {path}")
        else:
            print(f"[缺失] {name}: {path}")

    print()
    for nt in ["ppi", "kegg"]:
        cov = deg_coverage(nt, degs)
        if cov:
            print(f"{nt.upper()} 网络 DEG 覆盖度: {cov}")

    rows = []
    for d in ["AD", "PAH", "RA", "T2D"]:
        row = {"disease": d}
        for name, mat in mats.items():
            a, npos = auroc(mat, gold, d)
            row[f"AUROC_{name}"] = a
            row[f"n_pos_{name}"] = npos
        rows.append(row)

    df = pd.DataFrame(rows)
    auc_cols = [c for c in df.columns if c.startswith("AUROC_")]
    mean_row = {"disease": "MEAN"}
    for c in auc_cols:
        mean_row[c] = round(df[c].astype(float).mean(), 4)
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\n" + "=" * 78)
    print("PPI(原论文网络) vs KEGG(当前改动) —— FULL 协议 AUROC")
    print("=" * 78)
    print(df.to_string(index=False))
    print(f"\n[写入] {OUT}")


if __name__ == "__main__":
    main()
