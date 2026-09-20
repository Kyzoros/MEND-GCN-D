#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
靶标扩展方法对比: 相似度传播(当前HDCTI) vs 图卷积(GCN)
======================================================
回答建议5的核心问题: HDCTI 模块该维持相似度传播, 还是真的加上图卷积?

对比设计:
  基线(相似度传播): 用 ingredient_targets_enhanced.csv
                    (已含 HDCTI 相似度传播扩展的靶标) 做 enrichment 打分
  GCN(图卷积)    : 用 gcn_expanded_targets.csv
                    (原始靶标 + GCN 链路预测 top-K 新靶标) 做 enrichment 打分
  两者使用完全相同的疾病 DEG (config 的 top-K) 与金标准, 保证可比。

实现要点:
  enrichment 函数按 "成分->靶点->草药" 聚合, 因此为 GCN 草药级靶标
  构造"虚拟成分"映射 (每草药一个虚拟成分), 即可复用原函数而不改动它。

用法:
    python compare_target_expansion.py
    python compare_target_expansion.py --gcn-targets <path>
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


def build_deg():
    """按 config 构造疾病基因集 (与 regenerate_enrichment_scores 一致)"""
    dpath = os.path.join(cfg.PROCESSED_DIR, "disease_profiles_processed.csv")
    if not os.path.exists(dpath):
        dpath = os.path.join(cfg.DATA_DIR, "disease_profiles_processed.csv")
    df = pd.read_csv(dpath)
    out = {}
    for did in df["disease_id"].unique():
        sub = df[df["disease_id"] == did]
        if cfg.DISEASE_LOGFC_THRESHOLD > 0:
            sub = sub[sub["logFC"].abs() > cfg.DISEASE_LOGFC_THRESHOLD]
        if cfg.DISEASE_DEG_TOP_K and cfg.DISEASE_DEG_TOP_K > 0:
            if "P_value" in sub.columns:
                sub = sub.sort_values("P_value", ascending=True)
            sub = sub.head(cfg.DISEASE_DEG_TOP_K)
        elif cfg.DISEASE_PVALUE_THRESHOLD > 0 and "P_value" in sub.columns:
            sub = sub[sub["P_value"] < cfg.DISEASE_PVALUE_THRESHOLD]
        out[did] = set(sub["gene_symbol"])
    return out


def make_virtual_ingredient_files(gcn_path, out_dir):
    """把草药级 GCN 靶标包装成 enrichment 需要的 (成分-靶点, 草药-成分) 两个文件"""
    gcn = pd.read_csv(gcn_path)
    herbs = sorted(gcn["herb_id"].astype(str).unique())
    h2vid = {h: 900000 + i for i, h in enumerate(herbs)}

    ing_tgt = pd.DataFrame({
        "ingredient_id": gcn["herb_id"].astype(str).map(h2vid),
        "target_gene": gcn["target_gene"].astype(str),
        "confidence": gcn.get("confidence", 1.0),
    }).dropna(subset=["ingredient_id"])

    herb_ing = pd.DataFrame({
        "herb_id": herbs,
        "ingredient_id": [h2vid[h] for h in herbs],
    })

    p1 = os.path.join(out_dir, "_gcn_ing_targets.csv")
    p2 = os.path.join(out_dir, "_gcn_herb_ing.csv")
    ing_tgt.to_csv(p1, index=False)
    herb_ing.to_csv(p2, index=False)
    return p1, p2


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


def evaluate(mat, gold, label):
    herbs = list(mat.index.astype(str))
    print(f"\n--- {label} ---")
    print(f"{'disease':<8}{'n_pos':>7}{'AUROC':>9}{'CI_lo':>9}{'CI_hi':>9}"
          f"{'显著性':>12}")
    print("-" * 56)
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
        print(f"{d:<8}{len(pos):>7}{a:>9.4f}{lo:>9.4f}{hi:>9.4f}{sig:>12}")
    print("-" * 56)
    print(f"{'平均':<8}{'':>7}{np.mean(aucs):>9.4f}   显著 {nsig}/{len(aucs)}")
    return np.mean(aucs), nsig


def main():
    ap = argparse.ArgumentParser(description="靶标扩展方法对比")
    ap.add_argument("--gcn-targets", default=os.path.join(
        cfg.RESULTS_DIR, "weighted_kegg", "gcn_expanded_targets.csv"))
    args = ap.parse_args()

    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    print("[1/3] 构造疾病基因集 ...")
    degs = build_deg()

    cur = pd.read_csv(os.path.join(cfg.RESULTS_DIR, "weighted_kegg",
                                   "hdmap_s_scores.csv")).set_index("herb_id")
    herb_ids = list(cur.index.astype(str))

    print("[2/3] 基线: 相似度传播扩展 (ingredient_targets_enhanced) ...")
    base = pair_single_herb_all_diseases_enrichment(degs, herb_ids)
    base.index = base.index.astype(str)

    print("[3/3] GCN: 图卷积扩展 ...")
    if not os.path.exists(args.gcn_targets):
        print(f"[错误] 找不到 {args.gcn_targets}, 先跑 gcn_target_expansion.py")
        sys.exit(1)
    p1, p2 = make_virtual_ingredient_files(args.gcn_targets,
                                           os.path.dirname(args.gcn_targets))
    gcn_mat = pair_single_herb_all_diseases_enrichment(
        degs, herb_ids,
        ingredient_targets_path=p1,
        herb_ingredients_path=p2)
    gcn_mat.index = gcn_mat.index.astype(str)

    print("\n" + "=" * 60)
    print("对比结果 (FULL 协议, 扩充金标准 96 正例)")
    print("=" * 60)
    a1, s1 = evaluate(base, gold, "基线: 相似度传播 (当前 HDCTI)")
    a2, s2 = evaluate(gcn_mat, gold, "GCN: 图卷积扩展")

    print("\n" + "=" * 60)
    print(f"  相似度传播  平均 AUROC = {a1:.4f}   显著 {s1}/4")
    print(f"  GCN 图卷积  平均 AUROC = {a2:.4f}   显著 {s2}/4")
    diff = a2 - a1
    print(f"  差值        = {diff:+.4f}  "
          f"({'GCN 更优' if diff > 0 else '相似度传播更优'})")
    print("=" * 60)

    out = os.path.join(cfg.RESULTS_DIR, "paper_deliverables",
                       "target_expansion_comparison.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame([
        {"method": "similarity_propagation (current HDCTI)",
         "mean_AUROC": round(a1, 4), "n_significant": s1},
        {"method": "GCN_graph_convolution",
         "mean_AUROC": round(a2, 4), "n_significant": s2},
    ]).to_csv(out, index=False)
    print(f"[写入] {out}")


if __name__ == "__main__":
    main()
