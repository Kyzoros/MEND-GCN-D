#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCN-D 每病可解释性分析（冲二区：生物学可解释性 + 案例验证）
============================================================
目的
--------------------------------------------------------------------------
二区期刊（BIB）评审最看重的问题之一是: "你的模型除了给个分数, 能不能给出
**机制上说得通** 的生物学解释?" 本脚本用**纯本地数据**回答这个问题,
不新增任何数据下载, 不改变任何已有评测数字。

做法
--------------------------------------------------------------------------
对每一个疾病:
  1. 取 GCN-D 分数最高的 Top-K 味草药 (默认 20)
  2. 对每味草药, 取出它的**已知成分靶点集合**
  3. 计算该靶点集合与**该病疾病基因集**的重叠, 并做 Fisher 精确检验
     (背景 = 图中全部基因)
  4. 标注该草药是"金标准已知正例"还是"新预测"

  若一味草药的靶点显著富集在该病基因集上, 则它的高分具有**机制层面的解释**:
  "该草药的活性成分作用于该病相关的基因"。

为什么这能支撑论文
--------------------------------------------------------------------------
- **可解释性**: 不再是黑箱打分, 每个预测都能落到"草药成分 -> 靶点 -> 疾病基因"的通路
- **案例验证**: Top-K 中"已知正例"的比例 = 模型是否把文献已知的有效药排到了前面;
                "新预测且富集显著"的草药 = 可供后续实验验证的候选

输入（全部本地已有）
--------------------------------------------------------------------------
- output/results/paper_deliverables/gcn_d_scores.csv   (由 run_gcn_d_scores.py 生成)
- data/processed/gold_standard_final_8d_supplemented.csv
- data/processed/herb_ingredients_processed.csv
- data/processed/ingredient_targets_enhanced_recovered.csv
- data/processed/disease_profiles_processed.csv
- data/processed/herb_name_map.csv

输出
--------------------------------------------------------------------------
- paper_deliverables/per_disease_top_herbs.csv       每病 Top-K 逐草药明细
- paper_deliverables/per_disease_interpretation.csv  每病汇总 (已知命中率/富集概况)

用法
--------------------------------------------------------------------------
    python interpret_gcn_d.py
    python interpret_gcn_d.py --top 20
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg                                        # noqa: E402

PROC = cfg.PROCESSED_DIR
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")

SCORES = os.path.join(OUT_DIR, "gcn_d_scores.csv")
GOLD = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
HERB_ING = os.path.join(PROC, "herb_ingredients_processed.csv")
ING_TGT = os.path.join(PROC, "ingredient_targets_enhanced_recovered.csv")
DISEASE_PROF = os.path.join(PROC, "disease_profiles_processed.csv")
HERB_NAME = os.path.join(PROC, "herb_name_map.csv")


def is_gene_symbol(g):
    """判断 target_gene 的值是否为**基因符号**（而非蛋白质/酶名称）。

    数据缺陷（2026-09-15 自查发现）
    ------------------------------------------------------------------
    `ingredient_targets_enhanced_recovered.csv` 的 target_gene 列**混入了
    UniProt 的蛋白质/酶名称**，且包含细菌/病毒蛋白，例如：
        'Histidine decarboxylase'、'Gag-Pol polyprotein'、
        'Putative cytochrome P450 130'、
        '2-hydroxy-6-oxo-6-phenylhexa-2,4-dienoate hydrolase'
    实测：2039 个取值中 1171 个是基因符号、868 个是蛋白质名称（占 43%），
    而这 868 个与疾病 DEG 基因符号的交集**恒为 0**。

    若不剔除，草药靶点集合会被近一半的"永不可能匹配"的条目污染，
    直接稀释靶点-疾病基因的重叠信号，使富集检验失去功效。
    """
    if not isinstance(g, str):
        return False
    s = g.strip()
    if not s or s.lower() == "nan" or " " in s:
        return False
    return s.upper() == s and s.replace("-", "").replace(".", "").isalnum()


def load_herb_targets():
    """草药 -> 该草药全部成分靶点基因集合（只保留基因符号型靶点）"""
    hi = pd.read_csv(HERB_ING)
    it = pd.read_csv(ING_TGT)
    hi["herb_id"] = hi["herb_id"].astype(str)
    hi["ingredient_id"] = hi["ingredient_id"].astype(str)
    it["ingredient_id"] = it["ingredient_id"].astype(str)
    it["target_gene"] = it["target_gene"].astype(str)

    n_raw = n_kept = 0
    ing2genes = {}
    for ing, sub in it.groupby("ingredient_id"):
        raw = [str(g).strip() for g in sub["target_gene"].dropna()]
        n_raw += len(raw)
        # 只保留基因符号型, 剔除 UniProt 蛋白质/酶名称 (见 is_gene_symbol 说明)
        gs = {g for g in raw if is_gene_symbol(g)}
        n_kept += len(gs)
        if gs:
            ing2genes[ing] = gs
    print(f"[靶点清洗] 原始 {n_raw} 条 -> 保留基因符号 {n_kept} 条 "
          f"(剔除蛋白质/酶名称 {n_raw - n_kept} 条, "
          f"{100 * (n_raw - n_kept) / max(n_raw, 1):.1f}%)")

    herb2genes = {}
    for herb, sub in hi.groupby("herb_id"):
        gs = set()
        for ing in sub["ingredient_id"]:
            gs |= ing2genes.get(ing, set())
        if gs:
            herb2genes[herb] = gs
    return herb2genes, ing2genes


def main():
    ap = argparse.ArgumentParser(description="GCN-D 每病可解释性分析")
    ap.add_argument("--top", type=int, default=20,
                    help="每病取分数最高的前 K 味草药")
    ap.add_argument("--gene-set", choices=["top", "broad"], default="top",
                    help="疾病基因集口径: top=按显著性取最显著 N 个 (默认, 有判别力); "
                         "broad=该病全部 DEG (过宽, 富集无判别力)")
    ap.add_argument("--dge-top", type=int, default=200,
                    help="gene-set=top 时, 每病取最显著的多少个 DEG")
    ap.add_argument("--min-lfc", type=float, default=1.0,
                    help="gene-set=top 时, 要求的 |logFC| 下限")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    scores = pd.read_csv(SCORES)
    scores["herb_id"] = scores["herb_id"].astype(str)
    diseases = [c for c in scores.columns if c != "herb_id"]

    gold = pd.read_csv(GOLD)
    gold["herb_id"] = gold["herb_id"].astype(str)
    known = {(str(r.disease_id), str(r.herb_id))
             for r in gold[gold["label"] == 1].itertuples()}

    dp = pd.read_csv(DISEASE_PROF)
    dp["disease_id"] = dp["disease_id"].astype(str)
    dp["gene_symbol"] = dp["gene_symbol"].astype(str)
    dp["P_value"] = pd.to_numeric(dp.get("P_value"), errors="coerce")
    dp["logFC"] = pd.to_numeric(dp.get("logFC"), errors="coerce")

    def _clean(gs):
        return {g for g in gs.dropna()
                if str(g).strip() and str(g).lower() != "nan"}

    if args.gene_set == "broad":
        # 该病全部 DEG（默认 1500 个，约占背景 14%，过于宽泛，富集无判别力）
        dis_genes = {d: _clean(sub["gene_symbol"])
                     for d, sub in dp.groupby("disease_id")}
    else:
        # 疾病特征基因集：按显著性取最显著的 top-N 个（并要求 |logFC| 达到阈值）
        sig = dp.dropna(subset=["P_value"])
        if args.min_lfc:
            sig = sig[sig["logFC"].abs() >= args.min_lfc]
        sig = sig.sort_values("P_value")
        dis_genes = {d: set(_clean(sub["gene_symbol"].head(args.dge_top)))
                     for d, sub in sig.groupby("disease_id")}
        print(f"[基因集] 模式=top  每病取最显著 {args.dge_top} 个 DEG "
              f"(|logFC|>= {args.min_lfc})")
        for d, gs in dis_genes.items():
            print(f"    {d:<8} n={len(gs)}")

    herb2genes, _ = load_herb_targets()

    try:
        nm = pd.read_csv(HERB_NAME)
        name_map = dict(zip(nm["herb_id"].astype(str), nm["herb_latin"]))
        cn_map = dict(zip(nm["herb_id"].astype(str), nm["tcm_name"]))
    except Exception:
        name_map, cn_map = {}, {}

    # 背景基因总数 = 疾病基因集 ∪ 成分靶点集
    all_genes = set()
    for s in dis_genes.values():
        all_genes |= s
    for s in herb2genes.values():
        all_genes |= s
    N = len(all_genes)
    print(f"[背景] 全图基因数 N = {N}")
    print(f"[草药] 有靶点的草药数 = {len(herb2genes)}")

    rows, summary = [], []
    for d in diseases:
        dg = dis_genes.get(d, set())
        n_dg = len(dg)
        sub = scores[["herb_id", d]].rename(columns={d: "score"})
        sub = sub.sort_values("score", ascending=False).head(args.top)

        n_known = 0
        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            h = str(r["herb_id"])
            tg = herb2genes.get(h, set())
            ov = tg & dg
            a, b = len(ov), len(tg) - len(ov)
            c = n_dg - len(ov)
            dd = N - len(tg) - n_dg + len(ov)
            try:
                or_, p = fisher_exact([[a, b], [c, max(dd, 0)]],
                                      alternative="greater")
            except Exception:
                or_, p = float("nan"), float("nan")
            is_known = int((d, h) in known)
            n_known += is_known
            rows.append({
                "disease": d, "rank": rank, "herb_id": h,
                "herb_latin": name_map.get(h, ""),
                "tcm_name": cn_map.get(h, ""),
                "gcn_d_score": round(float(r["score"]), 4),
                "is_known_positive": is_known,
                "n_herb_targets": len(tg),
                "n_disease_genes": n_dg,
                "n_overlap": len(ov),
                "fisher_OR": round(float(or_), 3) if or_ == or_ else None,
                "fisher_p": float(f"{p:.3g}") if p == p else None,
                "enriched": int(p == p and p < 0.05),
            })

        sub_df = pd.DataFrame([r for r in rows if r["disease"] == d])
        summary.append({
            "disease": d,
            "n_disease_genes": n_dg,
            f"top{args.top}_n_known": n_known,
            f"top{args.top}_precision": round(n_known / len(sub_df), 3)
                                        if len(sub_df) else None,
            f"top{args.top}_n_enriched": int(sub_df["enriched"].sum()),
            "median_overlap": int(sub_df["n_overlap"].median())
                              if len(sub_df) else 0,
            "median_fisher_p": float(f"{sub_df['fisher_p'].median():.3g}")
                               if len(sub_df) else None,
        })
        print(f"{d:<8} 疾病基因={n_dg:<5} Top{args.top} 已知命中={n_known:<3} "
              f"富集显著={int(sub_df['enriched'].sum()):<3} "
              f"中位重叠={int(sub_df['n_overlap'].median())}")

    df = pd.DataFrame(rows)
    sm = pd.DataFrame(summary)
    f1 = os.path.join(OUT_DIR, "per_disease_top_herbs.csv")
    f2 = os.path.join(OUT_DIR, "per_disease_interpretation.csv")
    df.to_csv(f1, index=False, encoding="utf-8-sig")
    sm.to_csv(f2, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f"每病汇总 (Top{args.top})")
    print("=" * 78)
    print(sm.to_string(index=False))
    print(f"\n[写入] {f1}")
    print(f"[写入] {f2}")


if __name__ == "__main__":
    main()
