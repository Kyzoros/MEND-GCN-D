#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决定性检验: "草药靶点数量" 单独作为预测器, 能拿到多少 AUROC?
================================================================================
背景 (见 target_count_bias_analysis.md):
  enrichment 得分与草药靶点总数 Spearman ρ = 0.65~0.82 (ρ² 42%~67%);
  跨病中心化后 AUROC 从 0.622 -> 0.499 (疾病特异性信号≈0)。

本检验:
  若"靶点数"单独作为打分, 对金标准的 AUROC 已接近 enrichment 的 0.6009,
  则说明 enrichment 的表观性能【完全可由靶点数量混杂解释】—— 即它识别的是
  "靶点多的药/研究充分的药", 而不是"对该病有效的药"。

  对照: 同时报告金标准正例 vs 其余草药的靶点数中位数与 Mann-Whitney p。
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

BASE = "c:/Users/alikex/Desktop/Work/systems_mapping/"
PROC = os.path.join(BASE, "data", "processed")
sys.path.insert(0, BASE)
import config as cfg  # noqa: E402

PAPER = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")
OUT = os.path.join(PAPER, "target_count_confound_check.csv")


def herb_target_counts():
    """每味草药的去重靶点基因数"""
    it = pd.read_csv(os.path.join(PROC, "ingredient_targets_enhanced.csv"))
    ing_tg = {}
    for iid, g in zip(it["ingredient_id"], it["target_gene"]):
        g = str(g).strip()
        if g and g != "nan":
            ing_tg.setdefault(int(iid), set()).add(g)
    hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
    herb_tg = {}
    for hid, iid in zip(hi["herb_id"], hi["ingredient_id"]):
        herb_tg.setdefault(str(hid).strip(), set()).update(ing_tg.get(int(iid), set()))
    return {h: len(t) for h, t in herb_tg.items()}


def main():
    tc = herb_target_counts()
    print(f"草药靶点数: 中位数={np.median(list(tc.values())):.0f}, "
          f"均值={np.mean(list(tc.values())):.0f}, n={len(tc)}")

    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    main_mat = pd.read_csv(os.path.join(cfg.RESULTS_DIR, "weighted_kegg", "hdmap_s_scores.csv"), index_col=0)
    main_mat.index = main_mat.index.astype(str)
    herbs = list(main_mat.index)

    rows = []
    for d in ["AD", "PAH", "RA", "T2D"]:
        pos = set(gold[(gold["disease_id"] == d) & (gold["label"] == 1)]["herb_id"].astype(str))
        pos = {h for h in pos if h in set(herbs)}
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in herbs], dtype=float)

        s_tc = np.array([tc.get(h, 0) for h in herbs], dtype=float)      # 靶点数作为预测器
        s_en = main_mat.loc[herbs, d].fillna(0).values.astype(float)     # enrichment (当前方法)

        a_tc = float(roc_auc_score(y, s_tc))
        a_en = float(roc_auc_score(y, s_en))

        tc_pos = [tc.get(h, 0) for h in pos]
        tc_neg = [tc.get(h, 0) for h in herbs if h not in pos]
        try:
            p = float(mannwhitneyu(tc_pos, tc_neg, alternative="greater").pvalue)
        except Exception:
            p = float("nan")

        rows.append({
            "disease": d,
            "n_pos": len(pos),
            "AUROC_靶点数(混杂变量)": round(a_tc, 4),
            "AUROC_enrichment(当前方法)": round(a_en, 4),
            "差值(enrichment-靶点数)": round(a_en - a_tc, 4),
            "正例靶点数中位数": int(np.median(tc_pos)),
            "负例靶点数中位数": int(np.median(tc_neg)),
            "MannWhitney_p": f"{p:.3g}",
        })

    df = pd.DataFrame(rows)
    mean_row = {"disease": "MEAN", "n_pos": "-"}
    for c in ["AUROC_靶点数(混杂变量)", "AUROC_enrichment(当前方法)", "差值(enrichment-靶点数)"]:
        mean_row[c] = round(df[c].astype(float).mean(), 4)
    mean_row["正例靶点数中位数"] = "-"
    mean_row["负例靶点数中位数"] = "-"
    mean_row["MannWhitney_p"] = "-"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    os.makedirs(PAPER, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 100)
    print("决定性检验: 靶点数量(混杂变量) 单独当预测器 vs 当前 enrichment 方法")
    print("=" * 100)
    print(df.to_string(index=False))
    print("\n判读: 若'靶点数'的 AUROC 已接近 enrichment, 则表观性能主要由混杂驱动, 非真实疗效预测")
    print(f"\n[写入] {OUT}")


if __name__ == "__main__":
    main()
