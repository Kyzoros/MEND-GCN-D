"""
run_tms_size_correction.py
==========================
方案A 快速验证：对现有 enrichment TMS 做"靶点数量校正"(size-corrected),
验证偏倚是否消除 + AUROC 是否提升。不重跑管线(直接用已算好的 enrichment 分数)。

校正逻辑(大白话): 对每个病, 拟合 "TMS分数 ~ 靶点数量" 的直线,
用残差(实际分 - 直线上同靶点数的预期分)作为校正分。
  - 残差>0 = 比"同样靶点数的草药"得分更高 = 真富集信号(非数量驱动)
  - 残差去除了"靶点多=分高"的机械偏心
"""
import os
import sys

import numpy as np
import pandas as pd
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
PROC = os.path.join(HERE, "data", "processed")
RES = os.path.join(HERE, "output", "results")

RAW_TMS = os.path.join(RES, "AD_kegg_results", "hdmap_s_scores_enrichment.csv")
CURATED = os.path.join(HERE, "data", "gold_standard_final.csv")
TARGETS = os.path.join(PROC, "ingredient_targets_processed.csv")
HERB_ING = os.path.join(PROC, "herb_ingredients_processed.csv")


def auc_score(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    rpos = ranks[y == 1].sum()
    n1, n0 = len(pos), len(neg)
    return (rpos - n1 * (n1 + 1) / 2) / (n1 * n0)


def spearman(a, b):
    return np.corrcoef(pd.Series(a).rank().values,
                       pd.Series(b).rank().values)[0, 1]


def main():
    raw = pd.read_csv(RAW_TMS).set_index("herb_id")
    curated = pd.read_csv(CURATED)
    pos = {d: set(g["herb_id"])
           for d, g in curated[curated["label"] == 1].groupby("disease_id")}

    # 每草药靶点数量 (经 herb_ingredients 桥接)
    ing = pd.read_csv(HERB_ING)
    tg = pd.read_csv(TARGETS)
    hi = "herb_id" if "herb_id" in ing.columns else ing.columns[0]
    ii = "ingredient_id" if "ingredient_id" in ing.columns else ing.columns[1]
    ti = "ingredient_id" if "ingredient_id" in tg.columns else tg.columns[0]
    tgc = "target_gene" if "target_gene" in tg.columns else tg.columns[1]
    m = ing[[hi, ii]].merge(tg[[ti, tgc]], left_on=ii, right_on=ti)
    tcount = m.groupby(hi)[tgc].nunique()
    tc = tcount.reindex(raw.index)

    rows = []
    print("=== 方案A 验证: 靶点数量校正 (size-corrected) ===\n")
    for d in raw.columns:
        s = raw[d].values.astype(float)
        x = tc.values.astype(float)
        herbs = raw.index.tolist()
        y = np.array([1 if h in pos.get(d, set()) else 0 for h in herbs])
        mask = ~np.isnan(s) & ~np.isnan(x)

        # size-corrected 残差
        coef = np.polyfit(x[mask], s[mask], 1)
        corr = s - np.polyval(coef, x)

        rho_raw = spearman(x[mask], s[mask])
        rho_corr = spearman(x[mask], corr[mask])
        auc_raw = auc_score(y, s) if y.sum() > 0 else float("nan")
        auc_corr = auc_score(y, corr) if y.sum() > 0 else float("nan")
        rows.append({
            "disease": d, "rho_raw": rho_raw, "rho_corrected": rho_corr,
            "auc_raw": auc_raw, "auc_corrected": auc_corr,
        })
        print(f"  {d}: 偏倚 rho {rho_raw:+.3f} -> {rho_corr:+.3f} | "
              f"AUROC {auc_raw:.3f} -> {auc_corr:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PROC, "tms_size_correction_check.csv"), index=False)
    print("\n解读: rho_corrected 应趋近 0 (偏心消除); "
          "auc_corrected 应 >= auc_raw (判别力提升)")
    print(f"写入: {os.path.join(PROC, 'tms_size_correction_check.csv')}")


if __name__ == "__main__":
    main()
