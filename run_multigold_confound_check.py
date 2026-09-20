"""
run_multigold_confound_check.py
===============================
多套金标准混杂检验 (论文核心诊断)

对比两套独立来源金标准下 TMS 的判别力与一致性:
  G1 = curated           人工整理        (data/gold_standard_final.csv, 4 病)
  G2 = tcmbank_indication TCMBank 主治文本 (data/processed/gold_tcmbank_indication.csv, 8 病)

并做混杂诊断:
  - 两套金标准正例集合的 Jaccard 重叠 -> 独立来源是否一致 (高=可信)
  - 靶点数量 (经 herb_ingredients 桥接) 与 TMS 的 Spearman 相关 -> TMS 是否只是"靶点多"的代理

TMS 来源: 优先 AD_kegg_results/hdmap_s_scores_enrichment.csv (含 4 病 enrichment TMS, 方法一致);
         该文件缺失时回退扫描 output/results 下所有 hdmap_s_scores*.csv。
注意: 当前 TMS 仅覆盖 4 病 (AD/PAH/RA/T2D); 新 4 病需先用主流程重算 TMS 后重跑本脚本。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROC = os.path.join(HERE, "data", "processed")
RES = os.path.join(HERE, "output", "results")

CURATED = os.path.join(HERE, "data", "gold_standard_final.csv")
TCMBANK = os.path.join(PROC, "gold_tcmbank_indication.csv")
TARGETS = os.path.join(PROC, "ingredient_targets_processed.csv")
HERB_ING = os.path.join(PROC, "herb_ingredients_processed.csv")
PRIORITY_TMS = os.path.join(RES, "AD_kegg_results", "hdmap_s_scores_enrichment.csv")


def auc_score(y, s):
    """Manual Mann-Whitney AUROC. y: 0/1, s: score (higher=better)."""
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
    n_pos, n_neg = len(pos), len(neg)
    return (rpos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return np.corrcoef(ra, rb)[0, 1]


def load_tms():
    if os.path.exists(PRIORITY_TMS):
        df = pd.read_csv(PRIORITY_TMS)
        tms = {c: df.set_index("herb_id")[c] for c in df.columns if c != "herb_id"}
        return tms, "AD_kegg_results/hdmap_s_scores_enrichment.csv"
    tms = {}
    for f in glob.glob(os.path.join(RES, "**", "hdmap_s_scores*.csv"), recursive=True):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "herb_id" not in d.columns:
            continue
        for c in d.columns:
            if c != "herb_id" and c not in tms:
                tms[c] = d.set_index("herb_id")[c]
    return tms, "glob scan"


def main():
    tms, src = load_tms()
    print(f"TMS 来源: {src}")
    print(f"TMS 覆盖病种: {sorted(tms.keys())}")

    curated = pd.read_csv(CURATED)
    tcmbank = pd.read_csv(TCMBANK)
    curated_pos = {d: set(g["herb_id"])
                   for d, g in curated[curated["label"] == 1].groupby("disease_id")}
    tcm_pos = {d: set(g["herb_id"]) for d, g in tcmbank.groupby("disease_id")}

    diseases = [d for d in tms if d in curated_pos or d in tcm_pos]
    rows = []
    for d in sorted(diseases):
        s = tms[d].dropna()
        herbs = s.index.tolist()
        sc = s.values
        y = np.array([1 if h in curated_pos.get(d, set()) else 0 for h in herbs])
        yt = np.array([1 if h in tcm_pos.get(d, set()) else 0 for h in herbs])
        auc_c = auc_score(y, sc) if y.sum() > 0 else float("nan")
        auc_t = auc_score(yt, sc) if yt.sum() > 0 else float("nan")
        cs, ts = curated_pos.get(d, set()), tcm_pos.get(d, set())
        union = cs | ts
        jac = len(cs & ts) / len(union) if union else float("nan")
        rows.append({
            "disease": d, "n_curated": len(cs), "n_tcmbank": len(ts),
            "overlap": len(cs & ts), "auc_curated": auc_c,
            "auc_tcmbank": auc_t, "jaccard": jac,
        })
    res = pd.DataFrame(rows)
    print("\n=== 多金标准 AUROC 对比 (curated vs tcmbank_indication) ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- 混杂诊断: 靶点数量 vs TMS ----
    if os.path.exists(TARGETS) and os.path.exists(HERB_ING):
        ing = pd.read_csv(HERB_ING)
        tg = pd.read_csv(TARGETS)
        hi = "herb_id" if "herb_id" in ing.columns else ing.columns[0]
        ii = "ingredient_id" if "ingredient_id" in ing.columns else ing.columns[1]
        ti = "ingredient_id" if "ingredient_id" in tg.columns else tg.columns[0]
        tg_col = "target_gene" if "target_gene" in tg.columns else tg.columns[1]
        m = ing[[hi, ii]].merge(tg[[ti, tg_col]], left_on=ii, right_on=ti)
        tcount = m.groupby(hi)[tg_col].nunique()
        print("\n=== 混杂诊断: 靶点数量 vs TMS (Spearman rho) ===")
        for d in sorted(diseases):
            s = tms[d].dropna()
            common = s.index.intersection(tcount.index)
            if len(common) < 10:
                print(f"  {d:7s}: 样本不足, 跳过")
                continue
            rho = spearman(tcount[common].values, s[common].values)
            flag = "  <- 偏倚风险" if abs(rho) > 0.3 else ""
            print(f"  {d:7s}: rho = {rho:+.3f}  (n={len(common)}){flag}")
    else:
        print("\n[跳过] 靶点混杂诊断: 缺少 herb_ingredients 或 ingredient_targets 文件")

    out = os.path.join(PROC, "multigold_confound_report.csv")
    res.to_csv(out, index=False)
    print(f"\n写入: {out}")
    print("[提示] 新 4 病 (HCC/CRC/ASTHMA/OA) 的 TMS 尚未计算; "
          "请用主流程重算 8 病 TMS 后重跑本脚本以覆盖全部病种。")


if __name__ == "__main__":
    main()
