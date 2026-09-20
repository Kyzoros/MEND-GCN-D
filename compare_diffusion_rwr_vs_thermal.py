#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 2 步: RWR vs 热扩散(thermal_2step) 对比实验
================================================================================
控制变量:
  网络 = PPI (processed/ppi_network.csv)
  打分 = ndcg (原论文 top/bottom 极端基因匹配, hdmap_s.compute_tms)
  K    = 250 (原论文 optimal signature size)
  疾病谱 = 按 logFC 降序 (原论文按 fold change 排序)
自变量:
  扩散 = thermal_2step (原论文) vs rwr (当前 config)

关键: use_cache=False —— 签名缓存只按 network_type 分离, 不区分扩散方法。
      若不绕开缓存, 第二次跑会错误复用第一次的签名, 导致两者结果完全相同。

评估: FULL 协议 AUROC (阳性=金标准草药, 阴性=其余全部)
      必报对照: 靶点数基线 (0.5933)
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = "c:/Users/alikex/Desktop/Work/systems_mapping/"
PROC = os.path.join(BASE, "data", "processed")
OUT_DIR = os.path.join(BASE, "output", "results", "paper_deliverables")
sys.path.insert(0, BASE)

import config as cfg  # noqa: E402
import perturbation_signature as ps  # noqa: E402
import hdmap_s as hs_mod  # noqa: E402
from hdmap_s import HDmapS, pair_single_herb_all_diseases  # noqa: E402

OUT = os.path.join(OUT_DIR, "diffusion_rwr_vs_thermal.csv")

# 原论文参数: optimal signature size = 250 (top 250 + bottom 250)
hs_mod.TOP_BOTTOM_K = 250


def load_data():
    hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
    # PPI 版靶点 (与网络匹配)
    it_path = os.path.join(PROC, "ingredient_targets_enhanced_ppi.csv")
    if not os.path.exists(it_path):
        it_path = os.path.join(PROC, "ingredient_targets_enhanced.csv")
    it = pd.read_csv(it_path, low_memory=False)
    ppi = pd.read_csv(os.path.join(PROC, "ppi_network.csv"))
    prof = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    print(f"[数据] 草药成分 {hi.shape}, 靶点 {it.shape}, PPI {ppi.shape}, 疾病谱 {prof.shape}")
    return hi, it, ppi, prof, gold


def build_disease_profiles(prof):
    """疾病谱: 按 logFC 降序 (原论文按 fold change 排序), 不截断"""
    out = {}
    for did, g in prof.groupby("disease_id"):
        sub = g.dropna(subset=["logFC"]).sort_values("logFC", ascending=False)
        out[did] = sub["gene_symbol"].astype(str).tolist()
    return out


def run_diffusion(diffusion, hi, it, ppi, disease_profiles, herb_ids, gold):
    """用指定扩散方法构建签名并评估 ndcg 打分"""
    print(f"\n{'=' * 70}")
    print(f"[扩散] {diffusion}")
    print("=" * 70)
    ps.DIFFUSION_METHOD = diffusion  # 覆盖模块级变量 (关键)

    sigs, builder = ps.batch_build_herb_signatures(
        hi, it, ppi, use_cache=False, network_type="ppi")
    print(f"[签名] {len(sigs)} 草药 × {builder.get_all_genes().__len__() if hasattr(builder, 'get_all_genes') else '?'} 基因")

    hdmap = HDmapS(sigs, builder)

    herb_ids_all = [h for h in herb_ids if h in sigs]
    tms_df = pair_single_herb_all_diseases(hdmap, herb_ids_all, disease_profiles, k=250)
    # ndcg 版 TMS: 负值 = 治疗方向 -> 取负后"越高越治疗"
    score = -tms_df

    rows = []
    for d in score.columns:
        pos = set(gold[(gold["disease_id"] == d) & (gold["label"] == 1)]["herb_id"].astype(str))
        herbs = list(score.index)
        pos = {h for h in pos if h in set(herbs)}
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in herbs], dtype=float)
        s = score.loc[herbs, d].fillna(0).values.astype(float)
        if len(set(y)) < 2:
            continue
        a = float(roc_auc_score(y, s))
        rows.append({"disease": d, "n_pos": len(pos), "AUROC": round(a, 4)})
        print(f"  {d}: n_pos={len(pos)}, AUROC={a:.4f}")

    df = pd.DataFrame(rows)
    mean_auc = round(df["AUROC"].mean(), 4)
    print(f"  MEAN AUROC = {mean_auc}")
    return df, mean_auc


def target_count_baseline(hi, it, gold, herb_ids):
    """靶点数基线: 只用'该草药有多少个靶点'作为打分"""
    ing_tg = {}
    for iid, g in zip(it["ingredient_id"], it["target_gene"]):
        g = str(g).strip()
        if g and g != "nan":
            ing_tg.setdefault(int(iid), set()).add(g)
    herb_tg = {}
    for hid, iid in zip(hi["herb_id"], hi["ingredient_id"]):
        herb_tg.setdefault(str(hid).strip(), set()).update(ing_tg.get(int(iid), set()))
    tc = {h: len(t) for h, t in herb_tg.items()}

    rows = []
    for d in ["AD", "PAH", "RA", "T2D"]:
        pos = set(gold[(gold["disease_id"] == d) & (gold["label"] == 1)]["herb_id"].astype(str))
        pos = {h for h in pos if h in set(herb_ids)}
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in herb_ids], dtype=float)
        s = np.array([tc.get(h, 0) for h in herb_ids], dtype=float)
        rows.append({"disease": d, "AUROC": round(float(roc_auc_score(y, s)), 4)})
    df = pd.DataFrame(rows)
    return round(df["AUROC"].mean(), 4)


def main():
    hi, it, ppi, prof, gold = load_data()
    disease_profiles = build_disease_profiles(prof)
    herb_ids = list(hi["herb_id"].astype(str).unique())
    print(f"[草药] {len(herb_ids)} 味")

    # 靶点数基线 (必报对照)
    tc_base = target_count_baseline(hi, it, gold, herb_ids)
    print(f"\n[对照] 靶点数基线 MEAN AUROC = {tc_base}")

    # 两种扩散
    df_t, mean_t = run_diffusion("thermal", hi, it, ppi, disease_profiles, herb_ids, gold)
    df_r, mean_r = run_diffusion("rwr", hi, it, ppi, disease_profiles, herb_ids, gold)

    # 汇总
    cmp_df = pd.DataFrame({
        "disease": df_t["disease"],
        "thermal(原论文)": df_t["AUROC"].values,
        "rwr(当前)": df_r["AUROC"].values,
    })
    cmp_df["差值(thermal-rwr)"] = (cmp_df["thermal(原论文)"] - cmp_df["rwr(当前)"]).round(4)
    cmp_df.loc["MEAN"] = ["MEAN",
                          round(cmp_df["thermal(原论文)"].mean(), 4),
                          round(cmp_df["rwr(当前)"].mean(), 4),
                          round(cmp_df["thermal(原论文)"].mean() - cmp_df["rwr(当前)"].mean(), 4)]

    os.makedirs(OUT_DIR, exist_ok=True)
    cmp_df.to_csv(OUT, index=False)

    print("\n" + "=" * 78)
    print("第2步结果: thermal(原论文) vs rwr(当前) —— FULL 协议 AUROC")
    print("=" * 78)
    print(cmp_df.to_string(index=False))
    print(f"\n[对照] 靶点数基线 = {tc_base}  "
          f"(方法需明显超过它才算学到真实信号)")
    print(f"[写入] {OUT}")


if __name__ == "__main__":
    main()
