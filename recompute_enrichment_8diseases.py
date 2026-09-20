"""
recompute_enrichment_8diseases.py
==================================
只为"补算新 4 病(HCC/CRC/ASTHMA/OA)的 enrichment TMS"而写, 不跑慢速 GA。
读取已处理好的 8 病 disease_profiles_processed.csv, 用与 main.py step3 完全一致的
top-K 过滤 + pair_single_herb_all_diseases_enrichment, 产出 8 病 TMS 分数。
输出: data/processed/hdmap_s_scores_enrichment_8d.csv
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfg
from hdmap_s import pair_single_herb_all_diseases_enrichment

PROC = cfg.PROCESSED_DIR
TOP_K = cfg.DISEASE_DEG_TOP_K
LOGFC = cfg.DISEASE_LOGFC_THRESHOLD
PV = cfg.DISEASE_PVALUE_THRESHOLD


def filter_genes(sub):
    if LOGFC > 0:
        sub = sub[sub['logFC'].abs() > LOGFC]
    if TOP_K and TOP_K > 0:
        if 'P_value' in sub.columns:
            sub = sub.sort_values('P_value', ascending=True)
        else:
            sub = sub.sort_values('logFC', key=lambda s: s.abs(), ascending=False)
        sub = sub.head(TOP_K)
    elif PV > 0 and 'P_value' in sub.columns:
        sub = sub[sub['P_value'] < PV]
    return sub


def main():
    profiles = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    print(f"profiles 疾病: {sorted(profiles['disease_id'].unique())}")

    disease_genes_dict = {}
    for did in profiles['disease_id'].unique():
        sub = profiles[profiles['disease_id'] == did]
        sub = filter_genes(sub).sort_values('logFC', ascending=False)
        disease_genes_dict[did] = sub['gene_symbol'].tolist()
        print(f"  {did}: {len(disease_genes_dict[did])} DEG 基因")

    hi = pd.read_csv(os.path.join(PROC, cfg.FILE_HERB_INGREDIENTS_PROCESSED))
    herb_ids = sorted(set(hi['herb_id'].astype(str)))
    print(f"  草药数: {len(herb_ids)}")

    tms = pair_single_herb_all_diseases_enrichment(disease_genes_dict, herb_ids)
    tms.index.name = "herb_id"
    OUT = os.path.join(PROC, "hdmap_s_scores_enrichment_8d.csv")
    tms.to_csv(OUT)
    print(f"\n8病 enrichment TMS 形状: {tms.shape}")
    print(f"疾病列: {list(tms.columns)}")
    print(f"写入: {OUT}")


if __name__ == "__main__":
    main()
