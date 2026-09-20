#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosine-TMS 基线 (杠杆③): 草药靶点向量 与 疾病 DEG 向量(logFC 加权) 的纯余弦相似度。
关键点: 不含方向校准 —— 作 DATE(方向校准) 的对照, 用于证明"方向校准"的必要性。
输出: output/results/weighted_kegg/hdmap_s_scores_cosine.csv
       (index=herb_id, columns=AD/PAH/RA/T2D, 值=cosine, 越高=越治疗方向一致)
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = "c:/Users/alikex/Desktop/Work/systems_mapping/"
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "output", "results")
OUT = os.path.join(RES, "weighted_kegg", "hdmap_s_scores_cosine.csv")
sys.path.insert(0, BASE)
import config as cfg  # noqa: E402

TOPK = 1500  # 与 regenerate_enrichment_scores.py 保持一致


def main():
    # 疾病 DEG (按 |logFC| top-K, 与 regenerate 一致)
    prof = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    dis_vec = {}
    for did, g in prof.groupby("disease_id"):
        sub = g.reindex(g["logFC"].abs().sort_values(ascending=False).index).head(TOPK)
        dis_vec[did] = (sub["gene_symbol"].astype(str).tolist(),
                        sub["logFC"].astype(float).tolist())

    # 草药 -> {gene: weight}
    it = pd.read_csv(os.path.join(PROC, "ingredient_targets_enhanced.csv"))
    ing_tg = {}
    for _, r in it.iterrows():
        g = str(r["target_gene"]).strip()
        if g and g != "nan":
            c = float(r.get("confidence", 0.5) or 0.5)
            ing_tg.setdefault(int(r["ingredient_id"]), {})[g] = max(
                ing_tg.get(int(r["ingredient_id"]), {}).get(g, 0.0), c)
    hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
    herb_tg = {}
    for _, r in hi.iterrows():
        hid = str(r["herb_id"]).strip()
        for g, c in ing_tg.get(int(r["ingredient_id"]), {}).items():
            herb_tg.setdefault(hid, {})[g] = herb_tg.get(hid, {}).get(g, 0.0) + c

    # 基因空间
    genes = set()
    for gl, _ in dis_vec.values():
        genes.update(gl)
    for tg in herb_tg.values():
        genes.update(tg.keys())
    g2i = {g: i for i, g in enumerate(sorted(genes))}
    n = len(g2i)

    # 疾病单位向量
    dvec = {}
    for did, (gl, lf) in dis_vec.items():
        v = np.zeros(n)
        for gene, fc in zip(gl, lf):
            if gene in g2i:
                v[g2i[gene]] = fc
        nv = np.linalg.norm(v)
        if nv > 0:
            v = v / nv
        dvec[did] = v

    # 统一草药集合到 DATE 主矩阵 (491 味), 保证各方法评估集合一致
    date_mat = pd.read_csv(os.path.join(RES, "weighted_kegg", "hdmap_s_scores.csv"), index_col=0)
    std_herbs = list(date_mat.index)
    herbs = std_herbs
    rows = []
    for hid in herbs:
        hv = np.zeros(n)
        for g, w in herb_tg[hid].items():
            if g in g2i:
                hv[g2i[g]] = w
        nh = np.linalg.norm(hv)
        if nh > 0:
            hv = hv / nh
        row = {"herb_id": hid}
        for did, v in dvec.items():
            row[did] = float(np.dot(hv, v))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("herb_id")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT)
    print("Cosine-TMS 矩阵:", df.shape)
    print(df.mean().round(4).to_string())
    print("[写入]", OUT)


if __name__ == "__main__":
    main()
