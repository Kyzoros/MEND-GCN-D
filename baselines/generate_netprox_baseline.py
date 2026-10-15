#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NetProx 基线 (杠杆③): 基于 PPI 网络的药物-疾病模块网络邻近度 (Menche 2015 风格)。
对每味草药, 以其靶点集合 S 为源做多源 BFS, 得到 PPI 上各节点到 S 的最短跳数;
疾病模块 D (top-K DEG) 中每个基因到 S 的距离均值即为 proximity, 取负(越近=越治疗)。
输出: output/results/weighted_kegg/hdmap_s_scores_netprox.csv
       (index=herb_id, columns=AD/PAH/RA/T2D, 值=-mean(shortest_path), 越高=越治疗)
"""
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "output", "results")
OUT = os.path.join(RES, "weighted_kegg", "hdmap_s_scores_netprox.csv")

TOPK = 1500  # 与 regenerate 一致


def load_ppi(path):
    adj = {}
    df = pd.read_csv(path)
    for a, b in zip(df.iloc[:, 0], df.iloc[:, 1]):
        a, b = str(a), str(b)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def bfs_multisource(adj, sources):
    dist = {s: 0 for s in sources if s in adj}
    q = deque(dist.keys())
    while q:
        u = q.popleft()
        du = dist[u]
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
    return dist


def main():
    adj = load_ppi(os.path.join(PROC, "ppi_network.csv"))

    prof = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    dis_mod = {}
    for did, g in prof.groupby("disease_id"):
        sub = g.reindex(g["logFC"].abs().sort_values(ascending=False).index).head(TOPK)
        dis_mod[did] = [str(x) for x in sub["gene_symbol"].tolist()]

    it = pd.read_csv(os.path.join(PROC, "ingredient_targets_enhanced.csv"))
    ing_tg = {}
    for _, r in it.iterrows():
        g = str(r["target_gene"]).strip()
        if g and g != "nan":
            ing_tg.setdefault(int(r["ingredient_id"]), set()).add(g)
    hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
    herb_tg = {}
    for _, r in hi.iterrows():
        hid = str(r["herb_id"]).strip()
        herb_tg.setdefault(hid, set()).update(ing_tg.get(int(r["ingredient_id"]), set()))

    # 统一草药集合到 DATE 主矩阵 (491 味), 保证各方法评估集合一致
    date_mat = pd.read_csv(os.path.join(RES, "weighted_kegg", "hdmap_s_scores.csv"), index_col=0)
    std_herbs = list(date_mat.index)
    herbs = std_herbs
    diseases = list(dis_mod.keys())
    data = {d: [] for d in diseases}
    for i, hid in enumerate(herbs, 1):
        S = herb_tg[hid]
        dist = bfs_multisource(adj, S)
        for d in diseases:
            vals = [dist[g] for g in dis_mod[d] if g in adj and g in dist]
            vals = [v for v in vals if np.isfinite(v)]
            data[d].append(-float(np.mean(vals)) if vals else 0.0)
        if i % 50 == 0:
            print(f"[NetProx] {i}/{len(herbs)}")

    df = pd.DataFrame(data, index=herbs)
    df.index.name = "herb_id"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT)
    print("NetProx 矩阵:", df.shape)
    print(df.mean().round(4).to_string())
    print("[写入]", OUT)


if __name__ == "__main__":
    main()
