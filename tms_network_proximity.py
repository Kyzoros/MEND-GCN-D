#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Proximity TMS (Barabasi 网络医学标准方法)
==================================================
为什么需要它 (2026-08-29 诊断结论):
  现有 enrichment 方法 (Fisher 富集, 得分=-log10 p) 的分数与"草药靶点数量"
  的 Spearman 相关高达 0.65~0.82, 说明分数主要反映"这味药有多少靶点"
  这个跨病通用的非特异信号; 疾病特异性极弱 (跨病中心化后 AUROC 掉到 0.50)。
  两种朴素修正 (回归掉靶点数 / 跨病中心化) 均失败:
      - 回归掉靶点数 -> AUROC 0.374 (暴跌, 说明靶点数含部分真实信号)
      - 跨病中心化   -> AUROC 0.499 (说明特异性信号本就微弱)
  因此需要换用具有疾病特异性的网络方法。

方法 (Barabasi 网络医学范式):
  1. 在 KEGG 共成员网络上, 用 multi-source BFS 求疾病模块到全图的最短距离
  2. 草药靶点集 T 到疾病模块 S 的邻近度
        d(S,T) = mean_{t in T} min_{s in S} d(s,t)
  3. 度匹配随机化 (degree-preserving) 给出随机期望与方差。
     对每个靶点, 参照"相同度"的节点集合的距离分布。样本量 n 较大时
     用解析近似 (与显式置换等价):
        mu_rand  = mean_t  mu[deg(t)]
        var_rand = sum_t var[deg(t)] / |T|^2
  4. z = (d_obs - mu_rand) / sqrt(var_rand)
     z 越负 = 草药靶点越贴近疾病模块 = 越可能治疗
  5. TMS = -z  (越高越治疗, 与下游 higher_is_better=True 一致)

相比 Fisher 富集的优势:
  - 使用网络拓扑, 而非单纯集合重叠
  - z-score 消除"靶点数量"与"节点度"的系统性偏置
  - 疾病特异性来自"相对于随机期望的偏离"

用法:
    python tms_network_proximity.py
    python tms_network_proximity.py --out output/results/weighted_kegg/hdmap_s_scores_proximity.csv
"""
import argparse
import os
import sys
from collections import deque

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import config as cfg
from kegg_network import build_kegg_adjacency


def load_disease_genes():
    """按 config 阈值加载每病 DEG 基因列表。

    【2026-09-02 修正】必须与 main.py 的 _filter_disease_genes 严格一致, 否则
    本方法用的疾病基因集与 enrichment 方法不同, 两者 AUROC 无法比较。
    原实现只用了 logFC + P 值双阈值, 漏掉了当前默认的 top-K 模式
    (DISEASE_DEG_TOP_K=1500, 优先级高于 P 值阈值)。
    """
    path = os.path.join(cfg.PROCESSED_DIR, "disease_profiles_processed.csv")
    if not os.path.exists(path):
        path = os.path.join(cfg.DATA_DIR, "disease_profiles_processed.csv")
    df = pd.read_csv(path)

    out = {}
    for did in df["disease_id"].unique():
        sub = df[df["disease_id"] == did]
        # 与 main.py:_filter_disease_genes 逐条对齐
        if cfg.DISEASE_LOGFC_THRESHOLD > 0 and "logFC" in sub.columns:
            sub = sub[sub["logFC"].abs() > cfg.DISEASE_LOGFC_THRESHOLD]
        if cfg.DISEASE_DEG_TOP_K and cfg.DISEASE_DEG_TOP_K > 0:
            # top-K 模式: 按 P 值取最显著的 K 个 (各病规模一致, 保证跨病可比)
            if "P_value" in sub.columns:
                sub = sub.sort_values("P_value", ascending=True)
            else:
                sub = sub.sort_values("logFC", key=lambda s: s.abs(), ascending=False)
            sub = sub.head(cfg.DISEASE_DEG_TOP_K)
        elif cfg.DISEASE_PVALUE_THRESHOLD > 0 and "P_value" in sub.columns:
            sub = sub[sub["P_value"] < cfg.DISEASE_PVALUE_THRESHOLD]
        out[did] = sub.sort_values("logFC", ascending=False)["gene_symbol"].tolist()
    return out


def load_herb_targets():
    """herb_id -> set(target_gene)

    【2026-09-02 修正】改走 hdmap_s.build_herb_target_sets(), 以复用
    **靶点基因符号规范化** (蛋白质全名 -> 基因符号)。
    原实现直接读 ingredient_targets_enhanced.csv, 其中 72.1% 的记录是蛋白质全名,
    在网络里根本查不到对应节点, 等于本方法一直在用不到三成的靶点数据。
    """
    from hdmap_s import build_herb_target_sets
    return build_herb_target_sets()


def multi_source_bfs(adj_csr, sources, n):
    """无向无权图 multi-source BFS, 返回 dist 数组 (-1 = 不可达)"""
    indptr, indices = adj_csr.indptr, adj_csr.indices
    dist = np.full(n, -1, dtype=np.int32)
    q = deque()
    for s in sources:
        if 0 <= s < n and dist[s] < 0:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        du = dist[u] + 1
        for v in indices[indptr[u]:indptr[u + 1]]:
            if dist[v] < 0:
                dist[v] = du
                q.append(v)
    return dist


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


def main():
    ap = argparse.ArgumentParser(description="Network Proximity TMS")
    ap.add_argument("--out", default=None, help="输出 CSV 路径")
    ap.add_argument("--min-targets", type=int, default=5,
                    help="草药在网络中的最少可达靶点数, 低于则置 0")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        cfg.RESULTS_DIR, "weighted_kegg", "hdmap_s_scores_proximity.csv")

    # ---- 1. 网络 (二值化 + 对称化) ----
    print("[1/4] 加载 KEGG 网络 ...")
    adj_norm, gene2idx = build_kegg_adjacency(
        cache_dir=cfg.PROCESSED_DIR, min_shared=cfg.KEGG_MIN_SHARED)
    n = adj_norm.shape[0]
    adj = adj_norm.copy()
    adj.data = np.ones_like(adj.data)          # 二值化
    adj = ((adj + adj.T) > 0).astype(np.int8).tocsr()   # 对称化(无向)
    adj.setdiag(0)
    adj.eliminate_zeros()
    deg = np.asarray(adj.sum(axis=1)).flatten().astype(np.int64)
    print(f"  网络: {n} 节点, {adj.nnz // 2} 无向边")

    # ---- 2. DEG ----
    print("[2/4] 加载疾病 DEG ...")
    disease_genes = load_disease_genes()
    for d, g in disease_genes.items():
        in_net = [x for x in g if x in gene2idx]
        print(f"  {d:<5} DEG={len(g):>5}  在网络中={len(in_net):>5}")

    # ---- 3. 草药靶点 ----
    print("[3/4] 加载草药靶点 ...")
    herb_targets = load_herb_targets()
    print(f"  草药数: {len(herb_targets)}")

    # ---- 4. 逐病计算 ----
    print("[4/4] 计算 Network Proximity ...")
    result = {}
    for dis, genes in disease_genes.items():
        src = [gene2idx[g] for g in genes if g in gene2idx]
        if not src:
            print(f"  [跳过] {dis}: 无 DEG 在网络中")
            continue
        dist = multi_source_bfs(adj, src, n)
        reach = dist >= 0
        if reach.sum() == 0:
            print(f"  [跳过] {dis}: 网络不可达")
            continue

        # 每度桶的距离均值/方差 (度匹配随机化的参照分布)
        g_mu = float(dist[reach].mean())
        g_var = float(dist[reach].astype(float).var())
        mu_by_deg, var_by_deg = {}, {}
        reach_deg = deg[reach]
        reach_dist = dist[reach].astype(float)
        for dv in np.unique(reach_deg):
            m = reach_deg == dv
            if m.sum() < 5:
                continue
            mu_by_deg[int(dv)] = float(reach_dist[m].mean())
            var_by_deg[int(dv)] = float(reach_dist[m].var())

        scores = {}
        for hid, tgts in herb_targets.items():
            tidx = [gene2idx[t] for t in tgts if t in gene2idx]
            tidx = [i for i in tidx if dist[i] >= 0]
            if len(tidx) < args.min_targets:
                scores[hid] = 0.0
                continue
            d_obs = float(dist[tidx].mean())
            dgs = [int(deg[i]) for i in tidx]
            mus = np.array([mu_by_deg.get(d, g_mu) for d in dgs])
            vs = np.array([var_by_deg.get(d, g_var) for d in dgs])
            mu_rand = float(mus.mean())
            var_rand = float(vs.sum()) / (len(tidx) ** 2)
            if var_rand <= 1e-12:
                scores[hid] = 0.0
            else:
                z = (d_obs - mu_rand) / np.sqrt(var_rand)
                scores[hid] = -float(z)      # z 越负越近 -> 取负使越高越治疗
        result[dis] = scores
        vals = np.array(list(scores.values()))
        print(f"  {dis:<5} 完成  score: min={vals.min():.3f} "
              f"max={vals.max():.3f} mean={vals.mean():.3f}")

    df = pd.DataFrame(result)
    df.index.name = "herb_id"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path)
    print(f"\n[写入] {out_path}   shape={df.shape}")

    # ---- 评估 (FULL 协议: 阳性 vs 全部草药) ----
    try:
        from sklearn.metrics import roc_auc_score
        gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
        herbs = list(df.index.astype(str))
        print("\n" + "=" * 74)
        print("评估 (FULL 协议: 阳性 vs 全部草药, 越高越治疗)")
        print("=" * 74)
        print(f"{'disease':<8}{'n_pos':>7}{'AUROC':>9}{'CI_lo':>9}{'CI_hi':>9}{'显著性':>16}")
        print("-" * 74)
        aucs = []
        for d in df.columns:
            pos = set(gold[(gold["disease_id"] == d) &
                           (gold["label"] == 1)]["herb_id"].astype(str))
            pos = {h for h in pos if h in df.index}
            if len(pos) < 3:
                continue
            y = np.array([1 if h in pos else 0 for h in herbs], dtype=float)
            s = df.loc[herbs, d].values.astype(float)
            auc, lo, hi = bootstrap_ci(y, s)
            sig = "显著" if lo > 0.5 else ("反向" if hi < 0.5 else "不显著")
            aucs.append(auc)
            print(f"{d:<8}{len(pos):>7}{auc:>9.4f}{lo:>9.4f}{hi:>9.4f}{sig:>16}")
        print("-" * 74)
        print(f"{'平均':<8}{'':>7}{np.mean(aucs):>9.4f}")
        print("(对比: 2026-09-02 修正后重测 —— 富集法 0.4753 / 富集z 0.4696 / "
              "网络邻近度 0.4835, 均**低于**靶点数平凡基线 0.5809)")
    except Exception as e:
        print(f"[提示] 评估跳过: {e}")


if __name__ == "__main__":
    main()
