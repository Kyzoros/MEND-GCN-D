#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K 分层评估协议 (剥离注释偏倚) —— 论文主评估协议
================================================
为什么需要它 (2026-09-02):
  FULL 协议 (阳性 vs 全部草药) 下, "靶点数 K" 这个**与疾病无关**的平凡特征
  就能拿到平均 AUROC 0.5809, 而三种疾病特异性方法只有 0.47~0.48。
  原因是金标准(文献 curated)与靶点注释共享**文献关注度偏倚**:
  研究得透的草药靶点多、也更容易被金标准收录。FULL 协议让 K 极易占优。

  本协议把草药按靶点数分层, **只在靶点数相近的草药之间比较**,
  从而剥离"研究热度", 回答真正的问题:
      **控制靶点数后, 打分还能不能识别出真正有效的草药?**

指标 (分层 AUC / stratified AUC):
  1. 按 K 分 q 个分位箱 (默认 4)
  2. 只统计同一箱内的 (阳性, 阴性) 配对: score(阳性) > score(阴性) 记 1, 平局 0.5
  3. 分层 AUC = Σ concordant / Σ pairs, 按配对数跨箱加权
  4. 零假设 = 0.5

有效性自检 (必须看):
  把 K 本身当打分跑同一协议, 结果必须 ≈ 0.5。
  若明显偏离 0.5, 说明分箱没剥离干净, 协议结论不可信。

用法:
    python run_k_stratified_eval.py
    python run_k_stratified_eval.py --conf 0.7          # 只用置信度>=0.7的靶点
    python run_k_stratified_eval.py --bins 5 --boot 1000
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import config as cfg
from hdmap_s import load_gene_symbol_map
from evaluation_metrics import load_gold_standard
from data_loader import load_herb_ingredients

PROC = cfg.PROCESSED_DIR
TOP_K = cfg.DISEASE_DEG_TOP_K
LOGFC = cfg.DISEASE_LOGFC_THRESHOLD
PV = cfg.DISEASE_PVALUE_THRESHOLD


# ----------------------------------------------------------------------------
# 疾病基因集 / 靶点
# ----------------------------------------------------------------------------
def filter_genes(sub):
    """与 main.py:_filter_disease_genes 严格一致 (top-K 优先)。"""
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


def _normalize(name, gmap, known):
    """蛋白质全名/别名 -> 基因符号; 无法规范的返回 None。"""
    s = str(name).strip()
    if not s or s.lower() == 'nan':
        return None
    if s in known:
        return s
    hit = gmap.get(s.lower())
    if hit:
        return hit
    if ' ' not in s and 2 <= len(s) <= 15:
        return s
    return None


def load_disease_genes():
    df = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    out = {}
    for did in df['disease_id'].unique():
        sub = filter_genes(df[df['disease_id'] == did])
        genes = sub.sort_values('logFC', ascending=False)['gene_symbol'].tolist()
        out[did] = set(g for g in genes if isinstance(g, str) and g and g != 'nan')
    return out


def build_herb_targets(conf_thr=0.0):
    """草药 -> 靶点基因集合 (已做基因符号规范化, 可按置信度过滤)。"""
    it = pd.read_csv(os.path.join(PROC, cfg.FILE_INGREDIENT_TARGETS_ENHANCED))
    hi = pd.read_csv(os.path.join(PROC, cfg.FILE_HERB_INGREDIENTS_PROCESSED))
    gmap = load_gene_symbol_map()
    known = set(gmap.values())

    if conf_thr > 0 and 'confidence' in it.columns:
        it = it[it['confidence'].astype(float) >= conf_thr]

    ing2t = {}
    for _, r in it.iterrows():
        g = _normalize(r['target_gene'], gmap, known)
        if g:
            ing2t.setdefault(int(r['ingredient_id']), set()).add(g)

    herb = {}
    for _, r in hi.iterrows():
        hid = str(r['herb_id']).strip()
        tg = ing2t.get(int(r['ingredient_id']))
        if tg:
            herb.setdefault(hid, set()).update(tg)
    return herb


# ----------------------------------------------------------------------------
# 打分
# ----------------------------------------------------------------------------
def enrichment_scores(herb_targets, deg, universe):
    """返回 (pvalue 列, z 列)。背景(null) = 靶点宇宙, 与 hdmap_s 修正后一致。"""
    N_U = len(universe)
    n_d = len(deg)
    pc, zc = {}, {}
    for h, T_all in herb_targets.items():
        T = T_all & universe
        k = len(T)
        a = len(T & deg)
        if a == 0 or k == 0 or n_d == 0:
            pc[h], zc[h] = 0.0, 0.0
            continue
        b, c = k - a, n_d - a
        dd = N_U - a - b - c
        if dd < 0:
            dd = 0
        pc[h] = -np.log10(max(fisher_exact([[a, b], [c, dd]])[1], 1e-300))
        pp = n_d / N_U
        var = k * pp * (1 - pp) * (N_U - k) / max(N_U - 1, 1)
        zc[h] = (a - k * pp) / np.sqrt(var) if var > 0 else 0.0
    return pc, zc


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def stratified_auc(y, s, bins):
    """只统计同箱内的 (pos,neg) 配对。返回 (分层AUC, 总配对数)。"""
    concordant = 0.0
    total = 0
    for b in np.unique(bins):
        m = bins == b
        yy, ss = y[m], s[m]
        pos, neg = ss[yy == 1], ss[yy == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        for p in pos:
            concordant += float(np.sum(p > neg) + 0.5 * np.sum(p == neg))
        total += len(pos) * len(neg)
    return (concordant / total if total else float('nan')), total


def full_auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2 or len(set(s)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="K 分层评估协议")
    ap.add_argument("--bins", type=int, default=4, help="靶点数分层箱数 (默认 4)")
    ap.add_argument("--conf", type=float, default=0.0,
                    help="靶点置信度下限 (默认 0=不过滤; 0.7 实测最优)")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap 次数 (默认 500)")
    ap.add_argument("--out", default=None, help="结果 CSV 输出路径")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        "output", "results", "paper_deliverables", "k_stratified_evaluation.csv")

    print("=" * 90)
    print("K 分层评估协议 (剥离注释偏倚)")
    print("=" * 90)

    dgd = load_disease_genes()
    herb_targets = build_herb_targets(args.conf)
    U = set().union(*herb_targets.values())
    N_U = len(U)
    order = sorted(herb_targets.keys())
    K = np.array([float(len(herb_targets[h])) for h in order])
    print(f"靶点置信度阈值: {args.conf} | 唯一靶点 {N_U} | 草药 {len(order)} "
          f"| 平均靶点/药 {K.mean():.0f}")

    # 分箱
    qs = np.quantile(K, np.linspace(0, 1, args.bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    bins = np.digitize(K, qs[1:-1])
    print(f"\n[分箱] K 分 {args.bins} 箱:")
    for b in range(args.bins):
        kk = K[bins == b]
        if len(kk):
            print(f"  箱{b}: n={len(kk):3d}  K∈[{kk.min():.0f},{kk.max():.0f}]  "
                  f"中位 {np.median(kk):.0f}")

    # 网络邻近度(若已产出)
    prox_path = "output/results/network_proximity/hdmap_s_scores_proximity_8d.csv"
    prox = None
    if os.path.exists(prox_path):
        prox = pd.read_csv(prox_path, index_col=0)
        prox.index = prox.index.astype(str)

    gold, _, _ = load_gold_standard(cfg.GOLD_STANDARD_PATH, load_herb_ingredients())

    rows = []
    boot_ready = []
    for d in dgd:
        if d not in gold:
            continue
        pos = set(str(h) for h in gold[d]) & set(order)
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in order])
        pc, zc = enrichment_scores(herb_targets, dgd[d] & U, U)
        scores = {
            'K': K,
            'enrichment_p': np.array([float(pc.get(h, 0.0)) for h in order]),
            'enrichment_z': np.array([float(zc.get(h, 0.0)) for h in order]),
        }
        if prox is not None and d in prox.columns:
            scores['network_proximity'] = prox[d].reindex(order).fillna(0.0).values.astype(float)
        boot_ready.append((d, y, scores))

        for m, s in scores.items():
            st, npair = stratified_auc(y, s, bins)
            rows.append({'disease_id': d, 'method': m, 'stratified_auc': round(st, 4),
                         'full_auc': round(full_auc(y, s), 4),
                         'n_positive': int(len(pos)), 'n_pairs': int(npair)})

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("分层 AUC (主指标; 0.5 = 随机)")
    print("=" * 90)
    piv = df.pivot(index='disease_id', columns='method', values='stratified_auc')
    print(piv.round(4).to_string())
    print("-" * 90)
    print("平均:")
    for m in piv.columns:
        print(f"  {m:<20} {piv[m].mean():.4f}")

    # 自检
    k_strat = float(piv['K'].mean())
    print("\n" + "=" * 90)
    print(f"有效性自检: K 的分层 AUC = {k_strat:.4f} "
          f"({'通过, 偏倚已剥离' if abs(k_strat - 0.5) < 0.05 else '未通过, 分箱不足'})")
    print("=" * 90)

    # bootstrap CI
    rng = np.random.default_rng(42)
    n = len(order)
    methods = list(boot_ready[0][2].keys())
    boot = {m: [] for m in methods}
    for _ in range(args.boot):
        idx = rng.integers(0, n, n)
        per = {m: [] for m in methods}
        for d, y, sc in boot_ready:
            yy = y[idx]
            if len(set(yy)) < 2:
                continue
            bb = bins[idx]
            for m, s in sc.items():
                st, _ = stratified_auc(yy, s[idx], bb)
                if not np.isnan(st):
                    per[m].append(st)
        for m in methods:
            if per[m]:
                boot[m].append(float(np.mean(per[m])))

    print(f"\n{args.boot} 次 bootstrap 95% CI:")
    print(f"{'方法':<22}{'分层AUC':>10}{'CI_lo':>10}{'CI_hi':>10}{'判定':>16}")
    print("-" * 68)
    verdicts = {}
    for m in methods:
        est = float(piv[m].mean())
        arr = np.array(boot[m])
        lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan))
        if m == 'K':
            v = "自检(应≈0.5)"
        elif lo > 0.5:
            v = "优于随机"
        elif hi < 0.5:
            v = "差于随机"
        else:
            v = "与随机无异"
        verdicts[m] = (round(lo, 4), round(hi, 4), v)
        print(f"{m:<22}{est:>10.4f}{lo:>10.4f}{hi:>10.4f}{v:>16}")

    df['ci_lo'] = df['method'].map(lambda m: verdicts.get(m, (np.nan,) * 3)[0])
    df['ci_hi'] = df['method'].map(lambda m: verdicts.get(m, (np.nan,) * 3)[1])
    df['verdict'] = df['method'].map(lambda m: verdicts.get(m, (np.nan,) * 3)[2])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {out_path}")


if __name__ == "__main__":
    main()
