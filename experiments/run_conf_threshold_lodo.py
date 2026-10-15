#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
靶点置信度阈值的 留一病验证 (Leave-One-Disease-Out, LODO)
=========================================================
背景 (2026-09-02):
  K 分层协议下, 用 confidence>=0.7 的靶点能把富集z从 0.4628 提到 0.5268 (+0.064)。
  但**该阈值是在同样这 8 个病上挑出来的**, 属事后选择 (post-hoc selection),
  有过拟合风险 —— 不能据此声称"置信度过滤有效"。

本脚本做诚实的样本外验证:
  外层: 依次留出 1 个病作为测试集 (8 折)
  内层: 在其余 7 个病上, 从候选阈值中选分层 AUC 最高的 t*
  评估: 用 t* 在留出病上算分层 AUC
  对照: 同一留出病、用现行设定 (t=0, 不过滤) 的基线

若 LODO 结果 ≈ 基线, 说明 +0.064 是过拟合, 不应写进论文;
若 LODO 仍明显优于基线, 说明置信度过滤是真改进。

同时报告:
  - 每个折选中的阈值 (一致性: 若各折选的阈值乱跳, 说明选择不稳定)
  - 阈值-性能曲线 (看是否存在稳定的平台期, 而非尖峰)

用法:
    python run_conf_threshold_lodo.py
    python run_conf_threshold_lodo.py --thrs 0 0.3 0.5 0.7 0.9
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg
from hdmap_s import load_gene_symbol_map
from evaluation_metrics import load_gold_standard
from data_loader import load_herb_ingredients

PROC = cfg.PROCESSED_DIR
TOP_K = cfg.DISEASE_DEG_TOP_K
LOGFC = cfg.DISEASE_LOGFC_THRESHOLD
PV = cfg.DISEASE_PVALUE_THRESHOLD
N_BINS = 4


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


def _norm(name, gmap, known):
    s = str(name).strip()
    if not s or s.lower() == 'nan':
        return None
    if s in known:
        return s
    h = gmap.get(s.lower())
    if h:
        return h
    if ' ' not in s and 2 <= len(s) <= 15:
        return s
    return None


def build_herb_targets(conf_thr):
    it = pd.read_csv(os.path.join(PROC, cfg.FILE_INGREDIENT_TARGETS_ENHANCED))
    hi = pd.read_csv(os.path.join(PROC, cfg.FILE_HERB_INGREDIENTS_PROCESSED))
    gmap = load_gene_symbol_map()
    known = set(gmap.values())
    if conf_thr > 0 and 'confidence' in it.columns:
        it = it[it['confidence'].astype(float) >= conf_thr]
    ing2t = {}
    for _, r in it.iterrows():
        g = _norm(r['target_gene'], gmap, known)
        if g:
            ing2t.setdefault(int(r['ingredient_id']), set()).add(g)
    herb = {}
    for _, r in hi.iterrows():
        hid = str(r['herb_id']).strip()
        tg = ing2t.get(int(r['ingredient_id']))
        if tg:
            herb.setdefault(hid, set()).update(tg)
    return herb


def strat_auc(y, s, bins):
    conc, tot = 0.0, 0
    for b in np.unique(bins):
        m = bins == b
        yy, ss = y[m], s[m]
        pos, neg = ss[yy == 1], ss[yy == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        for p in pos:
            conc += float(np.sum(p > neg) + 0.5 * np.sum(p == neg))
        tot += len(pos) * len(neg)
    return (conc / tot if tot else float('nan')), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thrs", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 0.9, 0.95])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        "output", "results", "paper_deliverables", "conf_threshold_lodo.csv")

    dgd_all = {}
    df = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    for did in df['disease_id'].unique():
        sub = filter_genes(df[df['disease_id'] == did])
        genes = sub.sort_values('logFC', ascending=False)['gene_symbol'].tolist()
        dgd_all[did] = set(g for g in genes if isinstance(g, str) and g and g != 'nan')

    gold, _, _ = load_gold_standard(cfg.GOLD_STANDARD_PATH, load_herb_ingredients())
    diseases = sorted([d for d in dgd_all if d in gold])

    # ---- 预计算: 每个阈值 -> (bins, {病: 富集z}, K, order) ----
    print("=" * 88)
    print("预计算各置信度阈值下的分层结构与富集z")
    print("=" * 88)
    per_thr = {}
    for t in args.thrs:
        ht = build_herb_targets(t)
        U = set().union(*ht.values())
        N_U = len(U)
        order = sorted(ht.keys())
        if N_U < 20 or len(order) < 50:
            print(f"  阈值 {t}: 靶点过少 (|U|={N_U}), 跳过")
            continue
        K = np.array([float(len(ht[h])) for h in order])
        qs = np.quantile(K, np.linspace(0, 1, N_BINS + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        bins = np.digitize(K, qs[1:-1])
        zmap = {}
        for d in diseases:
            deg = dgd_all[d] & U
            n_d = len(deg)
            zc = {}
            for h in order:
                T = ht[h] & U
                k = len(T)
                a = len(T & deg)
                if a == 0 or k == 0 or n_d == 0:
                    zc[h] = 0.0
                    continue
                pp = n_d / N_U
                var = k * pp * (1 - pp) * (N_U - k) / max(N_U - 1, 1)
                zc[h] = (a - k * pp) / np.sqrt(var) if var > 0 else 0.0
            zmap[d] = np.array([float(zc.get(h, 0.0)) for h in order])
        # K 自检 (每阈值都应 ≈0.5)
        k_self = []
        for d in diseases:
            pos = set(str(h) for h in gold[d]) & set(order)
            if len(pos) < 3:
                continue
            y = np.array([1 if h in pos else 0 for h in order])
            st, _ = strat_auc(y, K, bins)
            k_self.append(st)
        per_thr[t] = {'bins': bins, 'z': zmap, 'K': K, 'order': order,
                      'U': U, 'k_self': float(np.nanmean(k_self))}
        mean_auc = float(np.nanmean([
            strat_auc(np.array([1 if h in set(str(x) for x in gold[d]) & set(order) else 0
                                for h in order]), zmap[d], bins)[0]
            for d in diseases
            if len(set(str(x) for x in gold[d]) & set(order)) >= 3]))
        print(f"  阈值 {t:<5} |U|={N_U:<5} 均靶点/药={K.mean():>4.0f}  "
              f"富集z(全8病)={mean_auc:.4f}  K自检={per_thr[t]['k_self']:.4f}")

    thr_list = sorted(per_thr.keys())
    base_t = 0.0 if 0.0 in per_thr else thr_list[0]

    # ---- LODO ----
    print("\n" + "=" * 88)
    print("留一病验证 (LODO): 在其余 7 病上选阈值, 在留出病上评估")
    print("=" * 88)
    print(f"{'留出病':<9}{'内层选中阈值':>14}{'LODO AUC':>12}{'基线(t=0)':>12}{'差值':>10}")
    print("-" * 88)

    rows = []
    lodo_aucs, base_aucs, chosen = [], [], []
    for dtest in diseases:
        train = [d for d in diseases if d != dtest]
        # 内层: 在 7 个训练病上评估每个阈值
        inner = {}
        for t in thr_list:
            P = per_thr[t]
            aucs = []
            for d in train:
                pos = set(str(h) for h in gold[d]) & set(P['order'])
                if len(pos) < 3:
                    continue
                y = np.array([1 if h in pos else 0 for h in P['order']])
                st, _ = strat_auc(y, P['z'][d], P['bins'])
                aucs.append(st)
            inner[t] = float(np.nanmean(aucs)) if aucs else float('nan')
        t_star = max(inner, key=lambda t: inner[t])

        # 外层: 用 t_star 在留出病上评估
        Pt = per_thr[t_star]
        Pb = per_thr[base_t]
        pos_t = set(str(h) for h in gold[dtest]) & set(Pt['order'])
        y_t = np.array([1 if h in pos_t else 0 for h in Pt['order']])
        lodo, _ = strat_auc(y_t, Pt['z'][dtest], Pt['bins'])

        pos_b = set(str(h) for h in gold[dtest]) & set(Pb['order'])
        y_b = np.array([1 if h in pos_b else 0 for h in Pb['order']])
        base, _ = strat_auc(y_b, Pb['z'][dtest], Pb['bins'])

        lodo_aucs.append(lodo)
        base_aucs.append(base)
        chosen.append(t_star)
        print(f"{dtest:<9}{t_star:>14}{lodo:>12.4f}{base:>12.4f}{lodo - base:>+10.4f}")
        rows.append({'held_out_disease': dtest, 'selected_threshold': t_star,
                     'lodo_auc': round(lodo, 4), 'baseline_auc': round(base, 4),
                     'delta': round(lodo - base, 4),
                     'inner_scores': ';'.join(f"{t}:{inner[t]:.3f}" for t in thr_list)})

    print("-" * 88)
    ml, mb = float(np.nanmean(lodo_aucs)), float(np.nanmean(base_aucs))
    print(f"{'平均':<9}{'':>14}{ml:>12.4f}{mb:>12.4f}{ml - mb:>+10.4f}")

    print("\n" + "=" * 88)
    print("结论")
    print("=" * 88)
    from collections import Counter
    print(f"  各折选中的阈值: {dict(Counter(chosen))}")
    print(f"  LODO 平均     : {ml:.4f}")
    print(f"  基线平均      : {mb:.4f}")
    print(f"  样本外增益    : {ml - mb:+.4f}")
    if ml - mb > 0.02:
        print("  >>> 置信度过滤在样本外**仍有增益**, 是真改进, 可写进论文。")
    elif abs(ml - mb) <= 0.02:
        print("  >>> 样本外增益消失 —— 之前看到的 +0.064 **是过拟合**, 不应声称有效。")
    else:
        print("  >>> 样本外反而更差 —— 阈值选择有害。")

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {out_path}")


if __name__ == "__main__":
    main()
