#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5 折交叉验证的干净对级评估 (2026-09-04, v5)
============================================
为什么需要它
--------------------------------------------------------------------------
§3.4 的"干净留出"只用 1/5 正例对 (61 对) 做测试, 存在两个问题:
  1. **估计不稳** —— 换一个随机种子同一配置能差 0.04 (seed0 在两次运行中分别得到
     0.6557 与 0.6132), 因为 61 对的 AUROC 本身噪声大;
  2. **覆盖率低** —— 80% 的正例对从未被当作测试对, 结论建立在 1/5 的样本上。

本脚本做 **5 折 CV**: 306 个正例对划分成 5 个互不相交的折, 每折轮流作测试,
其余 4 折训练。于是**每一对正例恰好被测试一次**, 最终在全部 306 对上汇总,
并用 bootstrap 给 95% CI。无监督方法不需要训练, 直接在全部 306 对上评估即可,
与有监督方法口径完全一致 (候选负例 = {全部草药} \\ {该病全部已知正例})。

产出: output/results/paper_deliverables/Table_cv_clean.csv
用法:
    python run_cv_final.py --epochs 3000
    python run_cv_final.py --epochs 300 --methods GCN_DistMult   # 冒烟
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg
from run_fair_comparison import (build_graph, build_exclude, train_gcn,
                                 score_fisher, score_proximity,
                                 score_targetcount)
from run_fair_final import score_rwr

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")

BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)
N_FOLDS = 5
FOLD_SEED = 12345


def make_folds(herbs, disease_ids, gold, n_folds=N_FOLDS, seed=FOLD_SEED):
    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}
    pairs = sorted({
        (hidx[str(r.herb_id)], didx[str(r.disease_id)])
        for r in gold[gold["label"] == 1].itertuples()
        if str(r.herb_id) in hidx and str(r.disease_id) in didx})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    folds = []
    for k in range(n_folds):
        folds.append([pairs[i] for i in perm[k::n_folds]])
    return pairs, folds


def pair_metrics(S, herbs, disease_ids, gold, test_pairs, topk=(10, 20)):
    """对级干净评估: 每个测试正例 (h,d) 与该病全部已知正例之外的草药比较。
    同时返回全局 AUPRC（把所有测试对的正例与负例分数合并计算，PU 设定下信息量高于 AUROC）。"""
    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}
    pos_by_d = {d: set() for d in range(len(disease_ids))}
    for r in gold[gold["label"] == 1].itertuples():
        h, d = str(r.herb_id), str(r.disease_id)
        if h in hidx and d in didx:
            pos_by_d[didx[d]].add(hidx[h])

    recs = []
    all_labels, all_scores = [], []
    for h, d in test_pairs:
        negs = np.array([i for i in range(len(herbs)) if i not in pos_by_d[d]],
                        dtype=np.int64)
        if len(negs) == 0:
            continue
        s_neg = np.nan_to_num(S[negs, d].astype(float), nan=-1e9)
        s_pos = float(np.nan_to_num(S[h, d].astype(float), nan=-1e9))
        auc = float((s_neg < s_pos).mean() + 0.5 * (s_neg == s_pos).mean())
        rank = int((s_neg > s_pos).sum()) + 1
        recs.append({"disease": disease_ids[d], "herb": herbs[h], "auc": auc,
                     "rank": rank, **{f"hit{k}": 1.0 if rank <= k else 0.0
                                      for k in topk}})
        all_labels.extend([1] + [0] * len(negs))
        all_scores.extend([s_pos] + s_neg.tolist())
    auprc = (average_precision_score(all_labels, all_scores)
             if all_labels else float("nan"))
    return pd.DataFrame(recs), float(auprc)


def summarise(df, auprc, name, n_boot=2000, seed=7):
    aucs = df["auc"].values
    rng = np.random.default_rng(seed)
    boots = np.array([aucs[rng.integers(0, len(aucs), len(aucs))].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = {"method": name, "n_pairs": int(len(aucs)),
           "pair_AUROC": round(float(aucs.mean()), 4),
           "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4),
           "AUPRC": round(auprc, 4),
           "MRR": round(float((1.0 / df["rank"]).mean()), 4),
           "Hits@10": round(float(df["hit10"].mean()), 4),
           "Hits@20": round(float(df["hit20"].mean()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--methods", nargs="+",
                    default=["GCN_DistMult", "MLP_no_graph"])
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    print("=" * 78)
    print(f"5 折交叉验证 · 干净对级评估  (epochs={args.epochs})")
    print("=" * 78)

    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, _g = build_graph(use_logfc=True)
    print(f"[图] 节点={N}  草药={nH}  疾病={nD}")
    EXCL = build_exclude(gold, herbs, disease_ids)
    all_pairs, folds = make_folds(herbs, disease_ids, gold)
    print(f"[划分] 正例对={len(all_pairs)}  折数={N_FOLDS}  每折≈{len(folds[0])} 对")
    print("[负采样] filtered: 排除该病全部已知正例 (含测试折)\n")

    rows, per_pair = [], []

    # ---------------- 无监督基线: 无需训练, 直接在全部 306 对上评估 ----------
    print("[1/2] 无监督基线 (全 306 对)")
    for name, S in [("D-TMS_Fisher", score_fisher(herbs, disease_ids).values),
                    ("RWR_diffusion", score_rwr(A, N, off_h, nH, off_d, nD)),
                    ("Network_proximity", score_proximity(herbs, disease_ids).values),
                    ("Target_count", score_targetcount(herbs, disease_ids).values)]:
        df, auprc = pair_metrics(S, herbs, disease_ids, gold, all_pairs)
        r = summarise(df, auprc, name)
        rows.append(r)
        print(f"  {name:<20} pair AUROC={r['pair_AUROC']:.4f} "
              f"[{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}]  AUPRC={r['AUPRC']:.4f}  "
              f"MRR={r['MRR']:.4f}  Hits@10={r['Hits@10']:.3f}")

    # ---------------- 有监督: 5 折 CV ------------------------------------
    print("\n[2/2] 有监督 5 折 CV")
    for mname in args.methods:
        c = dict(BEST)
        if mname == "MLP_no_graph":
            c["n_layers"] = 1
        dfs = []
        for k in range(N_FOLDS):
            test = folds[k]
            train = [p for j, f in enumerate(folds) if j != k for p in f]
            S, loss = train_gcn(A, X, off_h, off_d, nH, nD, train, nH,
                                epochs=args.epochs, seed=42,
                                exclude_pos=EXCL, **c)
            dfk, auprc_k = pair_metrics(S, herbs, disease_ids, gold, test)
            dfk["fold"] = k
            dfs.append((dfk, auprc_k))
            print(f"  {mname:<20} fold {k + 1}/{N_FOLDS}  "
                  f"n_test={len(dfk)}  fold_AUROC={dfk['auc'].mean():.4f}  "
                  f"fold_AUPRC={auprc_k:.4f}  loss={loss:.4f}")
        df = pd.concat([x for x, _ in dfs], ignore_index=True)
        macro_auprc = float(np.mean([a for _, a in dfs]))
        r = summarise(df, macro_auprc, mname)
        r["n_folds"] = N_FOLDS
        rows.append(r)
        df["method"] = mname
        per_pair.append(df)
        print(f"  --> {mname}: pair AUROC={r['pair_AUROC']:.4f} "
              f"[{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}]  AUPRC={r['AUPRC']:.4f}  "
              f"MRR={r['MRR']:.4f}  Hits@10={r['Hits@10']:.3f}\n")

    out = os.path.join(OUT_DIR, "Table_cv_clean.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[写入] {out}")
    if per_pair:
        p = os.path.join(OUT_DIR, "Table_cv_per_pair.csv")
        pd.concat(per_pair, ignore_index=True).to_csv(p, index=False,
                                                      encoding="utf-8-sig")
        print(f"[写入] {p}")
    print(f"\n[完成] 总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
