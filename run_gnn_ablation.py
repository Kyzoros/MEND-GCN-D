#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCN 消融实验 (2026-09-04)
==========================
在完全相同的训练/留出划分 (seed=42) 下, 逐项改变一个超参, 观察:
  - 8 病平均 FULL AUROC
  - 8 病平均 labeled AUROC
  - 留出泛化 FULL AUROC (1/5 正例对, 训练时从未见过)

消融维度:
  A. 层数 n_layers      : 2 (默认) / 3
  B. 隐层维度 hid       : 32 / 64 (默认) / 128
  C. 负采样比例 k_neg   : 1 / 5 (默认) / 10
  D. 基因 logFC 特征    : 有 (默认) / 无
  E. 训练轮数 epochs    : 100 / 300 (默认) / 600

输出: output/results/paper_deliverables/Table_gnn_ablation.csv
用法:
    python run_gnn_ablation.py
    python run_gnn_ablation.py --quick      # 只跑关键 4 组
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg
from run_fair_comparison import build_graph, train_gcn, build_exclude

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")

# 默认配置 (与 run_fair_comparison.py 一致)
DEFAULT = dict(epochs=300, lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2, seed=42)


def build_split(herbs, disease_ids, gold, seed=42):
    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}
    pos_pairs = sorted({
        (hidx[str(r.herb_id)], didx[str(r.disease_id)])
        for r in gold[gold["label"] == 1].itertuples()
        if str(r.herb_id) in hidx and str(r.disease_id) in didx})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pos_pairs))
    n_hold = len(pos_pairs) // 5
    hold = set(perm[:n_hold].tolist())
    train = [p for i, p in enumerate(pos_pairs) if i not in hold]
    test = [p for i, p in enumerate(pos_pairs) if i in hold]
    return pos_pairs, train, test


def metrics(S, herbs, disease_ids, gold, test_pairs):
    hidx = {h: i for i, h in enumerate(herbs)}
    fulls, labs = [], []
    for di, dis in enumerate(disease_ids):
        sub = gold[gold["disease_id"] == dis]
        pos = {str(r.herb_id) for r in sub[sub["label"] == 1].itertuples()
               if str(r.herb_id) in hidx}
        neg = {str(r.herb_id) for r in sub[sub["label"] == 0].itertuples()
               if str(r.herb_id) in hidx}
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in herbs], dtype=float)
        s = np.nan_to_num(S[:, di].astype(float), nan=0.0)
        fulls.append(roc_auc_score(y, s))
        li = [hidx[h] for h in (pos | neg)]
        yl = np.array([1 if herbs[i] in pos else 0 for i in li], dtype=float)
        if len(set(yl)) == 2:
            labs.append(roc_auc_score(yl, s[li]))
    hold = []
    for h, d in test_pairs:
        sub = gold[gold["disease_id"] == disease_ids[d]]
        pos = {str(r.herb_id) for r in sub[sub["label"] == 1].itertuples()
               if str(r.herb_id) in hidx}
        y = np.array([1 if hh in pos else 0 for hh in herbs], dtype=float)
        if len(set(y)) == 2:
            hold.append(roc_auc_score(y, np.nan_to_num(S[:, d].astype(float), nan=0.0)))
    return (float(np.mean(fulls)), float(np.mean(labs)),
            float(np.mean(hold)) if hold else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑关键 4 组")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖默认 epochs")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    print("=" * 78)
    print("GCN 消融实验")
    print("=" * 78)

    # 两张图: 带 / 不带 logFC 特征 (拓扑相同, 仅 X 不同)
    A, X_log, N, off_h, off_d, nH, nD, herbs, disease_ids, _g2i = build_graph(use_logfc=True)
    _, X_nolog, *_ = build_graph(use_logfc=False)
    print(f"[图] 节点={N}  草药={nH}  疾病={nD}")
    EXCL = build_exclude(gold, herbs, disease_ids)
    print(f"[负采样] filtered setting: 排除全部已知正例 ({sum(len(v) for v in EXCL.values())} 对)")

    pos_pairs, train_pairs, test_pairs = build_split(herbs, disease_ids, gold)
    print(f"[划分] 正例对={len(pos_pairs)}  训练={len(train_pairs)}  留出={len(test_pairs)}\n")

    def cfg_of(**kw):
        c = dict(DEFAULT)
        c.update(kw)
        return c

    # 注: epoch 效应单独由 run_gnn_final.py 的 sweep (§3.5) 覆盖, 这里不再设 E 维;
    # 所有变体在相同 epoch 预算 (--epochs, 默认 1500) 下比较架构差异。
    if args.quick:
        variants = [
            ("默认 (2层/hid64/k5/logFC)", cfg_of()),
            ("A 层数=3", cfg_of(n_layers=3)),
            ("D 无 logFC 特征", cfg_of()),
            ("C k_neg=1", cfg_of(k_neg=1)),
        ]
        noflag_idx = {2}
    else:
        variants = [
            ("默认 (2层/hid64/k5/logFC)", cfg_of()),
            ("A 层数=3", cfg_of(n_layers=3)),
            ("B1 隐层=32", cfg_of(hid=32)),
            ("B2 隐层=128", cfg_of(hid=128)),
            ("C1 k_neg=1", cfg_of(k_neg=1)),
            ("C2 k_neg=10", cfg_of(k_neg=10)),
            ("D 无 logFC 特征", cfg_of()),
        ]
        noflag_idx = {6}

    if args.epochs:
        for i, (nm, c) in enumerate(variants):
            c["epochs"] = args.epochs
            variants[i] = (nm, c)

    rows = []
    for vi, (name, c) in enumerate(variants):
        X = X_nolog if vi in noflag_idx else X_log
        torch.manual_seed(c["seed"])
        np.random.seed(c["seed"])
        S, loss = train_gcn(A, X, off_h, off_d, nH, nD, train_pairs, nH,
                            epochs=c["epochs"], lr=c["lr"], hid=c["hid"],
                            out=c["out"], k_neg=c["k_neg"],
                            n_layers=c["n_layers"], seed=c["seed"],
                            exclude_pos=EXCL)
        f, l, h = metrics(S, herbs, disease_ids, gold, test_pairs)
        rows.append({"variant": name, "layers": c["n_layers"], "hid": c["hid"],
                     "k_neg": c["k_neg"], "logfc": vi not in noflag_idx,
                     "epochs": c["epochs"], "final_loss": round(loss, 4),
                     "AUROC_full": round(f, 4), "AUROC_labeled": round(l, 4),
                     "holdout_full": round(h, 4)})
        print(f"  {name:<34} FULL={f:.4f}  labeled={l:.4f}  "
              f"留出={h:.4f}  loss={loss:.4f}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("消融汇总")
    print("=" * 78)
    print(f"{'variant':<34}{'FULL':>9}{'labeled':>10}{'留出':>9}")
    print("-" * 78)
    for _, r in df.iterrows():
        print(f"{r['variant']:<34}{r['AUROC_full']:>9.4f}"
              f"{r['AUROC_labeled']:>10.4f}{r['holdout_full']:>9.4f}")
    print("-" * 78)
    base = df.iloc[0]
    print(f"默认配置: FULL={base['AUROC_full']:.4f}  "
          f"labeled={base['AUROC_labeled']:.4f}  留出={base['holdout_full']:.4f}")

    out = os.path.join(OUT_DIR, "Table_gnn_ablation.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {out}")


if __name__ == "__main__":
    main()
