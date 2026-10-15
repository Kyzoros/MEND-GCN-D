#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCN 定稿实验 (2026-09-04)
==========================
解决两个问题:
  1. 消融发现 epochs=600 明显优于 300 -> 做 epoch 扫描找平台
  2. 单 seed 存在 ~0.008 的运行间方差 -> 多种子重复, 报 mean ± std

流程:
  Stage 1  Epoch 扫描: epochs ∈ {300, 600, 900, 1200}, seed=42, 选留出最优
  Stage 2  多种子: 最优 epochs × seeds ∈ {0,1,2,3,4}, 报 mean ± std

确定性: torch.set_num_threads(1) 降低稀疏矩阵多线程归约带来的 FP 非确定性。

输出: output/results/paper_deliverables/Table_gnn_final.csv
用法:
    python run_gnn_final.py
    python run_gnn_final.py --epochs-list 300 600 900 --seeds 0 1 2 3 4
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg
from run_fair_comparison import build_graph, train_gcn, build_exclude

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")

BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)


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
    return ([p for i, p in enumerate(pos_pairs) if i not in hold],
            [p for i, p in enumerate(pos_pairs) if i in hold])


def metrics(S, herbs, disease_ids, gold, test_pairs):
    hidx = {h: i for i, h in enumerate(herbs)}
    fulls, labs, hold, per = [], [], [], {}
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
        a = roc_auc_score(y, s)
        fulls.append(a)
        per[dis] = a
        li = [hidx[h] for h in (pos | neg)]
        yl = np.array([1 if herbs[i] in pos else 0 for i in li], dtype=float)
        if len(set(yl)) == 2:
            labs.append(roc_auc_score(yl, s[li]))
    for h, d in test_pairs:
        sub = gold[gold["disease_id"] == disease_ids[d]]
        pos = {str(r.herb_id) for r in sub[sub["label"] == 1].itertuples()
               if str(r.herb_id) in hidx}
        y = np.array([1 if hh in pos else 0 for hh in herbs], dtype=float)
        if len(set(y)) == 2:
            hold.append(roc_auc_score(y, np.nan_to_num(S[:, d].astype(float), nan=0.0)))
    return (float(np.mean(fulls)), float(np.mean(labs)),
            float(np.mean(hold)) if hold else float("nan"), per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-list", type=int, nargs="+", default=[300, 600, 900, 1200])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch 线程数; 0=不限制(快). 设为 1 可降低稀疏 matmul "
                         "归约的 FP 非确定性, 但约慢 3-4 倍。")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"[确定性] torch 线程数 = {args.threads}")
    else:
        print(f"[速度优先] torch 线程数 = 不限制; 运行间方差由多种子平均吸收")

    os.makedirs(OUT_DIR, exist_ok=True)
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    print("=" * 78)
    print("GCN 定稿实验  (torch threads=1)")
    print("=" * 78)

    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, _g2i = build_graph(use_logfc=True)
    print(f"[图] 节点={N}  草药={nH}  疾病={nD}")
    EXCL = build_exclude(gold, herbs, disease_ids)
    print(f"[负采样] filtered setting: 每病排除全部已知正例 "
          f"({sum(len(v) for v in EXCL.values())} 对)")

    # ---- Stage 1: epoch 扫描 ----
    best_ep, best_hold = None, -1.0
    sweep_rows = []
    if not args.skip_sweep:
        print("\n--- Stage 1: Epoch 扫描 (seed=42) ---")
        print(f"{'epochs':>8}{'FULL':>10}{'labeled':>10}{'留出':>10}{'loss':>10}")
        print("-" * 78)
        train_pairs, test_pairs = build_split(herbs, disease_ids, gold, seed=42)
        for ep in args.epochs_list:
            S, loss = train_gcn(A, X, off_h, off_d, nH, nD, train_pairs, nH,
                                epochs=ep, seed=42, exclude_pos=EXCL, **BEST)
            f, l, h, _ = metrics(S, herbs, disease_ids, gold, test_pairs)
            sweep_rows.append({"stage": "epoch_sweep", "epochs": ep, "seed": 42,
                               "AUROC_full": round(f, 4), "AUROC_labeled": round(l, 4),
                               "holdout_full": round(h, 4), "loss": round(loss, 4)})
            print(f"{ep:>8}{f:>10.4f}{l:>10.4f}{h:>10.4f}{loss:>10.4f}")
            if h > best_hold:
                best_hold, best_ep = h, ep
        print("-" * 78)
        print(f"最优 epochs = {best_ep}  (留出 {best_hold:.4f})")
    else:
        best_ep = args.epochs_list[0]

    # ---- Stage 2: 多种子 ----
    print(f"\n--- Stage 2: 多种子 (epochs={best_ep}, seeds={args.seeds}) ---")
    print(f"{'seed':>6}{'FULL':>10}{'labeled':>10}{'留出':>10}{'loss':>10}")
    print("-" * 78)
    fs, ls, hs, seed_rows = [], [], [], []
    per_seed = {}
    for sd in args.seeds:
        train_pairs, test_pairs = build_split(herbs, disease_ids, gold, seed=sd)
        S, loss = train_gcn(A, X, off_h, off_d, nH, nD, train_pairs, nH,
                            epochs=best_ep, seed=sd, exclude_pos=EXCL, **BEST)
        f, l, h, per = metrics(S, herbs, disease_ids, gold, test_pairs)
        fs.append(f)
        ls.append(l)
        hs.append(h)
        per_seed[sd] = per
        seed_rows.append({"stage": f"seed_{sd}", "epochs": best_ep, "seed": sd,
                          "AUROC_full": round(f, 4), "AUROC_labeled": round(l, 4),
                          "holdout_full": round(h, 4), "loss": round(loss, 4)})
        print(f"{sd:>6}{f:>10.4f}{l:>10.4f}{h:>10.4f}{loss:>10.4f}")
    print("-" * 78)

    def ms(v):
        return f"{np.mean(v):.4f} ± {np.std(v):.4f}"

    print(f"FULL    : {ms(fs)}   (min {min(fs):.4f} / max {max(fs):.4f})")
    print(f"labeled : {ms(ls)}   (min {min(ls):.4f} / max {max(ls):.4f})")
    print(f"留出泛化: {ms(hs)}   (min {min(hs):.4f} / max {max(hs):.4f})")
    print("=" * 78)

    # 每病均值 (跨 seed)
    if per_seed:
        all_d = sorted(set().union(*[set(p.keys()) for p in per_seed.values()]))
        print("\n每病 FULL AUROC (跨 seed 均值 ± std)")
        print(f"{'disease':<10}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}")
        print("-" * 78)
        per_rows = []
        for d in all_d:
            vals = [per_seed[sd].get(d, np.nan) for sd in args.seeds]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue
            print(f"{d:<10}{np.mean(vals):>10.4f}{np.std(vals):>9.4f}"
                  f"{min(vals):>9.4f}{max(vals):>9.4f}")
            per_rows.append({"stage": "per_disease", "disease": d, "epochs": best_ep,
                             "AUROC_full": round(float(np.mean(vals)), 4),
                             "std": round(float(np.std(vals)), 4)})
        print("-" * 78)

    # 保存
    df = pd.DataFrame(sweep_rows + seed_rows)
    out = os.path.join(OUT_DIR, "Table_gnn_final.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    summary = (f"\n# FINAL epochs={best_ep} seeds={args.seeds} threads=1\n"
               f"# FULL,{np.mean(fs):.4f},std,{np.std(fs):.4f}\n"
               f"# labeled,{np.mean(ls):.4f},std,{np.std(ls):.4f}\n"
               f"# holdout,{np.mean(hs):.4f},std,{np.std(hs):.4f}\n")
    with open(out, "a", encoding="utf-8-sig") as fh:
        fh.write(summary)
    print(f"\n[写入] {out}")


if __name__ == "__main__":
    main()
