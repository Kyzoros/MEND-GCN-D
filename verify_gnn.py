#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCN-D 论文可复现性校验 v5 (2026-09-04)
=======================================
用与论文相同的代码重算主表数字, 与论文声明值比对, 打 PASS / FAIL。

设计原则
--------------------------------------------------------------------------
v4 版本试图重算 GCN-D 的"留出 0.8032", 但那个数字**已被本稿 §3.4 撤回**
(它把训练正例留在评估正例集里)。v5 因此重新划分校验对象:

  A 组 (确定性, 严格复现, 容差 ±0.005)
     4 个无监督基线的 每病 FULL / labeled 均值, 以及 306 对干净对级 AUROC。
     这些量不含任何随机性, 必须逐位复现。

  B 组 (随机, 冒烟级, 只设下界)
     GCN-D 与 MLP 各跑 1 折 CV × 300 epoch, 只要求
       - GCN-D 明显高于最强无监督基线 (0.5387)
       - GCN-D 明显高于 MLP 无图传播
     精确的论文数字由 run_final_tables.py 与 run_cv_final.py 全量重跑得到
     (GCN-D 约 18 分钟, 5 折 CV 约 55 分钟), 不放在快速校验里。

  C 组 (混杂度检查)
     D-TMS 分数与靶点数的 Spearman ρ 应落在论文声称的 0.65–0.82 区间。

用法:
    python verify_gnn.py                 # 全量 (约 5 分钟)
    python verify_gnn.py --quick         # 只跑 A 组确定性基线
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg
from run_fair_comparison import (build_graph, build_exclude, train_gcn,
                                 score_fisher, score_proximity,
                                 score_targetcount)
from run_fair_final import score_rwr
from run_cv_final import make_folds, pair_metrics

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")

# ---- 论文声明值 (v5) ----
# A 组: 确定性基线 (Table1_main_results.csv / Table_cv_clean.csv)
EXPECTED_FULL = {
    "D-TMS_Fisher":      0.5080,
    "RWR_diffusion":     0.5605,
    "Network_proximity": 0.5103,
    "Target_count":      0.5328,
}
EXPECTED_LABELED = {
    "D-TMS_Fisher":      0.4807,
    "RWR_diffusion":     0.5502,
    "Network_proximity": 0.4890,
    "Target_count":      0.5367,
}
EXPECTED_PAIR = {          # 306 对干净对级 AUROC
    "D-TMS_Fisher":      0.5080,
    "RWR_diffusion":     0.5387,
    "Network_proximity": 0.5033,
    "Target_count":      0.5203,
}
TOL_DETERMINISTIC = 0.005

# C 组: 靶点数混杂 —— §3.7 结论 (旧声称 0.65-0.82 已被证伪并更正)
#   直接富集 D-TMS: ρ≈0.07 (低); RWR 扩散: ρ≈0.79 (高); GCN-D: ρ≈0.17 (低)
EXPECTED_RHO = {
    "D-TMS_Fisher":      (0.0741, 0.05),
    "RWR_diffusion":     (0.7930, 0.05),
    "Network_proximity": (0.2336, 0.05),
}
TOL_DETERMINISTIC = 0.005


def per_disease_mean(S, herbs, disease_ids, gold, key):
    hidx = {h: i for i, h in enumerate(herbs)}
    vals = []
    for di, dis in enumerate(disease_ids):
        sub = gold[gold["disease_id"] == dis]
        pos = {str(r.herb_id) for r in sub[sub["label"] == 1].itertuples()
               if str(r.herb_id) in hidx}
        neg = {str(r.herb_id) for r in sub[sub["label"] == 0].itertuples()
               if str(r.herb_id) in hidx}
        if len(pos) < 3:
            continue
        s = np.nan_to_num(S[:, di].astype(float), nan=0.0)
        if key == "full":
            y = np.array([1.0 if h in pos else 0.0 for h in herbs])
        else:
            li = [hidx[h] for h in (pos | neg)]
            y = np.array([1.0 if herbs[i] in pos else 0.0 for i in li])
            s = s[li]
        if len(set(y)) == 2:
            vals.append(roc_auc_score(y, s))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke-epochs", type=int, default=300)
    args = ap.parse_args()

    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    print("=" * 78)
    print("verify_gnn.py v5 — GCN-D 论文数字复现校验")
    print("=" * 78)

    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, _g = build_graph(use_logfc=True)
    print(f"[图] 节点={N}  草药={nH}  疾病={nD}")

    scores = {}
    print("\n[A/3] 确定性无监督基线 ...")
    scores["D-TMS_Fisher"] = score_fisher(herbs, disease_ids).values
    scores["RWR_diffusion"] = score_rwr(A, N, off_h, nH, off_d, nD)
    scores["Network_proximity"] = score_proximity(herbs, disease_ids).values
    scores["Target_count"] = score_targetcount(herbs, disease_ids).values
    print("  完成")

    rows, fails = [], 0
    all_pairs, folds = make_folds(herbs, disease_ids, gold)

    for name, S in scores.items():
        got_f = per_disease_mean(S, herbs, disease_ids, gold, "full")
        got_l = per_disease_mean(S, herbs, disease_ids, gold, "labeled")
        df = pair_metrics(S, herbs, disease_ids, gold, all_pairs)
        got_p = float(df["auc"].mean())
        for key, got, exp in [("FULL", got_f, EXPECTED_FULL[name]),
                              ("labeled", got_l, EXPECTED_LABELED[name]),
                              ("306对干净", got_p, EXPECTED_PAIR[name])]:
            ok = abs(got - exp) <= TOL_DETERMINISTIC
            fails += 0 if ok else 1
            rows.append((f"{name}_{key}", exp, got, abs(got - exp),
                         "PASS" if ok else "FAIL"))

    # ---------------- C 组: 靶点数混杂 ----------------
    print("\n[C/3] 靶点数混杂度 ...")
    tc = score_targetcount(herbs, disease_ids).values[:, 0]
    for name in ["D-TMS_Fisher", "RWR_diffusion", "Network_proximity"]:
        rhos = []
        for di in range(len(disease_ids)):
            s = np.nan_to_num(scores[name][:, di].astype(float), nan=0.0)
            if np.std(s) > 0:
                rhos.append(spearmanr(s, tc).statistic)
        rho = float(np.nanmean(rhos))
        exp, tol = EXPECTED_RHO[name]
        ok = abs(rho - exp) <= tol
        fails += 0 if ok else 1
        rows.append((f"{name}_rho_vs_targetcount", exp, rho, abs(rho - exp),
                     "PASS" if ok else "FAIL"))

    # ---------------- B 组: GCN-D 冒烟 ----------------
    if not args.quick:
        print(f"\n[B/3] GCN-D / MLP 冒烟测试 (1 折 × {args.smoke_epochs} ep) ...")
        EXCL = build_exclude(gold, herbs, disease_ids)
        test, train = folds[0], [p for f in folds[1:] for p in f]
        res = {}
        for tag, kw in [("GCN_DistMult", {}), ("MLP_no_graph", {"n_layers": 1})]:
            S, _ = train_gcn(A, X, off_h, off_d, nH, nD, train, nH,
                             epochs=args.smoke_epochs, seed=42,
                             exclude_pos=EXCL, **kw)
            res[tag] = float(pair_metrics(S, herbs, disease_ids, gold,
                                          test)["auc"].mean())
            print(f"  {tag:<16} pair AUROC={res[tag]:.4f}  (n_test={len(test)})")
        best_unsup = max(EXPECTED_PAIR.values())
        ok1 = res["GCN_DistMult"] > best_unsup
        ok2 = res["GCN_DistMult"] > res["MLP_no_graph"]
        fails += 0 if ok1 else 1
        fails += 0 if ok2 else 1
        rows.append(("GCN_D > best unsupervised", f"> {best_unsup:.4f}",
                     res["GCN_DistMult"], 0.0, "PASS" if ok1 else "FAIL"))
        rows.append(("GCN_D > MLP(no propagation)", "> MLP",
                     res["GCN_DistMult"] - res["MLP_no_graph"], 0.0,
                     "PASS" if ok2 else "FAIL"))
        print("  (精确数字由 run_final_tables.py / run_cv_final.py 全量重跑得到)")
    else:
        print("\n[B/3] 跳过 (--quick)")

    # ---------------- 报告 ----------------
    print("\n" + "=" * 78)
    print("复现校验报告")
    print("=" * 78)
    print(f"{'指标':<34}{'论文已知':>12}{'重算值':>12}{'偏差':>10}{'判定':>8}")
    print("-" * 78)
    for name, exp, got, diff, verdict in rows:
        es = f"{exp}" if isinstance(exp, str) else f"{exp:.4f}"
        print(f"{name:<34}{es:>12}{got:>12.4f}{diff:>10.4f}{verdict:>8}")
    print("-" * 78)
    if fails == 0:
        print("[OK] 全部 PASS")
    else:
        print(f"[WARN] {fails} 项 FAIL — 检查输入数据 / config / 脚本是否漂移")
    print("=" * 78)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
