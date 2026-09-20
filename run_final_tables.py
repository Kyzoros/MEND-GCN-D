#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文主表的**唯一权威生成脚本** (2026-09-04, v5)
=================================================
此前 Table 1 的数字来自 3 个不同脚本在不同时间跑出来的产物手工拼接
(Table_fair_comparison.csv 的基线 + Table_gnn_final.csv 的 GCN-D +
Table_baseline_holdout.csv 的基线留出), 基线用 300 epoch 而 GCN-D 用 3000 epoch,
本身就不公平, 且跨进程拼接难以复现。本脚本一次性端到端生成全部主表数字。

同一份数据 / 同一套评估代码 / 同一批留出对, 比较 6 个方法:

  无监督 (全程不用标签)
    1. D-TMS / Fisher_enrichment   单跳集合富集
    2. RWR_diffusion               重启随机游走 (同一张异质图, 疾病节点为种子)
    3. Network_proximity           度匹配距离 z-score
    4. Target_count                平凡对照 |targets(h)|, 与疾病无关
  有监督 (相同划分 / 相同 filtered 负采样 / **相同 epoch 预算**)
    5. MLP_no_graph                n_layers=1, 去掉图传播 —— 隔离"图"的贡献
    6. GCN_DistMult                本文方法

三个协议都报:
  - FULL     正例 = 全部已知治疗草药, 负例 = 全库其余 (直推式, 含训练对)
  - labeled  仅在金标准显式 label∈{0,1} 的条目上
  - clean    对级干净留出: 对每个留出正例 (h,d), 把 h 与该病全部已知正例之外的
             草药排序比较 —— 评估正例集不含任何训练对, 才是真正的泛化指标

产出:
  Table1_main_results.csv  6 方法 × 8 病 FULL/labeled
  Table2_clean_holdout.csv 6 方法 干净留出 AUROC/MRR/Hits@K (3 种子 mean±std)
  Table3_per_disease.csv   各方法每病 FULL
  FINAL_NUMBERS.txt        论文字速查 (所有主数字集中一处)

用法:
    python run_final_tables.py                       # 默认 3000ep × seeds 0 1 2
    python run_final_tables.py --epochs 300 --seeds 0  # 冒烟测试
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg
from run_fair_comparison import (build_graph, build_exclude, train_gcn,
                                 score_fisher, score_proximity,
                                 score_targetcount)
from run_gnn_final import build_split
from run_fair_final import clean_holdout, score_rwr

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")

BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)
UNSUP = ["D-TMS_Fisher", "RWR_diffusion", "Network_proximity", "Target_count"]


def ms(v):
    return float(np.mean(v)), float(np.std(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    print("=" * 78)
    print("论文主表端到端生成 (v5)")
    print("=" * 78)
    print(f"金标准 {os.path.basename(GOLD_PATH)}: {len(gold)} 行, "
          f"正例 {(gold['label'] == 1).sum()}")
    print(f"epochs={args.epochs}  seeds={args.seeds}")

    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, _g = build_graph(use_logfc=True)
    print(f"[图] 节点={N}  草药={nH}  疾病={nD}")
    EXCL = build_exclude(gold, herbs, disease_ids)

    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}

    def per_disease(S):
        rows = []
        for di, dis in enumerate(disease_ids):
            sub = gold[gold["disease_id"] == dis]
            pos = {str(r.herb_id) for r in sub[sub["label"] == 1].itertuples()
                   if str(r.herb_id) in hidx}
            neg = {str(r.herb_id) for r in sub[sub["label"] == 0].itertuples()
                   if str(r.herb_id) in hidx}
            if len(pos) < 3:
                continue
            s = np.nan_to_num(S[:, di].astype(float), nan=0.0)
            y = np.array([1.0 if h in pos else 0.0 for h in herbs])
            a_full = roc_auc_score(y, s) if len(set(y)) == 2 else np.nan
            li = [hidx[h] for h in (pos | neg)]
            yl = np.array([1.0 if herbs[i] in pos else 0.0 for i in li])
            a_lab = roc_auc_score(yl, s[li]) if len(set(yl)) == 2 else np.nan
            rows.append({"disease": dis, "n_pos": len(pos), "n_neg": len(neg),
                         "AUROC_full": a_full, "AUROC_labeled": a_lab})
        return rows

    t1_rows, t2_rows, t3_rows = [], [], []

    # ================================================================= 无监督
    print("\n" + "=" * 78)
    print("[1/3] 无监督基线")
    print("=" * 78)
    print("  Fisher 富集 ...")
    S_fisher = score_fisher(herbs, disease_ids).values
    print("  RWR 扩散 ...")
    S_rwr = score_rwr(A, N, off_h, nH, off_d, nD)
    print("  网络邻近度 ...")
    S_prox = score_proximity(herbs, disease_ids).values
    print("  靶点数量 ...")
    S_tc = score_targetcount(herbs, disease_ids).values

    for name, S in zip(UNSUP, [S_fisher, S_rwr, S_prox, S_tc]):
        rows = per_disease(S)
        for r in rows:
            t1_rows.append({"method": name, "class": "unsupervised", **r})
            t3_rows.append({"method": name, "disease": r["disease"],
                            "AUROC_full": r["AUROC_full"]})
        hm_all = [clean_holdout(S, herbs, disease_ids, gold,
                                build_split(herbs, disease_ids, gold, seed=sd)[1])
                  for sd in args.seeds]
        auc, auc_s = ms([h["holdout_AUROC"] for h in hm_all])
        mrr, _ = ms([h["MRR"] for h in hm_all])
        h10, _ = ms([h["Hits@10"] for h in hm_all])
        h20, _ = ms([h["Hits@20"] for h in hm_all])
        t2_rows.append({"method": name, "class": "unsupervised",
                        "clean_holdout_mean": round(auc, 4),
                        "clean_holdout_std": round(auc_s, 4),
                        "MRR": round(mrr, 4), "Hits@10": round(h10, 4),
                        "Hits@20": round(h20, 4),
                        "n_pairs": int(hm_all[0]["n_pairs"])})
        print(f"  {name:<20} FULL={np.mean([r['AUROC_full'] for r in rows]):.4f}  "
              f"干净留出={auc:.4f} ± {auc_s:.4f}")

    # ================================================================= 有监督
    print("\n" + "=" * 78)
    print(f"[2/3] 有监督 (epochs={args.epochs})")
    print("=" * 78)
    sup_variants = [("GCN_DistMult", BEST)]
    if not args.skip_mlp:
        mlp = dict(BEST)
        mlp["n_layers"] = 1
        sup_variants.append(("MLP_no_graph", mlp))

    for name, cfgd in sup_variants:
        per_seed = {}
        n_pairs = 0
        f_all, l_all, h_all, mrr_all, h10_all, h20_all = [], [], [], [], [], []
        for sd in args.seeds:
            tr, te = build_split(herbs, disease_ids, gold, seed=sd)
            S, loss = train_gcn(A, X, off_h, off_d, nH, nD, tr, nH,
                                epochs=args.epochs, seed=sd,
                                exclude_pos=EXCL, **cfgd)
            rows = per_disease(S)
            hm = clean_holdout(S, herbs, disease_ids, gold, te)
            per_seed[sd] = {r["disease"]: r["AUROC_full"] for r in rows}
            f_all.append(np.nanmean([r["AUROC_full"] for r in rows]))
            l_all.append(np.nanmean([r["AUROC_labeled"] for r in rows]))
            h_all.append(hm["holdout_AUROC"])
            mrr_all.append(hm["MRR"])
            h10_all.append(hm["Hits@10"])
            h20_all.append(hm["Hits@20"])
            n_pairs = int(hm["n_pairs"])
            print(f"  {name:<20} seed={sd}  FULL={f_all[-1]:.4f}  "
                  f"labeled={l_all[-1]:.4f}  干净留出={h_all[-1]:.4f}  loss={loss:.4f}")
            if sd == args.seeds[0]:
                for r in rows:
                    t1_rows.append({"method": name, "class": "supervised",
                                    "note": f"seed{sd}", **r})
        fm, fs = ms(f_all)
        lm, ls = ms(l_all)
        hm_, hs_ = ms(h_all)
        t2_rows.append({"method": name, "class": "supervised",
                        "clean_holdout_mean": round(hm_, 4),
                        "clean_holdout_std": round(hs_, 4),
                        "MRR": round(ms(mrr_all)[0], 4),
                        "Hits@10": round(ms(h10_all)[0], 4),
                        "Hits@20": round(ms(h20_all)[0], 4),
                        "n_pairs": n_pairs})
        all_d = sorted(set().union(*[set(p.keys()) for p in per_seed.values()]))
        for d in all_d:
            vals = [per_seed[sd].get(d, np.nan) for sd in args.seeds]
            vals = [v for v in vals if not np.isnan(v)]
            t3_rows.append({"method": name, "disease": d,
                            "AUROC_full": round(float(np.mean(vals)), 4),
                            "std": round(float(np.std(vals)), 4)})
        print(f"  --> {name}: FULL={fm:.4f}±{fs:.4f}  labeled={lm:.4f}±{ls:.4f}  "
              f"干净留出={hm_:.4f}±{hs_:.4f}")

    # ================================================================= 写出
    print("\n" + "=" * 78)
    print("[3/3] 写出主表")
    print("=" * 78)
    df1 = pd.DataFrame(t1_rows)
    df2 = pd.DataFrame(t2_rows)
    df3 = pd.DataFrame(t3_rows)

    lines = [f"# run_final_tables.py  epochs={args.epochs} seeds={args.seeds}",
             f"# 金标准 {os.path.basename(GOLD_PATH)}  正例 "
             f"{(gold['label'] == 1).sum()} 草药 {nH} 疾病 {nD}", ""]
    lines.append("== 干净留出 (对级, 未见过的正例对) ==")
    for _, r in df2.iterrows():
        lines.append(f"{r['method']:<20} {r['clean_holdout_mean']:.4f} "
                     f"± {r['clean_holdout_std']:.4f}   MRR={r['MRR']:.4f}  "
                     f"Hits@10={r['Hits@10']:.4f}  Hits@20={r['Hits@20']:.4f}")
    lines.append("")
    lines.append("== 每病 FULL ==")
    for _, r in df1.iterrows():
        if pd.isna(r.get("AUROC_full")):
            continue
        lines.append(f"{r['method']:<20} {r['disease']:<8} {r['AUROC_full']:.4f}")

    for nm, df in [("Table1_main_results.csv", df1),
                   ("Table2_clean_holdout.csv", df2),
                   ("Table3_per_disease.csv", df3)]:
        p = os.path.join(OUT_DIR, nm)
        df.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"[写入] {p}")
    p = os.path.join(OUT_DIR, "FINAL_NUMBERS.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[写入] {p}")
    print(f"\n[完成] 总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
