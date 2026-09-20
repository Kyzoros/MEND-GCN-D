#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无监督基线在**与 GCN-D 完全相同的留出对**上的干净对级评估
=========================================================
动机: run_fair_final.py 的基线阶段把无监督方法放在**全部 306 个正例对**上评估,
而 GCN-D 的干净留出只在其**未见过的 61 个留出对**上评估。两者候选负例构造相同
(排除该病全部已知正例), 但评估对集合不同, 直接并列并不严格可比。

本脚本把 4 个无监督基线放到每个 seed 的**同一批留出对**上重算, 得到与 GCN-D
严格可比的干净对级 AUROC / MRR / Hits@K。

输出: output/results/paper_deliverables/Table_baseline_holdout.csv
用法: python run_baseline_holdout.py --seeds 0 1 2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg
from run_fair_comparison import (build_graph, score_fisher, score_proximity,
                                 score_targetcount)
from run_gnn_final import build_split
from run_fair_final import clean_holdout, score_rwr

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    print("=" * 78)
    print("无监督基线 —— 与 GCN-D 同批留出对的干净评估")
    print("=" * 78)

    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, _g = build_graph(use_logfc=True)
    print(f"[图] 节点={N}  草药={nH}  疾病={nD}")

    print("  计算 Fisher / RWR / 网络邻近 / 靶点数 ...")
    methods = {
        "Fisher_enrichment": score_fisher(herbs, disease_ids).values,
        "RWR_diffusion": score_rwr(A, N, off_h, nH, off_d, nD),
        "Network_proximity": score_proximity(herbs, disease_ids).values,
        "Target_count": score_targetcount(herbs, disease_ids).values,
    }

    rows = []
    for name, S in methods.items():
        per_seed = {}
        for sd in args.seeds:
            _tr, te = build_split(herbs, disease_ids, gold, seed=sd)
            hm = clean_holdout(S, herbs, disease_ids, gold, te)
            per_seed[sd] = hm
        aucs = [per_seed[sd]["holdout_AUROC"] for sd in args.seeds]
        mrrs = [per_seed[sd]["MRR"] for sd in args.seeds]
        h10 = [per_seed[sd]["Hits@10"] for sd in args.seeds]
        rows.append({"method": name,
                     "clean_holdout_mean": round(float(np.mean(aucs)), 4),
                     "clean_holdout_std": round(float(np.std(aucs)), 4),
                     "MRR": round(float(np.mean(mrrs)), 4),
                     "Hits@10": round(float(np.mean(h10)), 4),
                     "n_pairs": int(np.mean([per_seed[sd]["n_pairs"] for sd in args.seeds]))})
        print(f"  {name:<20} 干净留出={np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
              f"MRR={np.mean(mrrs):.4f}  Hits@10={np.mean(h10):.3f}")

    out = os.path.join(OUT_DIR, "Table_baseline_holdout.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
