#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 GCN-D 的每病 Top 预测 + 金标准命中排名 (供 8 病 case study 引用)
========================================================================
在**全图**上训练一个 GCN-D (与论文主表同配置), 取其 herb×disease 分数 S,
对每病取 Top-20 草药并标注是否为金标准已知有效药, 同时给出每病 FULL AUROC
与已知药的中位排名/富集倍数。输出:
  output/results/paper_deliverables/gcn_per_disease_top20.csv
  output/results/paper_deliverables/gcn_gold_ranks_8d.csv
用法:
  python run_gcn_case_predict.py --epochs 3000
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
from run_fair_comparison import (build_graph, build_exclude, train_gcn,
                                 score_fisher)
from run_fair_final import score_rwr

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")
BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)
    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, _ = build_graph(use_logfc=True)
    hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
    name_col = "herb_name" if "herb_name" in hi.columns else hi.columns[-1]
    name_map = dict(zip(hi["herb_id"].astype(str), hi[name_col].astype(str)))

    EXCL = build_exclude(gold, herbs, disease_ids)
    S, loss = train_gcn(A, X, off_h, off_d, nH, nD,
                        [(int(herbs.index(str(r.herb_id))), disease_ids.index(str(r.disease_id)))
                         for r in gold[gold["label"] == 1].itertuples()
                         if str(r.herb_id) in set(herbs)
                         and str(r.disease_id) in set(disease_ids)],
                        nH, epochs=args.epochs, seed=42, exclude_pos=EXCL, **BEST)
    print(f"[GCN-D] 末轮 loss={loss:.4f}")

    gold_pos = gold[gold["label"] == 1]
    top_rows, rank_rows = [], []
    print("=" * 78)
    print("GCN-D 每病 Top-20 与金标准命中")
    print("=" * 78)
    for di, d in enumerate(disease_ids):
        s = S[:, di]
        order = np.argsort(-s)
        pos = {str(r.herb_id) for r in gold_pos[gold_pos["disease_id"] == d].itertuples()}
        for rank, hidx in enumerate(order[:20], 1):
            h = herbs[hidx]
            top_rows.append({"disease": d, "rank": rank, "herb_id": h,
                             "herb_name": name_map.get(h, h),
                             "score": round(float(s[hidx]), 4),
                             "is_gold_pos": 1 if h in pos else 0})
        # 金标命中排名
        ranks = []
        for h in pos:
            hi_i = herbs.index(h)
            ranks.append(int(np.searchsorted(-s, -s[hi_i]) + 1))
        if ranks:
            med = float(np.median(ranks))
            enrich = (nH + 1) / 2.0 / med if med > 0 else float("nan")
            y = np.array([1 if herbs[i] in pos else 0 for i in range(nH)], float)
            auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
            print(f"  {d:<7} AUROC={auc:.4f}  已知药中位排名={med:.0f}/{nH}  "
                  f"富集={enrich:.2f}x  命中前20的已知药="
                  f"{sum(1 for r in top_rows[-20:] if r['is_gold_pos'])}")
            for h, rk in sorted(zip(pos, ranks), key=lambda t: t[1])[:25]:
                rank_rows.append({"disease": d, "herb_id": h,
                                  "herb_name": name_map.get(h, h), "rank": rk,
                                  "is_gold": 1})

    pd.DataFrame(top_rows).to_csv(os.path.join(OUT_DIR, "gcn_per_disease_top20.csv"),
                                  index=False, encoding="utf-8-sig")
    pd.DataFrame(rank_rows).to_csv(os.path.join(OUT_DIR, "gcn_gold_ranks_8d.csv"),
                                  index=False, encoding="utf-8-sig")
    print(f"[写入] gcn_per_disease_top20.csv / gcn_gold_ranks_8d.csv")


if __name__ == "__main__":
    main()
