#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
靶点数混杂控制实验 (2026-09-04, v5)
====================================
问题: "草药被研究得越充分 / 靶点越多, 越容易被任何打分排到前面" 是本领域的
已知混杂 (target-count confound)。论文 §3.7 声称 ρ = 0.65–0.82, 但重算发现
该值只对**扩散类**打分成立 (RWR ρ = 0.79), 对直接富集 (Fisher ρ = 0.07) 不成立。

更关键的问题: **GCN-D 自己是不是也在追热度?**
本脚本做两件事:
  1. 计算各方法分数与靶点数的 Spearman ρ (每病, 取均值);
  2. **残差化控制**: 对每个病的分数向量, 用秩回归剔除靶点数的影响, 取残差后
     重新做 5 折 CV 的干净对级评估。若残差化后 GCN-D 仍显著高于基线, 说明它的
     优势不是"追热度"。

产出: output/results/paper_deliverables/Table_confound_control.csv
用法: python run_confound_check.py --epochs 3000
      python run_confound_check.py --epochs 300 --folds 2   # 冒烟
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg
from run_fair_comparison import (build_graph, build_exclude, train_gcn,
                                 score_fisher, score_proximity,
                                 score_targetcount)
from run_fair_final import score_rwr
from run_cv_final import make_folds, pair_metrics

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")
BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)


def rho_vs_targetcount(S, tc):
    rs = []
    for di in range(S.shape[1]):
        s = np.nan_to_num(S[:, di].astype(float), nan=0.0)
        if np.std(s) > 0:
            rs.append(spearmanr(s, tc).statistic)
    return float(np.nanmean(rs)), float(np.nanmin(rs)), float(np.nanmax(rs))


def residualize(S, tc):
    """逐列秩回归剔除靶点数影响, 返回残差分数。"""
    R = np.zeros_like(S, dtype=float)
    n = S.shape[0]
    t = rankdata(tc) / n
    A = np.column_stack([np.ones(n), t])
    for di in range(S.shape[1]):
        s = rankdata(np.nan_to_num(S[:, di].astype(float), nan=0.0)) / n
        beta, *_ = np.linalg.lstsq(A, s, rcond=None)
        R[:, di] = s - A @ beta
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    print("=" * 78)
    print("靶点数混杂控制实验")
    print("=" * 78)

    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, _g = build_graph(use_logfc=True)
    print(f"[图] 节点={N}  草药={nH}  疾病={nD}")
    EXCL = build_exclude(gold, herbs, disease_ids)

    tc = score_targetcount(herbs, disease_ids).values[:, 0]
    all_pairs, folds = make_folds(herbs, disease_ids, gold)
    folds = folds[:args.folds]

    rows = []
    unsup = {"D-TMS_Fisher": score_fisher(herbs, disease_ids).values,
             "RWR_diffusion": score_rwr(A, N, off_h, nH, off_d, nD),
             "Network_proximity": score_proximity(herbs, disease_ids).values}

    print("\n[1/2] 无监督方法: ρ 与残差化后干净对级 AUROC")
    for name, S in unsup.items():
        r, lo, hi = rho_vs_targetcount(S, tc)
        raw = float(pair_metrics(S, herbs, disease_ids, gold, all_pairs)["auc"].mean())
        res = float(pair_metrics(residualize(S, tc), herbs, disease_ids, gold,
                                 all_pairs)["auc"].mean())
        rows.append({"method": name, "rho_vs_targetcount": round(r, 4),
                     "rho_range": f"[{lo:.3f},{hi:.3f}]",
                     "pair_AUROC_raw": round(raw, 4),
                     "pair_AUROC_residualized": round(res, 4),
                     "delta": round(res - raw, 4), "n_pairs": len(all_pairs)})
        print(f"  {name:<20} ρ={r:.4f}  原始={raw:.4f}  残差化={res:.4f}  "
              f"Δ={res - raw:+.4f}")

    print(f"\n[2/2] GCN-D {len(folds)} 折 CV (epochs={args.epochs})")
    dfs_raw, dfs_res = [], []
    for k, test in enumerate(folds):
        train = [p for j, f in enumerate(folds) if j != k for p in f]
        S, loss = train_gcn(A, X, off_h, off_d, nH, nD, train, nH,
                            epochs=args.epochs, seed=42, exclude_pos=EXCL, **BEST)
        dfs_raw.append(pair_metrics(S, herbs, disease_ids, gold, test))
        dfs_res.append(pair_metrics(residualize(S, tc), herbs, disease_ids,
                                    gold, test))
        print(f"  fold {k + 1}/{len(folds)}  loss={loss:.4f}")
    raw = float(pd.concat(dfs_raw)["auc"].mean())
    res = float(pd.concat(dfs_res)["auc"].mean())
    # ρ 用最后一折模型为代表 (分数尺度跨折可比)
    r, lo, hi = rho_vs_targetcount(S, tc)
    rows.append({"method": "GCN_DistMult", "rho_vs_targetcount": round(r, 4),
                 "rho_range": f"[{lo:.3f},{hi:.3f}]",
                 "pair_AUROC_raw": round(raw, 4),
                 "pair_AUROC_residualized": round(res, 4),
                 "delta": round(res - raw, 4),
                 "n_pairs": int(len(pd.concat(dfs_raw)))})
    print(f"  GCN_DistMult        ρ={r:.4f}  原始={raw:.4f}  残差化={res:.4f}  "
          f"Δ={res - raw:+.4f}")

    out = os.path.join(OUT_DIR, "Table_confound_control.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {out}")
    print(f"[完成] 总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
