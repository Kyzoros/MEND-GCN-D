#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金标准 AUROC 评估 (standalone)
================================
在已有的 TMS 得分矩阵上计算金标准 AUROC / AUPRC / Precision@K / F1@K，
无需重跑昂贵的 main.py 管线。这是回答"我的模型是否只是瞎蒙"的核心脚本。

用法:
    python evaluate_gold_standard.py
    python evaluate_gold_standard.py --protocol full      # 主协议(与 main.py 一致)
    python evaluate_gold_standard.py --protocol labeled   # 子集协议(仅诊断)
    python evaluate_gold_standard.py --tms output/results/weighted_kegg/hdmap_s_scores.csv

协议:
    full    : 阳性 vs 全部草药。药物重定位标准做法, 论文主协议。
              负样本多(约 490), 统计功效高。
    labeled : 阳性 vs 已知阴性(label=0)。更严格, 但样本仅约 25, 置信区间极宽,
              仅作诊断参考, 不作主结论。

方向 (2026-08-29 修复):
    hdmap_s_scores.csv 在 dual/enrichment 模式下语义是"越高越治疗",
    必须传 higher_is_better=True。
    此前本脚本未传该参数(用默认 False), 导致 y_pred 取负,
    所有 AUROC 被报成 1 - 真实值 (例: 真实 0.461 被报成 0.539)。
    现已与 main.py 第 890 行保持一致。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
from config import DATA_DIR, RESULTS_DIR, GOLD_STANDARD_PATH, TMS_METHOD
from data_loader import load_herb_ingredients
from evaluation_metrics import load_gold_standard, evaluate_gold_standard


def build_default_tms_path() -> str:
    cand = [
        os.path.join(RESULTS_DIR, "weighted_kegg", "hdmap_s_scores.csv"),
        os.path.join(RESULTS_DIR, "hdmap_s_scores.csv"),
    ]
    for c in cand:
        if os.path.exists(c):
            return c
    return cand[0]


def bootstrap_ci(y, s, n_boot=2000, seed=42):
    """AUROC 的 bootstrap 95% 置信区间, 返回 (auroc, lo, hi)"""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    if len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    auc = float(roc_auc_score(y, s))
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        bs.append(roc_auc_score(y[idx], s[idx]))
    if not bs:
        return auc, float("nan"), float("nan")
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return auc, float(lo), float(hi)


def significance_table(tms_df, gold, gold_neg, higher_is_better,
                       protocol, n_boot=2000):
    """逐病计算 AUROC + bootstrap 95% CI + 显著性判定"""
    tms = tms_df.set_index("herb_id") if "herb_id" in tms_df.columns else tms_df.copy()
    tms.index = tms.index.astype(str)

    rows = []
    for col in tms.columns:
        pos = [str(h) for h in gold.get(col, [])]
        if not pos:
            continue
        if protocol == "labeled" and gold_neg:
            negs = [str(h) for h in gold_neg.get(col, [])]
            herbs = [h for h in (pos + negs) if h in tms.index]
        else:
            herbs = [h for h in tms.index]
        if len(herbs) < 5:
            continue

        y = np.array([1 if h in set(pos) else 0 for h in herbs], dtype=float)
        s = tms.loc[herbs, col].values.astype(float)
        if not higher_is_better:
            s = -s
        if len(set(y)) < 2:
            continue

        auc, lo, hi = bootstrap_ci(y, s, n_boot=n_boot)
        if lo > 0.5:
            sig = "显著优于随机"
        elif hi < 0.5:
            sig = "显著差于随机"
        else:
            sig = "不显著(CI跨0.5)"
        rows.append({
            "disease": col,
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "AUROC": round(auc, 4),
            "CI_lo": round(lo, 4),
            "CI_hi": round(hi, 4),
            "显著性": sig,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="金标准 AUROC 评估 (standalone)")
    ap.add_argument("--tms", default=build_default_tms_path(),
                    help="TMS 得分矩阵 CSV 路径")
    ap.add_argument("--gold", default=GOLD_STANDARD_PATH,
                    help="金标准表 CSV 路径")
    ap.add_argument("--top-k", type=int, default=10, help="Top-K 指标中的 K")
    ap.add_argument("--protocol", choices=["full", "labeled"], default="full",
                    help="full=阳性vs全部草药(主协议, 与main.py一致); "
                         "labeled=阳性vs已知阴性(仅诊断)")
    args = ap.parse_args()

    if not os.path.exists(args.tms):
        print(f"[错误] 找不到 TMS 矩阵: {args.tms}")
        sys.exit(1)

    tms_df = pd.read_csv(args.tms)
    herb_ing = load_herb_ingredients()
    gold, gold_neg, warns = load_gold_standard(args.gold, herb_ing)
    for w in warns:
        print(f"[警告] {w}")

    if not gold:
        print("[错误] 未加载任何金标准条目, 请检查 data/gold_standard.csv")
        sys.exit(1)

    # 关键修复: 与 main.py 第 890 行一致
    higher_is_better = TMS_METHOD in ("dual", "enrichment")
    use_neg = gold_neg if (args.protocol == "labeled" and gold_neg) else None

    print(f"\n金标准验证: {args.tms}")
    print(f"[协议] {args.protocol.upper()}   "
          f"[方向] higher_is_better={higher_is_better}   "
          f"(TMS_METHOD={TMS_METHOD})")
    print("=" * 64)

    gs_df = evaluate_gold_standard(
        tms_df, gold, top_k=args.top_k,
        higher_is_better=higher_is_better,
        negative_standard=use_neg,
    )
    print("=" * 64)

    if gs_df.empty:
        print("无可用评估 (金标准疾病列与 TMS 矩阵无交集)")
        sys.exit(0)

    mean_auroc = gs_df["AUROC"].mean(skipna=True)
    print(f"平均 AUROC = {mean_auroc:.4f}   (随机基线 0.5, >0.5 即优于随机)")
    better = (gs_df["AUROC"] > 0.5).sum()
    print(f"优于随机的疾病数: {better}/{len(gs_df)}")

    # ---- 显著性诊断 (bootstrap 95% CI) ----
    print("\n" + "=" * 72)
    print("统计显著性诊断 (bootstrap 95% CI, 2000 次重采样)")
    print("=" * 72)
    sig_df = significance_table(tms_df, gold, gold_neg,
                                higher_is_better, args.protocol)
    if not sig_df.empty:
        try:
            print(sig_df.to_markdown(index=False))
        except Exception:
            print(sig_df.to_string(index=False))
        n_sig = (sig_df["显著性"] == "显著优于随机").sum()
        print(f"\n统计显著的疾病数: {n_sig}/{len(sig_df)}"
              f"   (CI 下界 > 0.5 才算显著)")
        if n_sig == 0:
            print("[提示] 全部不显著 —— 样本量太小, 现有数字不足以支撑结论。")

    out = os.path.join(os.path.dirname(args.tms), "gold_standard_evaluation.csv")
    gs_df.to_csv(out, index=False)
    print(f"\n结果已保存: {out}")

    try:
        print("\n" + gs_df.to_markdown(index=False))
    except Exception:
        print(gs_df.to_string(index=False))


if __name__ == "__main__":
    main()
