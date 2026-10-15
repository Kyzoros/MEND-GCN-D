#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成双通道TMS 原生输出 (standalone, 无需重跑管线)
=================================================
从已有的 TMS(cos) 矩阵直接派生双通道三件套并写入 RESULTS_DIR,
等价于 main.py 在 tms_method="dual" 下 Step3 的产物。

产物 (与 main.py dual 分支一致):
  - hdmap_s_scores_dual_reverse.csv   (tms_reverse = -cos, 越高=拮抗越强)
  - hdmap_s_scores_dual_engage.csv    (tms_engage  = +cos, 越高=嵌网越强)
  - hdmap_s_scores_dual_combined.csv  (按疾病机制先验合并, 越高=越治疗)
  - hdmap_s_scores_dual_cos.csv       (原始 cos, 便于诊断/回退)
  - hdmap_s_scores.csv                (= combined, 供 Step5/下游一致读取)

适用: 已有 output/results/weighted_kegg/hdmap_s_scores.csv (cos 矩阵) 时,
用本脚本秒级生成双通道文件, 不必重跑昂贵的签名构建。

用法 (PyCharm 直接 Run 本文件):
    python generate_dual_outputs.py
    python generate_dual_outputs.py --tms <path/to/hdmap_s_scores.csv>
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
from config import RESULTS_DIR, DISEASE_TMS_MECHANISM
from evaluate_dual_channel import auroc


def build_default_tms_path() -> str:
    """默认输入 = 原始 cos 矩阵 (稳定来源)。

    严重 bug 修复 (2026-08-29): 此前默认读取主文件 hdmap_s_scores.csv,
    而本脚本又把 combined 写回 hdmap_s_scores.csv, 构成"读自己写的文件"。
    后果: 每运行一次, 上一轮的 combined 就被当作新的 cos 再变换一次,
    rev/eng 的符号被反复翻转 -> 数据逐步污染,
    方向校准(direction_calibration)与金标准评估的数字随之失真。
    现在输入固定为 hdmap_s_scores_dual_cos.csv (只由 main.py Step3 写一次),
    且本脚本不再回写该文件。
    """
    cand = [
        os.path.join(RESULTS_DIR, "weighted_kegg", "hdmap_s_scores_dual_cos.csv"),
        os.path.join(RESULTS_DIR, "hdmap_s_scores_dual_cos.csv"),
    ]
    for c in cand:
        if os.path.exists(c):
            return c
    # 首次运行且尚无原始 cos 时才回退到主文件
    return os.path.join(RESULTS_DIR, "weighted_kegg", "hdmap_s_scores.csv")


def main():
    ap = argparse.ArgumentParser(description="生成双通道TMS 原生输出")
    ap.add_argument("--tms", default=build_default_tms_path(),
                    help="TMS 得分矩阵 CSV (cos 矩阵) 路径")
    args = ap.parse_args()

    if not os.path.exists(args.tms):
        print(f"[错误] 找不到 TMS 矩阵: {args.tms}")
        print("请先运行 main.py (weighted/dual) 生成 hdmap_s_scores.csv")
        sys.exit(1)

    out_dir = os.path.dirname(args.tms)
    cos_df = pd.read_csv(args.tms)
    if "herb_id" in cos_df.columns:
        cos_df = cos_df.set_index("herb_id")

    # 双通道派生
    rev_df = -cos_df            # 越高=拮抗(逆转)越强
    eng_df = cos_df.copy()      # 越高=嵌网(增益)越强
    rev_df.index.name = "herb_id"
    eng_df.index.name = "herb_id"

    # 按疾病机制先验合并
    combined = pd.DataFrame(index=rev_df.index)
    for d in rev_df.columns:
        mech = str(DISEASE_TMS_MECHANISM.get(d, "engagement")).strip().lower()
        combined[d] = rev_df[d] if mech == "antagonist" else eng_df[d]
    combined.index.name = "herb_id"

    rev_df.to_csv(os.path.join(out_dir, "hdmap_s_scores_dual_reverse.csv"))
    eng_df.to_csv(os.path.join(out_dir, "hdmap_s_scores_dual_engage.csv"))
    combined.to_csv(os.path.join(out_dir, "hdmap_s_scores_dual_combined.csv"))
    # 不再回写 hdmap_s_scores_dual_cos.csv: 它是本脚本的输入源,
    # 回写会让下一轮把 combined 当成原始 cos (自我污染), 见 build_default_tms_path 说明。
    # 原始 cos 只由 main.py Step3 写入。
    # 主文件 = combined (与 main.py dual 分支一致, 供 Step5 重载)
    combined.to_csv(os.path.join(out_dir, "hdmap_s_scores.csv"))

    print(f"已写入双通道文件到: {out_dir}")
    print("  - hdmap_s_scores_dual_reverse.csv")
    print("  - hdmap_s_scores_dual_engage.csv")
    print("  - hdmap_s_scores_dual_combined.csv")
    print("  - hdmap_s_scores_dual_cos.csv")
    print("  - hdmap_s_scores.csv (= combined)")

    # 即时验证 AUROC (复用金标准)
    try:
        from config import DATA_DIR, GOLD_STANDARD_PATH
        from data_loader import load_herb_ingredients
        from evaluation_metrics import load_gold_standard_mechanism
        herb_ing = load_herb_ingredients()
        gold, _, _, warns = load_gold_standard_mechanism(GOLD_STANDARD_PATH, herb_ing)
        rows = []
        for disease, herb_list in gold.items():
            if disease not in cos_df.columns:
                continue
            col = cos_df[disease]
            all_herbs = list(col.index)
            present = [h for h in herb_list if h in col.index]
            if len(present) < 2:
                continue
            y = np.array([1 if h in set(present) else 0 for h in all_herbs], dtype=float)
            tms = col.loc[all_herbs].values.astype(float)
            auc_rev = auroc(y, -tms)
            auc_eng = auroc(y, +tms)
            mech = str(DISEASE_TMS_MECHANISM.get(disease, "engagement")).strip().lower()
            auc_comb = auc_rev if mech == "antagonist" else auc_eng
            rows.append((disease, mech, auc_comb))
        if rows:
            avg = float(np.nanmean([r[2] for r in rows]))
            print(f"\n即时双通道 AUROC 校验 (合并): {avg:.4f}  (随机基线 0.5)")
            for d, mech, a in rows:
                print(f"  {d:<8} mech={mech:<11} AUROC={a:.4f}")
    except Exception as e:
        print(f"[提示] AUROC 校验跳过: {e}")


if __name__ == "__main__":
    main()
