#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugAI 方向验证结果分析 —— 混淆矩阵 / 召回率 / 多数类基线 / 置信度校准
================================================================================
输入: data/processed/drugai_direction_validation.csv (由 validate_drugai_direction.py 生成)
输出: data/processed/drugai_calibration_analysis.csv + 控制台报告
"""
import os
import sys

import pandas as pd

BASE = "c:/Users/alikex/Desktop/Work/systems_mapping/"
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(PROC, "drugai_calibration_analysis.csv")


def main():
    df = pd.read_csv(os.path.join(PROC, "drugai_direction_validation.csv"))
    print(f"total rows: {len(df)}")

    ev = df[df["action_pred"].isin([-1, 1])].copy()
    print(f"evaluable (pred in {{-1, +1}}): {len(ev)}  "
          f"(弃权/失败 {len(df) - len(ev)})")

    # 混淆矩阵
    ct = pd.crosstab(ev["action_true"], ev["action_pred"],
                     rownames=["true"], colnames=["pred"])
    print("\n=== 混淆矩阵 (true \\ pred) ===")
    print(ct.to_string())

    # 逐类召回率 / 精确率
    rows = []
    for cls, label in [(-1, "抑制"), (1, "激活")]:
        tp = int(((ev["action_true"] == cls) & (ev["action_pred"] == cls)).sum())
        fn = int(((ev["action_true"] == cls) & (ev["action_pred"] != cls)).sum())
        fp = int(((ev["action_true"] != cls) & (ev["action_pred"] == cls)).sum())
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rows.append({"class": label, "value": cls, "tp": tp, "fn": fn, "fp": fp,
                     "recall": round(rec, 4), "precision": round(prec, 4)})
        print(f"{label}({cls:+}): recall={rec:.4f}  precision={prec:.4f}  (tp={tp}, fn={fn}, fp={fp})")

    # 多数类基线 vs 模型
    maj = ev["action_true"].value_counts().max() / len(ev)
    acc = (ev["action_pred"] == ev["action_true"]).mean()
    print(f"\n多数类基线 (全预测 -1): {maj:.4f}")
    print(f"DrugAI 准确率:          {acc:.4f}")
    print(f"相对多数类提升:          {acc - maj:+.4f}")

    # 校准: 按 confidence 分块
    print("\n=== 置信度校准 (conf vs 实际准确率) ===")
    bins = [(0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.001)]
    cal_rows = []
    for lo, hi in bins:
        sub = ev[(ev["confidence"] >= lo) & (ev["confidence"] < hi)]
        if len(sub) == 0:
            continue
        a = (sub["action_pred"] == sub["action_true"]).mean()
        mc = sub["confidence"].mean()
        gap = mc - a  # 自信度 - 实际准确率, 越大越过度自信
        cal_rows.append({"conf_lo": lo, "conf_hi": hi, "n": len(sub),
                         "acc": round(a, 4), "mean_conf": round(mc, 4),
                         "overconfidence_gap": round(gap, 4)})
        print(f"conf[{lo}, {hi}): n={len(sub):>3}, acc={a:.4f}, "
              f"mean_conf={mc:.4f}, 过度自信差={gap:+.4f}")

    cal = pd.DataFrame(cal_rows)
    # 合并输出
    out_df = pd.concat([pd.DataFrame(rows), cal], ignore_index=True)
    out_df.to_csv(OUT, index=False)
    print(f"\n[写入] {OUT}")

    # 总体过度自信
    overall_conf = ev["confidence"].mean()
    print(f"\n=== 总体 ===")
    print(f"平均 confidence = {overall_conf:.4f}")
    print(f"实际准确率      = {acc:.4f}")
    print(f"过度自信差      = {overall_conf - acc:+.4f}")
    if overall_conf - acc > 0.1:
        print(">>> 结论: 模型严重过度自信。confidence 被用作边权重(action_type × confidence)时, "
              "几乎所有边权重≈1, 加权机制形同虚设, 无法抑制低质量方向/噪声靶点。")


if __name__ == "__main__":
    main()
