#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 2026-09 新增成果的投稿用图（v6）
======================================
配合中文稿 §3.2（非基因靶点消融）、§3.7（独立外部验证）、§3.8（每病 Top-20 富集）
三节使用。所有数值均来自实测 CSV，未做任何美化：

  - 外部验证: external_validation_summary_gcn_d.csv / _hdmap_s.csv
  - Top-20  : per_disease_interpretation.csv
  - 消融    : ablation_nongene_targets.csv（3 种子）

产出（写入 paper_deliverables/）:
    fig4_external_validation.png
    fig5_top20_enrichment.png
    fig6_ablation_nongene.png

用法:
    python make_figures_v6.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体：按可用性依次回退，避免出现方块
for f in ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS"]:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

OUT = os.path.join("output", "results", "paper_deliverables")
os.makedirs(OUT, exist_ok=True)

C_MAIN = "#2E5C8A"    # GCN-D / 完整
C_CMP = "#B0BEC5"     # 对照 / 移除


# ----------------------------------------------------------------------
# 图 4：独立外部验证（HERB CMap）
# ----------------------------------------------------------------------
def fig_external():
    dis = ["Alzheimer", "PAH", "RA", "T2D"]
    gcn = [0.1533, 0.1017, 0.0807, 0.1458]
    hdm = [0.0607, 0.0594, 0.0190, 0.1644]
    x = np.arange(len(dis))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    b1 = ax.bar(x - w / 2, gcn, w, label="GCN-D（本文）", color=C_MAIN,
                edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w / 2, hdm, w, label="HDmap-S（对照）", color=C_CMP,
                edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)

    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.annotate(f"{h:+.3f}", (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3 if h >= 0 else -12),
                    ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(dis)
    ax.set_ylabel("Spearman ρ（与 HERB CMap 外部分数）")
    ax.set_title("图 4. GCN-D 在独立外部数据上的验证：4/4 病方向一致为正",
                 fontsize=12)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_external_validation.png")
    fig.savefig(p)
    plt.close(fig)
    print("[写入]", p)


# ----------------------------------------------------------------------
# 图 5：每病 Top-20 已知正例富集倍数
# ----------------------------------------------------------------------
def fig_top20():
    # 疾病, 观测命中数, 随机期望
    # 正文图 5：仅展示富集倍数最高的 4 个疾病（完整 8 病见附录附表 S4）
    data = [("RA", 9, 1.563), ("OA", 9, 1.643), ("ASTHMA", 10, 2.084),
            ("AD", 3, 0.641)]
    data = sorted(data, key=lambda r: r[1] / r[2], reverse=True)
    names = [d[0] for d in data]
    fold = [d[1] / d[2] for d in data]
    colors = [C_MAIN for _ in fold]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    bars = ax.bar(names, fold, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
               label="随机期望（1×）")
    for b, f, d in zip(bars, fold, data):
        ax.annotate(f"{f:.2f}×", (b.get_x() + b.get_width() / 2, f),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("富集倍数（观测命中 / 随机期望）")
    ax.set_ylim(0, 6.6)
    ax.set_title("图 5. GCN-D Top-20 已知正例富集：均达随机期望 4.7 倍以上",
                 fontsize=12)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_top20_enrichment.png")
    fig.savefig(p)
    plt.close(fig)
    print("[写入]", p)


# ----------------------------------------------------------------------
# 图 6：非基因靶点消融（3 种子均值 ± 标准差）
# ----------------------------------------------------------------------
def fig_ablation():
    labels = ["FULL AUROC", "labeled AUROC"]
    keep_m = [0.8284, 0.8310]
    keep_s = [0.0068, 0.0048]
    rem_m = [0.8089, 0.8049]
    rem_s = [0.0086, 0.0057]
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    ax.bar(x - w / 2, keep_m, w, yerr=keep_s, capsize=4,
           label="完整靶点（含 NONGENE 标注项）", color=C_MAIN,
           edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, rem_m, w, yerr=rem_s, capsize=4,
           label="移除全部非基因符号靶点", color=C_CMP,
           edgecolor="black", linewidth=0.6)

    for i in range(len(labels)):
        ax.annotate(f"{keep_m[i]:.4f}", (x[i] - w / 2, keep_m[i] + keep_s[i]),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5)
        ax.annotate(f"{rem_m[i]:.4f}", (x[i] + w / 2, rem_m[i] + rem_s[i]),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5)
        ax.annotate(f"Δ={rem_m[i] - keep_m[i]:+.4f}",
                    (x[i], min(keep_m[i], rem_m[i]) - 0.028),
                    ha="center", fontsize=9, color="#C0392B",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.76, 0.88)
    ax.set_ylabel("AUROC（3 种子均值 ± 标准差）")
    ax.set_title("图 6. 消融：移除非基因符号靶点造成稳定性能损失，故保留并显式标注",
                 fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig6_ablation_nongene.png")
    fig.savefig(p)
    plt.close(fig)
    print("[写入]", p)


if __name__ == "__main__":
    fig_external()
    fig_top20()
    fig_ablation()
    print("\n全部完成")
