#!/usr/bin/env python3
"""
方案1 vs 方案2: 差异表达基因双阈值过滤对比
==============================================

方案1: |logFC| > logFC_threshold  AND  P_value < 0.05   (原始P值)
方案2: |logFC| > logFC_threshold  AND  adj.P.Val < 0.05  (FDR校正P值)

输出:
  1. 各疾病在两种方案下的基因数量和上调/下调统计
  2. 两种方案的基因重叠情况 (Venn/UpSet 数据)
  3. 过滤后的疾病谱 CSV 文件
  4. 对比可视化 PNG 图表

用法:
    python apply_filtering.py                      # 默认logFC阈值 0.05
    python apply_filtering.py --logfc 0.1          # 自定义logFC阈值
    python apply_filtering.py --logfc 0.0          # 仅用P值过滤 (不做logFC过滤)
    python apply_filtering.py --convert-first      # 先转换原始GEO2R数据
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# 设置matplotlib中文显示
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 项目路径 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import RAW_DIR, PROCESSED_DIR, TOP_BOTTOM_K

# ---- 疾病配置 ----
DISEASE_CONFIG = {
    "PAH": {
        "name_cn": "肺动脉高压",
        "name_en": "Pulmonary Arterial Hypertension",
        "raw_file": "tT2.csv",
        "processed_file": "disease_profiles_PAH.csv",
    },
    "RA": {
        "name_cn": "类风湿关节炎",
        "name_en": "Rheumatoid Arthritis",
        "raw_file": "GSE77298_RA.top.table.csv",
        "processed_file": "disease_profiles_RA.csv",
    },
    "T2D": {
        "name_cn": "2型糖尿病",
        "name_en": "Type 2 Diabetes",
        "raw_file": "GSE20966_T2D.top.table.csv",
        "processed_file": "disease_profiles_T2D.csv",
    },
    "AD": {
        "name_cn": "阿尔茨海默病",
        "name_en": "Alzheimer's Disease",
        "raw_file": "GSE63060_AD.top.table.csv",
        "processed_file": "disease_profiles_AD.csv",
    },
}

RAW_DIR_EXTERNAL = r"C:\Users\alikex\Desktop\Work\差异表达基因"


# ==================== 数据加载 ====================

def load_raw_geo2r_data(disease_id: str) -> pd.DataFrame:
    """从原始GEO2R CSV加载数据，保留 adj.P.Val 列"""
    config = DISEASE_CONFIG[disease_id]
    path = os.path.join(RAW_DIR_EXTERNAL, config["raw_file"])

    print(f"  加载原始数据: {config['raw_file']}")
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")

    # 标准化列名
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_").replace(".", "_")
        if col_lower in ("gene_symbol", "genesymbol", "gene.symbol"):
            rename_map[col] = "gene_symbol"
        elif col_lower in ("logfc", "log_fc", "log2fc"):
            rename_map[col] = "logFC"
        elif col_lower in ("p_value", "pvalue", "p.val", "p.value"):
            rename_map[col] = "P_value"
        elif col_lower in ("adj_p_val", "adj_p_value", "adj_pvalue",
                           "adj_p_val", "fdr", "q_value"):
            rename_map[col] = "adj_P_value"
    df = df.rename(columns=rename_map)

    # 确保必要列
    required = ["gene_symbol", "logFC", "P_value"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}. 可用列: {list(df.columns)}")

    # 过滤
    df = df[df["gene_symbol"].notna() & (df["gene_symbol"].astype(str).str.strip() != "")]
    numeric_mask = df["gene_symbol"].astype(str).str.match(r'^\d+$')
    df = df[~numeric_mask]

    # 类型转换
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df["P_value"] = pd.to_numeric(df["P_value"], errors="coerce")
    if "adj_P_value" in df.columns:
        df["adj_P_value"] = pd.to_numeric(df["adj_P_value"], errors="coerce")
    df = df.dropna(subset=["logFC", "P_value"])

    # 去重（每个基因保留最小P值）
    df = df.sort_values("P_value").drop_duplicates(
        subset=["gene_symbol"], keep="first"
    )

    print(f"    原始基因数: {len(df)}, 列: {list(df.columns)}")
    return df


def apply_scheme1(df: pd.DataFrame, logfc_threshold: float = 0.05) -> pd.DataFrame:
    """
    方案1: |logFC| > logfc_threshold AND P_value < 0.05
    """
    mask = (df["logFC"].abs() > logfc_threshold) & (df["P_value"] < 0.05)
    return df[mask].copy()


def apply_scheme2(df: pd.DataFrame, logfc_threshold: float = 0.05) -> pd.DataFrame:
    """
    方案2: |logFC| > logfc_threshold AND adj.P.Val < 0.05
    （需要原始数据中有 adj_P_value 列）
    """
    if "adj_P_value" not in df.columns:
        print("    [警告] 数据中无 adj_P_value 列，方案2回退到方案1")
        return apply_scheme1(df, logfc_threshold)

    mask = (df["logFC"].abs() > logfc_threshold) & (df["adj_P_value"] < 0.05)
    return df[mask].copy()


# ==================== 统计分析 ====================

def analyze_disease(raw_df: pd.DataFrame, disease_id: str,
                     logfc_threshold: float = 0.05) -> dict:
    """对单个疾病执行方案1和方案2的分析"""
    config = DISEASE_CONFIG[disease_id]
    print(f"\n{'='*60}")
    print(f"  {disease_id} - {config['name_cn']} ({config['name_en']})")
    print(f"{'='*60}")

    has_adj_p = "adj_P_value" in raw_df.columns
    print(f"  原始总基因数: {len(raw_df)}")
    print(f"  adj_P_value 列: {'有' if has_adj_p else '无'}")

    s1 = apply_scheme1(raw_df, logfc_threshold)
    s2 = apply_scheme2(raw_df, logfc_threshold)

    s1_up = (s1["logFC"] > 0).sum()
    s1_down = (s1["logFC"] < 0).sum()
    s2_up = (s2["logFC"] > 0).sum()
    s2_down = (s2["logFC"] < 0).sum()

    s1_genes = set(s1["gene_symbol"])
    s2_genes = set(s2["gene_symbol"])
    overlap = s1_genes & s2_genes
    s1_only = s1_genes - s2_genes
    s2_only = s2_genes - s1_genes

    print(f"\n  方案1 (|logFC|>{logfc_threshold}, P<0.05):")
    print(f"    差异基因: {len(s1)} (上调:{s1_up}, 下调:{s1_down})")
    print(f"    P_value 范围: [{s1['P_value'].min():.2e}, {s1['P_value'].max():.2e}]")
    print(f"    |logFC| 范围: [{s1['logFC'].abs().min():.4f}, {s1['logFC'].abs().max():.4f}]")

    print(f"\n  方案2 (|logFC|>{logfc_threshold}, adj.P<0.05):")
    print(f"    差异基因: {len(s2)} (上调:{s2_up}, 下调:{s2_down})")
    if has_adj_p and len(s2) > 0:
        print(f"    adj_P 范围: [{s2['adj_P_value'].min():.2e}, {s2['adj_P_value'].max():.2e}]")
        print(f"    |logFC| 范围: [{s2['logFC'].abs().min():.4f}, {s2['logFC'].abs().max():.4f}]")

    print(f"\n  方案1 vs 方案2 对比:")
    print(f"    重叠基因: {len(overlap)} ({100*len(overlap)/max(len(s1_genes|s2_genes),1):.1f}%)")
    print(f"    仅方案1:  {len(s1_only)}")
    print(f"    仅方案2:  {len(s2_only)}")

    # 被 FDR 校正剔除的基因（在方案1中但不在方案2中）
    excluded_by_fdr = s1_only
    if excluded_by_fdr:
        excluded_info = raw_df[raw_df["gene_symbol"].isin(excluded_by_fdr)]
        print(f"\n  被FDR校正剔除的基因示例 (P显著但adj_P不显著, 前5个):")
        for _, row in excluded_info.head(5).iterrows():
            adj_str = f"adj.P={row.get('adj_P_value', 'N/A'):.2e}" if has_adj_p else "N/A"
            print(f"    {row['gene_symbol']:<12} logFC={row['logFC']:+.4f}  "
                  f"P={row['P_value']:.2e}  {adj_str}")

    return {
        "disease_id": disease_id,
        "name_cn": config["name_cn"],
        "raw_count": len(raw_df),
        "has_adj_p": has_adj_p,
        "s1_count": len(s1),
        "s1_up": s1_up,
        "s1_down": s1_down,
        "s2_count": len(s2),
        "s2_up": s2_up,
        "s2_down": s2_down,
        "overlap": len(overlap),
        "s1_only": len(s1_only),
        "s2_only": len(s2_only),
        "s1_df": s1,
        "s2_df": s2,
    }


# ==================== 可视化 ====================

def plot_comparison(results: list, logfc_threshold: float, output_dir: str):
    """绘制方案1 vs 方案2 的对比图"""
    n = len(results)
    diseases = [r["disease_id"] for r in results]
    names_cn = [r["name_cn"] for r in results]

    fig = plt.figure(figsize=(16, 5 + 3 * n))

    # ---- 子图1: 基因数量对比柱状图 ----
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(n)
    width = 0.35
    bars1 = ax1.bar(x - width/2, [r["s1_count"] for r in results],
                     width, label="方案1 (P<0.05)", color="#2196F3", edgecolor="white")
    bars2 = ax1.bar(x + width/2, [r["s2_count"] for r in results],
                     width, label="方案2 (adj.P<0.05)", color="#FF9800", edgecolor="white")
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(diseases, fontsize=11)
    ax1.set_ylabel("差异表达基因数", fontsize=12)
    ax1.set_title(f"方案1 vs 方案2 基因数量对比 (|logFC|>{logfc_threshold})", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # ---- 子图2: 上调/下调堆叠柱状图 ----
    ax2 = fig.add_subplot(2, 2, 2)
    x2 = np.arange(n * 2)
    up_vals = []
    down_vals = []
    labels = []
    for r in results:
        up_vals.extend([r["s1_up"], r["s2_up"]])
        down_vals.extend([-r["s1_down"], -r["s2_down"]])
        labels.extend([f"{r['disease_id']}\n方案1", f"{r['disease_id']}\n方案2"])
    ax2.bar(x2, up_vals, color="#4CAF50", label="上调", edgecolor="white")
    ax2.bar(x2, down_vals, color="#F44336", label="下调", edgecolor="white")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("基因数", fontsize=12)
    ax2.set_title("上调/下调基因分布", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # ---- 子图3: 重叠/差异 Venn 风格堆叠 ----
    ax3 = fig.add_subplot(2, 2, 3)
    x3 = np.arange(n)
    overlap_vals = [r["overlap"] for r in results]
    s1_only_vals = [r["s1_only"] for r in results]
    s2_only_vals = [r["s2_only"] for r in results]
    b1 = ax3.bar(x3, s1_only_vals, label="仅方案1", color="#2196F3", edgecolor="white")
    b2 = ax3.bar(x3, overlap_vals, bottom=s1_only_vals, label="重叠", color="#9C27B0", edgecolor="white")
    b3 = ax3.bar(x3, s2_only_vals, bottom=[a+b for a,b in zip(s1_only_vals, overlap_vals)],
                  label="仅方案2", color="#FF9800", edgecolor="white")
    for i in range(n):
        total = s1_only_vals[i] + overlap_vals[i] + s2_only_vals[i]
        ax3.text(i, total + 10, str(total), ha="center", fontsize=9, fontweight="bold")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(diseases, fontsize=11)
    ax3.set_ylabel("基因数", fontsize=12)
    ax3.set_title("方案1 vs 方案2 基因重叠分析", fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(axis="y", alpha=0.3)

    # ---- 子图4: 过滤率对比 ----
    ax4 = fig.add_subplot(2, 2, 4)
    raw_counts = [r["raw_count"] for r in results]
    s1_pct = [100 * r["s1_count"] / max(r["raw_count"], 1) for r in results]
    s2_pct = [100 * r["s2_count"] / max(r["raw_count"], 1) for r in results]
    ax4.plot(diseases, s1_pct, "o-", color="#2196F3", linewidth=2, markersize=8, label="方案1")
    ax4.plot(diseases, s2_pct, "s-", color="#FF9800", linewidth=2, markersize=8, label="方案2")
    for i, (p1, p2) in enumerate(zip(s1_pct, s2_pct)):
        ax4.annotate(f"{p1:.1f}%", (diseases[i], p1), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, color="#2196F3")
        ax4.annotate(f"{p2:.1f}%", (diseases[i], p2), textcoords="offset points",
                     xytext=(0, -18), ha="center", fontsize=9, color="#FF9800")
    ax4.set_ylabel("保留比例 (%)", fontsize=12)
    ax4.set_title(f"基因保留率对比 (原始N={raw_counts})", fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    plt.tight_layout(pad=2)
    fig_path = os.path.join(output_dir, "scheme1_vs_scheme2_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n对比图已保存: {fig_path}")
    return fig_path


def plot_logfc_distribution(results: list, output_dir: str):
    """绘制每种疾病在两种方案下的logFC分布直方图"""
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, r in enumerate(results):
        s1 = r["s1_df"]
        s2 = r["s2_df"]
        name = f"{r['disease_id']} ({r['name_cn']})"

        for j, (df, scheme, color) in enumerate([
            (s1, "方案1 (P<0.05)", "#2196F3"),
            (s2, "方案2 (adj.P<0.05)", "#FF9800"),
        ]):
            ax = axes[i][j]
            up = df[df["logFC"] > 0]["logFC"]
            down = df[df["logFC"] < 0]["logFC"]
            ax.hist(up, bins=30, alpha=0.7, color="#4CAF50", label=f"上调 ({len(up)})")
            ax.hist(abs(down), bins=30, alpha=0.7, color="#F44336",
                    label=f"下调 ({len(down)})")
            ax.axvline(x=0.05, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{name} - {scheme}", fontsize=11)
            ax.set_xlabel("|logFC|", fontsize=10)
            ax.set_ylabel("基因数", fontsize=10)
            ax.legend(fontsize=8)

    plt.tight_layout(pad=2)
    fig_path = os.path.join(output_dir, "logfc_distribution_by_scheme.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"LogFC分布图已保存: {fig_path}")
    return fig_path


# ==================== 输出保存 ====================

def save_filtered_profiles(results: list, output_dir: str, logfc_threshold: float):
    """保存方案1和方案2过滤后的疾病谱文件"""
    s1_dir = os.path.join(output_dir, "scheme1_pvalue")
    s2_dir = os.path.join(output_dir, "scheme2_adj_pvalue")
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)

    all_s1 = []
    all_s2 = []

    for r in results:
        did = r["disease_id"]

        # 方案1
        s1_df = r["s1_df"][["gene_symbol", "logFC", "P_value"]].copy()
        s1_df["disease_id"] = did
        s1_df = s1_df[["disease_id", "gene_symbol", "logFC", "P_value"]]
        s1_path = os.path.join(s1_dir, f"disease_profiles_{did}_scheme1.csv")
        s1_df.to_csv(s1_path, index=False)
        all_s1.append(s1_df)
        print(f"  方案1 [{did}]: {len(s1_df)} 基因 -> {s1_path}")

        # 方案2
        s2_df = r["s2_df"][["gene_symbol", "logFC", "P_value"]].copy()
        if "adj_P_value" in r["s2_df"].columns:
            s2_df["adj_P_value"] = r["s2_df"]["adj_P_value"]
        s2_df["disease_id"] = did
        cols = ["disease_id", "gene_symbol", "logFC", "P_value"]
        if "adj_P_value" in s2_df.columns:
            cols.append("adj_P_value")
        s2_df = s2_df[cols]
        s2_path = os.path.join(s2_dir, f"disease_profiles_{did}_scheme2.csv")
        s2_df.to_csv(s2_path, index=False)
        all_s2.append(s2_df)
        print(f"  方案2 [{did}]: {len(s2_df)} 基因 -> {s2_path}")

    # 合并文件
    combined_s1 = pd.concat(all_s1, ignore_index=True)
    combined_s2 = pd.concat(all_s2, ignore_index=True)
    combined_s1.to_csv(os.path.join(s1_dir, "all_diseases_merged_scheme1.csv"), index=False)
    combined_s2.to_csv(os.path.join(s2_dir, "all_diseases_merged_scheme2.csv"), index=False)
    print(f"\n合并方案1: {len(combined_s1)} 条, 方案2: {len(combined_s2)} 条")


def print_summary_table(results: list):
    """打印汇总对比表"""
    print("\n")
    print("=" * 100)
    print("                    方案1 vs 方案2 汇总对比表")
    print("=" * 100)
    header = f"{'疾病':<8} {'原始':<8} {'方案1(P)':<10} {'↑':<6} {'↓':<6} {'方案2(adjP)':<12} {'↑':<6} {'↓':<6} {'重叠':<8} {'仅S1':<8} {'仅S2':<8} {'FDR剔除%':<10}"
    print(header)
    print("-" * 100)
    for r in results:
        s1_only = r["s1_only"]
        s1_total = r["s1_count"]
        fdr_removed_pct = 100 * s1_only / max(s1_total, 1)
        row = (f"{r['disease_id']:<8} "
               f"{r['raw_count']:<8} "
               f"{r['s1_count']:<10} "
               f"{r['s1_up']:<6} "
               f"{r['s1_down']:<6} "
               f"{r['s2_count']:<12} "
               f"{r['s2_up']:<6} "
               f"{r['s2_down']:<6} "
               f"{r['overlap']:<8} "
               f"{r['s1_only']:<8} "
               f"{r['s2_only']:<8} "
               f"{fdr_removed_pct:.1f}%")
        print(row)
    print("-" * 100)
    print("FDR剔除% = 方案1中有但方案2中没有的基因占比 (即被FDR校正过滤掉的比例)")
    print("=" * 100)


# ==================== 复用配置写入 ====================

def write_summary_csv(results: list, output_dir: str):
    """写入CSV格式的汇总表"""
    import csv
    csv_path = os.path.join(output_dir, "scheme_comparison_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "疾病", "原始基因数", "方案1_总数", "方案1_上调", "方案1_下调",
            "方案2_总数", "方案2_上调", "方案2_下调",
            "重叠基因", "仅方案1", "仅方案2", "FDR剔除比例(%)",
            "方案1_保留率(%)", "方案2_保留率(%)"
        ])
        for r in results:
            s1_pct = 100 * r["s1_count"] / max(r["raw_count"], 1)
            s2_pct = 100 * r["s2_count"] / max(r["raw_count"], 1)
            fdr_removed = 100 * r["s1_only"] / max(r["s1_count"], 1)
            writer.writerow([
                r["disease_id"], r["raw_count"],
                r["s1_count"], r["s1_up"], r["s1_down"],
                r["s2_count"], r["s2_up"], r["s2_down"],
                r["overlap"], r["s1_only"], r["s2_only"],
                f"{fdr_removed:.1f}",
                f"{s1_pct:.1f}", f"{s2_pct:.1f}",
            ])
    print(f"汇总CSV已保存: {csv_path}")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="方案1 vs 方案2: 差异表达基因双阈值过滤对比"
    )
    parser.add_argument("--logfc", type=float, default=0.05,
                        help="|logFC|阈值 (默认: 0.05), 设为0则不做logFC过滤")
    parser.add_argument("--diseases", nargs="+",
                        default=["PAH", "RA", "T2D", "AD"],
                        help="要分析的疾病列表 (默认: PAH RA T2D AD)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认: output/filtering_comparison/)")
    parser.add_argument("--convert-first", action="store_true",
                        help="先运行 convert_diseases.py 转换原始数据")
    parser.add_argument("--no-plots", action="store_true",
                        help="不生成图表")
    args = parser.parse_args()

    # ---- 先转换 ----
    if args.convert_first:
        print("=" * 60)
        print("步骤0: 转换原始GEO2R数据")
        print("=" * 60)
        from convert_diseases import convert_geo2r_to_standard
        for did in args.diseases:
            config = DISEASE_CONFIG[did]
            convert_geo2r_to_standard(
                os.path.join(RAW_DIR_EXTERNAL, config["raw_file"]),
                os.path.join(PROCESSED_DIR, config["processed_file"]),
                did
            )

    # ---- 输出目录 ----
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(PROCESSED_DIR), "output", "filtering_comparison",
            f"logfc_{str(args.logfc).replace('.', 'p')}_{timestamp}"
        )
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"方案1: |logFC| > {args.logfc}  AND  P_value < 0.05")
    print(f"方案2: |logFC| > {args.logfc}  AND  adj.P.Val < 0.05")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # ---- 分析所有疾病 ----
    results = []
    for did in args.diseases:
        if did not in DISEASE_CONFIG:
            print(f"  未知疾病: {did}, 跳过")
            continue
        try:
            raw_df = load_raw_geo2r_data(did)
            r = analyze_disease(raw_df, did, args.logfc)
            results.append(r)
        except Exception as e:
            print(f"  [错误] {did}: {e}")
            continue

    if not results:
        print("没有成功分析任何疾病，退出。")
        return

    # ---- 输出 ----
    print_summary_table(results)
    write_summary_csv(results, output_dir)
    save_filtered_profiles(results, output_dir, args.logfc)

    if not args.no_plots:
        print("\n生成可视化图表...")
        plot_comparison(results, args.logfc, output_dir)
        plot_logfc_distribution(results, output_dir)

    print(f"\n{'='*60}")
    print(f"  分析完成! 所有输出在: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
