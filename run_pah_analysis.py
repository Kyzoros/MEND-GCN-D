#!/usr/bin/env python3
"""
疾病草药推荐分析脚本
=====================
将用户的差异表达基因数据转换为模型格式，
利用已缓存的草药扰动签名，计算HDmap-S TMS得分，
输出 Top-50 推荐草药。

用法:
    python run_pah_analysis.py --disease PAH --input C:/path/to/tT2.csv
    python run_pah_analysis.py --disease AD --input C:/path/to/ad_deg.csv

或在代码中调用:
    from run_pah_analysis import run_analysis
    result = run_analysis("PAH", "C:/path/to/tT2.csv")
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# 切换到项目目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

from config import (
    TOP_BOTTOM_K, PERMUTATION_N, RESULTS_DIR,
    RAW_DIR, PROCESSED_DIR,
    DIFFUSION_METHOD, RWR_RESTART_PROB, RWR_MAX_ITER, RWR_TOL,
)
from data_loader import (
    load_ppi_network, load_herb_ingredients, load_ingredient_targets,
    build_ppi_adjacency,
)
from perturbation_signature import (
    PerturbationSignatureBuilder, RWRPerturbationSignatureBuilder,
    batch_build_herb_signatures,
)
from hdmap_s import HDmapS


def run_analysis(disease_name: str = "PAH",
                 input_csv: str = r"C:\Users\alikex\Desktop\tT2.csv",
                 gene_symbol_col: str = "Gene.symbol",
                 logfc_col: str = "logFC",
                 pvalue_col: str = "P.Value",
                 verbose: bool = True):
    """
    对指定疾病运行 HDmap-S 分析，输出 Top-50 草药推荐。

    Parameters
    ----------
    disease_name : str
        疾病名称（如 "PAH", "AD"），用于输出文件命名。
    input_csv : str
        用户差异表达基因CSV文件路径。
    gene_symbol_col : str
        基因名列名，默认 "Gene.symbol"。
    logfc_col : str
        logFC列名，默认 "logFC"。
    pvalue_col : str
        P值列名，默认 "P.Value"。
    verbose : bool
        是否打印详细日志。

    Returns
    -------
    tuple : (top50_df, tms_df, hdmap_s, builder)
    """
    # ---- 第1步：转换用户数据 ----
    if verbose:
        print("=" * 60)
        print(f"第1步: 转换 {disease_name} 差异表达基因数据")
        print("=" * 60)
    
    raw_df = pd.read_csv(input_csv)
    if verbose:
        print(f"  原始数据: {raw_df.shape[0]} 行, 列: {list(raw_df.columns)}")
    
    # 过滤空基因名
    raw_df = raw_df[raw_df[gene_symbol_col].notna() & (raw_df[gene_symbol_col] != "")]
    if verbose:
        print(f"  过滤空基因名后: {raw_df.shape[0]} 行")
    
    # 转换为模型格式
    disease_df = pd.DataFrame({
        "disease_id": disease_name,
        "gene_symbol": raw_df[gene_symbol_col].astype(str).str.strip(),
        "logFC": pd.to_numeric(raw_df[logfc_col], errors="coerce"),
        "P_value": pd.to_numeric(raw_df[pvalue_col], errors="coerce"),
    })
    
    # 去重 + 按logFC降序
    disease_df = disease_df.sort_values("P_value").drop_duplicates(
        subset=["disease_id", "gene_symbol"], keep="first"
    )
    disease_df = disease_df.sort_values(["disease_id", "logFC"], ascending=[True, False])
    
    if verbose:
        print(f"  最终数据: {disease_df.shape[0]} 个差异基因")
        print(f"  logFC 范围: [{disease_df['logFC'].min():.4f}, {disease_df['logFC'].max():.4f}]")
    
    disease_path = os.path.join(PROCESSED_DIR, f"disease_profiles_{disease_name}.csv")
    disease_df.to_csv(disease_path, index=False)
    if verbose:
        print(f"  已保存: {disease_path}")
    
    # ---- 第2步：加载PPI & 签名 ----
    if verbose:
        print("\n" + "=" * 60)
        print("第2步: 加载PPI网络 & 草药扰动签名")
        print("=" * 60)
    
    ppi_network = load_ppi_network()
    adj, gene2idx = build_ppi_adjacency(ppi_network)
    if verbose:
        print(f"  PPI网络: {len(gene2idx)} 个蛋白, {ppi_network.shape[0]} 条互作")
    
    if DIFFUSION_METHOD == "rwr":
        builder = RWRPerturbationSignatureBuilder(
            adj, gene2idx,
            restart_prob=RWR_RESTART_PROB,
            max_iter=RWR_MAX_ITER,
            tol=RWR_TOL
        )
        if verbose:
            print(f"  扩散方法: RWR (r={RWR_RESTART_PROB})")
    else:
        builder = PerturbationSignatureBuilder(adj, gene2idx)
        if verbose:
            print(f"  扩散方法: 两阶段热扩散")
    
    herb_ingredients = load_herb_ingredients()
    ingredient_targets = load_ingredient_targets()
    if verbose:
        print(f"  草药数: {herb_ingredients['herb_id'].nunique()}")
        print(f"  成分数: {ingredient_targets['ingredient_id'].nunique()}")
    
    if verbose:
        print("\n  构建草药扰动签名（已有缓存则自动加载）...")
    herb_signatures, builder = batch_build_herb_signatures(
        herb_ingredients, ingredient_targets, ppi_network, builder,
        use_cache=True,
    )
    if verbose:
        print(f"  已加载 {len(herb_signatures)} 种草药的扰动签名")
    
    # ---- 第3步：HDmap-S TMS计算 ----
    if verbose:
        print("\n" + "=" * 60)
        print(f"第3步: HDmap-S 计算 {disease_name} 的TMS得分")
        print("=" * 60)
    
    disease_genes = disease_df["gene_symbol"].tolist()
    if verbose:
        print(f"  疾病基因数: {len(disease_genes)}")
    
    ppi_genes = set(gene2idx.keys())
    overlap = set(disease_genes) & ppi_genes
    if verbose:
        print(f"  与PPI网络重叠基因: {len(overlap)} / {len(disease_genes)} "
              f"({100*len(overlap)/max(len(disease_genes),1):.1f}%)")
    
    hdmap_s = HDmapS(herb_signatures, builder)
    herb_ids = list(herb_signatures.keys())
    
    scores = []
    for herb_id in herb_ids:
        tms = hdmap_s.compute_tms(herb_id, disease_genes, k=TOP_BOTTOM_K)
        scores.append(tms)
    
    tms_df = pd.DataFrame({disease_name: scores}, index=herb_ids)
    tms_df.index.name = "herb_id"
    tms_df = tms_df.sort_values(disease_name, ascending=True)
    
    disease_subdir = os.path.join(RESULTS_DIR, f"{disease_name}_results")
    os.makedirs(disease_subdir, exist_ok=True)
    tms_path = os.path.join(disease_subdir, f"hdmap_s_{disease_name}_scores.csv")
    tms_df.to_csv(tms_path)
    if verbose:
        print(f"  完整TMS结果已保存: {tms_path}")
    
    # ---- 第4步：Top-50 推荐 ----
    if verbose:
        print("\n" + "=" * 60)
        print(f"第4步: Top-50 推荐草药 ({disease_name})")
        print("=" * 60)
    
    herb_names = herb_ingredients[["herb_id", "herb_name"]].drop_duplicates()
    herb_name_map = dict(zip(herb_names["herb_id"], herb_names["herb_name"]))
    
    top50 = tms_df.head(50)
    top50["herb_name"] = top50.index.map(lambda x: herb_name_map.get(x, "未知"))
    
    if verbose:
        print(f"\n{'='*72}")
        print(f"  {disease_name} - Top 50 草药推荐")
        print(f"{'='*72}")
        print(f"  {'排名':<6}{'草药ID':<16}{'草药名':<30}{'TMS(越负越强)':<14}{'P_value(估计)'}")
        print(f"  {'-'*72}")
    
    output_records = []
    for rank, (herb_id, row) in enumerate(top50.iterrows(), 1):
        tms_obs, p_val = hdmap_s.permutation_test(
            herb_id, disease_genes, n_perm=min(PERMUTATION_N, 1000), k=TOP_BOTTOM_K,
            early_stop_extreme=20
        )
        output_records.append({
            "rank": rank,
            "herb_id": herb_id,
            "herb_name": row["herb_name"],
            "TMS": round(tms_obs, 4),
            "P_value": round(p_val, 4),
            "significant": "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        })
        if verbose:
            sig_mark = output_records[-1]["significant"]
            print(f"  {rank:<6}{herb_id:<16}{row['herb_name']:<30}{tms_obs:<10.4f}{p_val:<10.4f}  {sig_mark}")
    
    if verbose:
        print(f"  {'-'*72}")
        print(f"  显著性标记: *** p<0.001, ** p<0.01, * p<0.05")
    
    result_path = os.path.join(disease_subdir, f"{disease_name}_top50_herbs.csv")
    result_df = pd.DataFrame(output_records)
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    
    if verbose:
        print(f"\n  Top-50结果已保存: {result_path}")
        sig_count = sum(1 for r in output_records if r["P_value"] < 0.05)
        print(f"  显著草药数 (p<0.05): {sig_count} / 50")
        print(f"\n{'='*72}")
        print(f"  分析完成! ({disease_name})")
        print(f"{'='*72}")
    
    return result_df, tms_df, hdmap_s, builder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="疾病草药推荐分析 (HDmap-S)")
    parser.add_argument("--disease", "-d", type=str, default="PAH",
                        help="疾病名称，如 PAH, AD (默认: PAH)")
    parser.add_argument("--input", "-i", type=str,
                        default=r"C:\Users\alikex\Desktop\tT2.csv",
                        help="差异表达基因CSV文件路径")
    parser.add_argument("--gene-col", type=str, default="Gene.symbol",
                        help="基因名列名 (默认: Gene.symbol)")
    parser.add_argument("--logfc-col", type=str, default="logFC",
                        help="logFC列名 (默认: logFC)")
    parser.add_argument("--pvalue-col", type=str, default="P.Value",
                        help="P值列名 (默认: P.Value)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="静默模式")
    args = parser.parse_args()
    
    run_analysis(
        disease_name=args.disease,
        input_csv=args.input,
        gene_symbol_col=args.gene_col,
        logfc_col=args.logfc_col,
        pvalue_col=args.pvalue_col,
        verbose=not args.quiet,
    )
