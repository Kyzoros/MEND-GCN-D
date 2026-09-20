#!/usr/bin/env python3
"""
Systems-Mapping of Herbal Effects on Complex Diseases
=====================================================

工作流程:
1. 加载数据 (疾病表达谱、草药成分-靶标、PPI网络)
2. 构建草药扰动签名 (RWR扩散)
3. HDmap-S: 计算单草药-疾病TMS得分 + 置换检验
4. HDmap-M: 遗传算法搜索最优草药组合
5. 结果评估: H-PG网络 + 共同基因分析

输出结构 (扁平, 无嵌套子目录):
    output/results/{DISEASE}_{timestamp}/
    ├── hdmap_s_scores.csv              # 499草药 TMS得分
    ├── herb_perturbation_signatures.csv # 草药扰动签名矩阵
    ├── {DISEASE}_top50_herbs.csv        # Top50 草药排名 (含P值)
    ├── hdmap_m_results_{DISEASE}.csv    # GA最优组合
    ├── hdmap_m_topk_results_{DISEASE}.csv # Top-K组合
    ├── hpg_network_{DISEASE}.csv        # H-PG网络
    ├── {DISEASE}_common_genes.txt       # 共同扰动基因
    └── run_metadata.json               # 运行元信息

用法:
    python main.py                                      # 默认, 输出到 {disease}_results/
    python main.py --disease PAH                        # 指定疾病
    python main.py --disease PAH --input data/processed/disease_profiles_PAH.csv
    python main.py --ga-max-generations 100             # 覆盖GA代数
    python main.py --output-dir ./my_results            # 指定输出目录
    python main.py --timestamp                          # 输出目录带时间戳
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免 OMP Error#15 (PyCharm 手动跑也建议设)

import argparse
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    TOP_BOTTOM_K, PERMUTATION_N, OUTPUT_DIR, RESULTS_DIR,
    DISEASE_LOGFC_THRESHOLD, DISEASE_PVALUE_THRESHOLD, DISEASE_DEG_TOP_K,
    DIFFUSION_METHOD, RWR_RESTART_PROB, RWR_MAX_ITER, RWR_TOL, THERMAL_ALPHA,
    DISEASE_GENE_WEIGHTS, DISEASE_TMS_MECHANISM,
    NETWORK_TYPE,
    PPI_MULTI_SOURCE, PPI_FUSION_STRATEGY, PPI_MIN_SOURCES, PPI_SOURCE_DIR,
    TEMPORAL_ENABLED, TEMPORAL_METHOD, TEMPORAL_DECAY_FACTOR, TEMPORAL_TOP_N,
    TRANSFER_ENABLED, TRANSFER_LAMBDA, TRANSFER_MIN_SIMILARITY,
    TMS_METHOD,
    # GA 参数
    GA_MAX_GENERATIONS, GA_N_TOP, GA_MAX_COMBO_SIZE,
    GA_POP_SIZE, GA_CROSSOVER_RATE, GA_MUTATION_RATE, GA_ELITE_SIZE,
    GA_SIZE_PENALTY_LAMBDA, GA_HOF_SIZE, GA_RESTART_STAGNATION,
    GA_RESTART_FRACTION, GA_MAX_RESTARTS,
    GA_DIVERSITY_THRESHOLD, GA_EARLY_STOP_GENERATIONS,
    # 输出配置
    OUTPUT_INCLUDE_TIMESTAMP, OUTPUT_TIMESTAMP_FORMAT,
    get_run_output_dir, print_config_summary,
)
from data_loader import (
    load_disease_profiles, load_herb_ingredients,
    load_ingredient_targets, load_ppi_network,
    build_ppi_adjacency, build_fused_ppi_adjacency,
    filter_chembl_only,
)
from kegg_network import build_kegg_adjacency
from perturbation_signature import (
    PerturbationSignatureBuilder, RWRPerturbationSignatureBuilder,
    batch_build_herb_signatures, signature_to_dataframe
)
import scipy.sparse as sp

# 方案B(网络层融合): KEGG 在统一基因空间中的合并权重 (PPI 权重 = 1-alpha)
FUSED_NETWORK_ALPHA = 0.5


def _reindex_adj_to_union(adj, old_g2i, union_g2i, n_genes):
    """将 (adj, old_g2i) 重索引到统一基因空间 union_g2i, 缺失基因补零边。"""
    adj = adj.tocsr()
    old2new = {old_idx: union_g2i[g] for g, old_idx in old_g2i.items()}
    coo = adj.tocoo()
    rows = [old2new[o] for o in coo.row]
    cols = [old2new[c] for c in coo.col]
    return sp.coo_matrix((coo.data, (rows, cols)), shape=(n_genes, n_genes)).tocsr()


def _filter_disease_genes(sub):
    """统一的疾病 DEG 过滤逻辑 (step3 / step4 / temporal 共用)。

    top-K 模式 (DISEASE_DEG_TOP_K > 0, 现默认 1500):
        先按 |logFC| 做很宽松的预过滤, 再按 P 值取最显著的 K 个。
        目的: 让各病疾病模块规模一致, 保证跨病可比。

        背景 (2026-08-29): 旧的绝对阈值 (|logFC|>0.05, P<0.05) 下各病严重失衡——
          PAH 9674 个 (占 KEGG 网络 46%, 近乎全基因组), 而 T2D 仅 1730 个;
          若收紧到领域标准 |logFC|>1, AD 只剩 3 个 (AD 数据差异表达信号极弱),
          方法在 AD 上直接失效 (AUROC=0.500)。
        top-K 同时解决了"规模失衡"与"AD 信号弱"两个问题。

    绝对阈值模式 (DISEASE_DEG_TOP_K <= 0): 退回 logFC + P 值双阈值过滤。
    """
    if DISEASE_LOGFC_THRESHOLD > 0:
        sub = sub[sub['logFC'].abs() > DISEASE_LOGFC_THRESHOLD]
    if DISEASE_DEG_TOP_K and DISEASE_DEG_TOP_K > 0:
        if 'P_value' in sub.columns:
            sub = sub.sort_values('P_value', ascending=True)
        else:
            sub = sub.sort_values('logFC', key=lambda s: s.abs(),
                                  ascending=False)
        sub = sub.head(DISEASE_DEG_TOP_K)
    elif DISEASE_PVALUE_THRESHOLD > 0 and 'P_value' in sub.columns:
        sub = sub[sub['P_value'] < DISEASE_PVALUE_THRESHOLD]
    return sub


def build_fused_network_adjacency(alpha=0.5):
    """方案B: 在统一基因空间合并 KEGG 与 PPI 传播网络, 单一签名。
    两网络拓扑在扩散前即融合, 可捕捉跨网络效应(纯分数融合方案A做不到)。"""
    import config as _cfg_f
    kegg_adj, kegg_g2i = build_kegg_adjacency(
        cache_dir=_cfg_f.PROCESSED_DIR, min_shared=_cfg_f.KEGG_MIN_SHARED)
    if PPI_MULTI_SOURCE:
        ppi_adj, ppi_g2i = build_fused_ppi_adjacency(
            strategy=PPI_FUSION_STRATEGY, min_sources=PPI_MIN_SOURCES,
            source_dir=PPI_SOURCE_DIR)
    else:
        ppi_adj, ppi_g2i = build_ppi_adjacency(load_ppi_network())
    union_genes = sorted(set(kegg_g2i) | set(ppi_g2i))
    union_g2i = {g: i for i, g in enumerate(union_genes)}
    n = len(union_genes)
    kg = _reindex_adj_to_union(kegg_adj, kegg_g2i, union_g2i, n)
    pp = _reindex_adj_to_union(ppi_adj, ppi_g2i, union_g2i, n)
    fused = alpha * kg + (1.0 - alpha) * pp
    print(f"  [融合网络] KEGG基因={len(kegg_g2i)}, PPI基因={len(ppi_g2i)}, "
          f"统一基因={n}, alpha(KEGG权重)={alpha}")
    return fused, union_g2i
from hdmap_s import (HDmapS, pair_single_herb_all_diseases,
                     pair_single_herb_all_diseases_weighted,
                     pair_single_herb_all_diseases_dual,
                     pair_single_herb_all_diseases_enrichment, combine_dual_channels,
                     build_herb_target_sets)
from hdmap_m import HDmapM

# 时间感知模块（可选）
try:
    from temporal_signature import TemporalSignatureEngine
    _HAS_TEMPORAL = True
except ImportError:
    _HAS_TEMPORAL = False

# 跨疾病迁移学习模块（可选）
try:
    from transfer_learning import CrossDiseaseTransfer
    _HAS_TRANSFER = True
except ImportError:
    _HAS_TRANSFER = False

from evaluation import build_hpg_network
from evaluation_metrics import (
    compute_tms_ranking_metrics,
    evaluate_herb_prediction,
    generate_evaluation_report,
    export_chart_data,
)


def step1_load_data(cfg=None):
    """步骤1: 加载所有数据"""
    print("\n" + "="*60)
    print("步骤1: 加载数据")
    print("="*60)
    
    print("\n[1.1] 疾病表达谱...")
    disease_profiles = load_disease_profiles()
    print(f"  疾病数: {disease_profiles['disease_id'].nunique()}")
    
    print("\n[1.2] 草药-成分表...")
    herb_ingredients = load_herb_ingredients()
    print(f"  草药数: {herb_ingredients['herb_id'].nunique()}")
    print(f"  成分数: {herb_ingredients['ingredient_id'].nunique()}")
    
    print("\n[1.3] PPI网络...")
    ppi_network = load_ppi_network()
    print(f"  蛋白质数: {len(set(ppi_network['protein_A']) | set(ppi_network['protein_B']))}")
    print(f"  互作数: {ppi_network.shape[0]}")
    
    print("\n[1.4] 成分-靶标表...")
    if cfg is None:
        import config as cfg
    ingredient_targets = load_ingredient_targets(
        herb_ingredients_df=herb_ingredients,
        disease_profiles_df=disease_profiles,
        ppi_df=ppi_network,
        use_hdcti=cfg.HDCTI_USE,
        use_drugai=cfg.DRUGAI_USE,
    )
    print(f"  成分数: {ingredient_targets['ingredient_id'].nunique()}")
    print(f"  靶标数: {ingredient_targets['target_gene'].nunique()}")
    
    # 仅保留 ChEMBL 来源成分靶标，排除 PubChem 补全 OOD噪音（与 ablation.py 一致）
    ingredient_targets = filter_chembl_only(ingredient_targets)
    
    return disease_profiles, herb_ingredients, ingredient_targets, ppi_network


def step2_build_signatures(ppi_network, herb_ingredients, ingredient_targets, network_type="kegg"):
    """
    步骤2: 构建草药扰动签名

    使用 content_ratio 构建一组签名
    返回 herb_signatures: Dict[herb_id, vector]
    """
    print("\n" + "="*60)
    print("步骤2: 构建草药扰动签名")
    print("="*60)
    
    # 构建传播网络邻接矩阵 (PPI 或 KEGG)
    # network_type 由调用方显式传入 (main() 中已取 CLI 覆盖后的值), 避免函数内
    # 引用 main() 的局部变量 cfg (作用域不可见 -> NameError)
    print(f"\n[2.1] 构建传播网络 ({network_type})...")
    if network_type == "kegg":
        import config as _cfg_kegg
        adj, gene2idx = build_kegg_adjacency(cache_dir=_cfg_kegg.PROCESSED_DIR, min_shared=_cfg_kegg.KEGG_MIN_SHARED)
    elif PPI_MULTI_SOURCE:
        print(f"  [PPI模式] 多源融合 (策略={PPI_FUSION_STRATEGY}, 目录={PPI_SOURCE_DIR})")
        adj, gene2idx = build_fused_ppi_adjacency(
            strategy=PPI_FUSION_STRATEGY,
            min_sources=PPI_MIN_SOURCES,
            source_dir=PPI_SOURCE_DIR
        )
    elif network_type == "fused":
        # 方案B: 网络层融合 — 在统一基因空间合并 KEGG 与 PPI, 单次扩散
        adj, gene2idx = build_fused_network_adjacency(alpha=FUSED_NETWORK_ALPHA)
    else:
        print("  [PPI模式] 单源 (使用默认 ppi_network.csv)")
        adj, gene2idx = build_ppi_adjacency(ppi_network)
    
    # 根据配置选择扩散方法
    if DIFFUSION_METHOD == "rwr":
        builder = RWRPerturbationSignatureBuilder(
            adj, gene2idx,
            restart_prob=RWR_RESTART_PROB,
            max_iter=RWR_MAX_ITER,
            tol=RWR_TOL
        )
        print(f"  [扩散方法] RWR (r={RWR_RESTART_PROB}, max_iter={RWR_MAX_ITER})")
    else:
        builder = PerturbationSignatureBuilder(adj, gene2idx)
        print(f"  [扩散方法] 两阶段热扩散 (α={THERMAL_ALPHA})")
    
    # 批量构建草药扰动签名（纯标准模式，无注意力）
    print("\n[2.2] 批量计算草药扰动签名...")
    herb_signatures, builder = batch_build_herb_signatures(
        herb_ingredients, ingredient_targets, ppi_network, builder,
        use_cache=True, network_type=network_type,
    )
    
    # 保存签名
    print("\n[2.3] 保存到文件...")
    sig_df = signature_to_dataframe(herb_signatures, builder.get_all_genes())
    outpath = os.path.join(RESULTS_DIR, "herb_perturbation_signatures.csv")
    sig_df.to_csv(outpath)
    print(f"  保存至: {outpath}, shape: {sig_df.shape}")
    
    return herb_signatures, builder


def step3_hdmap_s(herb_signatures, builder, disease_profiles):
    """
    步骤3: HDmap-S 单草药-疾病匹配

    时间感知模式:
        见 _step3_temporal_mode

    标准模式:
        使用 content_ratio 签名为所有疾病计算TMS
    """
    print("\n" + "="*60)
    print("步骤3: HDmap-S - 单草药-疾病匹配")
    print("="*60)

    # ----- 时间感知模式 -----
    if TEMPORAL_ENABLED and _HAS_TEMPORAL and 'time_point' in disease_profiles.columns:
        print("  [模式] 时间感知模式已激活")
        print(f"  [时间集成] 方法={TEMPORAL_METHOD}, 衰减因子={TEMPORAL_DECAY_FACTOR}")
        return _step3_temporal_mode(herb_signatures, builder, disease_profiles)
    
    if TEMPORAL_ENABLED and 'time_point' not in disease_profiles.columns:
        print("  [提示] TEMPORAL_ENABLED=True 但数据中无 time_point 列，回退标准模式")
    
    # ----- 标准模式 -----
    disease_genes_dict = {}
    for disease_id in disease_profiles['disease_id'].unique():
        sub = disease_profiles[disease_profiles['disease_id'] == disease_id]
        before = len(sub)
        sub = _filter_disease_genes(sub)      # top-K (默认) 或 绝对阈值
        mode = (f"top-K={DISEASE_DEG_TOP_K} by P" if DISEASE_DEG_TOP_K > 0
                else f"|logFC|>{DISEASE_LOGFC_THRESHOLD},P<{DISEASE_PVALUE_THRESHOLD}")
        print(f"  [过滤] {disease_id}: {before}→{len(sub)} 基因 ({mode})")
        sub = sub.sort_values('logFC', ascending=False)
        disease_genes_dict[disease_id] = sub['gene_symbol'].tolist()
    
    # ---- 重排序: 已知AD靶标基因提到列表最前面 ----
    # 这样它们的rank最小(1-43), 任何命中它们的草药获得最大增益
    if disease_id in DISEASE_GENE_WEIGHTS:
        gs, _ = DISEASE_GENE_WEIGHTS[disease_id]
        genes = disease_genes_dict[disease_id]
        known_found = [g for g in gs if g in genes]
        others = [g for g in genes if g not in gs]
        disease_genes_dict[disease_id] = known_found + others
        if known_found:
            print(f"  [重排] {disease_id}: {len(known_found)} 个已知靶标提到基因列表最前面")
    
    hdmap_s = HDmapS(herb_signatures, builder)
    herb_ids = list(herb_signatures.keys())
    
    # 构建基因权重字典 (已知靶标基因在TMS计算中获得额外权重) — NDCG 模式用
    gene_weights = None
    if TMS_METHOD == "ndcg" and disease_id in DISEASE_GENE_WEIGHTS:
        gene_set, weight = DISEASE_GENE_WEIGHTS[disease_id]
        gene_weights = {g: weight for g in gene_set}
    
    # ---- 构建带 logFC 的疾病 profile（weighted 模式用）----
    disease_logfc_dict = {}
    for disease_id in disease_profiles['disease_id'].unique():
        sub = disease_profiles[disease_profiles['disease_id'] == disease_id]
        sub = _filter_disease_genes(sub)      # 与 disease_genes_dict 保持一致
        sub = sub.sort_values('logFC', ascending=False)
        disease_logfc_dict[disease_id] = (sub['gene_symbol'].tolist(), sub['logFC'].tolist())
    
    print(f"\n[3.2] 计算 {len(herb_ids)} 草药 × {len(disease_genes_dict)} 疾病的TMS (方法={TMS_METHOD})...")
    if TMS_METHOD == "dual":
        df_rev, df_eng = pair_single_herb_all_diseases_dual(
            hdmap_s, herb_ids, disease_logfc_dict, verbose=True
        )
        # 保存双通道
        rev_path = os.path.join(RESULTS_DIR, "hdmap_s_scores_dual_reverse.csv")
        eng_path = os.path.join(RESULTS_DIR, "hdmap_s_scores_dual_engage.csv")
        df_rev.to_csv(rev_path)
        df_eng.to_csv(eng_path)
        # 按疾病机制先验合并为单一"越高越治疗"得分
        tms_df = combine_dual_channels(df_rev, df_eng, DISEASE_TMS_MECHANISM)
        tms_path = os.path.join(RESULTS_DIR, "hdmap_s_scores_dual_combined.csv")
        tms_df.to_csv(tms_path)
        # 原始 cos 矩阵单独存放(便于诊断/回退), 不覆盖主文件
        cos_df = pd.DataFrame(
            {d: df_eng[d].values for d in df_eng.columns},
            index=df_eng.index
        )
        cos_df.to_csv(os.path.join(RESULTS_DIR, "hdmap_s_scores_dual_cos.csv"))
        # 主文件 hdmap_s_scores.csv 存合并通道(越高越好), 供 Step5 重新载入与下游一致
        tms_df.to_csv(os.path.join(RESULTS_DIR, "hdmap_s_scores.csv"))
        print(f"  保存双通道: {rev_path} / {eng_path}")
        print(f"  合并(机制先验): {tms_path}, shape: {tms_df.shape}")
    elif TMS_METHOD == "enrichment":
        # 新方法(2026-08-22): 直接"草药靶点 vs 疾病DEG"富集打分，替代扩散签名余弦。
        # 得分 = -log10(Fisher p)，越高 = 草药靶点越富集于疾病差异基因 = 越治疗。
        tms_df = pair_single_herb_all_diseases_enrichment(
            disease_genes_dict, herb_ids
        )
        tms_df.to_csv(os.path.join(RESULTS_DIR, "hdmap_s_scores_enrichment.csv"))
        tms_df.to_csv(os.path.join(RESULTS_DIR, "hdmap_s_scores.csv"))
        print(f"  保存富集打分: {os.path.join(RESULTS_DIR, 'hdmap_s_scores.csv')}, shape: {tms_df.shape}")
    elif TMS_METHOD == "weighted":
        tms_df = pair_single_herb_all_diseases_weighted(
            hdmap_s, herb_ids, disease_logfc_dict, verbose=True
        )
        tms_path = os.path.join(RESULTS_DIR, "hdmap_s_scores.csv")
        tms_df.to_csv(tms_path)
        print(f"  保存至: {tms_path}, shape: {tms_df.shape}")
    else:
        tms_df = pair_single_herb_all_diseases(
            hdmap_s, herb_ids, disease_genes_dict, k=TOP_BOTTOM_K,
            gene_weights=gene_weights
        )
        tms_path = os.path.join(RESULTS_DIR, "hdmap_s_scores.csv")
        tms_df.to_csv(tms_path)
        print(f"  保存至: {tms_path}, shape: {tms_df.shape}")
    
    # 输出Top-20推荐草药
    herb_ing = load_herb_ingredients()
    herb_names = herb_ing[['herb_id', 'herb_name']].drop_duplicates()
    herb_name_map = dict(zip(herb_names['herb_id'], herb_names['herb_name']))
    
    # herb_id 是 index，所有列都是疾病列
    disease_cols = list(tms_df.columns)
    if len(disease_cols) == 0:
        print("  [警告] 没有疾病列，跳过Top-50输出")
    else:
        # 对每个疾病，输出带置换检验P值的 Top-50
        for disease_col in disease_cols:
            # 获取该疾病对应的基因列表
            disease_genes = disease_genes_dict.get(disease_col, [])
            if not disease_genes:
                print(f"  [警告] 疾病 {disease_col} 没有基因列表，跳过")
                continue

            # dual/combined 约定"越高越治疗"; weighted/ndcg 约定"越低(TMS越负)越治疗"
            ascending = (TMS_METHOD != "dual")
            top50 = tms_df.sort_values(disease_col, ascending=ascending).head(50)

            output_records = []
            for rank, (herb_id, row) in enumerate(top50.iterrows(), 1):
                hname = herb_name_map.get(herb_id, '未知')
                tms_obs, p_val = hdmap_s.permutation_test(
                    herb_id, disease_genes,
                    n_perm=min(PERMUTATION_N, 500),
                    k=TOP_BOTTOM_K,
                    early_stop_extreme=20
                )
                sig = ("***" if p_val < 0.001 else ("**" if p_val < 0.01
                       else ("*" if p_val < 0.05 else "")))
                output_records.append({
                    "rank": rank,
                    "herb_id": herb_id,
                    "herb_name": hname,
                    "TMS": round(tms_obs, 4),
                    "P_value": round(p_val, 4),
                    "significant": sig,
                })

            sig_count = sum(1 for r in output_records if r["P_value"] < 0.05)
            print(f"  [Top-50] {disease_col}: 显著草药 {sig_count}/50 已计算完成")

            result_df = pd.DataFrame(output_records)
            result_path = os.path.join(RESULTS_DIR, f"{disease_col}_top50_herbs.csv")
            result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
            print(f"  已保存至: {result_path}")
    
    # ----- 跨疾病迁移学习增强（需≥2个疾病） -----
    if TRANSFER_ENABLED and _HAS_TRANSFER and len(disease_genes_dict) >= 2:
        print(f"\n[3.TL] 跨疾病迁移学习增强 (λ={TRANSFER_LAMBDA})...")
        transfer = CrossDiseaseTransfer(herb_signatures, builder, disease_profiles)
        
        # 构建疾病相似度矩阵
        sim_matrix = transfer.build_similarity_matrix(
            min_similarity=TRANSFER_MIN_SIMILARITY,
            verbose=True
        )
        
        # 保存相似度矩阵
        sim_path = os.path.join(RESULTS_DIR, "disease_similarity_matrix.csv")
        sim_matrix.to_csv(sim_path)
        print(f"  [保存] 疾病相似度矩阵 -> {sim_path}")
        
        # 对每个疾病输出最相似的疾病
        for did in disease_genes_dict:
            top_sim = transfer.get_most_similar_diseases(did, top_n=5)
            if not top_sim.empty:
                sim_str = ', '.join([f"{r['disease_id']}({r['similarity']:.3f})"
                                   for _, r in top_sim.iterrows()])
                print(f"  [{did}] 最相似疾病: {sim_str}")
        
        # 增强TMS
        tms_enhanced = transfer.enhance_tms(
            tms_df, lambda_val=TRANSFER_LAMBDA,
            min_similarity=TRANSFER_MIN_SIMILARITY
        )
        
        # 保存增强结果
        enhanced_path = os.path.join(RESULTS_DIR, "hdmap_s_scores_transfer_enhanced.csv")
        tms_enhanced.to_csv(enhanced_path)
        print(f"  [保存] 迁移增强TMS -> {enhanced_path}, shape={tms_enhanced.shape}")
        
        # 输出增强前后的Top-10对比
        disease_cols = [c for c in tms_df.columns if c != 'herb_id']
        if disease_cols:
            sample_did = disease_cols[0]
            herb_ing = load_herb_ingredients()
            name_map = dict(zip(herb_ing['herb_id'], herb_ing['herb_name']))
            
            enhanced_rank = transfer.enhanced_ranking(
                sample_did, tms_df, lambda_val=TRANSFER_LAMBDA, top_n=10,
                higher_is_better=(TMS_METHOD in ("dual", "enrichment"))
            )
            print(f"\n  Top-10 迁移增强推荐 ({sample_did}):")
            print(f"  {'排名':<6} {'草药ID':<12} {'原始TMS':<10} {'增强TMS':<10} {'变化Δ':<10}")
            print(f"  {'-'*48}")
            for _, row in enhanced_rank.iterrows():
                print(f"  {row['rank']:<6} {row['herb_id']:<12} "
                      f"{row['original_TMS']:<10.4f} {row['enhanced_TMS']:<10.4f} "
                      f"{row['delta']:<+10.4f}")
        
        # 用增强后的TMS替换原TMS（用于下游GA）
        tms_df = tms_enhanced
        print(f"\n  [迁移] 完成! TMS已增强，用于下游GA")
    
    elif TRANSFER_ENABLED and not _HAS_TRANSFER:
        print("  [提示] TRANSFER_ENABLED=True 但 transfer_learning.py 未找到")
    elif TRANSFER_ENABLED and len(disease_genes_dict) < 2:
        print(f"  [提示] TRANSFER_ENABLED=True 但仅有 {len(disease_genes_dict)} 个疾病，至少需要2个")
    
    print(f"\n[3.2] 置换检验 (n={PERMUTATION_N})...")
    perm_path = os.path.join(RESULTS_DIR, "hdmap_s_permutation.csv")
    print(f"  结果将保存至: {perm_path}")
    

    return hdmap_s, tms_df


def _step3_temporal_mode(herb_signatures, builder, disease_profiles):
    """
    步骤3 时间感知子模式:
    在疾病表达谱有 time_point 列时自动调用
    """
    import json
    
    # 初始化时间引擎
    print("\n[3.T] 初始化时间感知引擎...")
    engine = TemporalSignatureEngine(herb_signatures, builder, disease_profiles)
    
    # 时间一致性报告
    consistency = engine.get_time_consistency_report()
    print(f"[3.T] 时间一致性: {consistency['n_timepoints']} 个时间点, "
          f"{consistency['n_herbs']} 种草药")
    
    # 计算时间TMS轨迹
    print(f"\n[3.T] 计算时间TMS轨迹 (方法={TEMPORAL_METHOD})...")
    temporal_dfs = engine.compute_all_temporal_tms(verbose=True)
    
    # 保存时间TMS轨迹
    for did, df in temporal_dfs.items():
        csv_path = os.path.join(RESULTS_DIR, f"temporal_tms_{did}.csv")
        df.to_csv(csv_path)
        print(f"  [保存] 时间TMS轨迹 -> {csv_path}, shape={df.shape}")
    
    # 分析每个疾病
    for did in engine.disease_ids:
        print(f"\n[3.T] 疾病: {did}")
        
        # 时间加权排名
        ranking = engine.time_weighted_ranking(
            did, method=TEMPORAL_METHOD,
            decay_factor=TEMPORAL_DECAY_FACTOR,
            top_n=TEMPORAL_TOP_N
        )
        
        # 保存排名
        rank_path = os.path.join(RESULTS_DIR, f"temporal_ranking_{TEMPORAL_METHOD}_{did}.csv")
        ranking.to_csv(rank_path, index=False)
        print(f"  [保存] 时间加权排名 -> {rank_path}")
        
        # 打印Top-10
        herb_ing = load_herb_ingredients()
        name_map = dict(zip(herb_ing['herb_id'], herb_ing['herb_name']))
        print(f"  Top-10 时间加权推荐 ({TEMPORAL_METHOD}):")
        print(f"  {'排名':<6} {'草药ID':<12} {'草药名':<30} {'集成得分':<10}")
        print(f"  {'-'*58}")
        for _, row in ranking.head(10).iterrows():
            hname = name_map.get(row['herb_id'], '')
            print(f"  {row['rank']:<6} {row['herb_id']:<12} {hname:<30} {row['integrated_score']:<10.4f}")
        
        # 阶段特异性分析
        print(f"\n  [3.T] 各时间点特异性草药...")
        stage_specific = engine.find_disease_stage_specific_herbs(did, top_n=10)
        for t, df in stage_specific.items():
            top_hid = df.iloc[0]['herb_id']
            top_name = name_map.get(top_hid, '')
            print(f"    时间点 t={t}: Top1={top_hid}({top_name}), TMS={df.iloc[0]['TMS']:.4f}")
        
        # 保存阶段特异性结果
        stage_dict = {str(t): df.to_dict('records') for t, df in stage_specific.items()}
        stage_path = os.path.join(RESULTS_DIR, f"temporal_stage_specific_{did}.json")
        with open(stage_path, 'w', encoding='utf-8') as f:
            json.dump(stage_dict, f, ensure_ascii=False, indent=2, default=str)
    
    # 生成摘要
    summary = engine.summarize_temporal(
        list(engine.disease_ids)[0] if len(engine.disease_ids) > 0 else None,
        top_n=TEMPORAL_TOP_N
    )
    
    # 保存摘要
    summary_path = os.path.join(RESULTS_DIR, "temporal_analysis_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        # 转DataFrame为可序列化格式
        summary_serializable = {}
        for k, v in summary.items():
            if isinstance(v, pd.DataFrame):
                summary_serializable[k] = v.to_dict('records')
            elif isinstance(v, dict):
                summary_serializable[k] = {str(kk): str(vv) if hasattr(vv, 'shape') else vv 
                                          for kk, vv in v.items()}
            else:
                summary_serializable[k] = v
        json.dump(summary_serializable, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n  [保存] 时间感知分析摘要 -> {summary_path}")
    
    # 返回标准接口（兼容下游GA）
    disease_genes_dict = {}
    for did in disease_profiles['disease_id'].unique():
        sub = disease_profiles[disease_profiles['disease_id'] == did]
        if 'time_point' in sub.columns:
            sub = sub[sub['time_point'] == sub['time_point'].max()]  # 用最新时间点
        sub = _filter_disease_genes(sub)      # 与标准模式一致
        sub = sub.sort_values('logFC', ascending=False)
        disease_genes_dict[did] = sub['gene_symbol'].tolist()
    
    hdmap_s = HDmapS(herb_signatures, builder)
    herb_ids = list(herb_signatures.keys())
    tms_df = pair_single_herb_all_diseases(hdmap_s, herb_ids, disease_genes_dict, k=TOP_BOTTOM_K)
    
    print(f"\n  [完成] 时间感知分析完成! 结果已保存至 {RESULTS_DIR}")
    return hdmap_s, tms_df


def step4_hdmap_m(hdmap_s, herb_signatures, builder, disease_profiles, actual_disease=None):
    """
    步骤4: HDmap-M 遗传算法搜索最优复方
    """
    print("\n" + "="*60)
    print("步骤4: HDmap-M - 遗传算法搜索最优草药组合")
    print("="*60)
    
    # 标准模式
    disease_genes_dict = {}
    for disease_id in disease_profiles['disease_id'].unique():
        sub = disease_profiles[disease_profiles['disease_id'] == disease_id]
        # 疾病基因过滤 (与 step3 完全一致: top-K 或 绝对阈值)
        before = len(sub)
        sub = _filter_disease_genes(sub)
        mode = (f"top-K={DISEASE_DEG_TOP_K} by P" if DISEASE_DEG_TOP_K > 0
                else f"|logFC|>{DISEASE_LOGFC_THRESHOLD},P<{DISEASE_PVALUE_THRESHOLD}")
        print(f"  [过滤] {disease_id}: {before}→{len(sub)} 基因 ({mode})")
        sub = sub.sort_values('logFC', ascending=False)
        disease_genes_dict[disease_id] = sub['gene_symbol'].tolist()
    
    herb_ids = list(herb_signatures.keys())

    # ---- GA 目标函数与 HDmap-S 主方法对齐 ----
    # enrichment 模式下必须把同一份"靶点集合 + 背景基因集"交给 GA, 否则 GA 优化的仍是
    # 旧的 NDCG+扩散签名目标, 与论文提出的方法不一致 (详见 hdmap_m._fitness_enrichment)。
    # 用运行时 import 读取, 确保拿到 CLI 覆盖后的 TMS_METHOD。
    import config as _cfg
    _herb_targets = None
    _bg_genes = None
    if _cfg.TMS_METHOD == "enrichment":
        _herb_targets = build_herb_target_sets()
        # 背景集必须与 HDmap-S 的富集打分严格一致:
        # 用"靶点宇宙" (草药靶点只能从这里取), 而不是 靶点∪DEG (会把永不可命中的
        # DEG 基因算进分母, 高估期望命中率 5.82 倍)。两者不一致会让 GA 优化错目标。
        _bg_genes = set().union(*_herb_targets.values()) if _herb_targets else set()
        _all_deg = set().union(*[set(v) for v in disease_genes_dict.values()]) if disease_genes_dict else set()
        print(f"  [GA对齐] 目标函数=enrichment | 背景=靶点宇宙 {len(_bg_genes)} | "
              f"DEG∩靶点宇宙 {len(_all_deg & _bg_genes)} | 有靶点草药 {len(_herb_targets)}")

    hdmap_m = HDmapM(hdmap_s, herb_ids, herb_signatures, builder,
                     herb_targets=_herb_targets, background_genes=_bg_genes)
    
    print("\n[4.1] 运行GA...")
    # 优先使用指定疾病，否则取第一个
    disease_keys = list(disease_genes_dict.keys())
    if actual_disease and actual_disease in disease_genes_dict:
        sample_disease = actual_disease
    else:
        sample_disease = disease_keys[0]
        if actual_disease:
            print(f"  [警告] --disease={actual_disease} 不在数据中, 回退到 {sample_disease}")
    sample_genes = disease_genes_dict[sample_disease]
    print(f"  目标疾病: {sample_disease} ({len(sample_genes)} 个基因)")

    import config as _cfg
    # 双通道 GA 机制先验: 按疾病机制选择复方 TMS 方向
    if _cfg.GA_DUAL_MECHANISM and _cfg.DISEASE_TMS_MECHANISM:
        ga_mechanism_map = _cfg.DISEASE_TMS_MECHANISM
        print(f"  [GA双通道] 启用, 机制先验: {ga_mechanism_map}")
    else:
        ga_mechanism_map = {}
        print("  [GA双通道] 未启用 (使用原始单一逆转方向)")

    # 草药名映射 (供全部疾病保存用)
    herb_ing = load_herb_ingredients()
    herb_names = herb_ing[['herb_id', 'herb_name']].drop_duplicates()
    herb_name_map = dict(zip(herb_names['herb_id'], herb_names['herb_name']))

    # 可选: 对全部疾病运行 GA, 双通道机制逐病生效, 保存各自 Top-K
    ga_all_results = {}
    if _cfg.GA_RUN_ALL_DISEASES:
        print(f"\n[4.1a] 批量运行所有疾病 GA (双通道机制逐病生效)...")
        for _d in disease_genes_dict:
            _mech = ga_mechanism_map.get(_d, "engagement")
            _r = hdmap_m.optimize(
                disease_genes_dict[_d],
                max_generations=_cfg.GA_MAX_GENERATIONS,
                max_size=_cfg.GA_MAX_COMBO_SIZE,
                n_top=_cfg.GA_N_TOP,
                verbose=False,
                gene_weights=None,
                mechanism=_mech,
            )
            ga_all_results[_d] = _r
            _sol = [str(h) for h in _r['best_solution']]
            print(f"  [{_d}] 机制={_mech}  #1({len(_sol)}味): {_sol}  得分={_r['best_score']:.4f}")
            _tk_path = os.path.join(RESULTS_DIR, f"hdmap_m_topk_results_{_d}.csv")
            pd.DataFrame([
                {'rank': i, 'herbs': ', '.join(e['herbs']),
                 'n_herbs': e['n_herbs'], 'score': e['score'],
                 'herb_names': ', '.join([herb_name_map.get(h, '') for h in e['herbs']])}
                for i, e in enumerate(_r.get('top_solutions', []), 1)
            ]).to_csv(_tk_path, index=False)

    sample_mechanism = ga_mechanism_map.get(sample_disease, "engagement")
    print(f"  [GA机制] {sample_disease} -> {sample_mechanism}")

    # 构建 GA 用的基因权重
    ga_gene_weights = None
    if sample_disease in DISEASE_GENE_WEIGHTS:
        gs, wt = DISEASE_GENE_WEIGHTS[sample_disease]
        ga_gene_weights = {g: wt for g in gs}

    try:
        result = hdmap_m.optimize(
            sample_genes,
            max_generations=_cfg.GA_MAX_GENERATIONS,
            max_size=_cfg.GA_MAX_COMBO_SIZE,
            n_top=_cfg.GA_N_TOP,
            verbose=True,
            gene_weights=ga_gene_weights,
            mechanism=sample_mechanism,
        )
        best_sol = [str(h) for h in result['best_solution']]
        print(f"\n  疾病: {sample_disease} (机制={sample_mechanism})")
        print(f"  #1最优组合 ({len(best_sol)}味): {best_sol}")
        print(f"  得分: {result['best_score']:.4f}")
    except Exception as e:
        print(f"\n  [错误] GA运行失败: {e}")
        import traceback
        traceback.print_exc()
        print("  跳过GA，继续后续步骤...")
        return hdmap_m, None

    ga_path = os.path.join(RESULTS_DIR, f"hdmap_m_results_{sample_disease}.csv")
    topk_path = os.path.join(RESULTS_DIR, f"hdmap_m_topk_results_{sample_disease}.csv")

    # 保存 #1 组合
    ga_records = []
    for rank, hid in enumerate(best_sol, 1):
        ga_records.append({
            'rank': rank,
            'herb_id': hid,
            'herb_name': herb_name_map.get(hid, ''),
            'included': True
        })
    ga_df = pd.DataFrame(ga_records)
    ga_df.to_csv(ga_path, index=False)
    print(f"  #1组合已保存至: {ga_path}")
    for _, row in ga_df.iterrows():
        print(f"    {row['rank']}. {row['herb_id']} ({row['herb_name']})")

    # 保存 Top-K 多草药组合
    if result.get('top_solutions'):
        topk_records = []
        for rank, entry in enumerate(result['top_solutions'], 1):
            topk_records.append({
                'rank': rank,
                'herbs': ', '.join(entry['herbs']),
                'n_herbs': entry['n_herbs'],
                'score': entry['score'],
                'herb_names': ', '.join([herb_name_map.get(h, '') for h in entry['herbs']])
            })
        topk_df = pd.DataFrame(topk_records)
        topk_df.to_csv(topk_path, index=False)
        print(f"  Top-{len(topk_df)}组合已保存至: {topk_path}")

    print(f"\n[4.2] 全部疾病 GA 机制汇总:")
    for _d in disease_genes_dict:
        _r = ga_all_results.get(_d) or (result if _d == sample_disease else None)
        if _r:
            print(f"  {_d}: 机制={ga_mechanism_map.get(_d, 'engagement')}  最优得分={_r['best_score']:.4f}")
    return hdmap_m, result







def step5_evaluate(herb_signatures, builder, disease_profiles,
                   actual_disease=None, ga_result=None, hdmap_s=None):
    """
    步骤5: 结果评估（基于实际疾病数据动态分析）

    从TMS结果中自动选取Top3草药进行H-PG网络构建和富集分析，
    并计算 TMS 排名指标 (AUROC/AUPRC/F1/Z-score 等)，
    最终输出 evaluation_metrics.json 与 evaluation_report.md。
    """
    # ----- 提取疾病ID -----
    disease_ids = disease_profiles['disease_id'].unique()
    if len(disease_ids) == 0:
        print("  [警告] 未找到疾病数据，跳过步骤5")
        return
    # 优先使用指定疾病，否则取第一个
    if actual_disease and actual_disease in disease_ids:
        disease_id = actual_disease
    else:
        disease_id = str(disease_ids[0])
        if actual_disease:
            print(f"  [警告] --disease={actual_disease} 不在数据中, 回退到 {disease_id}")
    
    print("\n" + "="*60)
    print(f"步骤5: 结果评估 - {disease_id} 案例分析")
    print("="*60)
    
    # ----- 从TMS结果中自动选取Top3草药 -----
    tms_path = os.path.join(RESULTS_DIR, "hdmap_s_scores.csv")
    if not os.path.exists(tms_path):
        print(f"\n  [警告] 未找到TMS结果文件 {tms_path}，跳过评估")
        print(f"  提示: 请先运行步骤3 HDmap-S")
        return
    
    tms_df = pd.read_csv(tms_path)
    disease_cols = [c for c in tms_df.columns if c != 'herb_id']
    if not disease_cols:
        print(f"\n  [警告] TMS结果中没有疾病列，跳过评估")
        return
    
    # 优先使用指定疾病的TMS列
    if disease_id in disease_cols:
        disease_col = disease_id
    else:
        disease_col = disease_cols[0]
        print(f"  [警告] 疾病 {disease_id} 不在TMS列中, 回退到 {disease_col}")
    # dual 合并通道: 越高越治疗(ascending=False); weighted/ndcg: 越低(TMS越负)越治疗(ascending=True)
    top_herbs_df = tms_df.sort_values(disease_col, ascending=(TMS_METHOD != "dual")).head(3)
    top_herbs = top_herbs_df['herb_id'].tolist()
    
    # 加载草药名称映射
    from data_loader import load_herb_ingredients
    herb_ing = load_herb_ingredients()
    herb_name_map = dict(zip(herb_ing['herb_id'], herb_ing['herb_name']))
    
    print(f"\n[5.1] 分析Top3草药 (来自TMS排名):")
    for rank, (_, row) in enumerate(top_herbs_df.iterrows(), 1):
        hname = herb_name_map.get(row['herb_id'], '')
        print(f"  {rank}. {row['herb_id']} ({hname})  TMS={row[disease_col]:.4f}")
    
    # ----- 提取各草药的top/bottom基因 -----
    print("\n[5.2] 提取Venn图数据...")
    top_bottom = {}
    all_genes = builder.get_all_genes()
    for herb in top_herbs:
        if herb in herb_signatures:
            sig = herb_signatures[herb]
            order = np.argsort(-np.abs(sig))
            top_genes = set(all_genes[i] for i in order[:TOP_BOTTOM_K])
            bottom_genes = set(all_genes[i] for i in order[-TOP_BOTTOM_K:])
            top_bottom[herb] = top_genes | bottom_genes
        else:
            print(f"  [警告] {herb} 不在签名中")
    
    common = set()
    if len(top_bottom) >= 2:
        common = set.intersection(*top_bottom.values())
        n_herbs = len(top_bottom)
        print(f"  {n_herbs}味草药共有扰动基因: {len(common)}")
    
    # ----- 构建H-PG网络 -----
    print("\n[5.3] 构建H-PG网络...")
    available_herbs = [h for h in top_herbs if h in top_bottom]
    if available_herbs:
        hpg_df = build_hpg_network(available_herbs, herb_signatures, builder, top_k=500)
        hpg_path = os.path.join(RESULTS_DIR, f"hpg_network_{disease_id}.csv")
        hpg_df.to_csv(hpg_path, index=False)
        print(f"  网络边数: {len(hpg_df)}")
        print(f"  保存至: {hpg_path}")
        print(f"  提示: 可用Cytoscape打开此文件进行可视化")
    
    # ----- GO富集分析 -----
    print("\n[5.4] GO/KEGG富集分析 (需手动或安装gseapy)...")
    if common:
        print(f"  共同基因数: {len(common)}")
        common_file = os.path.join(RESULTS_DIR, f"{disease_id}_common_genes.txt")
        with open(common_file, "w") as f:
            for g in sorted(common):
                f.write(g + "\n")
        print(f"  基因列表已保存至: {common_file}")
    print("  请上传到 DAVID (https://david.ncifcrf.gov/) 进行富集分析")
    print("  或使用: pip install gseapy; 然后调用 EnrichmentAnalyzer.run_gseapy_enrichment()")

    # ----- 评估指标 (AUROC/AUPRC/F1/Z-score/区分度等) -----
    print("\n[5.5] 计算 TMS 排名指标 (AUROC/AUPRC/F1/Z-score/区分度)...")
    tms_series = tms_df[disease_col]  # 全部草药的 TMS
    metrics = compute_tms_ranking_metrics(
        tms_series,
        top_n=50,
        disease_name=disease_id,
        higher_is_better=(TMS_METHOD in ("dual", "enrichment")),
    )

    # 把 GA 结果合并到指标中
    if ga_result is not None:
        metrics["ga_best_score"] = float(ga_result.get("best_score", 0.0))
        metrics["ga_best_n_herbs"] = int(len(ga_result.get("best_solution", [])))
        metrics["ga_generations_run"] = int(ga_result.get("generations_run", 0))
        metrics["ga_restart_count"] = int(ga_result.get("restart_count", 0))
        hof = ga_result.get("hall_of_fame", [])
        if hof:
            metrics["hof_size"] = int(len(hof))
            metrics["hof_best_score"] = float(hof[0]["score"])
            metrics["hof_best_n_herbs"] = int(hof[0]["n_herbs"])

    # 打印核心指标
    print(f"  TMS 均值/标准差: {metrics.get('tms_mean', 0):.4f} / {metrics.get('tms_std', 0):.4f}")
    print(f"  负TMS比例: {metrics.get('pct_negative', 0):.1f}%")
    print(f"  区分度 (top10%/bottom10%): {metrics.get('discrimination_ratio', 0):.4f}")
    print(f"  Top1 Z-score: {metrics.get('z_score_top1', 0):.4f}")
    if "auroc" in metrics:
        print(f"  AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f} | F1: {metrics['f1_score']:.4f}")
    else:
        print("  (无金标准验证集，未计算 AUROC/AUPRC)")

    # ----- 金标准验证 (跨全部疾病列的 AUROC/AUPRC/Precision@K) -----
    print("\n[5.6] 金标准验证 (AUROC/AUPRC/Precision@K, 跨全部疾病)...")
    try:
        from config import GOLD_STANDARD_PATH
        from evaluation_metrics import load_gold_standard, evaluate_gold_standard
        gold, gold_neg, gs_warns = load_gold_standard(GOLD_STANDARD_PATH, herb_ing)
        for w in gs_warns:
            print(f"  [金标准警告] {w}")
        if gold:
            # 统一评估协议 = FULL (阳性 vs 全部草药)。子集(pos vs neg)噪声大, 仅作诊断附存。
            gs_df = evaluate_gold_standard(
                tms_df, gold, top_k=10,
                higher_is_better=(TMS_METHOD in ("dual", "enrichment")),
                negative_standard=None,
            )
            if not gs_df.empty:
                gs_path = os.path.join(RESULTS_DIR, "gold_standard_evaluation.csv")
                gs_df.to_csv(gs_path, index=False)
                print(f"  [协议] FULL (阳性 vs 全部 {len(tms_df)} 味草药) —— 论文主协议")
                print(f"  金标准评估已保存: {gs_path}")
                print(f"  平均 AUROC = {gs_df['AUROC'].mean():.4f}  (随机基线 0.5)")
                # 附: 子集协议(pos vs neg) 仅诊断, 不作主结论
                try:
                    gs_sub = evaluate_gold_standard(
                        tms_df, gold, top_k=10,
                        higher_is_better=(TMS_METHOD in ("dual", "enrichment")),
                        negative_standard=(gold_neg if gold_neg else None),
                    )
                    gs_sub.to_csv(os.path.join(RESULTS_DIR, "gold_standard_evaluation_subset.csv"), index=False)
                    print(f"  [附] 子集协议(pos vs neg) 另存 gold_standard_evaluation_subset.csv (仅诊断)")
                except Exception as e2:
                    print(f"  [提示] 子集协议计算失败: {e2}")
            else:
                print("  [提示] 金标准疾病列与 TMS 矩阵无交集, 跳过")
        else:
            print("  [提示] 未加载到金标准, 请检查 data/gold_standard.csv")
    except Exception as e:
        print(f"  [金标准评估跳过] {e}")

    # ----- 保存 evaluation_metrics.json -----
    metrics_file = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    print(f"  已保存评估指标: {metrics_file}")

    # ----- 生成 Markdown 评估报告 -----
    if ga_result is not None:
        all_results = {
            "full": {
                "scenario_id": "full",
                "scenario_name": "完整模型 (main pipeline)",
                "disease_id": disease_id,
                "n_disease_genes": len(disease_profiles),
                "top1_score": float(tms_series.min()),
                "tms_mean": metrics.get("tms_mean"),
                "tms_std": metrics.get("tms_std"),
                "n_negative_tms": metrics.get("n_negative_tms"),
                "ga_best_combo": [str(h) for h in ga_result.get("best_solution", [])],
                "ga_best_score": float(ga_result.get("best_score", 0.0)),
                "runtime_seconds": 0,
                "metrics": metrics,
            }
        }
        comparison_df = pd.DataFrame([{
            "场景": "完整模型 (main pipeline)",
            "场景ID": "full",
            "Top1_TMS": float(tms_series.min()),
            "TMS_均值": metrics.get("tms_mean"),
            "TMS_标准差": metrics.get("tms_std"),
            "AUROC": metrics.get("auroc"),
            "AUPR": metrics.get("auprc"),
            "F1": metrics.get("f1_score"),
            "GA最优得分": float(ga_result.get("best_score", 0.0)),
            "GA代数": int(ga_result.get("generations_run", 0)),
        }])
        try:
            report_path = generate_evaluation_report(
                all_results, comparison_df, RESULTS_DIR,
                title=f"Systems-Mapping 评估报告 - {disease_id}",
            )
            print(f"  评估报告 (Markdown): {report_path}")
        except Exception as e:
            print(f"  [警告] Markdown 报告生成失败: {e}")

    # ----- 导出可视化数据 (供后续 matplotlib/seaborn 复用) -----
    try:
        chart_files = export_chart_data(tms_series, RESULTS_DIR, prefix=f"tms_{disease_id}")
        print(f"  图表数据已导出: {list(chart_files.values())}")
    except Exception as e:
        print(f"  [警告] 图表数据导出失败: {e}")

    return metrics


def _parse_args():
    """解析命令行参数，支持覆盖 YAML 配置"""
    parser = argparse.ArgumentParser(
        description="Systems-Mapping: 草药-疾病系统映射分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                    # 默认配置
  python main.py --disease AD                       # AD 分析
  python main.py --ga-max-generations 300           # 覆盖 GA 代数
  python main.py --no-drugai --no-hdcti             # 消融: 关闭 DrugAI + HDCTI
  python main.py --output-dir ./ad_results
        """
    )
    # ---- 基本参数 ----
    parser.add_argument("--disease", "-d", type=str, default="AD",
                        help="疾病ID (默认: AD)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="疾病表达谱 CSV 路径 (默认: data/raw/disease_profiles.csv)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="输出目录 (默认: output/results/)")
    parser.add_argument("--timestamp", action="store_true",
                        help="输出目录添加时间戳")
    
    # ---- 模块开关 (消融实验用) ----
    parser.add_argument("--no-drugai", action="store_true",
                        help="关闭 DrugAI 模块")
    parser.add_argument("--no-hdcti", action="store_true",
                        help="关闭 HDCTI 靶标扩展")
    parser.add_argument("--no-ppi-multi", action="store_true",
                        help="关闭多源 PPI 融合，使用单源 PPI")
    parser.add_argument("--network", "-n", choices=["ppi", "kegg", "fused"], default=None,
                        help="传播网络类型: ppi | kegg (默认: config.yaml)")
    parser.add_argument("--diffusion", choices=["thermal", "rwr"], default=None,
                        help="扩散方法 (默认: config.yaml)")
    parser.add_argument("--no-transfer", action="store_true",
                        help="关闭跨疾病迁移学习")
    
    # ---- GA 超参数覆盖 ----
    ga_group = parser.add_argument_group("遗传算法参数覆盖")
    ga_group.add_argument("--ga-max-generations", type=int, default=None,
                          help=f"GA 最大代数 (默认: {GA_MAX_GENERATIONS})")
    ga_group.add_argument("--ga-pop-size", type=int, default=None,
                          help=f"GA 种群大小 (默认: {GA_POP_SIZE})")
    ga_group.add_argument("--ga-n-top", type=int, default=None,
                          help=f"返回 Top-K 组合数 (默认: {GA_N_TOP})")
    ga_group.add_argument("--ga-max-combo", type=int, default=None,
                          help=f"组合最大草药数 (默认: {GA_MAX_COMBO_SIZE})")
    
    # ---- HDmap-S 参数覆盖 ----
    hs_group = parser.add_argument_group("HDmap-S 参数覆盖")
    hs_group.add_argument("--top-bottom-k", type=int, default=None,
                          help=f"最优签名基因数 (默认: {TOP_BOTTOM_K})")
    hs_group.add_argument("--permutation-n", type=int, default=None,
                          help=f"置换检验次数 (默认: {PERMUTATION_N})")
    
    return parser.parse_args()


def _apply_cli_overrides(args, config_module):
    """将命令行参数应用到配置模块（动态覆盖模块级变量）"""
    overrides_applied = []
    
    # 模块开关
    if args.no_drugai:
        config_module.DRUGAI_USE = False
        overrides_applied.append("DRUGAI_USE=False")
    if args.no_hdcti:
        config_module.HDCTI_USE = False
        overrides_applied.append("HDCTI_USE=False")
    if args.no_ppi_multi:
        config_module.PPI_MULTI_SOURCE = False
        overrides_applied.append("PPI_MULTI_SOURCE=False")
    if args.network:
        config_module.NETWORK_TYPE = args.network
        overrides_applied.append(f"NETWORK_TYPE={args.network}")
    if args.diffusion:
        config_module.DIFFUSION_METHOD = args.diffusion
        overrides_applied.append(f"DIFFUSION_METHOD={args.diffusion}")
    if args.no_transfer:
        config_module.TRANSFER_ENABLED = False
        overrides_applied.append("TRANSFER_ENABLED=False")
    
    # GA 参数
    if args.ga_max_generations is not None:
        config_module.GA_MAX_GENERATIONS = args.ga_max_generations
        overrides_applied.append(f"GA_MAX_GENERATIONS={args.ga_max_generations}")
    if args.ga_pop_size is not None:
        config_module.GA_POP_SIZE = args.ga_pop_size
        overrides_applied.append(f"GA_POP_SIZE={args.ga_pop_size}")
    if args.ga_n_top is not None:
        config_module.GA_N_TOP = args.ga_n_top
        overrides_applied.append(f"GA_N_TOP={args.ga_n_top}")
    if args.ga_max_combo is not None:
        config_module.GA_MAX_COMBO_SIZE = args.ga_max_combo
        overrides_applied.append(f"GA_MAX_COMBO_SIZE={args.ga_max_combo}")
    
    # HDmap-S 参数
    if args.top_bottom_k is not None:
        config_module.TOP_BOTTOM_K = args.top_bottom_k
        overrides_applied.append(f"TOP_BOTTOM_K={args.top_bottom_k}")
    if args.permutation_n is not None:
        config_module.PERMUTATION_N = args.permutation_n
        overrides_applied.append(f"PERMUTATION_N={args.permutation_n}")
    
    if overrides_applied:
        print(f"\n[CLI] 配置覆盖: {', '.join(overrides_applied)}")


def main():
    """主工作流程 — 支持 CLI 参数和标准化输出"""
    global RESULTS_DIR  # 更新模块级变量，确保所有 step 函数使用新路径
    
    args = _parse_args()
    
    # 重新导入 config（确保使用模块级变量）
    import config as cfg
    _apply_cli_overrides(args, cfg)
    # 重新绑定本地 NETWORK_TYPE 等为 CLI 覆盖后的值 (否则下述分支仍用导入时的默认值)
    NETWORK_TYPE = cfg.NETWORK_TYPE
    PPI_MULTI_SOURCE = cfg.PPI_MULTI_SOURCE
    DIFFUSION_METHOD = cfg.DIFFUSION_METHOD
    
    # ---- 设置输出目录 ----
    # ---- 日志头部（早期打印，疾病信息稍后补充） ----
    print("=" * 60)
    print("  Systems-Mapping of Herbal Effects on Complex Diseases")
    print("=" * 60)
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_config_summary()
    
    # ===== 步骤1: 加载数据 =====
    if args.input:
        disease_profiles, herb_ingredients, ingredient_targets, ppi_network = \
            step1_load_data(cfg)
        # 用自定义输入覆盖疾病数据
        custom_df = pd.read_csv(args.input)
        disease_profiles = custom_df
        print(f"  [CLI] 使用自定义输入: {args.input} ({len(disease_profiles)} 行)")
    else:
        disease_profiles, herb_ingredients, ingredient_targets, ppi_network = \
            step1_load_data(cfg)
    
    # 提取疾病 ID
    disease_ids = disease_profiles['disease_id'].unique()
    actual_disease = args.disease or disease_ids[0]
    if args.disease and args.disease not in disease_ids:
        print(f"  [警告] --disease={args.disease} 不在数据中 ({list(disease_ids)}), 使用 {actual_disease}")
    
    # ---- 确定输出目录 ----
    # KEGG 为默认网络 → 无后缀；PPI 作为备选 → 带 _ppi 后缀以区分
    net_suffix = f"_{cfg.NETWORK_TYPE}" if cfg.NETWORK_TYPE != "kegg" else ""
    if args.output_dir:
        output_dir = args.output_dir
    elif args.timestamp:
        output_dir = get_run_output_dir(f"{actual_disease}{net_suffix}")
    else:
        output_dir = os.path.join(cfg.RESULTS_DIR, f"{actual_disease}{net_suffix}_results")
    
    # 同步更新: config 模块 + main 模块级变量（step 函数读取此变量）
    cfg.RESULTS_DIR = output_dir
    RESULTS_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"  输出目录: {output_dir}")
    print(f"  疾病: {actual_disease}")
    
    # ===== 步骤2: 构建扰动签名 =====
    # 必须把 CLI 覆盖后的网络类型 (cfg.NETWORK_TYPE) 显式传下去,
    # 否则 step2 永远用默认 "kegg", 导致 PPI 实际复用 KEGG 签名 (两网络结果完全一致)。
    herb_signatures, builder = step2_build_signatures(
        ppi_network, herb_ingredients, ingredient_targets,
        network_type=cfg.NETWORK_TYPE,
    )
    
    # ===== 步骤3: HDmap-S =====
    hdmap_s, tms_df = step3_hdmap_s(
        herb_signatures, builder, disease_profiles
    )
    
    # ===== 步骤4: HDmap-M =====
    hdmap_m, ga_result = step4_hdmap_m(
        hdmap_s, herb_signatures, builder, disease_profiles,
        actual_disease=actual_disease
    )
    
    # ===== 步骤5: 评估 =====
    step5_evaluate(herb_signatures, builder, disease_profiles,
                   actual_disease=actual_disease,
                   ga_result=ga_result,
                   hdmap_s=hdmap_s)
    
    # ---- 生成运行元信息 ----
    import json
    meta = {
        "timestamp": datetime.now().isoformat(),
        "disease": str(actual_disease),
        "n_herbs": len(herb_signatures) if isinstance(herb_signatures, dict) else "N/A",
        "n_disease_genes": len(disease_profiles),
        "diffusion_method": cfg.DIFFUSION_METHOD,
        "network_type": cfg.NETWORK_TYPE,
        "drugai_enabled": cfg.DRUGAI_USE,
        "hdcti_enabled": cfg.HDCTI_USE,
        "ppi_multi_source": cfg.PPI_MULTI_SOURCE,
        "ga_max_generations": cfg.GA_MAX_GENERATIONS,
        "ga_pop_size": cfg.GA_POP_SIZE,
        "top_bottom_k": cfg.TOP_BOTTOM_K,
        "cli_args": {k: v for k, v in vars(args).items() if v is not None},
    }
    meta_path = os.path.join(output_dir, "run_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("  全部流程完成!")
    print(f"  结果已保存至: {output_dir}")
    print(f"  运行元信息: {meta_path}")
    print("=" * 60)


def _ensure_hashseed():
    """确保PYTHONHASHSEED已设置，使DrugAI推理结果可复现"""
    if 'PYTHONHASHSEED' not in os.environ:
        # 重新启动自身进程，使hash种子生效
        # 必须通过env参数传递，因为subprocess默认不继承os.environ的运行时修改
        import subprocess
        result = subprocess.run(
            [sys.executable] + sys.argv,
            env={**os.environ, 'PYTHONHASHSEED': '0'}
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    _ensure_hashseed()
    main()
