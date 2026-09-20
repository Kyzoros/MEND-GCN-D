"""
评估验证模块

实现论文中的:
1. 文献验证 (PubMed超几何分布检验)
2. Jaccard距离计算 (评估扰动签名的差异性)
3. GO/KEGG富集分析接口
4. H-PG网络构建
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from scipy.stats import hypergeom

from config import (
    TOP_BOTTOM_K, HPG_EDGE_THRESHOLD, HPG_TOP_K,
    ENRICHMENT_BACKGROUND_GENES, PUBMED_TOTAL_ARTICLES,
    JACCARD_GENE_SET_SIZES,
)


# ==================== 文献共现验证 ====================

def hypergeometric_literature_test(
    herb_name: str, disease_name: str,
    N_total: int = PUBMED_TOTAL_ARTICLES,    # PubMed总文献数
    K_disease: Optional[int] = None,  # 疾病相关文献数
    n_herb: Optional[int] = None,     # 草药相关文献数
    k_cooccur: Optional[int] = None   # 共现文献数
) -> float:
    """
    超几何分布检验: 评估草药-疾病共现的显著性
    
    论文公式:
    P(h,d) = 1 - Σ_{i=0}^{k-1} [C(K,i) * C(N-K, n-i)] / C(N,n)
    
    其中:
    N = PubMed总文献数
    K = 疾病d的文献数
    n = 草药h的文献数
    k = 草药h和疾病d共现的文献数
    
    Parameters
    ----------
    herb_name    : 草药名称
    disease_name : 疾病名称
    N_total      : 总文献数
    K_disease    : 疾病相关文献数
    n_herb       : 草药相关文献数
    k_cooccur    : 共现文献数
    
    Returns
    -------
    p_value : float, 显著时P<0.05
    """
    if K_disease is None or n_herb is None or k_cooccur is None:
        # 需要用户提供文献统计数
        raise ValueError(
            "需要提供疾病/草药/共现文献计数.\n"
            "可以使用RISmed/R包从PubMed获取, 见 fetch_pubmed_stats()"
        )
    
    # 超几何分布: 从N中抽n个, 抽到≥k个属于K类的概率
    # P(X ≥ k) = 1 - P(X ≤ k-1)
    p_value = hypergeom.sf(k_cooccur - 1, N_total, K_disease, n_herb)
    
    return p_value


def batch_literature_verification(
    herb_disease_pairs: List[Tuple[str, str, int, int, int]],
    N_total: int = 22188039
) -> pd.DataFrame:
    """
    批量文献验证
    
    Parameters
    ----------
    herb_disease_pairs : List of (herb, disease, K_disease, n_herb, k_cooccur)
    
    Returns
    -------
    DataFrame with P_values
    """
    records = []
    for herb, disease, K, n, k in herb_disease_pairs:
        pval = hypergeometric_literature_test(
            herb, disease, N_total, K, n, k
        )
        records.append({
            "herb": herb,
            "disease": disease,
            "K_disease": K,
            "n_herb": n,
            "k_cooccur": k,
            "P_value": pval
        })
    return pd.DataFrame(records)


# ==================== Jaccard距离分析 ====================

def jaccard_distance(set1: Set, set2: Set) -> float:
    """计算两个集合的Jaccard距离"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 1.0
    return 1.0 - intersection / union


def compute_herb_jaccard_distances(
    herb_top_genes: Dict[str, List[str]],
    herb_bottom_genes: Dict[str, List[str]],
    gene_set_sizes: List[int] = None
) -> pd.DataFrame:
    """
    计算草药间的Jaccard距离 (论文Figure 3)
    
    Parameters
    ----------
    herb_top_genes    : Dict[herb_id, top_genes]
    herb_bottom_genes : Dict[herb_id, bottom_genes]
    gene_set_sizes    : 测试的不同基因集大小 [200, 400, 600, 800, 1000, all]
    
    Returns
    -------
    DataFrame with Jaccard distances
    """
    if gene_set_sizes is None:
        gene_set_sizes = JACCARD_GENE_SET_SIZES
    
    herb_ids = list(herb_top_genes.keys())
    results = []
    
    for size in gene_set_sizes:
        distances = []
        for i, h1 in enumerate(herb_ids):
            top1 = set(herb_top_genes[h1][:size // 2])
            bottom1 = set(herb_bottom_genes[h1][:size // 2])
            for h2 in herb_ids[i + 1:]:
                top2 = set(herb_top_genes[h2][:size // 2])
                bottom2 = set(herb_bottom_genes[h2][:size // 2])
                
                set1 = top1 | bottom1
                set2 = top2 | bottom2
                
                dist = jaccard_distance(set1, set2)
                distances.append({
                    "herb1": h1, "herb2": h2,
                    "gene_set_size": size, "jaccard_dist": dist
                })
        results.extend(distances)
    
    return pd.DataFrame(results)


# ==================== GO/KEGG富集分析 ====================

class EnrichmentAnalyzer:
    """
    基因富集分析接口
    
    论文使用DAVID数据库 (https://david.ncifcrf.gov/)
    
    阈值: GO BP P-value < 0.01, FDR < 0.05
    """
    
    # 简化的背景数据库 (实际应使用全基因组背景)
    # 这里给出API用于拟合DAVID/BioServices
    BACKGROUND_GENES = ENRICHMENT_BACKGROUND_GENES  # 人类基因近似数
    
    def __init__(self, species: str = "human"):
        self.species = species
    
    def fisher_exact_test(self, query_genes: List[str],
                          pathway_genes: Set[str],
                          background: int = None) -> float:
        """
        Fisher精确检验
        """
        if background is None:
            background = self.BACKGROUND_GENES
        
        k = len(set(query_genes) & pathway_genes)
        n = len(query_genes)
        K = len(pathway_genes)
        N = background
        
        pval = hypergeom.sf(k - 1, N, K, n)
        return pval
    
    def run_david_enrichment(gene_list_file: str):
        """
        使用DAVID在线工具进行富集分析
        
        请手动操作:
        1. 打开 https://david.ncifcrf.gov/
        2. 上传基因列表
        3. 选择Identifier: OFFICIAL_GENE_SYMBOL
        4. 选择Species: Homo Sapiens
        5. 提交并下载结果
        
        Parameters
        ----------
        gene_list_file : str, 基因列表文件路径
        """
        raise NotImplementedError(
            "DAVID在线分析需要手动操作.\n"
            f"请将基因列表保存到 {gene_list_file} 后上传到 https://david.ncifcrf.gov/\n"
            "或者使用Python库: pip install gseapy\n"
            "用法: import gseapy as gp; gp.enrichr()"
        )
    
    @staticmethod
    def run_gseapy_enrichment(genes: List[str], gene_set: str = "KEGG_2021_Human"):
        """
        使用gseapy进行富集分析
        需要: pip install gseapy
        
        Parameters
        ----------
        genes    : 基因列表
        gene_set : 基因集名称 (如 "KEGG_2021_Human", "GO_Biological_Process_2021")
        
        Returns
        -------
        DataFrame of enrichment results
        """
        try:
            import gseapy as gp
        except ImportError:
            raise ImportError("请安装gseapy: pip install gseapy")
        
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=gene_set,
            organism="human",
            outdir=None  # 不保存文件
        )
        return enr.results


# ==================== H-PG网络构建 ====================

def build_hpg_network(
    herbs: List[str],
    herb_signatures: Dict[str, np.ndarray],
    builder,
    top_k: int = 500
) -> pd.DataFrame:
    """
    构建草药-扰动基因网络 (H-PG网络)
    
    论文 Figure 6C:
    - 节点: 草药 + 靶标基因
    - 边: 草药→基因 (边宽=扰动强度)
    - 节点大小: 扰动强度
    
    Parameters
    ----------
    herbs           : 草药组合列表
    herb_signatures : Dict[herb_id, perturbation vector]
    builder         : PerturbationSignatureBuilder
    top_k           : 取前k个扰动基因 (论文: 500)
    
    Returns
    -------
    DataFrame for Cytoscape import:
        Source, Target, Weight, Source_Type
    """
    all_genes = builder.get_all_genes()
    records = []
    
    # 获取组合扰动签名 (使用 builder.diffuse 自适应 RWR/热扩散)
    combo_sig = np.zeros(builder.n_proteins)
    for herb in herbs:
        if herb in herb_signatures:
            combo_sig += herb_signatures[herb]
    combo_sig = builder.diffuse(combo_sig)
    
    # 取top_k基因
    order = np.argsort(-np.abs(combo_sig))[:top_k]
    top_gene_indices = set(order.tolist())
    
    for herb in herbs:
        if herb not in herb_signatures:
            continue
        sig = herb_signatures[herb]
        for idx in top_gene_indices:
            weight = float(sig[idx])
            if abs(weight) > HPG_EDGE_THRESHOLD:
                records.append({
                    "Source": str(herb),
                    "Target": all_genes[idx],
                    "Weight": weight,
                    "Source_Type": "herb",
                    "Target_Type": "gene"
                })
    
    return pd.DataFrame(records)
