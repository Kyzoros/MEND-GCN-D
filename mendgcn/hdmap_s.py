"""
HDmap-S: 单草药-疾病映射模块

论文方法:
1. 提取草药扰动签名的top 250 / bottom 250基因作为"最优签名"
2. 计算TMS (Total Matching Score):
   TMS(h,d,t) = (MS_top - MS_bottom) / 2
3. 置换检验评估显著性

优化:
  - 预提取所有草药的top/bottom基因（避免重复计算）
  - 置换检验使用提前终止策略
"""

import json
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy.stats import fisher_exact

import sys
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
from config import (TOP_BOTTOM_K, PERMUTATION_N, PROCESSED_DIR,
                     FILE_INGREDIENT_TARGETS_ENHANCED, FILE_HERB_INGREDIENTS_PROCESSED)
from perturbation_signature import PerturbationSignatureBuilder
import sys


class HDmapS:
    """
    Single Herb-Disease Mapping
    
    计算单个草药与疾病的匹配度得分
    
    改进: 在初始化时预提取所有草药的top/bottom基因集
    """
    
    def __init__(self, herb_signatures: Dict[str, np.ndarray],
                 builder: PerturbationSignatureBuilder):
        """
        Parameters
        ----------
        herb_signatures : Dict[herb_id, 扰动向量]
        builder         : 扰动签名构建器 (含top/bottom提取方法)
        """
        self.herb_signatures = herb_signatures
        self.builder = builder
        self.all_genes = builder.get_all_genes()
        self.gene_set = set(self.all_genes)
        
        # 预提取所有草药的top/bottom基因集（避免compute_tms中重复计算）
        self._herb_top_genes: Dict[str, List[str]] = {}
        self._herb_bottom_genes: Dict[str, List[str]] = {}
        self._precompute_top_bottom(k=TOP_BOTTOM_K)
    
    def _precompute_top_bottom(self, k: int = TOP_BOTTOM_K):
        """预提取所有草药的top/bottom基因"""
        for hid, sig in self.herb_signatures.items():
            top, bottom = self.builder.extract_top_bottom_genes(sig, k=k)
            self._herb_top_genes[hid] = top
            self._herb_bottom_genes[hid] = bottom
    
    def compute_tms(self, herb_id: str, disease_genes: List[str],
                    disease_logfc: Optional[List[float]] = None,
                    k: int = TOP_BOTTOM_K,
                    gene_weights: Optional[dict] = None) -> float:
        """
        计算 TMS (Total Matching Score) — NDCG 版本 + 基因权重

        Parameters
        ----------
        herb_id       : 草药ID
        disease_genes : 疾病差异基因列表 (按logFC降序排列)
        disease_logfc : 疾病基因的logFC值 (若为None则仅用排名)
        k             : top/bottom基因数
        gene_weights  : {gene_symbol: weight_multiplier}
                        已知靶标基因命中时 gain × weight
                        使银杏叶等有效草药获得更低(更好)的TMS

        Returns
        -------
        TMS : float, [-0.5, 0.5]区间, 负值 = 治疗方向
        """
        if herb_id not in self.herb_signatures:
            return 0.0
        
        # 使用预提取的top/bottom基因
        top_genes = self._herb_top_genes.get(herb_id)
        bottom_genes = self._herb_bottom_genes.get(herb_id)
        if top_genes is None or bottom_genes is None:
            return 0.0
        
        # 疾病基因排位字典: gene -> rank (1-based)
        disease_rank = {g: i + 1 for i, g in enumerate(disease_genes)
                        if g in self.gene_set}
        n_disease = len(disease_rank)
        
        if n_disease == 0:
            return 0.0
        
        # NDCG 增益: rank 越前增益越大, 已知靶标基因额外加权
        ranks_top = [disease_rank.get(g, n_disease + 1) for g in top_genes]
        ranks_bottom = [disease_rank.get(g, n_disease + 1) for g in bottom_genes]
        
        r_top = np.asarray(ranks_top, dtype=np.float64)
        r_bottom = np.asarray(ranks_bottom, dtype=np.float64)
        gain_top = np.where(r_top <= n_disease, 1.0 / np.log2(r_top + 1.0), 0.0)
        gain_bottom = np.where(r_bottom <= n_disease, 1.0 / np.log2(r_bottom + 1.0), 0.0)
        
        # 已知靶标基因加权: 命中关键基因 → gain 放大
        if gene_weights:
            weight_default = gene_weights.get("_default", 1.0)
            w_top = np.array([gene_weights.get(g, weight_default) for g in top_genes])
            w_bottom = np.array([gene_weights.get(g, weight_default) for g in bottom_genes])
            gain_top = gain_top * w_top
            gain_bottom = gain_bottom * w_bottom
        
        # IDCG: k 个基因理想情况 = rank 1..k
        k_eff = min(k, n_disease)
        idcg_k = float(np.sum(1.0 / np.log2(np.arange(1.0, k_eff + 1.0) + 1.0)))
        if idcg_k < 1e-12:
            idcg_k = 1.0
        
        ndcg_top = float(np.sum(gain_top)) / idcg_k
        ndcg_bottom = float(np.sum(gain_bottom)) / idcg_k
        
        tms = (ndcg_top - ndcg_bottom) / 2.0
        return tms
    
    def compute_tms_with_weights(self, herb_id: str,
                                  disease_genes: List[str],
                                  disease_logfc: List[float],
                                  k: int = TOP_BOTTOM_K) -> float:
        """
        使用logFC加权的TMS计算 (论文原始方法更准确的实现)
        
        论文描述:
        "the HDmap-S score was calculated based on the matching degree 
         between the perturbation signatures of the herbs at the top/bottom 
         of the disease-pathological state profiles"
        
        即: 检查草药top基因是否在疾病列表顶部, bottom基因是否在底部
        """
        if herb_id not in self.herb_signatures:
            return 0.0
        
        signature = self.herb_signatures[herb_id]
        top_genes, bottom_genes = self.builder.extract_top_bottom_genes(
            signature, k=k
        )
        
        # 疾病基因集 (取前1000个差异最大的基因)
        n_disease = min(len(disease_genes), 1000)
        disease_top = set(disease_genes[:n_disease])
        disease_bottom = set(disease_genes[-n_disease:])
        
        # 计算交集
        top_match = len(set(top_genes) & disease_top)
        bottom_match = len(set(bottom_genes) & disease_bottom)
        top_mis = len(set(top_genes) & disease_bottom)
        bottom_mis = len(set(bottom_genes) & disease_top)
        
        ms_top = (top_match - top_mis) / (2 * k)
        ms_bottom = (bottom_match - bottom_mis) / (2 * k)
        
        tms = (ms_top - ms_bottom) / 2.0
        
        return tms
    
    # ================================================================
    # 加权点积 TMS (2025-07-29)
    # 根本改进: 不用硬阈值Top500，所有基因按扰动强度×疾病效应加权参与
    # TMS = cos(sig, logFC_vector) ∈ [-1,1], 负值=治疗方向
    # ================================================================
    
    def _precompute_herb_norm_vectors(self):
        """预计算所有草药的L2归一化签名向量（用于加权TMS的矩阵乘法）"""
        n_herbs = len(self.herb_signatures)
        n_genes = len(self.all_genes)
        H = np.zeros((n_herbs, n_genes), dtype=np.float64)
        self._herb_order = []
        for i, (hid, sig) in enumerate(self.herb_signatures.items()):
            H[i, :] = sig
            self._herb_order.append(hid)
        # L2 逐行归一化
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self._H_norm = H / norms  # shape: (n_herbs, n_genes)
    
    def _build_disease_vector(self, disease_genes: List[str],
                               disease_logfc: List[float],
                               gene2idx: Dict[str, int],
                               n_genes: int) -> np.ndarray:
        """将疾病基因列表+logFC转换为与签名空间对齐的向量"""
        v = np.zeros(n_genes, dtype=np.float64)
        for gene, lfc in zip(disease_genes, disease_logfc):
            if gene in gene2idx:
                v[gene2idx[gene]] = lfc
        # L2 归一化: 使不同疾病的TMS量级可比
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            v /= v_norm
        return v
    
    def compute_tms_weighted(self, herb_id: str,
                              disease_vector: np.ndarray) -> float:
        """
        加权点积 TMS = cos(herb_signature, disease_logFC_vector)
        
        - 全基因参与，无硬阈值截断
        - 疾病特异基因按 |logFC| 自动获得高权重
        - 方向: sig×logFC > 0 = 加重, < 0 = 治疗
        - TMS ∈ [-1, 1], 负值 = 治疗方向（向后兼容）
        """
        if herb_id not in self.herb_signatures:
            return 0.0
        sig = self.herb_signatures[herb_id]
        s_norm = np.linalg.norm(sig)
        if s_norm < 1e-12:
            return 0.0
        # cosine = dot(sig, v_d) / (||sig|| × ||v_d||)
        # v_d is already L2-normalized by _build_disease_vector
        return float(np.dot(sig, disease_vector) / s_norm)
    
    def compute_tms_weighted_batch(self, disease_vector: np.ndarray) -> np.ndarray:
        """
        批量计算所有草药对同一疾病的加权TMS（矩阵乘法，极快）
        
        Returns: (n_herbs,) array of TMS values (cos), order matching self._herb_order
        """
        if not hasattr(self, '_H_norm'):
            self._precompute_herb_norm_vectors()
        # H_norm: (n_herbs, n_genes) × v_d: (n_genes,) → (n_herbs,)
        return self._H_norm @ disease_vector

    def compute_tms_dual_batch(self, disease_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        双通道TMS批量计算（核心方法修复: 解决"逆转 vs 嵌网"机制异构）。

        机制异构背景:
          - 逆转型(antagonist)草药 (如雷公藤/白芍): 把疾病基因反向压 → cos<0 → 负TMS=治疗
          - 嵌网型(engagement)草药 (如补益药): 参与并扶持疾病网络 → cos>0 → 正TMS=治疗
          单符号标量TMS无法同时表达两者, 故拆成两个方向独立的通道。

        返回两个 (n_herbs,) 通道, 均约定 **越高 = 越像治疗**(便于下游统一排序/进GA):
          - tms_reverse : -cos(sig, v_d)   ; 越高 = 拮抗(逆转)越强
          - tms_engage  : +cos(sig, v_d)   ; 越高 = 嵌网(增益)越强
        """
        if not hasattr(self, '_H_norm'):
            self._precompute_herb_norm_vectors()
        cos = self._H_norm @ disease_vector
        return -cos, cos

    def permutation_test(self, herb_id: str, disease_genes: List[str],
                         n_perm: int = PERMUTATION_N,
                         k: int = TOP_BOTTOM_K,
                         early_stop_extreme: int = 20) -> Tuple[float, float]:
        """
        置换检验评估TMS显著性（带提前终止）
        
        Parameters
        ----------
        herb_id             : 草药ID
        disease_genes       : 疾病差异基因列表
        n_perm              : 置换次数 (论文: 10,000)
        k                   : top/bottom基因数
        early_stop_extreme  : 提前终止阈值，当极端计数达到此值时提前终止
        
        Returns
        -------
        tms_obs  : 观测到的TMS值
        p_value  : 显著性P值
        """
        if herb_id not in self.herb_signatures:
            return 0.0, 1.0
        
        n = len(disease_genes)
        if n < 2 * k:
            return 0.0, 1.0
        
        # 观测值
        tms_obs = self.compute_tms(herb_id, disease_genes, k=k)

        # 置换检验: 打乱疾病基因顺序（带提前终止）
        # 单边下尾检验: TMS越负=治疗方向越强，只统计比观测值更负的随机TMS
        count_extreme = 0
        n_actual = n_perm

        # 预提取top/bottom基因（加速置换循环）
        top_genes = self._herb_top_genes.get(herb_id, [])
        bottom_genes = self._herb_bottom_genes.get(herb_id, [])
        # 过滤出在PPI网络中的基因，用连续索引（0-based）
        filtered_genes = [g for g in disease_genes if g in self.gene_set]
        disease_idx = {g: i for i, g in enumerate(filtered_genes)}
        n_disease = len(filtered_genes)
        n_disease_arr = float(n_disease)

        if n_disease == 0:
            return 0.0, 1.0

        for i in range(n_perm):
            # 打乱疾病基因顺序（仅打乱索引）
            perm_indices = np.random.permutation(n_disease)
            perm_rank = {g: perm_indices[idx] + 1 for g, idx in disease_idx.items()}

            # 快速计算TMS
            ranks_top = [perm_rank.get(g, n_disease + 1) for g in top_genes]
            ranks_bottom = [perm_rank.get(g, n_disease + 1) for g in bottom_genes]
            ms_top = 1.0 - np.mean([r / n_disease_arr for r in ranks_top])
            ms_bottom = 1.0 - np.mean([r / n_disease_arr for r in ranks_bottom])
            tms_perm = (ms_top - ms_bottom) / 2.0

            # 单边下尾: 只统计比观测值更负的随机TMS
            if tms_perm <= tms_obs:
                count_extreme += 1
                # 提前终止：如果极端值已经足够多，p值肯定不显著
                if count_extreme >= early_stop_extreme:
                    n_actual = i + 1
                    break

        p_value = count_extreme / n_actual

        return tms_obs, p_value


def pair_single_herb_all_diseases(
    hdmap_s: HDmapS,
    herb_ids: List[str],
    disease_profiles: Dict[str, List[str]],
    k: int = TOP_BOTTOM_K,
    verbose: bool = True,
    gene_weights: Optional[dict] = None
) -> pd.DataFrame:
    """
    计算所有草药-疾病对的TMS得分
    
    Returns
    -------
    DataFrame: index=disease_id, columns=herb_id, values=TMS
    """
    result = {}
    total = len(herb_ids) * len(disease_profiles)
    count = 0
    
    for disease_id, genes in disease_profiles.items():
        scores = []
        for herb_id in herb_ids:
            tms = hdmap_s.compute_tms(herb_id, genes, k=k, gene_weights=gene_weights)
            scores.append(tms)
            count += 1
        result[disease_id] = scores
        
        if verbose and len(herb_ids) > 0 and count % max(1, len(herb_ids) * 10) == 0:
            print(f"[HDmap-S] 进度: {count}/{total}")
    
    df = pd.DataFrame(result, index=herb_ids)
    df.index.name = "herb_id"
    return df


def pair_single_herb_all_diseases_weighted(
    hdmap_s: HDmapS,
    herb_ids: List[str],
    disease_profiles: Dict[str, Tuple[List[str], List[float]]],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    加权点积版本: 计算所有草药-疾病对的TMS得分
    
    Parameters
    ----------
    disease_profiles : Dict[disease_id, (gene_list, logfc_list)]
        gene_list 和 logfc_list 按logFC降序排列（同原始接口）
    
    Returns
    -------
    DataFrame: index=herb_id, columns=disease_id, values=TMS
    """
    gene2idx = {g: i for i, g in enumerate(hdmap_s.all_genes)}
    n_genes = len(hdmap_s.all_genes)
    
    # 预计算所有草药的归一化签名矩阵
    hdmap_s._precompute_herb_norm_vectors()
    herb_order = hdmap_s._herb_order
    herb_idx_map = {h: i for i, h in enumerate(herb_order)}
    
    result = {}
    for disease_id, (gene_list, logfc_list) in disease_profiles.items():
        if verbose:
            print(f"[HDmap-S weighted] {disease_id}: building disease vector...")
        v_d = hdmap_s._build_disease_vector(gene_list, logfc_list, gene2idx, n_genes)
        
        # 矩阵乘法: (n_herbs, n_genes) @ (n_genes,) → (n_herbs,)
        tms_all = hdmap_s.compute_tms_weighted_batch(v_d)
        
        # 按 herb_ids 顺序取出 (仅计算请求的herb)
        scores = [float(tms_all[herb_idx_map[h]]) if h in herb_idx_map else 0.0
                  for h in herb_ids]
        result[disease_id] = scores
    
    df = pd.DataFrame(result, index=herb_ids)
    df.index.name = "herb_id"
    return df


def pair_single_herb_all_diseases_dual(
    hdmap_s: HDmapS,
    herb_ids: List[str],
    disease_profiles: Dict[str, Tuple[List[str], List[float]]],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    双通道TMS: 计算所有草药-疾病对的 逆转/嵌网 两路得分。

    Parameters
    ----------
    disease_profiles : Dict[disease_id, (gene_list, logfc_list)]

    Returns
    -------
    (df_reverse, df_engage):
        df_reverse : index=herb_id, columns=disease_id, 值=-cos (越高=拮抗越强)
        df_engage  : index=herb_id, columns=disease_id, 值=+cos (越高=嵌网越强)
        两个通道均约定"越高=越治疗"。
    """
    gene2idx = {g: i for i, g in enumerate(hdmap_s.all_genes)}
    n_genes = len(hdmap_s.all_genes)

    hdmap_s._precompute_herb_norm_vectors()
    herb_order = hdmap_s._herb_order
    herb_idx_map = {h: i for i, h in enumerate(herb_order)}

    rev_result: Dict[str, List[float]] = {}
    eng_result: Dict[str, List[float]] = {}
    for disease_id, (gene_list, logfc_list) in disease_profiles.items():
        if verbose:
            print(f"[HDmap-S dual] {disease_id}: building disease vector...")
        v_d = hdmap_s._build_disease_vector(gene_list, logfc_list, gene2idx, n_genes)
        cos = hdmap_s.compute_tms_weighted_batch(v_d)              # (n_herbs,)
        tms_rev = -cos                                             # 越高=拮抗
        tms_eng = cos                                              # 越高=嵌网
        rev_result[disease_id] = [
            float(tms_rev[herb_idx_map[h]]) if h in herb_idx_map else 0.0
            for h in herb_ids
        ]
        eng_result[disease_id] = [
            float(tms_eng[herb_idx_map[h]]) if h in herb_idx_map else 0.0
            for h in herb_ids
        ]

    df_rev = pd.DataFrame(rev_result, index=herb_ids)
    df_rev.index.name = "herb_id"
    df_eng = pd.DataFrame(eng_result, index=herb_ids)
    df_eng.index.name = "herb_id"
    return df_rev, df_eng


# ============================================================================
# 靶点基因符号规范化
# ============================================================================
# 【2026-09-02 修复】ingredient_targets_enhanced.csv 的 target_gene 列
# **基因符号与蛋白质全名混用**: 同一个靶点会同时以 "PTGS2" 和
# "Prostaglandin G/H synthase 2" 两种写法出现, 被当成两个不同的基因。
# 1222 个唯一靶点中 823 个(67%)是蛋白质全名, 占全部记录的 72.1%
# (133,325 / 184,848)。而疾病 DEG 用的是**基因符号**, 因此这 72% 的记录
# 在富集分析时永远匹配不上、被静默丢弃 —— 富集打分实际只用到不到三成靶点数据。
# 修复方式: 用项目自带的映射表把蛋白质全名/别名统一转成基因符号。
_GENE_SYMBOL_MAP_CACHE: Optional[Dict[str, str]] = None

# 基因符号的粗略特征: 短且不含空格 (蛋白质全名几乎都含空格)
_GENE_SYMBOL_RE = re.compile(r'^[A-Za-z0-9\-_\.]{2,15}$')


def load_gene_symbol_map(processed_dir: str = None) -> Dict[str, str]:
    """
    载入 蛋白质全名/别名 -> 基因符号 的映射表 (多来源合并, 小写键, 结果缓存)。

    来源 (均在 data/processed/ 下, 项目自带):
      gene_alias.json            76,984 条   (主映射)
      mygene_mapping.json        14,287 条
      tcmsp_target_mapping.json   1,677 条
    """
    global _GENE_SYMBOL_MAP_CACHE
    if _GENE_SYMBOL_MAP_CACHE is not None:
        return _GENE_SYMBOL_MAP_CACHE
    base = processed_dir or PROCESSED_DIR
    merged: Dict[str, str] = {}
    for fname in ("gene_alias.json", "mygene_mapping.json", "tcmsp_target_mapping.json"):
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                kk, vv = str(k).strip().lower(), str(v).strip()
                if kk and vv and vv.lower() != "nan":
                    merged.setdefault(kk, vv)
    _GENE_SYMBOL_MAP_CACHE = merged
    return merged


def normalize_target_gene(name, gene_symbol_map: Dict[str, str],
                          known_symbols: Optional[set] = None) -> Optional[str]:
    """
    把单个靶点标识规范为基因符号; 无法规范且明显不是符号的返回 None(丢弃)。

    判定顺序:
      1. 空值 / NaN                -> None
      2. 已是已知基因符号          -> 原样返回
      3. 命中映射表 (忽略大小写)   -> 返回映射到的基因符号
      4. 形如基因符号 (短且无空格) -> 原样返回 (可能是映射表未收录的合法符号)
      5. 其余 (蛋白质全名但无映射) -> None
         (必须丢弃: 保留它们只会虚增靶点数 K, 却永远匹配不上 DEG, 从而稀释富集信号)
    """
    s = str(name).strip()
    if not s or s.lower() == "nan":
        return None
    if known_symbols and s in known_symbols:
        return s
    hit = gene_symbol_map.get(s.lower())
    if hit:
        return hit
    if " " not in s and _GENE_SYMBOL_RE.match(s):
        return s
    return None


def build_herb_target_sets(
    ingredient_targets_path: str = None,
    herb_ingredients_path: str = None,
) -> Dict[str, set]:
    """
    构建 草药 -> 靶点基因集合 (已统一规范为基因符号)。

    与 pair_single_herb_all_diseases_enrichment 口径完全一致:
      成分-靶点表按 ingredient_id 聚合 -> 草药经 herb_ingredients 汇总其所有成分的靶点。
    注意: 富集打分实际只用到"靶点基因集合"(原实现 tg = set(tv.keys())), 置信度权重
      并未参与打分, 故此处直接返回集合。抽出此函数供 HDmap-S 单药打分与 HDmap-M(GA)
      复方打分共用, 确保两者永远是同一份数据、同一套口径, 不会悄悄跑偏。

    Returns
    -------
    Dict[herb_id, Set[gene_symbol]]
    """
    if ingredient_targets_path is None:
        ingredient_targets_path = os.path.join(PROCESSED_DIR, FILE_INGREDIENT_TARGETS_ENHANCED)
    if herb_ingredients_path is None:
        herb_ingredients_path = os.path.join(PROCESSED_DIR, FILE_HERB_INGREDIENTS_PROCESSED)

    it = pd.read_csv(ingredient_targets_path)
    gene_symbol_map = load_gene_symbol_map()
    known_symbols = set(gene_symbol_map.values())

    ing_targets = defaultdict(set)
    n_total = 0
    n_dropped = 0
    for _, r in it.iterrows():
        n_total += 1
        g = normalize_target_gene(r['target_gene'], gene_symbol_map, known_symbols)
        if g:
            ing_targets[int(r['ingredient_id'])].add(g)
        else:
            n_dropped += 1

    hi = pd.read_csv(herb_ingredients_path)
    herb_targets = defaultdict(set)
    for _, r in hi.iterrows():
        hid = str(r['herb_id']).strip()
        tg = ing_targets.get(int(r['ingredient_id']))
        if tg:
            herb_targets[hid] |= tg

    if n_total:
        print(f"  [靶点规范化] 记录 {n_total:,} 条, 无法规范被丢弃 {n_dropped:,} 条 "
              f"({n_dropped / n_total:.1%}); 唯一靶点基因 {len(set().union(*ing_targets.values())) if ing_targets else 0}")
    return dict(herb_targets)


def pair_single_herb_all_diseases_enrichment(
    disease_genes_dict: Dict[str, set],
    herb_ids: List[str],
    ingredient_targets_path: str = None,
    herb_ingredients_path: str = None,
) -> pd.DataFrame:
    """
    新方法(2026-08-22): 直接"草药靶点 vs 疾病DEG"富集打分，替代扩散签名余弦。

    做法:
      1. 由 herb_ingredients_processed + ingredient_targets_enhanced 构建
         每草药 -> {target_gene: 最大置信度权重}
      2. 对每病取 disease_genes_dict[d]（已按 logFC/P 过滤的 DEG 基因集）
      3. 对每个草药-疾病对，用 Fisher 精确检验计算草药靶点与疾病 DEG 的富集，
         得分 = -log10(p)（越高 = 草药靶点越富集于疾病差异基因 = 越治疗）
      4. 背景基因集 = 所有 DEG 基因 ∪ 所有草药靶点基因

    生物学依据: 网络医学(Barabasi)表明治疗药物靶点富集于疾病扰动基因模块，
      故"靶点富集于DEG = 治疗"(antagonist/逆转方向)，方向可由独立文献先验支撑，
      非测试集上循环论证。

    Returns
    -------
    pd.DataFrame: index=herb_id, columns=disease_id, 值=富集得分(越高越好)
    """
    if ingredient_targets_path is None:
        ingredient_targets_path = os.path.join(PROCESSED_DIR, FILE_INGREDIENT_TARGETS_ENHANCED)
    if herb_ingredients_path is None:
        herb_ingredients_path = os.path.join(PROCESSED_DIR, FILE_HERB_INGREDIENTS_PROCESSED)

    # 1. 草药 -> 靶点基因集合 (与 GA 复方打分共用 build_herb_target_sets, 保证口径一致)
    herb_targets = build_herb_target_sets(ingredient_targets_path, herb_ingredients_path)

    # 2. 背景基因集 (null universe)
    # 【2026-09-02 修正】背景必须是"靶点宇宙" —— 草药靶点只能从这个集合里取,
    # 所以零假设"靶点随机命中疾病基因"的期望命中率 = |DEG ∩ 靶点宇宙| / |靶点宇宙|。
    #
    # 旧实现用 "靶点基因 ∪ DEG基因" 作背景, 把 8,924 个**永不可命中**的 DEG 基因
    # (它们不是任何草药的靶点) 算进了分母, 使期望命中率被高估 5.82 倍
    # (0.1477 vs 正确值 0.0254)。后果: 随机草药实测命中率 0.036 本应是富集的
    # (相对正确 null 的 fold=1.42), 在旧 null 下却显示为"严重贫化"(fold=0.24);
    # 双侧 Fisher 测出的"显著性"实为"偏离错误期望"的程度, 而非真正的富集。
    all_target_genes = set().union(*herb_targets.values()) if herb_targets else set()
    bg = all_target_genes
    N_BG = len(bg)
    if N_BG == 0:
        raise ValueError("靶点宇宙为空 (|U|=0), 无法进行富集检验")

    all_deg_genes = set().union(*[set(v) for v in disease_genes_dict.values()]) if disease_genes_dict else set()
    print(f"  [富集背景] 靶点宇宙 |U|={N_BG} | 全部DEG={len(all_deg_genes)} | "
          f"可被草药命中的 DEG (DEG∩U)={len(all_deg_genes & bg)}")

    diseases = list(disease_genes_dict.keys())
    deg_sets = {d: set(g for g in disease_genes_dict[d] if g in bg) for d in diseases}
    for d in diseases:
        if not deg_sets[d]:
            print(f"  [警告] 疾病 {d} 的 DEG 与靶点宇宙无交集, 该病富集分将全为 0")

    result = {d: [] for d in diseases}
    for d in diseases:
        dg = deg_sets[d]
        for h in herb_ids:
            tg = herb_targets.get(h)
            if not tg:
                result[d].append(0.0)
                continue
            a = len(tg & dg)
            if a == 0:
                result[d].append(0.0)
                continue
            b = len(tg - dg)
            c = len(dg - tg)
            dd = N_BG - a - b - c
            if dd < 0:
                dd = 0
            _, p = fisher_exact([[a, b], [c, dd]])
            result[d].append(-np.log10(max(p, 1e-300)))

    df = pd.DataFrame(result, index=herb_ids)
    df.index.name = "herb_id"
    return df


def combine_dual_channels(
    df_reverse: pd.DataFrame,
    df_engage: pd.DataFrame,
    disease_mechanism: Dict[str, str],
) -> pd.DataFrame:
    """
    按疾病机制先验，从双通道中选每病的最终治疗潜力得分。

    约定(下游统一): 值越大 = 越像治疗。
      - 疾病机制=antagonist : 用 df_reverse (逆转, 负cos者得分高)
      - 疾病机制=engagement : 用 df_engage  (嵌网, 正cos者得分高)

    Parameters
    ----------
    disease_mechanism : {disease_id: 'antagonist' | 'engagement'}
                        (缺省疾病按 engagement 处理)

    Returns
    -------
    df_combined : index=herb_id, columns=disease_id, 值越大=越治疗
    """
    combined = pd.DataFrame(index=df_reverse.index)
    for d in df_reverse.columns:
        mech = str(disease_mechanism.get(d, "engagement")).strip().lower()
        if mech == "antagonist":
            combined[d] = df_reverse[d]
        else:
            combined[d] = df_engage[d]
    combined.index.name = "herb_id"
    return combined


def permutation_batch(
    hdmap_s: HDmapS,
    herb_ids: List[str],
    disease_profiles: Dict[str, List[str]],
    k: int = TOP_BOTTOM_K,
    n_perm: int = 1000
) -> pd.DataFrame:
    """
    批量置换检验
    (完整10000次很慢, 可用1000次近似)
    
    Returns
    -------
    DataFrame: index=disease_id, columns=[herb_id, TMS, P_value]
    """
    records = []
    for disease_id, genes in disease_profiles.items():
        for herb_id in herb_ids:
            tms, pval = hdmap_s.permutation_test(
                herb_id, genes, n_perm=n_perm, k=k
            )
            records.append({
                "disease_id": disease_id,
                "herb_id": herb_id,
                "TMS": tms,
                "P_value": pval
            })
    return pd.DataFrame(records)
