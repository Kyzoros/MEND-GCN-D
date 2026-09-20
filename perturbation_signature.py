"""
扰动签名构建模块
实现:
1. 初始扰动签名构建 (成分级 → 草药级)
2. 两阶段热扩散模型（默认方案）
3. RWR 重启随机游走扩散（增强方案）
4. 基因别名扩展: 解决不同数据库基因命名不一致问题
"""

import os
import json
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Union
from scipy import sparse

from config import THERMAL_ALPHA, TOP_BOTTOM_K, DIFFUSION_METHOD, RWR_RESTART_PROB, RWR_MAX_ITER, RWR_TOL, PROCESSED_DIR, PRESENCE_FLOOR
from data_loader import (
    build_ppi_adjacency, get_all_protein_list,
    load_herb_ingredients, load_ingredient_targets, load_ppi_network
)

# 尝试导入joblib用于并行加速
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


# ==================== 基因别名扩展 ====================

_gene_alias_dict: Optional[Dict[str, str]] = None


def _load_gene_alias_dict() -> Dict[str, str]:
    """
    加载基因别名字典，支持三跳映射链条：
    
      蛋白质全名 → gene symbol → ENSP ID → PPI索引
    
    不同数据库(ChEMBL/STRING/UniProt)使用不同命名体系：
      - ingredient_targets.csv: 蛋白质全名 (如 "Alcohol dehydrogenase 1B")
      - mygene_mapping.json:   蛋白质全名 → gene symbol (如 "ADH1B")
      - string_gene_mapping.csv: ENSP ID ↔ gene symbol (双向)
      - PPI network gene2idx:   ENSP ID → 矩阵索引
    
    此函数构建两阶段别名：
      1. 蛋白质全名 → gene symbol (mygene)
      2. gene symbol → ENSP ID (string反向映射)
    然后 PerturbationSignatureBuilder._get_protein_index() 
    会通过两阶段查找最终定位到 PPI 索引。
    """
    global _gene_alias_dict
    if _gene_alias_dict is not None:
        return _gene_alias_dict

    alias = {}

    # 1. mygene_mapping.json: 蛋白质全名 → gene symbol
    mygene_path = os.path.join(PROCESSED_DIR, "mygene_mapping.json")
    if os.path.exists(mygene_path):
        with open(mygene_path, 'r', encoding='utf-8') as f:
            mygene = json.load(f)
        for protein_name, gene_symbol in mygene.items():
            alias[protein_name] = gene_symbol
            # 小写变体
            alias[protein_name.lower()] = gene_symbol
        print(f"[别名] mygene_mapping: {len(mygene)} 条")

    # 2. string_gene_mapping.csv: 双向映射 ENSP ↔ gene symbol
    #    - ENSP → symbol: 用于调试和直接匹配
    #    - symbol → ENSP: 关键！将 gene symbol 桥接到 PPI 的 ENSP 索引
    string_path = os.path.join(PROCESSED_DIR, "string_gene_mapping.csv")
    symbol_to_ensp = {}  # symbol → ENSP_ID (反向映射，关键新增)
    if os.path.exists(string_path):
        string_df = pd.read_csv(string_path)
        for _, row in string_df.iterrows():
            ensp = str(row['ensp_id']).strip()
            symbol = str(row['gene_symbol']).strip()
            if ensp and symbol:
                alias[ensp] = symbol  # ENSP → symbol (原有)
                # 反向映射: symbol → ENSP (新增，支持三跳查询)
                # 一个symbol可能对应多个ENSP，取第一个（最常见的转录本）
                if symbol not in symbol_to_ensp:
                    symbol_to_ensp[symbol] = ensp
        # 将反向映射合并到别名字典：gene_symbol → ENSP_ID
        alias.update(symbol_to_ensp)
        print(f"[别名] string_gene_mapping: {len(string_df)} ENSP↔symbol, "
              f"反向映射 {len(symbol_to_ensp)} symbol→ENSP")

    _gene_alias_dict = alias
    print(f"[别名] 总映射数: {len(alias)} 条, "
          f"覆盖 {len(set(alias.values()))} 种目标名称")
    return alias


class PerturbationSignatureBuilder:
    """
    扰动签名构建器（基类，使用两阶段热扩散）
    
    工作流程:
    1. 构建PPI邻接矩阵 + 蛋白索引
    2. 为每个成分构建初始扰动向量 (+1/-1)
    3. 叠加成分得到草药级初始扰动
    4. 执行扩散得到最终扰动签名
    """
    
    # 子类可重写此属性来切换扩散方法
    DIFFUSION_METHOD = "thermal_2step"  # "thermal_2step" 或 "rwr"
    
    def __init__(self, ppi_adj, gene2idx: Dict[str, int]):
        """
        Parameters
        ----------
        ppi_adj : scipy.sparse matrix, PPI邻接矩阵 (n x n)
        gene2idx: Dict[str, int], 基因名→索引
        """
        self.adj = ppi_adj
        self.gene2idx = gene2idx
        self.n_proteins = ppi_adj.shape[0]
        self._all_genes = list(gene2idx.keys())
        
        # 加载基因别名字典（全局单例）
        self._alias_dict = _load_gene_alias_dict()
        
        # 预计算每个蛋白的度 (D_i)
        import scipy.sparse as sp
        if sp.issparse(ppi_adj):
            self.degrees = np.array(ppi_adj.sum(axis=1)).flatten().astype(np.float64)
        else:
            self.degrees = np.sum(ppi_adj, axis=1).astype(np.float64)
        self.degrees[self.degrees == 0] = 1  # 避免除零
        
        # ---- 预计算 PPI hub 掩码：度数 top 5% 标记为 True ----
        # hub 节点（如 ACTB、TP53 等）与几乎所有蛋白有交互，
        # 它们出现在任何草药的 top/bottom 中都无区分度，
        # extract_top_bottom_genes 中将自动跳过这些节点。
        degree_95 = np.percentile(self.degrees, 95)
        self._hub_mask = self.degrees >= degree_95
        n_hub = int(self._hub_mask.sum())
        
        # 预缓存
        self._cache: Dict[str, np.ndarray] = {}
    
    def _get_protein_index(self, gene: str) -> int:
        """
        获取基因在PPI邻接矩阵中的索引，支持多路径别名扩展。
        
        查询路径（按优先级）:
        1. 精确匹配: 基因名直接是 ENSP ID → gene2idx
        2. 直接 sym2ensp: 基因名是标准symbol → ENSP → gene2idx
        3. 蛋白质全名 → gene symbol → ENSP → gene2idx (三跳)
        4. 仍找不到返回 -1
        
        关键: ingredient_targets.csv 使用蛋白质全名(如"Alcohol dehydrogenase 1B"),
        PPI网络使用ENSP ID(如"ENSP00000306606")。mygene_mapping + string_gene_mapping
        组成的双层别名字典实现了三跳桥接。
        """
        # 路径1: 精确匹配 (ENSP ID → index)
        idx = self.gene2idx.get(gene, -1)
        if idx >= 0:
            return idx
        
        # 辅助函数: 从别名值查找PPI索引
        def _resolve_alias(val: str) -> int:
            """解析别名值: 可能是ENSP或gene symbol"""
            # 尝试直接作为ENSP查找
            idx = self.gene2idx.get(val, -1)
            if idx >= 0:
                return idx
            # 值可能是gene symbol, 查另一层映射: symbol → ENSP → index
            ensp = self._alias_dict.get(val)
            if ensp is not None:
                return self.gene2idx.get(ensp, -1)
            return -1
        
        # 路径2: 直接sym2ensp (增强数据常用gene symbol, 如"CXCL6")
        alias_val = self._alias_dict.get(gene)
        if alias_val is not None:
            idx = _resolve_alias(alias_val)
            if idx >= 0:
                return idx
        
        # 路径2b: 小写尝试
        alias_lower = self._alias_dict.get(gene.lower())
        if alias_lower is not None and alias_lower != alias_val:
            idx = _resolve_alias(alias_lower)
            if idx >= 0:
                return idx
        
        # 路径3: 蛋白质全名 → canonical name → 递归解析
        #   (例: "Alcohol dehydrogenase 1B" → "ADH1B" → "ENSP..." → PPI index)
        canonical = alias_val or alias_lower
        if canonical is not None and canonical != gene:
            idx = _resolve_alias(canonical)
            if idx >= 0:
                return idx
        
        return idx
    
    def _vectorize_initial(self, targets: Dict[str, int]) -> np.ndarray:
        """
        将成分的靶标字典转为初始扰动向量
        
        Parameters
        ----------
        targets : Dict[gene, action], action=+1(激活) or -1(抑制)
        
        Returns
        -------
        init_vec : np.ndarray, shape (n_proteins,)
        """
        vec = np.zeros(self.n_proteins, dtype=np.float64)
        for gene, action in targets.items():
            idx = self._get_protein_index(gene)
            if idx >= 0:
                vec[idx] = action  # +1 或 -1
        return vec
    
    # ==================== 扩散方法（核心可重写接口） ====================
    
    def diffuse(self, init_vec: np.ndarray) -> np.ndarray:
        """
        执行信号扩散：将初始扰动向量沿PPI网络传播
        
        基类实现为两阶段热扩散，子类可重写为RWR等。
        输出归一化到 [-1, +1]，保留正负号：
            +1 = 最强激活, -1 = 最强抑制
        
        Parameters
        ----------
        init_vec : np.ndarray, 初始扰动向量
        
        Returns
        -------
        np.ndarray, 归一化到 [-1, +1] 的扩散后扰动签名
        """
        result = self.full_thermal_diffusion(init_vec)
        # 归一化到 [-1, +1]，保留正负号
        abs_max = np.max(np.abs(result))
        if abs_max > 1e-12:
            result = result / abs_max
        return result
    
    def thermal_diffusion_step1(self, init_vec: np.ndarray) -> np.ndarray:
        """
        热扩散第一步 (论文公式):
            P_step1_i = Σ_j (α * c_j / D_j) * A_ij
        
        即每个蛋白的扰动 = 其邻居的扰动 / 邻居的度 之和
        """
        c_norm = init_vec / self.degrees  # c_j / D_j
        step1 = THERMAL_ALPHA * (self.adj @ c_norm)
        return step1
    
    def thermal_diffusion_step2(self, step1_vec: np.ndarray) -> np.ndarray:
        """
        热扩散第二步 (论文公式):
            P_step2_i = Σ_j (α * c'_j / D_j) * A_ij
        
        与第一步相同, 将第一步结果作为输入再做一次扩散
        """
        c_norm = step1_vec / self.degrees
        step2 = THERMAL_ALPHA * (self.adj @ c_norm)
        return step2
    
    def full_thermal_diffusion(self, init_vec: np.ndarray) -> np.ndarray:
        """
        完整两阶段热扩散
        返回最终扰动签名向量
        """
        step1 = self.thermal_diffusion_step1(init_vec)
        final = self.thermal_diffusion_step2(step1)
        return final
    
    # ==================== 签名构建方法 ====================
    
    def build_ingredient_signature(self, targets: Dict[str, int]) -> np.ndarray:
        """
        构建单个成分的扰动签名
        
        Parameters
        ----------
        targets : Dict[gene, action]
        
        Returns
        -------
        最终扰动向量 shape (n_proteins,)
        """
        init_vec = self._vectorize_initial(targets)
        return self.diffuse(init_vec)
    
    def get_initial_vector(self, targets: Dict[str, int]) -> np.ndarray:
        """
        仅获取成分的初始扰动向量（不做扩散），用于叠加后再扩散
        
        Parameters
        ----------
        targets : Dict[gene, action]
        
        Returns
        -------
        初始扰动向量 shape (n_proteins,)
        """
        return self._vectorize_initial(targets)
    
    def build_herb_signature(
        self, ingredient_initial_vecs: List[np.ndarray],
        content_ratios: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        构建草药的扰动签名（论文公式：叠加成分初始扰动→扩散）
        
        所有成分的初始扰动按含量权重叠加，再做一次完整扩散。
        
        Parameters
        ----------
        ingredient_initial_vecs : List[np.ndarray], 各成分的初始扰动向量（未经扩散）
        content_ratios        : List[float] 或 None, 各成分的含量权重
        
        Returns
        -------
        草药的最终扰动向量
        """
        if content_ratios is None:
            content_ratios = [1.0] * len(ingredient_initial_vecs)
        
        init_combined = np.zeros(self.n_proteins, dtype=np.float64)
        for vec, ratio in zip(ingredient_initial_vecs, content_ratios):
            init_combined += ratio * vec
        
        return self.diffuse(init_combined)
    
    def build_combination_signature(
        self, herb_signatures: List[np.ndarray]
    ) -> np.ndarray:
        """
        构建草药组合的扰动签名
        
        所有草药的扰动签名直接叠加后做扩散
        
        Parameters
        ----------
        herb_signatures : List[np.ndarray], 各草药的最终扰动向量
        
        Returns
        -------
        组合的最终扰动向量
        """
        init_combined = np.sum(herb_signatures, axis=0)
        return self.diffuse(init_combined)
    
    def extract_top_bottom_genes(
        self, signature: np.ndarray, k: int = TOP_BOTTOM_K
    ) -> Tuple[List[str], List[str]]:
        """
        提取扰动签名中正向/负向扰动最强的基因
        
        top_genes:  签名值最大的K个基因 → 草药对该基因的"激活"效应最强
        bottom_genes: 签名值最小的K个基因 → 草药对该基因的"抑制"效应最强

        自动跳过 PPI hub 节点（度数 top 5%），避免无区分度的高连接基因
        稀释 TMS 范围。hub 如 ACTB/TP53 与数千蛋白交互，无生物学特异性。

        Returns
        -------
        top_genes    : 正向扰动最强的k个基因（已过滤hub）
        bottom_genes : 负向扰动最强的k个基因（已过滤hub）
        """
        n = len(signature)
        k = min(k, n // 2)
        
        # 按签名值排序（升序）
        order = np.argsort(signature)
        
        # 过滤 hub 节点：取非 hub 中最极端的 k 个
        non_hub_order = [i for i in order if not self._hub_mask[i]]
        # 备选：如果非 hub 不够 k 个，fallback 到全部节点
        if len(non_hub_order) < k * 2:
            non_hub_order = list(order)
        
        bottom_idx = non_hub_order[:k]      # 最小的K个 → 抑制最强
        top_idx = non_hub_order[-k:]        # 最大的K个 → 激活最强
        
        top_genes = [self._all_genes[i] for i in top_idx]
        bottom_genes = [self._all_genes[i] for i in bottom_idx]
        
        return top_genes, bottom_genes
    
    def get_all_genes(self) -> List[str]:
        return self._all_genes


class RWRPerturbationSignatureBuilder(PerturbationSignatureBuilder):
    """
    基于RWR（重启随机游走）的扰动签名构建器
    
    使用RWR替代两阶段热扩散，优势：
    - 信号传播到稳态（不限跳数），捕获更远的生物学关联
    - 保留初始扰动信号（通过重启概率r），防止信号淹没
    - 每跳指数衰减，更符合生物学信号传播规律
    
    论文参考:
    - RWR in PPI networks: Kohler et al., "Walking the Interactome for 
      Prioritization of Candidate Disease Genes", AJHG 2008
    
    Parameters
    ----------
    restart_prob : float, 重启概率 (0~1)
        越大越保守（信号集中在初始节点附近），推荐 0.3~0.7
        论文常用值: 0.7（强重启，70%保留初始信号）
    
    max_iter : int, 最大迭代次数
        默认 100，通常 20-50 次即可收敛
    
    tol : float, 收敛阈值
        当两次迭代的 L1 范数差 < tol 时提前停止
    
    lazy_evaluate : bool
        是否使用惰性评估（仅在需要时计算转移矩阵）
    """
    
    DIFFUSION_METHOD = "rwr"
    
    def __init__(self, ppi_adj, gene2idx: Dict[str, int],
                 restart_prob: float = 0.85,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 lazy_evaluate: bool = True):
        super().__init__(ppi_adj, gene2idx)
        
        self.restart_prob = restart_prob
        self.max_iter = max_iter
        self.tol = tol
        
        # RWR转移矩阵 P = D^(-1) * A (列归一化)
        self._P = None
        if not lazy_evaluate:
            self._build_transition_matrix()
    
    def _build_transition_matrix(self):
        """构建列归一化的转移矩阵 P = A @ D^(-1)"""
        from scipy.sparse import diags, csr_matrix
        d_inv = diags(1.0 / self.degrees, format='csr')
        self._P = self.adj @ d_inv
        # 确保是CSR格式（加速矩阵乘法）
        if not isinstance(self._P, csr_matrix):
            self._P = csr_matrix(self._P)
    
    @property
    def P(self):
        """惰性获取转移矩阵"""
        if self._P is None:
            self._build_transition_matrix()
        return self._P
    
    def diffuse(self, init_vec: np.ndarray) -> np.ndarray:
        """
        RWR 重启随机游走扩散 — 分离正负方向传播
        
        问题: 传统RWR在无符号网络上传播±1混合信号，假设所有边正相关，
              违反生物网络中激活/抑制交替传播的实际机制。
        解决: 将初始信号拆分为纯正(激活)和纯负(抑制)两路，
              各自独立在无符号网络上传播，最后合并:
                result = RWR(pos_only) - RWR(neg_only)
              避免方向信号在网络中被错误抵消。
        
        迭代公式 (每路独立执行):
            p_pos^{t+1} = (1 - r) * P @ p_pos^t + r * p0_pos
            p_neg^{t+1} = (1 - r) * P @ p_neg^t + r * p0_neg
            result = p_pos - p_neg
        
        Parameters
        ----------
        init_vec : np.ndarray, 初始扰动向量 (+1=激活, -1=抑制)
        
        Returns
        -------
        np.ndarray, 稳态扰动向量 (+ = 激活效应, - = 抑制效应)
        """
        p0 = init_vec.astype(np.float64)
        if np.all(p0 == 0):
            return p0
        
        # 拆分为正（激活）和负（抑制）两个纯非负信号
        p0_pos = np.maximum(p0, 0.0)   # 保留激活信号
        p0_neg = np.maximum(-p0, 0.0)  # 保留抑制信号，取绝对值用于传播
        
        # 各自独立RWR传播
        p_pos = self._rwr_nonneg(p0_pos)
        p_neg = self._rwr_nonneg(p0_neg)
        
        # 合并: 激活 - 抑制，恢复有符号语义
        result = p_pos - p_neg
        # 与基类 diffuse 契约一致: 归一化到 [-1, +1],
        # 避免输出尺度随网络密度/重启概率漂移, 保证下游 cosine/阈值语义稳定
        abs_max = np.max(np.abs(result))
        if abs_max > 1e-12:
            result = result / abs_max
        return result
    
    def _rwr_nonneg(self, p0_abs: np.ndarray) -> np.ndarray:
        """RWR传播纯非负信号 (无方向冲突，纯衰减扩散)"""
        p = p0_abs.copy()
        r = self.restart_prob
        one_minus_r = 1.0 - r
        P = self.P
        
        for _ in range(self.max_iter):
            p_new = one_minus_r * (P @ p) + r * p0_abs
            diff = np.linalg.norm(p_new - p, ord=1)
            p = p_new
            if diff < self.tol:
                break
        return p
    
    def get_info(self) -> str:
        """返回RWR配置信息"""
        return (f"RWR(r={self.restart_prob}, max_iter={self.max_iter}, "
                f"tol={self.tol})")


# ==================== 批量构建接口 ====================

SIGNATURE_CACHE_FILE = None  # 由batch_build_herb_signatures设置

def _get_signature_cache_path(network_type=None):
    """获取签名缓存文件路径 (按传播网络分离, 避免 KEGG/PPI 互相复用错误签名)

    注意: 必须传入显式 network_type, 不能直接依赖 config.NETWORK_TYPE ——
    main.py 的 CLI 覆盖只重绑定了局部变量, 不会修改模块级 config.NETWORK_TYPE,
    否则 PPI 运行会错误地加载 KEGG 的签名缓存 (已踩坑: 两网络结果完全一致)。
    """
    from config import PROCESSED_DIR
    import config as _cfg
    import os
    nt = network_type or getattr(_cfg, "NETWORK_TYPE", "kegg")
    return os.path.join(PROCESSED_DIR, f"herb_signature_cache_{nt}.npz")

def _get_signature_meta_path(network_type=None):
    """获取签名元数据路径 (与缓存文件一一对应)"""
    from config import PROCESSED_DIR
    import config as _cfg
    import os
    nt = network_type or getattr(_cfg, "NETWORK_TYPE", "kegg")
    return os.path.join(PROCESSED_DIR, f"herb_signature_meta_{nt}.json")

def save_herb_signatures(herb_signatures, gene_names, filepath=None, network_type=None):
    """保存草药签名到文件"""
    if filepath is None:
        filepath = _get_signature_cache_path(network_type)
    meta_path = _get_signature_meta_path(network_type)
    
    import json
    herb_ids = list(herb_signatures.keys())
    arr = np.array([herb_signatures[h] for h in herb_ids])
    np.savez_compressed(filepath, signatures=arr, gene_names=gene_names)
    with open(meta_path, 'w') as f:
        json.dump({"herb_ids": herb_ids, "n_herbs": len(herb_ids), 
                   "n_genes": len(gene_names), "version": 3}, f)
    print(f"[缓存] 签名已保存: {filepath} ({len(herb_ids)} 草药 × {len(gene_names)} 基因)")

def load_herb_signatures(filepath=None, network_type=None):
    """从文件加载草药签名"""
    if filepath is None:
        filepath = _get_signature_cache_path(network_type)
    meta_path = _get_signature_meta_path(network_type)
    
    import json, os
    if not os.path.exists(filepath) or not os.path.exists(meta_path):
        return None
    
    data = np.load(filepath)
    signatures = data['signatures']
    gene_names = list(data['gene_names'])
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    herb_ids = meta['herb_ids']
    
    herb_signatures = {hid: signatures[i] for i, hid in enumerate(herb_ids)}
    print(f"[缓存] 加载签名: {len(herb_ids)} 草药 × {len(gene_names)} 基因")
    return herb_signatures


def _create_diffusion_builder(adj, gene2idx):
    """根据配置文件中的 DIFFUSION_METHOD 创建对应的扩散构建器"""
    if DIFFUSION_METHOD == "rwr":
        builder = RWRPerturbationSignatureBuilder(
            adj, gene2idx,
            restart_prob=RWR_RESTART_PROB,
            max_iter=RWR_MAX_ITER,
            tol=RWR_TOL
        )
        print(f"[扩散] 使用 RWR 随机游走 (r={RWR_RESTART_PROB}, max_iter={RWR_MAX_ITER})")
    else:
        builder = PerturbationSignatureBuilder(adj, gene2idx)
        print(f"[扩散] 使用两阶段热扩散 (alpha={THERMAL_ALPHA})")
    return builder


def batch_build_herb_signatures(
    herb_ingredients_df: pd.DataFrame,
    ingredient_targets_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    builder: Optional[PerturbationSignatureBuilder] = None,
    use_cache: bool = True,
    network_type: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], PerturbationSignatureBuilder]:
    """
    批量构建所有草药的扰动签名

    使用 content_ratio 构建一组签名
    返回 herb_signatures: Dict[herb_id, vector]
    """
    # v6: DrugAI 方向预测重做 —— 修复训练/推理特征错配 + 改用真实RDKit指纹重训模型
    #     (新成分泛化 AUC 0.6143 -> 0.8901), 并叠加实验验证机制覆盖层。
    #     action_type 已整体变化, 旧签名缓存必须失效。
    _CACHE_VERSION = 6
    
    # ----- 缓存检查 -----
    if use_cache:
        cached = load_herb_signatures(network_type=network_type)
        if cached is not None:
            meta_path = _get_signature_meta_path(network_type)
            import json
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                cache_version = meta.get('version', 1)
            except:
                cache_version = 1
            
            if cache_version < _CACHE_VERSION:
                print(f"[签名] 缓存版本(v{cache_version})与当前代码(v{_CACHE_VERSION})不兼容，重新计算")
                import os
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                cache_path = _get_signature_cache_path()
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            else:
                cached_n = len(cached)
                current_n = herb_ingredients_df['herb_id'].nunique()
                if abs(cached_n - current_n) <= 2:
                    print(f"[签名] 使用缓存签名 (v{cache_version}, {cached_n} 种草药)")
                    if builder is None:
                        adj, gene2idx = build_ppi_adjacency(ppi_df)
                        builder = _create_diffusion_builder(adj, gene2idx)
                    return cached, builder
                else:
                    print(f"[签名] 缓存草药数({cached_n})与当前({current_n})不匹配，重新计算")
    
    # 构建PPI邻接矩阵
    if builder is None:
        adj, gene2idx = build_ppi_adjacency(ppi_df)
        builder = _create_diffusion_builder(adj, gene2idx)
    
    # 按成分分组靶标（带置信度加权）
    # 统一 ingredient_id 类型为 str, 避免 herb_ingredients (int) 与 ingredient_targets (str) 类型不一致
    ing_target_groups = {}
    for _, row in ingredient_targets_df.iterrows():
        ing_id = str(row["ingredient_id"]).strip()
        if ing_id not in ing_target_groups:
            ing_target_groups[ing_id] = {}
        # 加权: action_type * confidence；无 confidence 列时默认 1.0
        # 方向未知(action_type=0, 如被中性化的OOD补全成分)时保留靶标存在性信号
        # (PRESENCE_FLOOR × confidence), 使"覆盖率提升"能真正进入扰动签名而非被清零
        confidence = float(row.get("confidence", 1.0))
        action_type = float(row["action_type"])
        if action_type == 0:
            weight = PRESENCE_FLOOR * confidence
        else:
            weight = action_type * confidence
        ing_target_groups[ing_id][row["target_gene"]] = weight
    
    # 预计算每个成分的初始扰动向量
    n_ing = len(ing_target_groups)
    print(f"[签名] 正在计算各成分的初始扰动向量 (共 {n_ing} 个成分)...")
    t_start = time.time()
    ingredient_init_vecs = {}
    
    if _HAS_JOBLIB and n_ing > 100:
        ing_items = list(ing_target_groups.items())
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(builder.get_initial_vector)(targets)
            for _, targets in ing_items
        )
        ingredient_init_vecs = {ing_id: vec for (ing_id, _), vec in zip(ing_items, results)}
        elapsed = time.time() - t_start
        print(f"  并行计算完成: {n_ing} 个成分, 用时 {elapsed:.0f}s ({n_ing/elapsed:.1f} 个/秒)")
    else:
        if not _HAS_JOBLIB and n_ing > 100:
            print("  [提示] 安装 joblib 可加速: pip install joblib")
        for i, (ing_id, targets) in enumerate(ing_target_groups.items()):
            ingredient_init_vecs[ing_id] = builder.get_initial_vector(targets)
            if (i + 1) % 500 == 0 or i == n_ing - 1:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_ing - i - 1) / rate if rate > 0 else 0
                print(f"  成分进度: {i + 1}/{n_ing} ({100*(i+1)//n_ing}%), "
                      f"用时 {elapsed:.0f}s, 预计剩余 {eta:.0f}s")
    
    # 按草药分组
    herb_groups = herb_ingredients_df.groupby("herb_id")
    n_herbs = len(herb_groups)
    
    # ----- 构建草药ID→成分ID列表的索引（注意力模式需要）-----
    # 统一 ingredient_id 为 str, 与 ing_target_groups 的 key 类型一致
    herb_to_ingredients: Dict[str, List[str]] = {}
    herb_to_ratios: Dict[str, Optional[List[float]]] = {}
    for herb_id, group in herb_groups:
        ing_ids = [str(iid).strip() for iid in group["ingredient_id"].tolist()]
        herb_to_ingredients[herb_id] = ing_ids
        if "content_ratio" in group.columns:
            herb_to_ratios[herb_id] = group["content_ratio"].fillna(1.0).tolist()
        else:
            herb_to_ratios[herb_id] = None
    
    # ----- 标准模式（使用 content_ratio）-----
    herb_signatures = {}
    print(f"\n[签名] 正在叠加计算各草药的扰动签名 (共 {n_herbs} 种草药)...")
    t_start = time.time()
    for i, (herb_id, ing_ids) in enumerate(herb_to_ingredients.items()):
        sigs = [ingredient_init_vecs.get(iid) for iid in ing_ids]
        sigs = [s for s in sigs if s is not None]
        ratios = herb_to_ratios.get(herb_id)
        
        if sigs:
            herb_signatures[herb_id] = builder.build_herb_signature(sigs, ratios)
        if (i + 1) % 50 == 0 or i == n_herbs - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_herbs - i - 1) / rate if rate > 0 else 0
            print(f"  草药进度: {i + 1}/{n_herbs} ({100*(i+1)//n_herbs}%), "
                  f"用时 {elapsed:.0f}s, 预计剩余 {eta:.0f}s")
    
    print(f"[签名] 完成! 共构建 {len(herb_signatures)} 个草药的扰动签名")
    
    if use_cache:
        all_genes = builder.get_all_genes()
        save_herb_signatures(herb_signatures, all_genes, network_type=network_type)
    
    return herb_signatures, builder


def signature_to_dataframe(
    herb_signatures: Dict[str, np.ndarray],
    gene_names: List[str]
) -> pd.DataFrame:
    """
    将草药扰动签名字典转为DataFrame
    
    Returns
    -------
    DataFrame: index=herb_id, columns=gene_names, values=perturbation intensity
    """
    df = pd.DataFrame.from_dict(
        herb_signatures, orient="index", columns=gene_names
    )
    df.index.name = "herb_id"
    return df
