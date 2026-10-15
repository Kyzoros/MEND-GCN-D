"""
HDmap-M: 多草药组合映射模块

基于HDmap-S的遗传算法m,搜索最优草药组合

算法流程 (论文Figure 2B):
1. 初始化: 随机生成500个候选解 (草药组合)
2. 评估: 对每个候选解计算HDmap-S得分
3. 选择: 基于得分的轮盘赌选择
4. 交叉: 交换两个解的草药
5. 变异: 随机替换部分草药
6. 精英保留: 最优解直接保留
7. 重复直到收敛
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from typing import Dict, List, Optional, Set, Tuple

import os
import sys
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
from config import (
    GA_POP_SIZE, GA_N_HERBS, GA_MAX_GENERATIONS,
    GA_CROSSOVER_RATE, GA_MUTATION_RATE, GA_ELITE_SIZE,
    GA_HOF_SIZE, GA_RESTART_STAGNATION, GA_RESTART_FRACTION, GA_MAX_RESTARTS,
    GA_SIZE_PENALTY_LAMBDA, GA_DIVERSITY_THRESHOLD,
    GA_EARLY_STOP_GENERATIONS, GA_DYNAMIC_EARLY_STOP,
    GA_MIN_IMPROVEMENT_RATE, GA_EARLY_STOP_WINDOW,
    GA_FITNESS_EPSILON, GA_NORM_EPSILON,
    TMS_METHOD,
)
from hdmap_s import HDmapS
from perturbation_signature import PerturbationSignatureBuilder
import os
import sys


# 表达谱中缺失的 gene_symbol (pandas 读成 NaN) 在背景基因集中的哨兵键。
# 必须与 HDmap-S 的 enrichment 实现保持一致: 原始实现把缺失值当作一个普通背景基因
# 计入 N_BG, 且某病 DEG 若含缺失值也会计入该病的 c。若擅自剔除, Fisher 表的 d 会少 1,
# 导致 GA 分数相对 HDmap-S 主打分整体漂移 (实测约 0.2, 已由一致性测试捕获)。
# NaN 无法参与排序, 且 nan != nan 使其不能直接作 dict 键, 故统一映射到该哨兵。
_BG_MISSING_KEY = "\x00__MISSING__"


def _bg_key(g):
    """背景基因 -> 位图键; 缺失值统一映射到哨兵键。"""
    if isinstance(g, str) and g and g != 'nan':
        return g
    return _BG_MISSING_KEY


class HDmapM:
    """
    Multi-Herbs Disease Mapping via Genetic Algorithm
    """
    
    def __init__(
        self,
        hdmap_s: HDmapS,
        all_herb_ids: List[str],
        herb_signatures: Dict[str, np.ndarray],
        builder: PerturbationSignatureBuilder,
        contraindications: Optional[List[Set[str]]] = None,
        diversity_threshold: float = GA_DIVERSITY_THRESHOLD,
        size_penalty_lambda: float = GA_SIZE_PENALTY_LAMBDA,
        herb_targets: Optional[Dict[str, set]] = None,
        background_genes: Optional[Set[str]] = None,
        fitness_mode: Optional[str] = None,
        combo_stat: str = "fold",
    ):
        """
        Parameters
        ----------
        hdmap_s             : HDmapS实例
        all_herb_ids        : 所有草药ID列表
        herb_signatures     : 草药扰动签名 Dict[herb_id, vector]
        builder             : 扰动签名构建器
        contraindications   : 禁忌组合列表, 每个元素是一组不可共存的草药ID
        diversity_threshold : 多样性阈值, 余弦相似度>=此值的草药不得同时入选 (0~1)
        size_penalty_lambda : 组合大小正则化系数 λ, 越大越惩罚大组合 (0=不惩罚)
        herb_targets        : 草药->靶点基因集合 (enrichment 模式必需, 用
                             hdmap_s.build_herb_target_sets 构建, 保证与 HDmap-S 同口径)
        background_genes    : 富集背景基因集 (enrichment 模式必需, 通常为
                             全部草药靶点基因 ∪ 全部疾病DEG基因)
        fitness_mode        : "enrichment" | "ndcg"; None 时自动跟随 config.TMS_METHOD
        combo_stat          : 复方富集统计量 "fold"(默认, 效应量) | "pvalue"(显著性, 仅供对比/消融)
        """
        self.hdmap_s = hdmap_s
        self.all_herb_ids = list(all_herb_ids)
        self.n_herbs = len(self.all_herb_ids)
        self.herb_signatures = herb_signatures
        self.builder = builder
        self.contraindications = contraindications or []
        self.diversity_threshold = diversity_threshold
        self.size_penalty_lambda = size_penalty_lambda

        # ---- 目标函数模式: 必须与 HDmap-S 的主方法 (config.TMS_METHOD) 保持一致 ----
        # 历史问题: GA 一直用 NDCG + 扩散签名(旧方法), 而 HDmap-S 已切换到 enrichment,
        # 造成"论文宣称用新方法评价草药, 复方优化器却在优化旧目标"的方法论不一致。
        # 现在默认自动跟随 TMS_METHOD, 两者永远同源。
        self.fitness_mode = (fitness_mode or TMS_METHOD or "ndcg").lower()
        if self.fitness_mode not in ("enrichment", "ndcg"):
            self.fitness_mode = "ndcg"  # 未知取值回退旧行为, 不改变原有语义
        # 复方富集统计量: 默认用效应量(fold), 详见 _fitness_enrichment
        self.combo_stat = (combo_stat or "fold").lower()
        if self.combo_stat not in ("fold", "zscore", "pvalue"):
            self.combo_stat = "fold"
        # 复方最小味数: 1 味的"复方"无意义(变异去重或交叉可能产生单味解)
        self.min_combo_size = 2
        self._herb_target_mask: Optional[Dict[str, int]] = None
        self._bg_gene2bit: Optional[Dict[str, int]] = None
        self._N_BG = 0
        self._fisher_cache: Dict[Tuple[int, int, int], float] = {}
        if self.fitness_mode == "enrichment":
            if not herb_targets or background_genes is None:
                raise ValueError(
                    "fitness_mode='enrichment' 需要同时提供 herb_targets 与 background_genes"
                )
            self._build_enrichment_structures(herb_targets, background_genes)
        
        # 草药ID→索引 映射
        self.herb2idx = {h: i for i, h in enumerate(self.all_herb_ids)}
        
        # ---- Hall of Fame: 外部存档，永不参与交叉/变异，防止最优解丢失 ----
        self.hall_of_fame: List[Tuple[List[str], float]] = []  # [(herbs, fitness), ...]
        self._hof_seen: Set[Tuple[str, ...]] = set()           # HoF 去重键
        
        # ---- 重启计数器 ----
        self._restart_count = 0  # 累计重启次数
        
        # ---- 预计算草药间余弦相似度，构建多样性禁忌对 ----
        self._build_diversity_constraints()
        
    def _build_diversity_constraints(self):
        """
        预计算草药签名间的两两余弦相似度，生成为禁忌对集合。
        
        相似度 >= diversity_threshold 的草药对不得同时入选组合，
        避免组合中出现冗余草药（签名高度重叠=药理作用雷同）。
        """
        n = self.n_herbs
        self._diverse_pairs: List[Set[str]] = []
        
        # 构建签名矩阵 (n_herbs × n_genes)
        # 用 np.vstack 而非 np.array, 避免形状不一时退化为 1D
        # 双重保险: atleast_1d 防止 0D 标量, reshape 强制 2D
        sigs = [
            np.atleast_1d(self.herb_signatures.get(hid, np.zeros(self.builder.n_proteins)))
            for hid in self.all_herb_ids
        ]
        if not sigs:
            return  # 无草药, 不计算多样性约束
        sig_matrix = np.vstack(sigs)
        if sig_matrix.ndim == 1:
            sig_matrix = sig_matrix.reshape(1, -1)
        
        # 预归一化（加速余弦计算）
        norms = np.linalg.norm(sig_matrix, axis=1, keepdims=True)
        norms[norms < GA_NORM_EPSILON] = 1.0
        sig_normalized = sig_matrix / norms
        
        # 批量计算上三角余弦相似度
        for i in range(n):
            # 取当前草药之后的所有草药做点积（余弦相似度=归一化向量的点积）
            similarities = sig_normalized[i] @ sig_normalized[i+1:].T
            high_sim_idx = np.where(similarities >= self.diversity_threshold)[0]
            
            for j_offset in high_sim_idx:
                j = i + 1 + j_offset
                self._diverse_pairs.append({self.all_herb_ids[i], self.all_herb_ids[j]})
        
        n_forbidden = len(self._diverse_pairs)
        if n_forbidden > 0:
            print(f"[HDmapM] 多样性约束: {n_forbidden} 对草药相似度>={self.diversity_threshold:.1f}, 禁止同时入选")
        else:
            print(f"[HDmapM] 多样性约束: 无高相似度草药对 (threshold={self.diversity_threshold:.1f})")
    
    def _check_contraindications(self, herbs: List[str]) -> bool:
        """检查是否违反禁忌 (用药禁忌 + 多样性约束)"""
        herb_set = set(herbs)
        # 检查用药禁忌
        for contra in self.contraindications:
            if contra.issubset(herb_set):
                return True
        # 检查多样性约束
        for pair in self._diverse_pairs:
            if pair.issubset(herb_set):
                return True
        return False
    
    def _update_hall_of_fame(self, population: List[List[str]],
                              fitness_scores: List[float]):
        """
        更新 Hall of Fame 外部存档。
        
        HoF 独立于种群，永不参与交叉/变异操作，确保最优解不丢失。
        保持 GA_HOF_SIZE 个历史最优解。
        """
        for sol, fit in zip(population, fitness_scores):
            key = tuple(sorted(sol))
            if key in self._hof_seen or len(sol) == 0:
                continue
            # 如果 HoF 未满，直接加入
            if len(self.hall_of_fame) < GA_HOF_SIZE:
                self.hall_of_fame.append((sol.copy(), fit))
                self._hof_seen.add(key)
            else:
                # 如果比 HoF 中最差的更好，替换
                worst_idx = max(range(len(self.hall_of_fame)),
                               key=lambda i: self.hall_of_fame[i][1])
                if fit < self.hall_of_fame[worst_idx][1]:  # 更负 = 更好
                    old_key = tuple(sorted(self.hall_of_fame[worst_idx][0]))
                    self._hof_seen.discard(old_key)
                    self.hall_of_fame[worst_idx] = (sol.copy(), fit)
                    self._hof_seen.add(key)
        # 保持升序（最负在前）
        self.hall_of_fame.sort(key=lambda x: x[1])
    
    def _random_solution(self, max_size: int = 4) -> List[str]:
        """生成一个随机草药组合"""
        n = np.random.randint(2, max_size + 1)
        return list(np.random.choice(self.all_herb_ids, n, replace=False))
    
    def _combine_signatures(self, herbs: List[str]) -> np.ndarray:
        """
        叠加多个草药的扰动签名（线性叠加 + 1/√n 归一化）

        P_combo = Σ P_herb_i / √n

        为什么 1/√n 归一化（而不是 1/n 或不归一化）？
        - 1/n 归一化：等权平均 → 多味药被"稀释"，单味药永远最优
        - 不归一化：n 味药叠加的绝对值放大 n 倍 → 长组合在提取 top/bottom
          时被自己的高绝对值主导，方向互补被掩盖
        - 1/√n 归一化：保留方向互补的"几何增益"，但保持叠加强度与单味药
          同阶。这是"独立随机变量之和的方差缩放"的标准做法，
          让 n 味药组合的"新信息量"随 n 增长，而不是被幅度掩盖。

        草药签名已经过RWR扩散(r=0.85)，直接求和得到组合效应。
        此线性叠加假设对应原论文的药理可加性假设。
        """
        sigs = [self.herb_signatures[h] for h in herbs if h in self.herb_signatures]
        if not sigs:
            return np.zeros(self.builder.n_proteins)
        n = len(sigs)
        # 1/√n 归一化：让多味药组合保留方向互补，同时不放大绝对值
        return np.sum(sigs, axis=0) / np.sqrt(n)
    
    # ---- 富集目标函数 (与 HDmap-S 的 enrichment 口径严格一致) ----

    def _build_enrichment_structures(self, herb_targets: Dict[str, set],
                                     background_genes: Set[str]):
        """
        预计算富集打分所需的位图(bitset)结构。

        用 Python 大整数作位图: 背景基因每个占 1 bit, 一味草药的靶点集合编码成一个 int。
        于是 复方靶点并集 = 各草药位图按位或 (整数运算), 与疾病DEG的交集大小 =
        位与后的 popcount。这样每次适应度评估只需几次整数运算, 避免 GA 上万次
        评估时反复做集合运算拖慢速度。
        """
        genes = sorted({_bg_key(g) for g in background_genes})
        self._bg_gene2bit = {g: i for i, g in enumerate(genes)}
        self._N_BG = len(self._bg_gene2bit)
        masks: Dict[str, int] = {}
        for h, tg in herb_targets.items():
            m = 0
            for g in tg:
                b = self._bg_gene2bit.get(_bg_key(g))
                if b is not None:
                    m |= (1 << b)
            masks[str(h)] = m
        self._herb_target_mask = masks

    def _deg_mask(self, disease_genes: List[str]) -> Tuple[int, int]:
        """疾病 DEG 基因列表 -> (位图, 落在背景集内的 DEG 数量)"""
        m = 0
        n = 0
        for g in disease_genes:
            b = self._bg_gene2bit.get(_bg_key(g))
            if b is not None:
                bit = 1 << b
                if not (m & bit):   # 去重: DEG 列表中可能有重复基因符号
                    m |= bit
                    n += 1
        return m, n

    def _fitness_enrichment(self, herbs: List[str], deg_mask: int,
                            n_deg: int) -> float:
        """
        复方富集目标函数。两种统计量可选 (self.combo_stat):

        "fold" (默认) —— 效应量: 倍数富集 = (a/K) / (n_deg/N_BG)
            即"观测命中比例 / 随机期望比例"。
            **不随靶点总数 K 系统性增大**, 因此不同规模的复方可直接比较。
            无需 Fisher 检验, 纯整数运算, 比 p 值方案快得多。

        "pvalue" —— 显著性: -log10(双侧 Fisher p), 与 HDmap-S 单药打分同口径。
            但 -log10(p) 会随 K 单调变显著 (样本量效应), 用于复方时等价于
            "靶点越多越好", 实测会让 GA 退化为一味堆药 —— 即便加了 max_size 硬上限,
            Top-K 仍几乎全是 6 味 (均值 5.9)。仅保留作对比/消融。

        为什么复方不能沿用 p 值:
          单味药之间靶点数差异有限, 用显著性排序可接受; 但复方靶点是各味药的**并集**,
          K 随味数近似线性增长, -log10(p) 随之系统性膨胀, 于是"加药"永远占优。
          效应量消除这一混淆, 让 GA 去寻找"靶点最集中于疾病模块"的组合,
          而不是"靶点最多"的组合。

        方向说明:
          enrichment 模式下分值恒正, "越高越治疗"由网络医学先验支撑
          (Barabasi: 药物靶点富集于疾病模块 = 治疗), 不依赖
          antagonist/engagement 双通道标签 —— 那是 dual 模式 (-cos) 的语义,
          与富集分不可混用。故本函数不使用 mechanism 参数。

        Returns
        -------
        fitness : float, 越负 = 治疗方向越强 (GA 约定)
        """
        union = 0
        for h in herbs:
            m = self._herb_target_mask.get(h)
            if m:
                union |= m
        if not union or n_deg == 0:
            return self.size_penalty_lambda * len(herbs)

        k_targets = bin(union).count('1')       # 复方靶点总数 K
        a = bin(union & deg_mask).count('1')    # 命中疾病DEG的靶点数

        if a == 0:
            score = 0.0
        elif self.combo_stat == "fold":
            # 效应量: 观测比例 / 期望比例 (无样本量偏差)
            expected_ratio = n_deg / self._N_BG
            score = (a / k_targets) / expected_ratio if expected_ratio > 0 else 0.0
        elif self.combo_stat == "zscore":
            # 标准化富集 (超几何 z 值): (a - E[a]) / sd[a]
            #   常数富集倍数下 z ∝ sqrt(K), 即"覆盖"与"集中度"的折中:
            #   加药带来的收益按平方根衰减, 必须真的提升集中度才划算,
            #   因此既不像 p 值那样线性膨胀(一味堆药), 也不像 fold 那样
            #   反向偏好小组合(缩到单味)。实测规模趋势最平稳。
            p = n_deg / self._N_BG
            expected = k_targets * p
            var = k_targets * p * (1.0 - p) * (self._N_BG - k_targets) / max(self._N_BG - 1, 1)
            score = (a - expected) / np.sqrt(var) if var > 0 else 0.0
        else:
            b = k_targets - a
            c = n_deg - a
            d = self._N_BG - a - b - c
            if d < 0:
                d = 0
            # 同一疾病内 n_deg 与背景固定, 只需按 (a, k_targets, n_deg) 缓存 p 值;
            # GA 上万次评估中存在大量重复组合, 缓存可显著省时。
            cache_key = (a, k_targets, n_deg)
            p = self._fisher_cache.get(cache_key)
            if p is None:
                _, p = fisher_exact([[a, b], [c, d]])
                self._fisher_cache[cache_key] = p
            score = -np.log10(max(p, 1e-300))

        # GA 约定"越负 = 越好", 富集分"越高 = 越治疗", 故取负号。
        # 组合大小惩罚与旧实现一致 (正值 = 更差); 用 fold 后规模偏差已消除,
        # 该惩罚仅作为"同等富集下偏好更精简复方"的微弱正则。
        return -score + self.size_penalty_lambda * len(herbs)

    def _fitness(self, herbs: List[str], disease_rank: Dict[str, int],
                  n_disease: int, k: int = 250,
                  gene_weights: Optional[dict] = None,
                  mechanism: str = "engagement") -> float:
        """
        适应度函数 = NDCG-TMS + 组合大小正则化惩罚 + 基因权重

        双通道机制方向:
          复方 TMS 存在机制异构——逆转型(antagonist)病需要复方"反向纠正"
          疾病签名, 而嵌网型(engagement)病需要复方与疾病签名"同向嵌网"。
          单符号标量无法同时表达, 故按疾病机制先验选择 TMS 方向, 统一保持
          "越负 = 治疗方向越强" 的 GA 约定。

          antagonist(逆转): rev = (ndcg_top - ndcg_bottom)/2, 负数=治疗 (原行为)
          engagement(嵌网): -rev, 负数=治疗 (反向, 让同向嵌网的复方得分更负)

        Parameters
        ----------
        herbs          : 草药组合
        disease_rank   : 疾病基因排位字典 {gene: rank}
        n_disease      : 排名字典中的基因总数
        k              : top/bottom基因数
        gene_weights   : {gene: weight_mul} 已知靶标加权
        mechanism      : "antagonist" | "engagement" 双通道机制先验

        Returns
        -------
        fitness : float, 正则化TMS; 越负 = 治疗方向越强
        """
        combo_sig = self._combine_signatures(herbs)
        
        top_genes, bottom_genes = self.builder.extract_top_bottom_genes(
            combo_sig, k=k
        )
        
        if n_disease == 0:
            return self.size_penalty_lambda * len(herbs)
        
        ranks_top = [disease_rank.get(g, n_disease + 1) for g in top_genes]
        ranks_bottom = [disease_rank.get(g, n_disease + 1) for g in bottom_genes]
        
        # NDCG 增益: gain(rank) = 1/log2(rank+1), rank 越小 gain 越大
        r_top = np.array(ranks_top, dtype=float)
        r_bottom = np.array(ranks_bottom, dtype=float)
        gain_top = np.where(r_top <= n_disease, 1.0 / np.log2(r_top + 1), 0.0)
        gain_bottom = np.where(r_bottom <= n_disease, 1.0 / np.log2(r_bottom + 1), 0.0)
        
        # 已知靶标基因加权
        if gene_weights:
            w_def = gene_weights.get("_default", 1.0)
            w_top = np.array([gene_weights.get(g, w_def) for g in top_genes])
            w_bottom = np.array([gene_weights.get(g, w_def) for g in bottom_genes])
            gain_top = gain_top * w_top
            gain_bottom = gain_bottom * w_bottom
        
        # IDCG: k 个基因理想情况下 rank=1..k（全部命中疾病列表顶部）
        idcg_k = float(np.sum(1.0 / np.log2(np.arange(1, min(k, n_disease) + 1, dtype=float) + 1)))
        if idcg_k < 1e-12:
            idcg_k = 1.0
        
        ndcg_top = float(np.sum(gain_top)) / idcg_k       # ∈ [0, 1]
        ndcg_bottom = float(np.sum(gain_bottom)) / idcg_k  # ∈ [0, 1]
        
        # 双通道方向选择 (保持 "越负 = 治疗" 约定):
        #   rev<0 表示复方反向纠正疾病 (适配 antagonist 病)
        #   -rev<0 表示复方与疾病同向嵌网 (适配 engagement 病)
        rev = (ndcg_top - ndcg_bottom) / 2.0
        tms = rev if mechanism == "antagonist" else -rev
        
        # 组合大小正则化: 对大组合施加惩罚（使TMS变正=更差）
        adjusted = tms + self.size_penalty_lambda * len(herbs)
        return adjusted
    
    # ---- 遗传算子 ----
    
    def _selection(self, population: List[List[str]],
                   fitness_scores: List[float]) -> List[List[str]]:
        """
        轮盘赌选择 (论文公式)
        P_i = |TMS_i| / Σ|TMS_j|
        
        TMS为负值时取反，使越负的组合被选中概率越高。
        """
        fitness = np.array(fitness_scores)
        # 取负: 最负的TMS → 最大的正权重
        fitness = -fitness
        fitness = np.maximum(fitness, GA_FITNESS_EPSILON)  # 避免除零/负值
        probs = fitness / fitness.sum()
        
        n_keep = len(population) - GA_ELITE_SIZE
        if n_keep <= 0:
            return population
        
        indices = np.random.choice(
            len(population), size=n_keep, replace=True, p=probs
        )
        return [population[i] for i in indices]
    
    def _crossover(self, parent1: List[str],
                   parent2: List[str]) -> Tuple[List[str], List[str]]:
        """
        两点交叉 + 最小子代大小保证
        
        从两个父代中各取一部分组合成子代，确保:
        1. 子代大小在 [1, len(parent1)+len(parent2)] 范围内
        2. 子代不会退化为父代的精确拷贝
        3. 交叉操作产生真正的遗传重组
        """
        if np.random.random() > GA_CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()
        
        set1, set2 = set(parent1), set(parent2)
        common = set1 & set2
        unique1 = list(set1 - common)
        unique2 = list(set2 - common)
        
        if len(unique1) < 1 or len(unique2) < 1:
            return parent1.copy(), parent2.copy()
        
        # 两点交叉: 随机选择两个切分点，避免全取一边
        # split1: 从parent1独有的元素中取[0:split1]
        # split2: 从parent2独有的元素中取[split2:]  
        split1 = np.random.randint(1, len(unique1) + 1)  # 至少取1个
        split2 = np.random.randint(0, len(unique2))       # 至少留1个给另一个
        
        # 子代1: common + parent1的前半 + parent2的后半
        child1 = list(common) + unique1[:split1] + unique2[split2:]
        # 子代2: common + parent1的后半 + parent2的前半
        child2 = list(common) + unique1[split1:] + unique2[:split2]
        
        # 去重
        child1 = list(dict.fromkeys(child1))
        child2 = list(dict.fromkeys(child2))
        
        # 最小大小保证
        if len(child1) == 0:
            child1 = [unique1[0]] if unique1 else list(common)[:1] or parent1[:1]
        if len(child2) == 0:
            child2 = [unique2[0]] if unique2 else list(common)[:1] or parent2[:1]
        
        return child1, child2
    
    def _mutation(self, herbs: List[str]) -> List[str]:
        """
        变异: 以一定概率替换一个草药为另一个
        """
        if np.random.random() > GA_MUTATION_RATE or len(herbs) == 0:
            return herbs
        
        herbs = list(herbs)
        idx = np.random.randint(len(herbs))
        new_herb = np.random.choice(self.all_herb_ids)
        herbs[idx] = new_herb
        result = list(dict.fromkeys(herbs))
        # 防止变异后列表变空
        if len(result) == 0:
            result = [np.random.choice(self.all_herb_ids)]
        return result
    
    def _cap_size(self, solution: List[str], max_size: int) -> List[str]:
        """
        强制组合大小不超过 max_size。

        为什么必须有这一步: _crossover 的子代是两个父代的并集, 大小最多可达
        |parent1 ∪ parent2|, 即每代可接近翻倍。交叉/变异后若不截断, 组合规模会随
        代数指数爆炸 —— 实测 500 代后产出 34~64 味的"复方", 远超 GA_MAX_COMBO_SIZE=6。
        该缺陷长期潜伏: 旧 NDCG 打分下多味药相互稀释信号、大组合不占优, 问题被掩盖;
        换 enrichment 后复方靶点取并集, 加药只会让富集分单调不降, 缺陷立即暴露。

        同时保证下界: _mutation 去重或 _crossover 取子集都可能把组合缩到 1 味,
        而 1 味的"复方"没有意义, 故补足到 min_combo_size。
        """
        sol = list(solution)
        if len(sol) > max_size:
            sol = sol[:max_size]
        min_size = getattr(self, "min_combo_size", 2)
        if len(sol) < min_size:
            existing = set(sol)
            pool = [h for h in self.all_herb_ids if h not in existing]
            while len(sol) < min_size and pool:
                sol.append(pool[int(np.random.randint(len(pool)))])
                existing = set(sol)
                pool = [h for h in pool if h not in existing]
        return sol

    def _elite_preserve(self, population: List[List[str]],
                        fitness_scores: List[float],
                        elite_count: int) -> List[List[str]]:
        """精英保留: 保留最优的几个解 (TMS越负越好)"""
        ranked = sorted(zip(population, fitness_scores),
                       key=lambda x: x[1])  # 升序 = 最负优先
        elites = [sol.copy() for sol, _ in ranked[:elite_count]]
        return elites
    
    # ---- 主算法 ----
    
    def optimize(self, disease_genes: List[str],
                 max_generations: int = GA_MAX_GENERATIONS,
                 max_size: int = 4,
                 k: int = 250,
                 n_top: int = 10,
                 verbose: bool = True,
                 gene_weights: Optional[dict] = None,
                 pop_size: int = GA_POP_SIZE,
                 mechanism: str = "engagement") -> Dict:
        """
        运行遗传算法找到最优草药组合
        
        Parameters
        ----------
        disease_genes   : 疾病差异基因列表 (按logFC降序)
        max_generations : 最大迭代次数
        max_size        : 组合中最大草药数
        k               : top/bottom基因数
        n_top           : 返回Top-K组合数 (默认10)
        
        Returns
        -------
        dict with keys:
            - 'best_solution' : 最优草药组合 (herb_id列表)
            - 'best_score'    : 最优得分 (|TMS|)
            - 'top_solutions' : Top-K草药组合列表 [{'herbs':[...], 'score':...}, ...]
            - 'generations_run' : 实际运行代数
            - 'history'       : 每代最优得分记录
        """
        # 存储基因权重供 _fitness 使用
        self._gene_weights = gene_weights
        
        # ===== 初始化种群 =====
        population = []
        MAX_CONTRAIND_RETRY = 200  # 防止禁忌死循环
        for _ in range(pop_size):
            solution = self._random_solution(max_size)
            retries = 0
            while retries < MAX_CONTRAIND_RETRY and self._check_contraindications(solution):
                solution = self._random_solution(max_size)
                retries += 1
            population.append(solution)
        
        # 预计算疾病基因排位字典 (避免每次fitness调用重建)
        disease_rank = {g: i + 1 for i, g in enumerate(disease_genes)
                        if g in self.builder.gene2idx}
        n_disease = len(disease_rank)
        
        # ---- 目标函数分发: enrichment 走富集打分, ndcg 沿用旧 NDCG-TMS ----
        _deg_mask = 0
        _n_deg = 0
        if self.fitness_mode == "enrichment":
            _deg_mask, _n_deg = self._deg_mask(disease_genes)
            _n_with_targets = sum(1 for m in self._herb_target_mask.values() if m)
            stat_desc = ("效应量 倍数富集 (a/K)/(n_deg/N_BG), 无样本量偏差"
                         if self.combo_stat == "fold"
                         else "显著性 -log10(Fisher p), 会随靶点数K膨胀")
            print(f"[GA] 目标函数=enrichment | 复方统计量={self.combo_stat} ({stat_desc})")
            print(f"[GA]   背景基因 {self._N_BG} | 疾病DEG(背景内) {_n_deg} | "
                  f"有靶点草药 {_n_with_targets}/{len(self._herb_target_mask)}")
            print(f"[GA] 注: 不使用 mechanism={mechanism} "
                  f"(富集分恒正, 越高越治疗, 不依赖双通道方向标签)")

        def _eval(sol):
            """按 fitness_mode 分发适应度评估。"""
            if self.fitness_mode == "enrichment":
                return self._fitness_enrichment(sol, _deg_mask, _n_deg)
            return self._fitness(sol, disease_rank, n_disease, k=k,
                                 gene_weights=self._gene_weights,
                                 mechanism=mechanism)

        # 计算初始适应度 (双通道机制方向)
        fitness = [_eval(sol) for sol in population]
        
        # ---- 追踪 Top-K 历史最优解 ----
        all_solutions = list(zip(population, fitness))
        # 去重 (基于tuple排序的herb_id列表)
        seen = set()
        unique_solutions = []
        for sol, fit in sorted(all_solutions, key=lambda x: x[1]):  # 升序 = 最负优先
            key = tuple(sorted(sol))
            if key not in seen and len(sol) > 0:
                seen.add(key)
                unique_solutions.append((sol.copy(), fit))
        top_solutions = unique_solutions[:n_top]
        
        best_solution = top_solutions[0][0]
        best_score = top_solutions[0][1]
        
        # 防护：确保初始最优解非空
        if len(best_solution) == 0:
            for sol in population:
                if len(sol) > 0:
                    best_solution = sol.copy()
                    best_score = _eval(sol)
                    break
        
        history = [best_score]
        
        no_improve = 0
        
        # 自适应早停的滑动窗口
        _window_size = GA_EARLY_STOP_WINDOW
        _recent_scores = [best_score]
        
        # ---- 初始化 Hall of Fame ----
        self.hall_of_fame = []
        self._hof_seen = set()
        self._update_hall_of_fame(population, fitness)
        self._restart_count = 0
        
        # ===== 主循环 =====
        for gen in range(max_generations):
            # 精英保留
            elites = self._elite_preserve(population, fitness, GA_ELITE_SIZE)
            
            # 选择
            selected = self._selection(population, fitness)
            
            # 交叉
            new_population = elites.copy()
            np.random.shuffle(selected)
            for i in range(0, len(selected) - 1, 2):
                child1, child2 = self._crossover(selected[i], selected[i + 1])
                # 子代是两父代并集, 必须截断, 否则组合规模随代数指数增长
                new_population.append(self._cap_size(child1, max_size))
                if len(new_population) < GA_POP_SIZE:
                    new_population.append(self._cap_size(child2, max_size))
            
            # 补全不足
            while len(new_population) < GA_POP_SIZE:
                sol = self._random_solution(max_size)
                new_population.append(sol)
            
            # 截断到种群大小
            population = new_population[:GA_POP_SIZE]
            
            # 变异（精英保护：不变异前 GA_ELITE_SIZE 个精英个体）
            if len(population) > GA_ELITE_SIZE:
                elite_part = population[:GA_ELITE_SIZE]
                non_elite_part = [self._cap_size(self._mutation(sol), max_size)
                                  for sol in population[GA_ELITE_SIZE:]]
                population = elite_part + non_elite_part
            else:
                population = [self._cap_size(self._mutation(sol), max_size)
                              for sol in population]
            
            # 检查禁忌
            for i, sol in enumerate(population):
                if self._check_contraindications(sol):
                    population[i] = self._random_solution(max_size)
            
            # 重新评估
            fitness = [_eval(sol) for sol in population]
            gen_best_idx = np.argmin(fitness)  # 最小TMS = 最负 = 最佳
            gen_best_score = fitness[gen_best_idx]
            history.append(gen_best_score)
            
            # ---- 更新 Top-K ----
            for sol, fit in zip(population, fitness):
                key = tuple(sorted(sol))
                if key not in seen and len(sol) > 0:
                    seen.add(key)
                    top_solutions.append((sol.copy(), fit))
            # 按TMS升序（越负越好），只保留Top-K
            top_solutions = sorted(top_solutions, key=lambda x: x[1])[:n_top]
            
            # ---- 更新 Hall of Fame（外部存档，永不丢失）----
            self._update_hall_of_fame(population, fitness)
            
            if gen_best_score < best_score:  # 更负 = 更好
                best_score = gen_best_score
                best_solution = population[gen_best_idx].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # 滑动窗口记录
            _recent_scores.append(gen_best_score)
            if len(_recent_scores) > _window_size:
                _recent_scores.pop(0)
            
            if verbose and (gen % 20 == 0 or gen == max_generations - 1):
                print(f"[GA] 第 {gen + 1} 代 | 最佳得分: {gen_best_score:.4f} | "
                      f"全局最优: {best_score:.4f} | 停滞: {no_improve} | "
                      f"HoF: {len(self.hall_of_fame)} | TopK: {len(top_solutions)}")
            
            # ===== 种群重启机制 =====
            # 停滞超过阈值时，重新初始化部分种群以恢复多样性
            # 限制最大重启次数，避免在局部最优处无限循环重启
            if no_improve >= GA_RESTART_STAGNATION and gen < max_generations - 1 and self._restart_count < GA_MAX_RESTARTS:
                n_restart = max(1, int(GA_POP_SIZE * GA_RESTART_FRACTION))
                # 保留当前种群精英
                keep_size = GA_POP_SIZE - n_restart
                ranked_pop = sorted(zip(population, fitness), key=lambda x: x[1])
                population = [sol.copy() for sol, _ in ranked_pop[:keep_size]]
                # ---- 注入 HoF 最优解，确保种群始终持有历史最佳路径 ----
                existing_keys = {tuple(sorted(s)) for s in population}
                n_hof_injected = 0
                for hof_sol, _ in self.hall_of_fame:
                    if len(population) >= GA_POP_SIZE:
                        break
                    hof_key = tuple(sorted(hof_sol))
                    if hof_key not in existing_keys:
                        population.append(self._cap_size(hof_sol.copy(), max_size))
                        existing_keys.add(hof_key)
                        n_hof_injected += 1
                # 补充随机解
                n_added = 0
                while len(population) < GA_POP_SIZE:
                    sol = self._random_solution(max_size)
                    if not self._check_contraindications(sol):
                        population.append(sol)
                        n_added += 1
                    elif n_added > GA_POP_SIZE * 3:  # 防止死循环
                        population.append(sol)  # 强行加入
                        n_added += 1
                self._restart_count += 1
                no_improve = 0  # 重置停滞计数
                _recent_scores = [gen_best_score]  # 重置滑动窗口，防止旧数据误触发早停
                # 重新评估适应度
                fitness = [_eval(sol) for sol in population]
                # 更新 best_score/best_solution 以反映 HoF 注入后的真实全局最优
                injected_best_idx = np.argmin(fitness)
                if fitness[injected_best_idx] < best_score:
                    best_score = fitness[injected_best_idx]
                    best_solution = population[injected_best_idx].copy()
                if verbose:
                    print(f"[GA] [Restart] 种群重启 #{self._restart_count} | "
                          f"保留 {keep_size} 个精英 | 注入HoF {n_hof_injected} 个 | "
                          f"注入随机 {n_added} 个 | HoF最优: {self.hall_of_fame[0][1]:.4f}")
            
            # ===== 早停条件 (双重机制) =====
            # 条件1: 连续20代无绝对改善
            if no_improve >= GA_EARLY_STOP_GENERATIONS:
                if verbose:
                    print(f"[GA] 早停: 连续 {no_improve} 代无改善")
                break
            
            # 条件2: 滑动窗口平均改善率 < 0.1% (防止"爬坡"式收敛的无效迭代)
            if len(_recent_scores) >= _window_size and no_improve >= GA_DYNAMIC_EARLY_STOP:
                _improvement = (_recent_scores[-1] - _recent_scores[0]) / max(abs(_recent_scores[0]), 1e-10)
                if abs(_improvement) < GA_MIN_IMPROVEMENT_RATE:
                    if verbose:
                        print(f"[GA] 动态早停: 滑动窗口改善率 {_improvement:.6f} < {GA_MIN_IMPROVEMENT_RATE}")
                    break
        
        # ---- 合并 Hall of Fame 到 Top-K 结果 ----
        # HoF 可能包含种群中已丢失的最优解，确保它们进入最终输出
        for sol, fit in self.hall_of_fame:
            key = tuple(sorted(sol))
            if key not in seen and len(sol) > 0:
                seen.add(key)
                top_solutions.append((sol.copy(), fit))
        top_solutions = sorted(top_solutions, key=lambda x: x[1])[:n_top]
        
        # 如果 HoF 最优解比 best_solution 更好，用 HoF 覆盖
        if self.hall_of_fame and self.hall_of_fame[0][1] < best_score:
            best_solution = self.hall_of_fame[0][0].copy()
            best_score = self.hall_of_fame[0][1]
        
        # 构建 Top-K 输出
        top_list = [
            {"herbs": sol, "score": float(score), "n_herbs": len(sol)}
            for sol, score in top_solutions
        ]
        
        return {
            "best_solution": best_solution,
            "best_score": best_score,
            "top_solutions": top_list,
            "generations_run": gen + 1,
            "history": history,
            "hall_of_fame": [{"herbs": sol, "score": float(score), "n_herbs": len(sol)}
                           for sol, score in self.hall_of_fame],
            "restart_count": self._restart_count,
        }


def run_hdmap_m_for_all_diseases(
    hdmap_m: HDmapM,
    disease_profiles: Dict[str, List[str]],
    max_generations: int = 100,
    n_top: int = 10,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, List[Dict]]]:
    """
    对所有疾病运行HDmap-M, 找到每种疾病的最优复方组合列表
    
    Returns
    -------
    summary_df  : DataFrame, 每种疾病的#1组合摘要
    topk_dict   : Dict[disease_id, List[Dict]], 每种疾病的Top-K组合
    """
    summary = []
    topk_dict = {}
    for disease_id, genes in disease_profiles.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"[HDmap-M] 疾病: {disease_id}")
            print(f"{'='*60}")
        
        result = hdmap_m.optimize(genes, max_generations=max_generations, n_top=n_top)
        
        summary.append({
            "disease_id": disease_id,
            "herbs": ",".join(result["best_solution"]),
            "n_herbs": len(result["best_solution"]),
            "score": result["best_score"],
            "generations": result["generations_run"]
        })
        topk_dict[disease_id] = result["top_solutions"]
    
    return pd.DataFrame(summary), topk_dict
