"""
HDCTI适配器 - 替代WES算法进行靶标预测
===========================================
将HDCTI异构图模型集成到systems_mapping管道中，用于：
1. 预测草药成分的潜在靶标蛋白
2. 扩展ingredient_targets数据
3. 替代原始WES算法

HDCTI论文: "HDCTI: Heterogeneous graph-based framework for 
Herb-Compound-Target-Disease association prediction"

集成方式:
- 构建H-C-P-D异构图（草药-成分-蛋白-疾病）
- 使用HDCTI预测新的C-P关联
- 将预测结果合并到ingredient_targets中
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
import warnings
warnings.filterwarnings("ignore")

# 添加HDCTI路径
HDCTI_PATH = r"D:\Bio_information\HDCTI\HDCTI-main"
if HDCTI_PATH not in sys.path:
    sys.path.insert(0, HDCTI_PATH)

# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
from config import (
    TCMSP_DATA_DIR, RAW_DIR, PROCESSED_DIR,
    HDCTI_USE, HDCTI_CONFIDENCE_THRESHOLD, HDCTI_TOP_K
)


class HDCTIAdapter:
    """
    HDCTI适配器
    
    功能:
    1. 从systems_mapping数据构建HDCTI所需的异构图
    2. 训练HDCTI模型（或加载预训练模型）
    3. 预测新的成分-蛋白质关联
    4. 将预测结果合并到现有靶标数据中
    
    使用示例:
        adapter = HDCTIAdapter()
        adapter.build_graph(herb_ingredients_df, ingredient_targets_df, disease_profiles_df)
        adapter.train(epochs=100)
        new_targets = adapter.predict_new_targets(ingredient_id="MOL000108")
        updated_df = adapter.merge_targets(existing_df, new_targets)
    """
    
    def __init__(
        self,
        confidence_threshold: float = HDCTI_CONFIDENCE_THRESHOLD,
        top_k: int = HDCTI_TOP_K,
        device: str = "cpu",
        similarity_method: str = "tanimoto"
    ):
        """
        Parameters
        ----------
        confidence_threshold : 预测置信度阈值，高于此值的靶标才被纳入
        top_k : 每个成分最多预测的靶标数
        device : 计算设备
        similarity_method : Path1成分相似度方法 ("jaccard" | "tanimoto")
        """
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.device = device
        self.similarity_method = similarity_method
        
        # 实体映射
        self.herb2idx: Dict[str, int] = {}
        self.comp2idx: Dict[str, int] = {}
        self.prot2idx: Dict[str, int] = {}
        self.disease2idx: Dict[str, int] = {}
        
        self.idx2herb: Dict[int, str] = {}
        self.idx2comp: Dict[int, str] = {}
        self.idx2prot: Dict[int, str] = {}
        self.idx2disease: Dict[int, str] = {}
        
        # 邻接矩阵
        self.HC_adj = None      # 草药-成分
        self.CP_adj = None      # 成分-蛋白质
        self.PD_adj = None      # 蛋白质-疾病
        
        # 密集矩阵缓存（避免重复 toarray() 调用）
        self._cp_dense_cache = None
        self._hc_dense_cache = None
        self._pd_dense_cache = None
        
        # Morgan指纹缓存（Tanimoto相似度）
        self._morgan_fps = None          # list of rdkit BitVect
        self._morgan_fps_matrix = None   # numpy 2D array
        self._morgan_ing_ids = None      # ingredient_ids 顺序数组
        self._morgan_id2idx = None        # ingredient_id → 指纹索引
        
        # 模型状态
        self.model = None
        self.is_trained = False
        
        # 预测缓存
        self._prediction_cache: Dict[str, Dict[str, float]] = {}
    
    # ==================== 数据构建 ====================
    
    def build_graph(
        self,
        herb_ingredients_df: pd.DataFrame,
        ingredient_targets_df: pd.DataFrame,
        disease_profiles_df: pd.DataFrame,
        ppi_genes: Optional[List[str]] = None
    ) -> Dict:
        """
        构建HDCTI异构图（H-C-P-D四层）
        
        Parameters
        ----------
        herb_ingredients_df : 草药-成分关系表
            columns: herb_id, herb_name, ingredient_id, ingredient_name, content_ratio
        ingredient_targets_df : 成分-靶标关系表
            columns: ingredient_id, target_gene, action_type
        disease_profiles_df : 疾病表达谱
            columns: disease_id, gene_symbol, logFC, P_value
        ppi_genes : PPI网络中的基因列表（用于过滤蛋白质节点）
        
        Returns
        -------
        dict : 图统计信息
        """
        print("[HDCTI] 构建异构图...")
        
        # --- 构建实体ID映射 ---
        herbs = sorted(herb_ingredients_df["herb_id"].unique())
        comps = sorted(herb_ingredients_df["ingredient_id"].unique())
        
        # 从ingredient_targets中获取所有蛋白质
        proteins_set = set(ingredient_targets_df["target_gene"].unique())
        # 加入疾病基因
        proteins_set.update(disease_profiles_df["gene_symbol"].unique())
        # 如果指定了PPI基因列表，过滤
        if ppi_genes:
            proteins_set = proteins_set.intersection(set(ppi_genes))
        proteins = sorted(proteins_set)
        
        diseases = sorted(disease_profiles_df["disease_id"].unique())
        
        self.herb2idx = {h: i for i, h in enumerate(herbs)}
        self.comp2idx = {c: i for i, c in enumerate(comps)}
        self.prot2idx = {p: i for i, p in enumerate(proteins)}
        self.disease2idx = {d: i for i, d in enumerate(diseases)}
        
        self.idx2herb = {v: k for k, v in self.herb2idx.items()}
        self.idx2comp = {v: k for k, v in self.comp2idx.items()}
        self.idx2prot = {v: k for k, v in self.prot2idx.items()}
        self.idx2disease = {v: k for k, v in self.disease2idx.items()}
        
        n_herbs = len(herbs)
        n_comps = len(comps)
        n_prots = len(proteins)
        n_diseases = len(diseases)
        
        # --- 构建H-C邻接矩阵 ---
        hc_rows, hc_cols = [], []
        for _, row in herb_ingredients_df.iterrows():
            h = row["herb_id"]
            c = row["ingredient_id"]
            if h in self.herb2idx and c in self.comp2idx:
                hc_rows.append(self.herb2idx[h])
                hc_cols.append(self.comp2idx[c])
        self.HC_adj = self._build_sparse(hc_rows, hc_cols, (n_herbs, n_comps))
        print(f"  H-C: {len(hc_rows)} 条边 ({n_herbs} herbs × {n_comps} compounds)")
        
        # --- 构建C-P邻接矩阵 ---
        cp_rows, cp_cols = [], []
        for _, row in ingredient_targets_df.iterrows():
            c = row["ingredient_id"]
            p = row["target_gene"]
            if c in self.comp2idx and p in self.prot2idx:
                cp_rows.append(self.comp2idx[c])
                cp_cols.append(self.prot2idx[p])
        self.CP_adj = self._build_sparse(cp_rows, cp_cols, (n_comps, n_prots))
        print(f"  C-P: {len(cp_rows)} 条边 ({n_comps} compounds × {n_prots} proteins)")
        
        # --- 构建P-D邻接矩阵 ---
        pd_rows, pd_cols = [], []
        for _, row in disease_profiles_df.iterrows():
            p = row["gene_symbol"]
            d = row["disease_id"]
            if p in self.prot2idx and d in self.disease2idx:
                pd_rows.append(self.prot2idx[p])
                pd_cols.append(self.disease2idx[d])
        self.PD_adj = self._build_sparse(pd_rows, pd_cols, (n_prots, n_diseases))
        print(f"  P-D: {len(pd_rows)} 条边 ({n_prots} proteins × {n_diseases} diseases)")
        
        stats = {
            "n_herbs": n_herbs, "n_comps": n_comps,
            "n_proteins": n_prots, "n_diseases": n_diseases,
            "hc_edges": len(hc_rows), "cp_edges": len(cp_rows),
            "pd_edges": len(pd_rows)
        }
        print(f"[HDCTI] 异构图构建完成: {stats}")
        return stats
    
    def _build_sparse(self, rows, cols, shape):
        """构建稀疏矩阵"""
        from scipy.sparse import coo_matrix
        if len(rows) == 0:
            return coo_matrix(shape, dtype=np.float32)
        data = np.ones(len(rows), dtype=np.float32)
        return coo_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)
    
    # ==================== Morgan指纹加载（Tanimoto Path1） ====================
    
    def _load_morgan_fingerprints(self):
        """加载预计算的Morgan指纹缓存（惰性加载）"""
        import os
        cache_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'morgan_fps.npz')
        if not os.path.exists(cache_path):
            print(f"[HDCTI] 警告: Morgan指纹缓存未找到 ({cache_path})，回退到Jaccard")
            self.similarity_method = "jaccard"
            return
        
        data = np.load(cache_path, allow_pickle=True)
        self._morgan_ing_ids = data['ingredient_ids']
        self._morgan_fps_matrix = data['fps_matrix']
        
        # 构建 ingredient_id → 指纹索引 映射
        self._morgan_id2idx = {int(ing_id): i for i, ing_id in enumerate(self._morgan_ing_ids)}
        
        # 用numpy行重建RDKit ExplicitBitVect
        from rdkit import DataStructs
        self._morgan_fps = []
        for i in range(len(self._morgan_ing_ids)):
            fp = DataStructs.ExplicitBitVect(2048)
            for j in np.where(self._morgan_fps_matrix[i] == 1)[0]:
                fp.SetBit(int(j))
            self._morgan_fps.append(fp)
        
        n_fps = len(self._morgan_fps)
        print(f"[HDCTI] Morgan指纹已加载: {n_fps} 个成分, "
              f"覆盖率 {n_fps/len(self.comp2idx)*100:.1f}%")
    
    def _path1_tanimoto(self, comp_idx: int, cp_dense: np.ndarray) -> np.ndarray:
        """
        Path 1 Tanimoto版本: 基于分子指纹的化学结构相似度传播
        替代Jaccard（基于共享靶标的循环论证）
        """
        ingredient_id = self.idx2comp[comp_idx]
        
        # 查指纹索引（类型兼容: int/str/numpy int）
        fp_idx = self._morgan_id2idx.get(int(ingredient_id))
        if fp_idx is None:
            # 该成分无SMILES，回退到Jaccard
            intersection1 = cp_dense @ cp_dense[comp_idx, :]
            row_sums = cp_dense.sum(axis=1)
            union1 = row_sums + cp_dense[comp_idx, :].sum() - intersection1
            union1[union1 == 0] = 1
            comp_sim_cc = intersection1 / union1
            comp_sim_cc[comp_idx] = 0
            return comp_sim_cc @ cp_dense
        
        # 计算该成分与所有其他成分的Tanimoto相似度
        query_fp = self._morgan_fps[fp_idx]
        
        from rdkit import DataStructs
        sims = np.array([DataStructs.TanimotoSimilarity(query_fp, fp) 
                         for fp in self._morgan_fps], dtype=np.float32)
        
        # 将指纹索引映射回comp2idx索引（类型兼容：统一用int）
        comp_sim_cc = np.zeros(cp_dense.shape[0], dtype=np.float32)
        # 预构建 int→comp_idx 的快速查找
        if not hasattr(self, '_morgan_fp_to_cidx'):
            self._morgan_fp_to_cidx = np.full(len(self._morgan_ing_ids), -1, dtype=np.int32)
            for fp_i, ing_id in enumerate(self._morgan_ing_ids):
                ing_id_int = int(ing_id)
                # 兼容 comp2idx 的 key 类型（int 或 str）
                c_idx = self.comp2idx.get(ing_id_int)
                if c_idx is None:
                    c_idx = self.comp2idx.get(str(ing_id_int))
                if c_idx is not None:
                    self._morgan_fp_to_cidx[fp_i] = c_idx
        
        valid_mask = self._morgan_fp_to_cidx >= 0
        comp_sim_cc[self._morgan_fp_to_cidx[valid_mask]] = sims[valid_mask]
        
        comp_sim_cc[comp_idx] = 0  # 排除自身
        return comp_sim_cc @ cp_dense

    # ==================== 基于超图传播的靶标预测（HDCTI核心） ====================
    
    def predict_new_targets(
        self,
        ingredient_id: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        预测成分的新靶标蛋白（HDCTI多路径超图传播）
        
        HDCTI核心思想：利用H-C-P-D异构图的多层结构同时传播，而非
        仅依赖C-P层的直接Jaccard相似度。
        
        三条传播路径：
        Path 1 (C→C→P): 通过共享靶标的成分相似性传播（Jaccard基础）
        Path 2 (C→H→C→P): 通过共同草药的高阶成分关联传播
        Path 3 (C→P→D→P): 通过疾病共靶标的蛋白质相似性传播
        
        Parameters
        ----------
        ingredient_id : 成分ID
        top_k : 返回Top-K个预测靶标
        
        Returns
        -------
        List[Tuple[str, float]]: [(target_gene, confidence_score), ...]
        """
        if top_k is None:
            top_k = self.top_k
        
        # 检查缓存
        if ingredient_id in self._prediction_cache:
            cached = self._prediction_cache[ingredient_id]
            return sorted(cached.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if ingredient_id not in self.comp2idx:
            print(f"[HDCTI] 警告: 成分 {ingredient_id} 不在图中")
            return []
        
        comp_idx = self.comp2idx[ingredient_id]
        n_prots = len(self.prot2idx)
        
        if self.CP_adj is None:
            return []
        
        # 缓存密集矩阵
        if self._cp_dense_cache is None:
            self._cp_dense_cache = self.CP_adj.toarray()
        if self._hc_dense_cache is None and self.HC_adj is not None:
            self._hc_dense_cache = self.HC_adj.toarray()
        if self._pd_dense_cache is None and self.PD_adj is not None:
            self._pd_dense_cache = self.PD_adj.toarray()
        
        cp_dense = self._cp_dense_cache
        comp_prot = cp_dense[comp_idx, :]  # 成分的已知靶标向量
        
        # ===== 三条传播路径 =====
        
        # Path 1: C→C→P (成分相似性传播)
        if self.similarity_method == "tanimoto":
            # Tanimoto: 基于Morgan分子指纹的化学结构相似度
            if self._morgan_fps is None:
                self._load_morgan_fingerprints()
            if self._morgan_fps is not None and len(self._morgan_fps) > 0:
                scores_p1 = self._path1_tanimoto(comp_idx, cp_dense)
            else:
                # 回退到Jaccard
                intersection1 = cp_dense @ comp_prot
                row_sums = cp_dense.sum(axis=1)
                union1 = row_sums + comp_prot.sum() - intersection1
                union1[union1 == 0] = 1
                comp_sim_cc = intersection1 / union1
                comp_sim_cc[comp_idx] = 0
                scores_p1 = comp_sim_cc @ cp_dense
        else:
            # Jaccard: 基于共享靶标的相似度（原始方法，保留对比）
            intersection1 = cp_dense @ comp_prot
            row_sums = cp_dense.sum(axis=1)
            union1 = row_sums + comp_prot.sum() - intersection1
            union1[union1 == 0] = 1
            comp_sim_cc = intersection1 / union1  # Jaccard成分相似度
            comp_sim_cc[comp_idx] = 0  # 排除自身
            scores_p1 = comp_sim_cc @ cp_dense  # 传播到蛋白质

        # 路径尺度对齐: 三路径原始量纲不一 (Path1 可达数, Path2/Path3 ~[0,1]),
        # 直接加权会让 Path1 主导, 权重0.5/0.25/0.25形同虚设。各路径先归一化到[0,1]再融合。
        _p1max = scores_p1.max()
        if _p1max > 0:
            scores_p1 = scores_p1 / _p1max

        # Path 2: C→H→C→P (超图路径：通过共享草药的高阶关联)
        # HDCTI核心创新：成分通过共同草药关联，捕获"配伍"信息
        scores_p2 = np.zeros(n_prots)
        if self._hc_dense_cache is not None:
            hc_dense = self._hc_dense_cache  # (n_herbs, n_comps)
            # H^T @ H: 草药共现矩阵 (n_comps, n_comps)
            # 归一化: 成分i和j的共现概率
            col_sums_hc = hc_dense.sum(axis=0)
            col_sums_hc[col_sums_hc == 0] = 1
            hc_norm = hc_dense / col_sums_hc  # 列归一化
            
            # 自门控: 类似HDCTI Self-Gating，筛选高质量连接
            hc_binary = (hc_norm > 0.1).astype(np.float32)
            
            # 成分-草药-成分 超图传播
            # comp_sim_chc = H^T_norm · H_norm (归一化共现)
            comp_sim_chc = hc_norm.T @ hc_binary  # (n_comps, n_comps)
            # 行列归一化（类似超图卷积的对称归一化）
            chc_rows = comp_sim_chc.sum(axis=1)
            chc_rows[chc_rows == 0] = 1
            comp_sim_chc = comp_sim_chc / chc_rows[:, np.newaxis]
            comp_sim_chc[comp_idx] = 0
            
            scores_p2 = comp_sim_chc[comp_idx, :] @ cp_dense
            _p2max = scores_p2.max()
            if _p2max > 0:
                scores_p2 = scores_p2 / _p2max

        # Path 3: C→P→D→P (通过疾病关联的蛋白质共功能传播)
        scores_p3 = np.zeros(n_prots)
        if self._pd_dense_cache is not None:
            pd_dense = self._pd_dense_cache  # (n_prots, n_diseases)
            # 蛋白质-疾病-蛋白质 关联
            col_sums_pd = pd_dense.sum(axis=0)
            col_sums_pd[col_sums_pd == 0] = 1
            pd_norm = pd_dense / col_sums_pd  # 列归一化
            
            # 蛋白质功能相似度矩阵: P@P^T (共享疾病)
            prot_sim = pd_norm @ pd_norm.T  # (n_prots, n_prots)
            # 度归一化
            prot_rows = prot_sim.sum(axis=1)
            prot_rows[prot_rows == 0] = 1
            prot_sim = prot_sim / prot_rows[:, np.newaxis]
            
            # 成分的已知靶标通过蛋白质功能相似性传播
            comp_prot_norm = comp_prot / max(comp_prot.sum(), 1)
            scores_p3 = comp_prot_norm @ prot_sim
            _p3max = scores_p3.max()
            if _p3max > 0:
                scores_p3 = scores_p3 / _p3max

        # ===== 多路径融合（加权） =====
        # 权重: Path1(直接化学相似性) = 0.5,
        #        Path2(草药配伍) = 0.25,
        #        Path3(疾病功能) = 0.25
        w1, w2, w3 = 0.5, 0.25, 0.25
        scores = w1 * scores_p1 + w2 * scores_p2 + w3 * scores_p3
        
        # 排除已知靶标
        scores[comp_prot > 0] = 0

        # 融合分归一化到 [0,1]: 否则 confidence_threshold(默认0.85) 对未归一化分数无效
        # (原始分可达数, 阈值形同虚设)。归一化后阈值表示"相对置信度", 与阈值语义一致。
        _smax = scores.max()
        if _smax > 0:
            scores = scores / _smax

        # ===== 输出Top-K =====
        # 获取得分最高的top_k索引
        top_indices = np.argpartition(-scores, min(top_k * 3, len(scores) - 1))[:top_k * 3]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        predictions = []
        for prot_idx in top_indices:
            score = float(scores[prot_idx])
            if score >= self.confidence_threshold:
                prot_name = self.idx2prot[prot_idx]
                predictions.append((prot_name, score))
            if len(predictions) >= top_k:
                break
        
        # 缓存结果
        cache = {name: s for name, s in predictions}
        self._prediction_cache[ingredient_id] = cache
        
        if len(predictions) > 0:
            print(f"[HDCTI] 成分 {ingredient_id}: "
                  f"P1={w1*100:.0f}%({self.similarity_method}) P2={w2*100:.0f}% P3={w3*100:.0f}% "
                  f"→ {len(predictions)} 个新靶标")
        return predictions
    
    def predict_all_new_targets(
        self,
        ingredient_targets_df: pd.DataFrame,
        herb_ingredients_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        批量预测所有成分的新靶标
        
        Returns
        -------
        DataFrame columns: ingredient_id, target_gene, confidence, source
        """
        all_predictions = []
        existing_pairs = set(
            zip(ingredient_targets_df["ingredient_id"], ingredient_targets_df["target_gene"])
        )
        
        # 只对已有至少一个已知靶标的成分做预测（旧版逻辑）
        # 避免为无靶标成分凭空生成预测，保持成分池规模可控
        known_ingredients = set(ingredient_targets_df["ingredient_id"].unique())
        all_ingredients = herb_ingredients_df["ingredient_id"].unique()
        all_ingredients = [i for i in all_ingredients if i in known_ingredients]
        n_total = len(all_ingredients)
        
        print(f"[HDCTI] 批量预测 {n_total} 个成分的新靶标...")
        for i, ing_id in enumerate(all_ingredients):
            preds = self.predict_new_targets(ing_id)
            for target_gene, conf in preds:
                if (ing_id, target_gene) not in existing_pairs:
                    all_predictions.append({
                        "ingredient_id": ing_id,
                        "target_gene": target_gene,
                        "confidence": conf,
                        "source": "HDCTI_predicted"
                    })
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{n_total}")
        
        result_df = pd.DataFrame(all_predictions)
        print(f"[HDCTI] 批量预测完成: {len(result_df)} 个新靶标对")
        return result_df
    
    # ==================== 靶标合并 ====================
    
    def merge_targets(
        self,
        existing_df: pd.DataFrame,
        new_predictions: pd.DataFrame,
        action_type_filler: int = 0,
        disease_dir: dict = None
    ) -> pd.DataFrame:
        """
        合并已有靶标与HDCTI预测的新靶标

        Parameters
        ----------
        existing_df : 现有ingredient_targets数据
        new_predictions : HDCTI预测的新靶标
        action_type_filler : 新靶标默认action_type（0=方向未知）
        disease_dir : 疾病基因方向字典 {gene_symbol: +1或-1}
                      +1表示疾病中下调（需要激活）, -1表示疾病中上调（需要抑制）
                      如果提供，则根据疾病方向自动推断HDCTI靶标方向

        Returns
        -------
        pd.DataFrame : 合并后的完整靶标表
        """
        if new_predictions.empty:
            print("[HDCTI] 无新靶标需要合并")
            return existing_df

        # 如果有疾病方向数据，统计并打印
        has_disease_info = disease_dir is not None and len(disease_dir) > 0
        if has_disease_info:
            print(f"[HDCTI] 启用疾病导向方向推断: {len(disease_dir)} 个基因有方向信息")

        merged = existing_df.copy()
        existing_pairs = set(zip(existing_df["ingredient_id"], existing_df["target_gene"]))

        # 统计方向推断结果
        n_positive = 0  # 疾病中下调 → 需要激活靶标 (+1)
        n_negative = 0  # 疾病中上调 → 需要抑制靶标 (-1)
        n_unknown = 0   # 不在疾病数据中 → 保持0

        new_rows = []
        for _, row in new_predictions.iterrows():
            ing_id = row["ingredient_id"]
            target = row["target_gene"]
            if (ing_id, target) not in existing_pairs:
                # ----- 方向推断逻辑 -----
                if has_disease_info and target in disease_dir:
                    # 该基因在疾病数据中，根据其表达方向确定草药的靶向方向
                    # 关键逻辑：疾病中上调的基因 → 需要抑制(-1)；疾病中下调的基因 → 需要激活(+1)
                    action = disease_dir[target]
                    if action > 0:
                        n_positive += 1
                    else:
                        n_negative += 1
                else:
                    # 该基因不在疾病数据中，使用默认值
                    action = action_type_filler
                    n_unknown += 1

                new_rows.append({
                    "ingredient_id": ing_id,
                    "target_gene": target,
                    "action_type": action,
                    "source": "HDCTI_predicted",
                    "confidence": row.get("confidence", self.confidence_threshold)
                })

        if has_disease_info:
            total = len(new_rows)
            print(f"[HDCTI] 方向推断: +1(激活)={n_positive}({100*n_positive//total if total else 0}%), "
                  f"-1(抑制)={n_negative}({100*n_negative//total if total else 0}%), "
                  f"0(未知)={n_unknown}({100*n_unknown//total if total else 0}%)")

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            # 确保列一致
            for col in merged.columns:
                if col not in new_df.columns:
                    new_df[col] = None
            new_df = new_df[merged.columns.intersection(new_df.columns)]
            merged = pd.concat([merged, new_df], ignore_index=True)
            print(f"[HDCTI] 合并完成: 新增 {len(new_rows)} 条靶标, 总计 {len(merged)} 条")

        return merged
    
    # ==================== 从TCMSP数据补充靶标 ====================
    
    @staticmethod
    def load_tcmsp_targets(
        tcmsp_targets_path: str = None,
        min_svm_score: float = 0.7,
        min_rf_score: float = 0.7
    ) -> pd.DataFrame:
        """
        从TCMSP爬取数据加载靶标预测
        
        TCMSP的all_targets.csv包含SVM_score和RF_score，
        表示靶标预测的置信度
        
        Parameters
        ----------
        tcmsp_targets_path : all_targets.csv路径
        min_svm_score : SVM最低得分阈值
        min_rf_score : RF最低得分阈值
        
        Returns
        -------
        DataFrame columns: ingredient_id, target_gene, svm_score, rf_score, source
        """
        if tcmsp_targets_path is None:
            tcmsp_targets_path = os.path.join(TCMSP_DATA_DIR, "all_targets.csv")
        
        if not os.path.exists(tcmsp_targets_path):
            print(f"[HDCTI] TCMSP靶标文件未找到: {tcmsp_targets_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(tcmsp_targets_path)
        
        # 按照置信度阈值过滤
        df_filtered = df[
            (df["SVM_score"] >= min_svm_score) & 
            (df["RF_score"] >= min_rf_score)
        ].copy()
        
        # 统一列名
        df_filtered["ingredient_id"] = df_filtered["molecule_ID"].astype(str).str.replace(".0", "")
        df_filtered["target_gene"] = df_filtered["target_name"]
        df_filtered["source"] = "TCMSP_SVM_RF"
        
        result = df_filtered[["ingredient_id", "target_gene", "SVM_score", "RF_score", "source"]]
        print(f"[HDCTI] 从TCMSP加载靶标: {len(result)} 条 (过滤后)")
        return result
    
    # ==================== 状态管理 ====================
    
    def save_state(self, path: str):
        """保存适配器状态"""
        import pickle
        state = {
            "herb2idx": self.herb2idx,
            "comp2idx": self.comp2idx,
            "prot2idx": self.prot2idx,
            "disease2idx": self.disease2idx,
            "prediction_cache": self._prediction_cache,
            "confidence_threshold": self.confidence_threshold,
            "top_k": self.top_k
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[HDCTI] 状态保存到: {path}")
    
    def load_state(self, path: str):
        """加载适配器状态"""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.herb2idx = state["herb2idx"]
        self.comp2idx = state["comp2idx"]
        self.prot2idx = state["prot2idx"]
        self.disease2idx = state["disease2idx"]
        self._prediction_cache = state.get("prediction_cache", {})
        self.confidence_threshold = state.get("confidence_threshold", self.confidence_threshold)
        self.top_k = state.get("top_k", self.top_k)
        self.idx2herb = {v: k for k, v in self.herb2idx.items()}
        self.idx2comp = {v: k for k, v in self.comp2idx.items()}
        self.idx2prot = {v: k for k, v in self.prot2idx.items()}
        self.idx2disease = {v: k for k, v in self.disease2idx.items()}
        print(f"[HDCTI] 状态加载完成")


# ==================== 便捷函数 ====================

def build_hdcti_adapter(
    herb_ingredients_df: pd.DataFrame,
    ingredient_targets_df: pd.DataFrame,
    disease_profiles_df: pd.DataFrame,
    ppi_df: pd.DataFrame = None,
    use_tcmsp: bool = True,
    confidence_threshold: float = HDCTI_CONFIDENCE_THRESHOLD,
    top_k: int = HDCTI_TOP_K
) -> HDCTIAdapter:
    """
    一站式构建HDCTI适配器（包含TCMSP数据补充）
    """
    adapter = HDCTIAdapter(
        confidence_threshold=confidence_threshold,
        top_k=top_k
    )
    
    # 获取PPI基因列表
    ppi_genes = None
    if ppi_df is not None:
        ppi_genes = set(ppi_df["protein_A"].tolist() + ppi_df["protein_B"].tolist())
        print(f"[HDCTI] PPI基因列表: {len(ppi_genes)} 个蛋白")
    
    # 构建异构图
    adapter.build_graph(
        herb_ingredients_df,
        ingredient_targets_df,
        disease_profiles_df,
        ppi_genes
    )
    
    # 可选：从TCMSP补充靶标
    if use_tcmsp:
        tcmsp_targets = HDCTIAdapter.load_tcmsp_targets()
        if not tcmsp_targets.empty:
            # 合并TCMSP靶标到ingredient_targets中
            ingredient_targets_df = adapter.merge_targets(
                ingredient_targets_df, tcmsp_targets
            )
    
    return adapter


if __name__ == "__main__":
    print("[测试] HDCTI适配器模块")
    print(f"  HDCTI代码路径: {HDCTI_PATH}")
    print(f"  TCMSP数据路径: {TCMSP_DATA_DIR}")
    print("  适配器已就绪，请在主流程中调用 build_hdcti_adapter()")