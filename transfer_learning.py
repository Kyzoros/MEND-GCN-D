"""
跨疾病迁移学习模块
============================================================================
核心思想:
  不同疾病之间共享分子机制（如炎症、氧化应激、凋亡通路），
  因此可以从数据充足的"源疾病"迁移知识到数据稀缺的"目标疾病"。

实现方式（层次1: 疾病嵌入相似性）:
  1. 疾病嵌入: 每个疾病的差异基因集经PPI网络传播得到"疾病扰动签名"
  2. 疾病相似度: 疾病嵌入之间的cosine相似度矩阵
  3. 迁移增强TMS:
     TMS'_i = (1 - λ) * TMS_i + λ * Σ_j sim(i,j) * TMS_j / Σ_j sim(i,j)
     其中 λ 是迁移强度, sim(i,j) 是疾病i和j的相似度

使用方式:
  transfer = CrossDiseaseTransfer(herb_signatures, builder, disease_profiles_df)
  transfer.build_disease_similarity()
  
  # 增强TMS
  enhanced = transfer.enhance_tms(tms_df, lambda_val=0.3)
  
  # 或直接获取增强后的排名
  rankings = transfer.enhanced_ranking(disease_id, lambda_val=0.3)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cosine

from config import TOP_BOTTOM_K


class CrossDiseaseTransfer:
    """
    跨疾病迁移学习引擎
    
    通过疾病差异基因集的PPI传播嵌入计算疾病间相似度，
    用"相似疾病的TMS"补强"目标疾病的TMS"。
    
    Parameters
    ----------
    herb_signatures : Dict[str, np.ndarray] or Dict[str, Dict[str, np.ndarray]]
        标准模式: Dict[herb_id, vector]
        注意力模式: Dict[disease_id, Dict[herb_id, vector]]
    builder : PerturbationSignatureBuilder, 用于PPI扩散
    disease_df : pd.DataFrame, 疾病表达谱
        必须包含列: disease_id, gene_symbol [, logFC]
    """
    
    def __init__(self, herb_signatures, builder, disease_df: pd.DataFrame):
        self.herb_signatures = herb_signatures
        self.builder = builder
        self.disease_df = disease_df.copy()
        
        # 疾病列表
        self.disease_ids = sorted(disease_df['disease_id'].unique())
        self.n_diseases = len(self.disease_ids)
        
        # 疾病嵌入和相似度矩阵（惰性计算）
        self._disease_embeddings: Dict[str, np.ndarray] = {}
        self._similarity_matrix: Optional[pd.DataFrame] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def _get_disease_genes(self, disease_id: str) -> List[str]:
        """获取疾病的差异基因列表"""
        sub = self.disease_df[self.disease_df['disease_id'] == disease_id]
        sub = sub.sort_values('logFC', ascending=False)
        return sub['gene_symbol'].tolist()
    
    # ==================== 疾病嵌入 ====================
    
    def build_disease_embedding(self, disease_id: str,
                                use_ppi_propagation: bool = True) -> np.ndarray:
        """
        构建疾病嵌入向量
        
        将疾病的差异基因列表编码为一个向量:
        方法1 (use_ppi_propagation=True): 
            将差异基因+1/-1赋值后沿PPI网络扩散 → 疾病"表达扰动"签名
            与草药签名在同一空间，可直接比较
        方法2 (use_ppi_propagation=False):
            仅用疾病基因集合的指示向量
        
        Parameters
        ----------
        disease_id : str
        use_ppi_propagation : bool, 是否做PPI传播
        
        Returns
        -------
        np.ndarray, shape (n_proteins,) 或 (n_disease_genes,)
        """
        cache_key = f"{disease_id}_prop={use_ppi_propagation}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        genes = self._get_disease_genes(disease_id)
        
        if use_ppi_propagation:
            # 方法1: PPI传播 → 与草药签名同一空间
            n_prots = self.builder.n_proteins
            vec = np.zeros(n_prots, dtype=np.float64)
            n_genes = len(genes)
            
            # 前TOP_BOTTOM_K个上调基因=+1, 后TOP_BOTTOM_K个下调基因=-1
            k = min(TOP_BOTTOM_K, n_genes // 2)
            for i, g in enumerate(genes[:k]):
                idx = self.builder.gene2idx.get(g, -1)
                if idx >= 0:
                    vec[idx] = 1.0
            for i, g in enumerate(genes[-k:]):
                idx = self.builder.gene2idx.get(g, -1)
                if idx >= 0:
                    vec[idx] = -1.0
            
            # PPI扩散
            embedding = self.builder.diffuse(vec)
        else:
            # 方法2: 简单基因集合指示向量
            n_prots = self.builder.n_proteins
            embedding = np.zeros(n_prots, dtype=np.float64)
            for g in genes:
                idx = self.builder.gene2idx.get(g, -1)
                if idx >= 0:
                    embedding[idx] = 1.0
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self._embedding_cache[cache_key] = embedding
        return embedding
    
    def build_disease_embeddings(self, use_ppi_propagation: bool = True,
                                 verbose: bool = True) -> Dict[str, np.ndarray]:
        """批量构建所有疾病的嵌入"""
        if verbose:
            print(f"[迁移] 构建 {self.n_diseases} 个疾病的嵌入 (PPI传播={use_ppi_propagation})...")
        
        for i, did in enumerate(self.disease_ids):
            self.build_disease_embedding(did, use_ppi_propagation=use_ppi_propagation)
            if verbose and (i + 1) % 50 == 0:
                print(f"  进度: {i + 1}/{self.n_diseases}")
        
        self._disease_embeddings = {
            did: self._embedding_cache[f"{did}_prop={use_ppi_propagation}"]
            for did in self.disease_ids
        }
        return self._disease_embeddings
    
    # ==================== 疾病相似度 ====================
    
    def compute_disease_similarity(self, did1: str, did2: str) -> float:
        """
        计算两个疾病的cosine相似度
        
        Returns
        -------
        float, [-1, 1], 1=完全相同, 0=无关, -1=相反
        """
        emb1 = self.build_disease_embedding(did1)
        emb2 = self.build_disease_embedding(did2)
        
        sim = 1.0 - cosine(emb1, emb2)
        return max(-1.0, min(1.0, sim))
    
    def build_similarity_matrix(self, min_similarity: float = 0.0,
                                verbose: bool = True) -> pd.DataFrame:
        """
        构建疾病相似度矩阵
        
        Parameters
        ----------
        min_similarity : float, 相似度阈值，低于此值的设为0（稀疏化）
        
        Returns
        -------
        DataFrame: index=disease_id, columns=disease_id, values=similarity
        """
        if verbose:
            print(f"[迁移] 构建 {self.n_diseases}×{self.n_diseases} 相似度矩阵...")
        
        # 先构建所有嵌入
        self.build_disease_embeddings(verbose=verbose)
        
        # 计算成对相似度
        n = self.n_diseases
        sim_matrix = np.eye(n, dtype=np.float64)  # 自身相似度=1
        
        n_pairs = n * (n - 1) // 2
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_disease_similarity(
                    self.disease_ids[i], self.disease_ids[j]
                )
                if abs(sim) >= min_similarity:
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
                
                count += 1
                if verbose and count % 500 == 0:
                    print(f"  相似度计算: {count}/{n_pairs}")
        
        df = pd.DataFrame(sim_matrix, index=self.disease_ids, columns=self.disease_ids)
        
        if verbose:
            n_edges = (np.abs(sim_matrix) > 0).sum() - n  # 减去对角线
            print(f"  矩阵完成: {n_edges} 对非零相似度 (阈值>{min_similarity})")
            print(f"  平均相似度: {sim_matrix[sim_matrix < 1].mean():.4f}")
            print(f"  最大相似度: {sim_matrix[sim_matrix < 1].max():.4f}")
        
        self._similarity_matrix = df
        return df
    
    # ==================== TMS增强 ====================
    
    def enhance_tms(self, tms_df: pd.DataFrame,
                    lambda_val: float = 0.3,
                    min_similarity: float = 0.05) -> pd.DataFrame:
        """
        用跨疾病迁移学习增强TMS结果
        
        公式:
            TMS'_i = (1 - λ) * TMS_i + λ * Σ_j sim(i,j) * TMS_j / Σ_j sim(i,j)
        
        Parameters
        ----------
        tms_df : DataFrame, HDmap-S输出的TMS矩阵
            index=herb_id, columns=disease_id, values=TMS
        lambda_val : float, 迁移强度 [0, 1]
            0=完全依赖自身TMS, 1=完全依赖其他疾病的TMS
        min_similarity : float, 最小相似度阈值
        
        Returns
        -------
        DataFrame, 增强后的TMS矩阵（同形状）
        """
        if self._similarity_matrix is None:
            self.build_similarity_matrix(min_similarity=min_similarity, verbose=False)
        
        sim_df = self._similarity_matrix
        
        # 确保疾病对齐
        diseases_in_tms = [c for c in tms_df.columns if c in sim_df.index]
        diseases_in_tms = [c for c in diseases_in_tms if c != 'herb_id']
        
        if len(diseases_in_tms) < 2:
            print(f"[迁移] 只有 {len(diseases_in_tms)} 个疾病，无需迁移增强")
            return tms_df.copy()
        
        print(f"[迁移] 增强TMS: λ={lambda_val}, {len(diseases_in_tms)} 个疾病")
        
        result = tms_df.copy()
        
        for did in diseases_in_tms:
            if did not in sim_df.index:
                continue
            
            # 获取该疾病的相似度向量（排除自身）
            sim_vec = sim_df.loc[did].copy()
            sim_vec[did] = 0.0
            
            # 过滤低相似度
            sim_vec = sim_vec[sim_vec.abs() >= min_similarity]
            
            if sim_vec.sum() == 0:
                continue
            
            # 归一化相似度为权重
            weights = sim_vec / sim_vec.abs().sum()
            
            # 计算其他疾病TMS的加权和
            other_tms = np.zeros(len(result))
            for other_did, w in weights.items():
                if other_did in result.columns:
                    other_tms += w * result[other_did].values
            
            # 增强: TMS' = (1-λ) * TMS + λ * 加权迁移
            original = result[did].values
            result[did] = (1 - lambda_val) * original + lambda_val * other_tms
        
        return result
    
    def enhanced_ranking(self, disease_id: str,
                         tms_df: Optional[pd.DataFrame] = None,
                         lambda_val: float = 0.3,
                         top_n: int = 50,
                         min_similarity: float = 0.05,
                         higher_is_better: bool = False) -> pd.DataFrame:
        """
        获取迁移增强后的Top-N排名

        Parameters
        ----------
        disease_id : str, 目标疾病
        tms_df     : DataFrame or None, 原始TMS矩阵。None时需传入
        lambda_val : float, 迁移强度
        higher_is_better : True=越高越治疗(dual合并通道); False=越低(TMS越负)越治疗(weighted/ndcg)

        Returns
        -------
        DataFrame: rank, herb_id, original_TMS, enhanced_TMS
        """
        if tms_df is None:
            raise ValueError("需要提供tms_df")

        enhanced = self.enhance_tms(tms_df, lambda_val=lambda_val,
                                    min_similarity=min_similarity)

        if disease_id not in enhanced.columns:
            print(f"[迁移] 疾病 {disease_id} 不在TMS结果中")
            return pd.DataFrame()

        original_scores = tms_df[disease_id] if disease_id in tms_df.columns else 0
        enhanced_scores = enhanced[disease_id]

        records = []
        for hid in enhanced.index:
            orig = original_scores.get(hid, 0) if hasattr(original_scores, 'get') else 0
            records.append({
                'herb_id': hid,
                'original_TMS': orig,
                'enhanced_TMS': enhanced_scores[hid],
                'delta': enhanced_scores[hid] - orig
            })

        df = pd.DataFrame(records)
        # 排序方向: dual合并通道越高越好(降序); 其余越低(TMS越负)越好(升序)
        df = df.sort_values('enhanced_TMS', ascending=(not higher_is_better)).head(top_n)
        df['rank'] = range(1, len(df) + 1)
        return df[['rank', 'herb_id', 'original_TMS', 'enhanced_TMS', 'delta']]
    
    def get_most_similar_diseases(self, disease_id: str,
                                   top_n: int = 10) -> pd.DataFrame:
        """获取与目标疾病最相似的其他疾病"""
        if self._similarity_matrix is None:
            self.build_similarity_matrix(verbose=False)
        
        if disease_id not in self._similarity_matrix.index:
            return pd.DataFrame()
        
        sims = self._similarity_matrix.loc[disease_id].sort_values(ascending=False)
        # 排除自身
        sims = sims[sims.index != disease_id].head(top_n)
        
        return pd.DataFrame({
            'disease_id': sims.index,
            'similarity': sims.values
        }).reset_index(drop=True)
    
    def get_transfer_summary(self, disease_id: str,
                              tms_df: pd.DataFrame,
                              lambda_val: float = 0.3,
                              top_n: int = 20) -> Dict:
        """生成迁移学习摘要"""
        top_similar = self.get_most_similar_diseases(disease_id, top_n=5)
        enhanced_rank = self.enhanced_ranking(disease_id, tms_df, lambda_val, top_n)
        
        return {
            'target_disease': disease_id,
            'n_diseases_total': self.n_diseases,
            'lambda': lambda_val,
            'top_similar_diseases': top_similar.to_dict('records') if not top_similar.empty else [],
            'enhanced_top20': enhanced_rank.to_dict('records') if not enhanced_rank.empty else [],
        }
