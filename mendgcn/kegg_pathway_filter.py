"""
KEGG通路过滤器 - 基于2跳邻域过滤HDCTI预测靶标
只保留与疾病基因在KEGG网络中≤2跳可达的靶标蛋白
"""
import numpy as np
import json
import os
from scipy.sparse import load_npz, csr_matrix
from typing import Set, List, Optional
import sys
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)

class KEGGPathwayFilter:
    """
    KEGG通路过滤器
    
    策略: 在KEGG共通路网络上，只保留与疾病基因距离≤2跳的预测靶标。
    KEGG网络边 = 两个基因在同一个KEGG通路中，因此2跳可达到的含义是
    基因与疾病基因共享直接或间接通路关联。
    """
    
    def __init__(self, kegg_adj_path: str = None, gene2idx_path: str = None):
        if kegg_adj_path is None:
            kegg_adj_path = 'data/processed/kegg_adjacency.npz'
        if gene2idx_path is None:
            gene2idx_path = 'data/processed/kegg_gene2idx.json'
        
        self.kegg_adj = None
        self.gene2idx = {}
        self.idx2gene = {}
        self.reachable_set: Set[str] = set()
        self.is_built = False
        
        if os.path.exists(kegg_adj_path) and os.path.exists(gene2idx_path):
            self.kegg_adj = load_npz(kegg_adj_path)
            with open(gene2idx_path) as f:
                self.gene2idx = json.load(f)
            self.idx2gene = {v: k for k, v in self.gene2idx.items()}
            print(f"[KEGG Filter] 网络已加载: {self.kegg_adj.shape[0]} 基因, "
                  f"{self.kegg_adj.nnz} 条边")
        else:
            print(f"[KEGG Filter] 警告: KEGG网络文件未找到，过滤已禁用")
    
    def build_filter(self, disease_genes: List[str], max_hops: int = 2):
        """
        构建2跳可达到集合
        
        Parameters
        ----------
        disease_genes : 疾病相关基因列表
        max_hops : 最大跳数
        """
        if self.kegg_adj is None:
            self.is_built = True  # 无网络，跳过过滤
            return
        
        # 找到疾病基因在KEGG中的索引
        disease_indices = []
        kegg_genes_in = 0
        for g in disease_genes:
            if g in self.gene2idx:
                disease_indices.append(self.gene2idx[g])
                kegg_genes_in += 1
        
        print(f"[KEGG Filter] 疾病基因: {len(disease_genes)} 个, "
              f"在KEGG网络中: {kegg_genes_in} 个")
        
        if len(disease_indices) == 0:
            print(f"[KEGG Filter] 无疾病基因在KEGG网络中，过滤已禁用")
            self.is_built = True
            return
        
        # 构建种子向量
        n_genes = self.kegg_adj.shape[0]
        seed = np.zeros(n_genes, dtype=np.float32)
        for idx in disease_indices:
            seed[idx] = 1.0
        
        # 2跳传播: reachable = seed | (A @ seed > 0) | (A @ A @ seed > 0)
        reachable = seed.copy()
        
        # 1跳邻居
        one_hop = self.kegg_adj @ seed
        reachable[one_hop > 0] = 1
        
        # 2跳邻居
        two_hop = self.kegg_adj @ one_hop
        reachable[two_hop > 0] = 1
        
        # 排除种子自身（疾病基因已知，不需要过滤）
        reachable_indices = np.where(reachable > 0)[0]
        self.reachable_set = {self.idx2gene[int(i)] for i in reachable_indices}
        
        # 统计
        n_reachable = len(self.reachable_set)
        n_direct = int((one_hop > 0).sum())
        n_two_hop = n_reachable - n_direct - len(disease_indices)
        
        print(f"[KEGG Filter] 2跳可达基因: {n_reachable} "
              f"(直达:{n_direct}, 2跳:{max(0, n_two_hop)}, "
              f"占比:{n_reachable/n_genes*100:.1f}%)")
        
        self.is_built = True
    
    def filter_targets(self, target_genes: List[str]) -> List[str]:
        """
        过滤靶标列表，只保留在可达集合中的
        
        Returns
        -------
        List[str]: 过滤后的靶标基因列表
        """
        if not self.is_built:
            self._warn_not_built()
            return target_genes
        
        if self.kegg_adj is None or len(self.reachable_set) == 0:
            return target_genes  # 无网络/无疾病基因，不过滤
        
        filtered = [g for g in target_genes if g in self.reachable_set]
        n_removed = len(target_genes) - len(filtered)
        if n_removed > 0:
            print(f"[KEGG Filter] 过滤掉 {n_removed}/{len(target_genes)} 个靶标 "
                  f"(不在疾病2跳邻域内)")
        return filtered
    
    def is_reachable(self, gene: str) -> bool:
        """检查单个基因是否在2跳可达范围内"""
        if not self.is_built:
            return True  # 未构建，默认通过
        if self.kegg_adj is None:
            return True
        return len(self.reachable_set) == 0 or gene in self.reachable_set
    
    def _warn_not_built(self):
        print("[KEGG Filter] 警告: 过滤器未构建，请先调用 build_filter()")
    
    def get_stats(self) -> dict:
        return {
            "n_kegg_genes": self.kegg_adj.shape[0] if self.kegg_adj is not None else 0,
            "n_reachable": len(self.reachable_set),
            "is_active": self.is_built and self.kegg_adj is not None and len(self.reachable_set) > 0
        }


def create_ad_filter() -> KEGGPathwayFilter:
    """
    创建AD疾病专用的KEGG通路过滤器
    从疾病表达谱中提取AD相关基因作为种子
    """
    import pandas as pd
    
    # 加载AD疾病基因
    profiles_path = 'data/processed/disease_profiles_AD.csv'
    if not os.path.exists(profiles_path):
        print(f"[KEGG Filter] 警告: 未找到 {profiles_path}")
        f = KEGGPathwayFilter()
        f.is_built = True
        return f
    
    profiles = pd.read_csv(profiles_path)
    ad_genes = profiles[profiles['disease_id'] == 'AD']['gene_symbol'].unique().tolist()
    
    f = KEGGPathwayFilter()
    f.build_filter(ad_genes, max_hops=2)
    return f


if __name__ == '__main__':
    # 测试
    f = create_ad_filter()
    print(f"\n统计: {f.get_stats()}")
    test_genes = ['APP', 'PSEN1', 'MAPT', 'TNF', 'NOT_A_GENE']
    filtered = f.filter_targets(test_genes)
    print(f"过滤结果: {filtered}")
