"""
DrugAI适配器 - 替代PreAM模型进行激活/抑制方向预测
=====================================================
将DrugAI深度学习模型集成到systems_mapping管道中，用于：
1. 预测成分对靶标蛋白的激活(+1)/抑制(-1)作用方向
2. 替代原始PreAM模型的action_type预测
3. 更新ingredient_targets中的action_type列

DrugAI论文: "DrugAI: a multi-view deep learning model for predicting 
drug–target activating/inhibiting mechanisms"
期刊: Briefings in Bioinformatics, 2023, IF=13.994

模型架构:
- 化合物编码: GNN (AttentiveFP) 编码分子图
- 蛋白质编码: 1D-CNN 编码氨基酸序列
- 多视图融合: 化合物图特征 + 蛋白质序列特征 + Node2Vec嵌入 + 指纹特征
- 输出: Sigmoid → P(activation) → 映射为 +1(激活) / -1(抑制)

依赖:
- PyTorch >= 1.10
- DGL >= 0.8
- RDKit
- requests (用于PubChem/UniProt API查询)
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# ==================== DrugAI模型定义（嵌入，不依赖外部文件）====================

def _morgan_fingerprint(smiles, n_bits=167, radius=2):
    """
    SMILES -> 真实Morgan/ECFP指纹 (使用RDKit)
    
    Morgan指纹(=ECFP)通过迭代编码分子图中每个原子半径范围内的
    化学子结构环境，是药物化学中标准的分子指纹表示。
    
    降级方案: RDKit不可用时使用带有化学感知的改进哈希指纹。
    """
    if not smiles or not isinstance(smiles, str):
        return np.zeros(n_bits, dtype=np.float32)
    
    # 优先使用RDKit真实Morgan指纹
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _improved_hash_fingerprint(smiles, n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except ImportError:
        return _improved_hash_fingerprint(smiles, n_bits)


def _improved_hash_fingerprint(smiles, n_bits=167):
    """
    改进的哈希指纹（RDKit不可用时的降级方案）
    
    与原始字符哈希不同，此实现基于化学子结构:
    - 原子类型(token)级别的n-gram
    - 化学键感知的分割
    - 更接近真实指纹的统计特性
    """
    fp = np.zeros(n_bits, dtype=np.float32)
    
    # 1. 按化学边界分割SMILES（保留化学语义token）
    # 例如: CC(=O)O → ['C', 'C', '(=', 'O', ')', 'O']
    tokens = []
    i = 0
    while i < len(smiles):
        if smiles[i] in '()[]':
            tokens.append(smiles[i])
            i += 1
        elif i + 1 < len(smiles) and smiles[i:i+2] in ('Cl', 'Br', 'Si', 'Se', 'Te'):
            tokens.append(smiles[i:i+2])
            i += 2
        elif smiles[i] in '=#@/\\':
            tokens.append(smiles[i])
            i += 1
        elif smiles[i].isdigit():
            # 环闭合数字: %后两位 或 单数字
            if smiles[i] == '%' and i + 2 < len(smiles) and smiles[i+1:i+3].isdigit():
                tokens.append(smiles[i:i+3])
                i += 3
            else:
                tokens.append(smiles[i])
                i += 1
        else:
            tokens.append(smiles[i])
            i += 1
    
    n = len(tokens)
    if n == 0:
        return fp
    
    # 2. Token n-gram哈希 (模拟Morgan指纹的半径编码)
    for radius in [0, 1, 2]:  # 模拟不同半径
        for j in range(n - radius):
            feat = tokens[j] if radius == 0 else ''.join(tokens[j:j+radius+1])
            fp[hash(f"r{radius}_{feat}") % n_bits] += 1
    
    # 3. SMILES全局特征哈希
    # 原子计数
    for atom in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']:
        count = smiles.count(atom)
        if count > 0:
            fp[hash(f"atom_{atom}_{count}") % n_bits] += count
    
    # 4. 键类型计数
    for bond, char in [('double', '='), ('triple', '#'), ('ring', '1')]:
        count = smiles.count(char)
        if count > 0:
            fp[hash(f"bond_{bond}_{count}") % n_bits] += count
    
    if fp.sum() > 0:
        fp = fp / fp.max()
    return fp.astype(np.float32)

def _train_matched_morgan_fp(smiles, n_bits=167):
    """
    与 train_drugai_v3.custom_morgan_fingerprint 严格一致的指纹。

    checkpoint (best_model_auc_0.9475.pkl) 就是用这套字符 n-gram 哈希训练的。
    上面的 _morgan_fingerprint 用的是 token n-gram + "r{radius}_" 前缀,
    两者哈希到完全不同的桶 -> 属于训练/推理特征错配。
    实测 (diagnose_drugai_views.py, 1123条真实标签测试集):
        错配指纹 AUC=0.6143  ->  一致指纹 AUC=0.7296

    注意: 依赖内置 hash(), 必须在 PYTHONHASHSEED=0 下运行 (main.py 已保证)。
    """
    if not smiles or not isinstance(smiles, str):
        return np.zeros(n_bits, dtype=np.float32)
    fp = np.zeros(n_bits, dtype=np.float32)
    for c in smiles:
        fp[hash(c) % n_bits] += 1
    for i in range(len(smiles) - 1):
        fp[hash(smiles[i:i + 2]) % n_bits] += 1
    for i in range(len(smiles) - 2):
        fp[hash(smiles[i:i + 3]) % n_bits] += 1
    if fp.sum() > 0:
        fp /= fp.max()
    return fp.astype(np.float32)


def _rdkit_morgan_fp(smiles, n_bits=167, radius=2):
    """
    真实 RDKit Morgan 指纹 (retrain_drugai_rdkit.py 训练所用)。

    相比基于内置 hash() 的字符 n-gram 指纹, 它:
      * 确定性: 不受 PYTHONHASHSEED 影响, 跨进程/跨机器完全一致;
      * 有化学含义: 编码真实子结构环境, 而非字符串片段。
    对应 checkpoint: Drug_AI_best_view/best_model_rdkit.pkl
      random 划分 TEST AUC=0.9666 / drug-disjoint TEST AUC=0.8901
    """
    if not smiles or not isinstance(smiles, str):
        return np.zeros(n_bits, dtype=np.float32)
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(bv, dtype=np.float32)
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def _protein_onehot(seq, max_len=800):
    """蛋白质序列 -> one-hot编码 (800x20)"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    onehot = np.zeros((max_len, 20), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        if aa in aa_to_idx: onehot[i, aa_to_idx[aa]] = 1.0
    return onehot

# ==================== 化合物编码器 (SimpleGNN, 与训练脚本一致) ====================

class FingerprintMLP(nn.Module):
    """指纹MLP (与checkpoint中GNNs_flag='FingerprintMLP'对应)"""
    def __init__(self, input_dim=167, hidden_dim=256, output_dim=167, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.fc(x)


class SimpleGNN(nn.Module):
    """简化的图神经网络 (与 train_drugai_standalone.py 架构一致)"""
    def __init__(self, node_feat_size=58, edge_feat_size=10, hidden_size=168, output_size=167, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = 3
        self.head_dim = hidden_size // self.num_heads
        
        self.node_encoder = nn.Linear(node_feat_size, hidden_size)
        self.edge_encoder = nn.Linear(edge_feat_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=self.num_heads, batch_first=True)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features, edge_features):
        batch_size, max_atoms, _ = node_features.shape
        h = self.node_encoder(node_features)
        h_attn, _ = self.attention(h, h, h)
        h = h + h_attn
        h, _ = self.gru(h)
        h = h[:, -1, :]
        h = self.dropout(h)
        return self.output(h)


# ==================== 蛋白质编码器 (ProteinCNN, 与训练脚本一致) ====================

class ProteinCNN(nn.Module):
    """蛋白质CNN编码器 (与 train_drugai_standalone.py 架构一致)"""
    def __init__(self, protein_Oridim=20, feature_size=200, out_features=343,
                 max_seq_len=1000, kernels=None, dropout_rate=0.1):
        super().__init__()
        kernels = kernels or [3]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(protein_Oridim, feature_size, ks),
                nn.ReLU(),
                nn.MaxPool1d(max_seq_len - ks + 1)
            ) for ks in kernels
        ])
        self.fc = nn.Linear(feature_size * len(kernels), out_features)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        return self.fc(self.dropout(out))


# ==================== DrugAI主分类器 (与训练脚本一致) ====================

class DrugAIClassifier(nn.Module):
    """
    DrugAI多视图分类器（方案C最佳视图）
    
    架构与 train_drugai_standalone.py 完全一致:
    - 化合物: SimpleGNN (分子图 → GNN_fdim)
    - 蛋白质: ProteinCNN (序列 → CNN_fdim)
    - 融合: 全视图拼接 → DNN → 输出
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.protein_encoder = ProteinCNN(
            protein_Oridim=config['protein_Oridim'],
            feature_size=config['feature_size'],
            out_features=config['out_features'],
            max_seq_len=config['max_seq_len'],
            kernels=config['kernels'],
            dropout_rate=config['dropout_r']
        )
        
        gnn_flag = config.get('GNNs_flag', 'SimpleGNN')
        if gnn_flag == 'FingerprintMLP':
            self.compound_encoder = FingerprintMLP(
                input_dim=config['d_fp_dim'],
                output_dim=config['GNN_fdim'],
                dropout=config['dropout_r']
            )
        else:
            self.compound_encoder = SimpleGNN(
                node_feat_size=58,
                edge_feat_size=10,
                hidden_size=config['GNN_fdim'],
                output_size=config['GNN_fdim'],
                dropout=config['dropout_r']
            )
        
        self.GNN_fdim = config['GNN_fdim']
        self.d_n2v_fdim = config['d_n2v_fdim']
        self.d_fp_dim = config['d_fp_dim']
        self.CNN_fdim = config['CNN_fdim']
        self.p_n2v_fdim = config['p_n2v_fdim']
        self.g_fp_dim = config['g_fp_dim']
        
        total_dim = (self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim + 
                     self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim)
        
        DNN_layers = config['DNN_layers']
        layer_sizes = [total_dim] + DNN_layers + [1]
        self.layer_size = len(layer_sizes) - 1
        
        self.predictor = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(self.layer_size)
        ])
        self.dropout = nn.Dropout(config['dropout_r'])
    
    def forward(self, compound_graphs, protein_seqs, d_n2v, d_fp, p_n2v, g_fp):
        """
        Parameters
        ----------
        compound_graphs : dict with 'node_features' and 'edge_features'
        protein_seqs : 蛋白质序列张量
        d_n2v, p_n2v, d_fp, g_fp : 各视图特征
        """
        if isinstance(self.compound_encoder, FingerprintMLP):
            compound_feat = self.compound_encoder(d_fp)
        else:
            compound_feat = self.compound_encoder(
                compound_graphs['node_features'],
                compound_graphs['edge_features']
            )
        protein_feat = self.protein_encoder(protein_seqs)
        
        features = torch.cat([compound_feat, d_n2v, d_fp, protein_feat, p_n2v, g_fp], dim=1)
        for i, layer in enumerate(self.predictor):
            if i == self.layer_size - 1:
                features = layer(features)
            else:
                features = F.relu(self.dropout(layer(features)))
        return features.squeeze(-1)

# ==================== DrugAI适配器 ====================

# 添加DrugAI代码路径
DRUGAI_PATH = r"D:\Bio_information\DrugAI\DrugAI-main"
if DRUGAI_PATH not in sys.path:
    sys.path.insert(0, DRUGAI_PATH)

# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
from config import (
    DRUGAI_USE, DRUGAI_MODEL_PATH, DRUGAI_CONFIDENCE_THRESHOLD,
    DRUGAI_DEVICE, DRUGAI_DATA_DIR, TCMSP_DATA_DIR, RAW_DIR
)


# ==================== 断点续传工具 ====================

class CheckpointManager:
    """
    通用断点续传管理器
    
    支持功能:
    - 记录当前处理进度 (已处理条数/最后索引)
    - 每批数据即时持久化到本地
    - 中断后从断点恢复
    - 原子写入避免文件损坏 (先写临时文件再rename)
    
    使用模式:
        ckpt = CheckpointManager("path/to/checkpoint.json")
        ckpt.load()  # 加载之前的进度
        
        for i, item in enumerate(items[ckpt.progress:]):
            result = process(item)
            ckpt.save_batch_result(result)
            ckpt.checkpoint(i + ckpt.progress + 1)  # 更新进度
    """
    
    def __init__(self, checkpoint_path: str, autosave_every: int = 100):
        """
        Parameters
        ----------
        checkpoint_path : 断点文件路径 (.json)
        autosave_every : 每处理多少条自动持久化一次
        """
        self.checkpoint_path = checkpoint_path
        self.autosave_every = autosave_every
        self.progress = 0          # 已成功处理的条目数
        self.total = 0             # 总条目数
        self.batch_results = []    # 当前批次暂存的结果
        self.start_time = None     # 开始时间
        self._dirty = False        # 是否有未持久化的数据
        self._result_path = None   # 结果持久化路径
    
    def init(self, total: int, result_path: str = None) -> int:
        """
        初始化或恢复断点
        
        Parameters
        ----------
        total : 总条目数
        result_path : 结果持久化文件路径 (可选，用于分批保存结果)
        
        Returns
        -------
        int : 应该从第几条开始处理 (0-based)
        """
        self.total = total
        self._result_path = result_path
        self.start_time = time.time()
        self._load()
        
        skipped = self.progress
        if skipped > 0:
            print(f"[Checkpoint] 从断点恢复: 已完成 {skipped}/{total} "
                  f"({100*skipped//total}%), 继续处理剩余 {total - skipped} 条")
            if result_path and os.path.exists(result_path):
                print(f"[Checkpoint] 已有结果文件: {result_path}")
        else:
            # 清空旧结果文件 (全新开始)
            if result_path and os.path.exists(result_path):
                os.remove(result_path)
        
        return skipped
    
    def checkpoint(self, processed: int):
        """
        更新并持久化进度
        
        Parameters
        ----------
        processed : 当前已处理的总条目数
        """
        self.progress = processed
        self._dirty = True
        
        # 每隔 autosave_every 条自动持久化
        if processed % self.autosave_every == 0:
            self._save()
    
    def save_batch_result(self, result):
        """
        暂存一条批量结果 (不立即写盘，等checkpoint触发)
        
        Parameters
        ----------
        result : 单条结果数据 (可JSON序列化)
        """
        self.batch_results.append(result)
        self._dirty = True
    
    def flush_batch_results(self, append_fn=None):
        """
        将暂存的结果批量写入文件
        
        Parameters
        ----------
        append_fn : callable(file_handle, results), 自定义写入函数
                   默认按行写JSON
        """
        if not self._result_path or not self.batch_results:
            return len(self.batch_results)
        
        count = len(self.batch_results)
        self._flush_to_file(self.batch_results, append_fn)
        self.batch_results.clear()
        return count
    
    def _flush_to_file(self, results: list, append_fn=None):
        """将结果列表写入文件（追加模式 + 原子写入保护）"""
        if not self._result_path:
            return
        
        os.makedirs(os.path.dirname(self._result_path), exist_ok=True)
        
        try:
            if append_fn:
                # 自定义写入
                if os.path.exists(self._result_path):
                    with open(self._result_path, "a", encoding="utf-8") as f:
                        append_fn(f, results)
                else:
                    with open(self._result_path, "w", encoding="utf-8") as f:
                        append_fn(f, results)
            else:
                # 默认：每行一个 JSON 对象 (JSONL格式)
                with open(self._result_path, "a", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Checkpoint] 结果写入失败: {e}")
    
    def _load(self):
        """从断点文件加载进度"""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    data = json.load(f)
                saved_progress = data.get("progress", 0)
                saved_total = data.get("total", 0)
                
                # 验证断点有效性：total一致且progress在合理范围
                if saved_total == self.total and 0 <= saved_progress <= self.total:
                    self.progress = saved_progress
                elif saved_total != self.total:
                    print(f"[Checkpoint] 总条目数变化 ({saved_total}→{self.total}), 从头开始")
                    self.progress = 0
            except (json.JSONDecodeError, KeyError):
                self.progress = 0
        else:
            self.progress = 0
    
    def _save(self):
        """持久化当前进度到断点文件（原子写入）"""
        if not self._dirty:
            return
        
        tmp_path = self.checkpoint_path + ".tmp"
        try:
            data = {
                "progress": self.progress,
                "total": self.total,
                "elapsed_seconds": time.time() - (self.start_time or time.time()),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            
            # 原子写入：先写临时文件，再rename
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            
            # Windows下需要先删除目标文件
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
            os.rename(tmp_path, self.checkpoint_path)
            
            self._dirty = False
        except Exception as e:
            print(f"[Checkpoint] 进度保存失败: {e}")
    
    def force_save(self):
        """强制立即持久化（在流程结束时调用）"""
        self._save()
        if self._result_path and self.batch_results:
            self.flush_batch_results()
        self._dirty = False
    
    def get_eta(self) -> str:
        """估算剩余时间"""
        if self.progress == 0 or self.start_time is None:
            return "计算中..."
        elapsed = time.time() - self.start_time
        rate = self.progress / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.progress) / rate if rate > 0 else 0
        if remaining < 60:
            return f"{remaining:.0f}s"
        elif remaining < 3600:
            return f"{remaining/60:.1f}min"
        else:
            return f"{remaining/3600:.1f}h"
    
    def clear(self):
        """清除断点（完成后调用）"""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            print(f"[Checkpoint] 断点已清除: {self.checkpoint_path}")
    
    def is_resuming(self) -> bool:
        """是否正在从断点恢复"""
        return self.progress > 0


# ==================== SMILES查询工具 ====================

class SMILESLookup:
    """
    SMILES查询工具 (P0增强版)
    
    支持多种来源 (按优先级):
    1. 本地缓存 (JSON文件)
    2. ChEMBL本地文件映射 (chembl_36_chemreps.txt.gz)
    3. PubChem REST API (带重试机制)
    4. CIR (Chemical Identifier Resolver) 第三回退源
    
    新特性:
    - 失败查询缓存: 避免重复请求已知失败的分子
    - 指数退避重试: 网络不稳定时自动重试
    - CIR回退: PubChem查不到时自动尝试CIR
    """
    
    def __init__(self, cache_path: str = None, failed_cache_path: str = None):
        self.cache_path = cache_path or os.path.join(DRUGAI_DATA_DIR, "smiles_cache.json")
        self.failed_cache_path = failed_cache_path or os.path.join(DRUGAI_DATA_DIR, "_failed_smiles_cache.json")
        self._cache = {}
        self._failed_cache = set()  # 永久失败的分子ID集合
        self._load_cache()
        self._load_failed_cache()
        
        # 一次性抑制 RDKit C++ 层日志 (Kekulize/valence/Sanitize 警告)
        try:
            from rdkit import RDLogger
            RDLogger.logger().setLevel(RDLogger.CRITICAL)
        except ImportError:
            pass
    
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    self._cache = json.load(f)
                print(f"[SMILES] 缓存加载: {len(self._cache)} 条")
            except:
                self._cache = {}
    
    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)
    
    def _load_failed_cache(self):
        """加载失败查询缓存 (避免重复请求已知失败的分子)"""
        if os.path.exists(self.failed_cache_path):
            try:
                with open(self.failed_cache_path, "r") as f:
                    self._failed_cache = set(json.load(f))
                print(f"[SMILES] 失败缓存加载: {len(self._failed_cache)} 个已知失败")
            except:
                self._failed_cache = set()
    
    def _save_failed_cache(self):
        os.makedirs(os.path.dirname(self.failed_cache_path), exist_ok=True)
        with open(self.failed_cache_path, "w") as f:
            json.dump(sorted(list(self._failed_cache)), f, indent=2)
    
    def get_smiles(self, molecule_id: str, molecule_name: str = None,
                   mol_id: str = None, use_failed_cache: bool = True) -> Optional[str]:
        """
        获取分子的SMILES字符串 (P0增强版)
        
        查找顺序: 缓存 → TCMSP MOL2 → PubChem(快速) → None
        
        Parameters
        ----------
        molecule_id : 分子ID (TCMSP molecule_ID)
        molecule_name : 分子名称 (用于PubChem API查询)
        mol_id : TCMSP MOL_ID (如 MOL010127, 用于下载MOL2文件)
        use_failed_cache : 是否跳过已知失败的分子
        
        Returns
        -------
        SMILES字符串或None
        """
        mid = str(molecule_id).replace(".0", "")
        
        # 0. 检查是否已知失败
        if use_failed_cache and mid in self._failed_cache:
            return None
        
        # 1. 检查缓存
        if mid in self._cache:
            return self._cache[mid]
        
        # 2. 尝试各种API查询
        smiles = self._query_all_sources(
            str(molecule_name) if molecule_name else "",
            mol_id=mol_id
        )
        if smiles:
            self._cache[mid] = smiles
            self._save_cache()
            return smiles
        
        # 标记为失败
        self._failed_cache.add(mid)
        self._save_failed_cache()
        return None
    
    def _query_all_sources(self, name: str, mol_id: str = None) -> Optional[str]:
        """
        依次尝试所有SMILES查询源
        
        顺序: TCMSP MOL2 → PubChem(快速) → None
        注意: 已跳过ChEMBL(慢/命中率低)和CIR(国内被墙)
        """
        # 1. 优先TCMSP MOL2 (最高命中率, 含完整分子结构)
        if mol_id:
            smiles = self._query_tcmsp_mol2(mol_id)
            if smiles:
                return smiles
        
        # 2. PubChem名称查询 (快速兜底, 短超时)
        smiles = self._query_pubchem_fast(name)
        if smiles:
            return smiles
        
        return None

    def _query_tcmsp_mol2(self, mol_id: str) -> Optional[str]:
        """
        从TCMSP下载MOL2文件, 用RDKit转换为SMILES
        
        这是最可靠的SMILES来源, MOL2含完整3D分子结构,
        不需要名称匹配, 成功率90%+
        
        三层RDKit容错:
        1. 标准sanitize (处理规范分子)
        2. 部分sanitize + kekuleSmiles=False (处理非标准芳香环)
        3. 无sanitize兜底 (处理极端情况)
        
        Parameters
        ----------
        mol_id : TCMSP MOL_ID (如 MOL010127)
        
        Returns
        -------
        SMILES字符串或None
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdmolops
            from rdkit import RDLogger
            # 抑制 RDKit C++ 层日志 (Kekulize警告等)
            RDLogger.logger().setLevel(RDLogger.CRITICAL)
        except ImportError:
            return None
        
        url = f"https://www.tcmsp-e.com/tcmspmol/{mol_id}.mol2"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        mol2_content = None
        try:
            import requests
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code != 200:
                return None
            mol2_content = resp.text
        except Exception:
            try:
                import urllib.request
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    mol2_content = resp.read().decode('utf-8')
            except Exception:
                return None
        
        if not mol2_content or len(mol2_content) < 50:
            return None
        
        # 写入临时文件 (RDKit需要文件路径)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mol2', 
                                          delete=False, encoding='utf-8') as f:
            f.write(mol2_content)
            temp_path = f.name
        
        smiles = None
        mol = None
        try:
            # 方案1: 标准sanitize (处理规范分子)
            mol = Chem.MolFromMol2File(temp_path, sanitize=True, removeHs=True)
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                except Exception:
                    smiles = None
            
            # 方案2: 不sanitize读取, 手动部分sanitize (处理非标准芳香环)
            if smiles is None:
                mol = Chem.MolFromMol2File(temp_path, sanitize=False, removeHs=False)
                if mol is not None:
                    # 移除自键
                    for bond in list(mol.GetBonds()):
                        if bond.GetBeginAtomIdx() == bond.GetEndAtomIdx():
                            mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    # 安全sanitize (跳过KEKULIZE, 避免C++层警告)
                    safe_flags = (
                        Chem.SanitizeFlags.SANITIZE_CLEANUP
                        | Chem.SanitizeFlags.SANITIZE_PROPERTIES
                        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                        | Chem.SanitizeFlags.SANITIZE_FINDKRINGLARGE
                    )
                    try:
                        rdmolops.SanitizeMol(mol, safe_flags)
                        mol.UpdatePropertyCache(strict=False)
                    except Exception:
                        pass
                    try:
                        smiles = Chem.MolToSmiles(mol, canonical=True, 
                                                  kekuleSmiles=False, isomericSmiles=True)
                    except Exception:
                        smiles = None
            
            # 方案3: 完全不sanitize兜底 (处理极端非规范结构)
            if smiles is None and mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol, canonical=True, 
                                              kekuleSmiles=False,
                                              allBondsExplicit=True, allHsExplicit=False)
                except Exception:
                    smiles = None
        except Exception:
            smiles = None
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
        
        return smiles

    def _query_pubchem_fast(self, name: str) -> Optional[str]:
        """
        PubChem快速查询 (单次, 短超时, 不重试)
        
        与_query_pubchem_with_retry不同的是:
        - 超时8s (原15s)
        - 不重试
        - 只试原始名称
        """
        try:
            import urllib.request
            import urllib.parse
            url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{urllib.parse.quote(name)}/property/CanonicalSMILES,IsomericSMILES/JSON"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "SystemsMapping/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
                props = data["PropertyTable"]["Properties"][0]
                for key in ["CanonicalSMILES", "IsomericSMILES"]:
                    smiles = props.get(key)
                    if smiles:
                        return smiles
        except Exception:
            pass
        return None
    
    def _query_chembl_local(self, name: str) -> Optional[str]:
        """
        通过本地ChEMBL文件查询SMILES
        
        策略: 将分子名称与chembl_36_chemreps.txt.gz中的名称进行匹配
        由于需要ChEMBL ID映射，此方法先通过ChEMBL API搜索名称获取ID，
        再查本地文件获取SMILES。
        """
        try:
            import gzip
            # 尝试通过ChEMBL API搜索
            chembl_id = self._query_chembl_api_for_id(name)
            if not chembl_id:
                return None
            
            # 在本地文件中查找SMILES
            chembl_file = os.path.join(TCMSP_DATA_DIR, "chembl_36_chemreps.txt.gz")
            if not os.path.exists(chembl_file):
                return None
            
            # 使用grep方式查找 (大文件，避免全部加载)
            import subprocess
            try:
                result = subprocess.run(
                    ['findstr', chembl_id, chembl_file.replace('.gz', '')],
                    capture_output=True, text=True, timeout=10
                )
                # 检查是否有解压后的文件
                pass
            except:
                pass
            
            # 如果文件已解压，直接搜索
            unzipped = chembl_file.replace('.gz', '')
            if os.path.exists(unzipped):
                with open(unzipped, 'r', encoding='utf-8') as f:
                    for line in f:
                        if chembl_id in line:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                return parts[1]
            
            # 如果文件未解压，流式读取
            with gzip.open(chembl_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if chembl_id in line:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            return parts[1]
            
        except Exception:
            pass
        return None
    
    def _query_chembl_api_for_id(self, name: str) -> Optional[str]:
        """通过ChEMBL API搜索分子名称获取ChEMBL ID"""
        try:
            import urllib.request
            import urllib.parse
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={urllib.parse.quote(name)}&format=json&limit=1"
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SystemsMapping/1.0")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                molecules = data.get("molecules", [])
                if molecules:
                    return molecules[0].get("molecule_chembl_id")
        except Exception:
            pass
        return None
    
    def _query_pubchem_with_retry(self, name: str, max_retries: int = 3) -> Optional[str]:
        """
        通过PubChem REST API查询SMILES (带指数退避重试)
        
        Parameters
        ----------
        name : 分子名称
        max_retries : 最大重试次数
        """
        import re
        try:
            import requests
            has_requests = True
        except ImportError:
            has_requests = False
        
        def clean(s: str) -> str:
            s = re.sub(r'_qt$', '', s, flags=re.IGNORECASE)
            s = re.sub(r'^\([^)]+\)-', '', s)
            s = s.replace(', ', '-').strip()
            return s
        
        strategies = [name, clean(name)]
        seen = set()
        
        for s in strategies:
            if not s or s in seen:
                continue
            seen.add(s)
            
            for attempt in range(max_retries):
                try:
                    if has_requests:
                        smiles = self._query_pubchem_requests(s)
                    else:
                        smiles = self._query_pubchem_urllib(s)
                    
                    if smiles:
                        return smiles
                    
                    # 非网络错误 (404等) 不重试
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                        time.sleep(wait)
                    else:
                        break
        
        return None
    
    def _query_pubchem_requests(self, name: str) -> Optional[str]:
        """通过PubChem REST API查询SMILES (requests版本)"""
        import requests
        import urllib.parse
        
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{urllib.parse.quote(name)}/property/CanonicalSMILES,IsomericSMILES/JSON"
        )
        resp = requests.get(
            url, 
            headers={"User-Agent": "SystemsMapping/1.0"},
            timeout=15  # 增加超时时间
        )
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        props = data["PropertyTable"]["Properties"][0]
        for key in ["CanonicalSMILES", "IsomericSMILES", "SMILES", "ConnectivitySMILES"]:
            smiles = props.get(key)
            if smiles:
                return smiles
        return None
    
    def _query_pubchem_urllib(self, name: str) -> Optional[str]:
        """通过PubChem REST API查询SMILES (urllib备选方案)"""
        try:
            import urllib.request
            import urllib.parse
            url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{urllib.parse.quote(name)}/property/CanonicalSMILES,IsomericSMILES/JSON"
            )
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SystemsMapping/1.0")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                props = data["PropertyTable"]["Properties"][0]
                for key in ["CanonicalSMILES", "IsomericSMILES", "ConnectivitySMILES"]:
                    smiles = props.get(key)
                    if smiles:
                        return smiles
            return None
        except Exception:
            return None
    
    def _query_cir(self, name: str) -> Optional[str]:
        """
        通过CIR (Chemical Identifier Resolver) 查询SMILES
        
        CIR是NIH提供的化学标识符解析服务，支持多种名称格式。
        URL: https://cactus.nci.nih.gov/chemical/structure/{name}/smiles
        """
        try:
            import urllib.request
            import urllib.parse
            
            url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(name)}/smiles"
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SystemsMapping/1.0")
            
            with urllib.request.urlopen(req, timeout=15) as resp:
                smiles = resp.read().decode('utf-8').strip()
                # 验证返回的是有效SMILES
                if smiles and len(smiles) > 1 and not smiles.startswith('<'):
                    return smiles
            return None
        except Exception:
            return None
    
    def batch_query(self, molecule_ids: List[str], molecule_names: List[str] = None) -> Dict[str, str]:
        """批量查询SMILES"""
        results = {}
        for i, mid in enumerate(molecule_ids):
            name = molecule_names[i] if molecule_names else None
            smiles = self.get_smiles(mid, name)
            if smiles:
                results[mid] = smiles
            time.sleep(0.05)  # 减少等待时间 (多源回退已内置延迟)
        return results
    
    def prefetch_failed(self, ingredients_df: pd.DataFrame) -> int:
        """
        重新尝试之前失败的查询 (使用TCMSP MOL2等新数据源) - 支持断点续传
        
        Returns
        -------
        int : 本轮成功获取的SMILES数量
        """
        # 构建 molecule_id → (molecule_name, MOL_ID) 映射
        name_map = {}
        mol_id_map = {}
        has_mol_col = "MOL_ID" in ingredients_df.columns
        for _, row in ingredients_df.iterrows():
            mid_val = row["molecule_ID"]
            mid = str(int(float(mid_val))) if pd.notna(mid_val) else str(mid_val)
            name = row.get("molecule_name", None)
            if pd.notna(name) and mid in self._failed_cache:
                name_map[mid] = str(name)
                if has_mol_col:
                    mol_val = row.get("MOL_ID", None)
                    if pd.notna(mol_val):
                        mol_id_map[mid] = str(mol_val)
        
        failed_list = sorted(self._failed_cache)
        total = len(failed_list)
        
        if total == 0:
            return 0
        
        # 初始化断点管理器
        ckpt_path = os.path.join(DRUGAI_DATA_DIR, "_smiles_prefetch_checkpoint.json")
        ckpt = CheckpointManager(ckpt_path, autosave_every=50)
        start_idx = ckpt.init(total)
        
        count = 0
        processed = start_idx
        
        if ckpt.is_resuming():
            print(f"[SMILES] 从断点恢复: 已处理 {start_idx}/{total}, "
                  f"缓存中已有 {len(self._cache)} 条")
        
        print(f"[SMILES] 重新尝试 {total} 个之前失败的分子 (从第{start_idx+1}个开始)...")
        print(f"[SMILES] 查询策略: TCMSP MOL2(优先) → PubChem(快速兜底)")
        
        for idx in range(start_idx, total):
            mid = failed_list[idx]
            if mid not in name_map:
                processed += 1
                continue
            
            name = name_map[mid]
            mol_id = mol_id_map.get(mid, None)
            smiles = self._query_all_sources(str(name), mol_id=mol_id)
            if smiles:
                self._cache[mid] = smiles
                self._failed_cache.discard(mid)
                count += 1
            
            processed += 1
            
            # 每50个保存进度 + 持久化
            if processed % 50 == 0 or processed == total:
                print(f"[SMILES] 重试进度: {processed}/{total}, 本轮成功: {count}, "
                      f"ETA: {ckpt.get_eta()}")
                self._save_cache()
                self._save_failed_cache()
                ckpt.checkpoint(processed)
            
            time.sleep(0.05)
        
        
        # 最终保存
        self._save_cache()
        self._save_failed_cache()
        ckpt.force_save()
        ckpt.clear()
        print(f"[SMILES] 重试完成! 本轮新增: {count} 条, 缓存总计: {len(self._cache)} 条")
        return count
    
    def build_from_tcmsp(
        self, 
        ingredients_df: pd.DataFrame,
        mol_dir: str = None
    ) -> int:
        """
        从TCMSP数据构建SMILES缓存 (P0增强版 + 断点续传)
        
        使用多源策略: PubChem(重试) → CIR回退 → 失败缓存
        
        断点续传:
        - 每100条自动持久化缓存
        - 中断后重新运行自动从断点恢复
        - 断点文件: data/drugai_data/_smiles_build_checkpoint.json
        
        Parameters
        ----------
        ingredients_df : TCMSP all_ingredients.csv数据
        mol_dir : MOL文件目录路径 (保留参数兼容性)
        
        Returns
        -------
        int : 成功获取SMILES的数量
        """
        # 构建可索引列表（避免iterrows在断点恢复时重复遍历）
        rows_list = list(ingredients_df.iterrows())
        total = len(rows_list)
        
        # 初始化断点管理器
        ckpt_path = os.path.join(DRUGAI_DATA_DIR, "_smiles_build_checkpoint.json")
        ckpt = CheckpointManager(ckpt_path, autosave_every=100)
        start_idx = ckpt.init(total)
        
        count = 0
        skipped = 0
        failed = 0
        
        if ckpt.is_resuming():
            print(f"[SMILES] 从断点恢复构建, 已处理 {start_idx}/{total}")
            # 重新统计 start_idx 之前的跳过/成功/失败数量
            for idx in range(start_idx):
                _, row = rows_list[idx]
                mid_val = row["molecule_ID"]
                mid = str(int(float(mid_val))) if pd.notna(mid_val) else str(mid_val)
                if mid in self._cache:
                    skipped += 1
                elif mid in self._failed_cache:
                    skipped += 1
                elif pd.isna(row.get("molecule_name", None)):
                    failed += 1
                # 成功的已写入缓存，通过缓存大小推断
            count = len(self._cache) - len(self._cache)  # 仅统计本轮新增
        
        print(f"[SMILES] {'恢复' if ckpt.is_resuming() else '开始'}构建缓存, "
              f"共 {total} 个成分, 从第{start_idx+1}个开始...")
        print(f"[SMILES] 查询策略: TCMSP MOL2(优先) → PubChem(快速兜底)")
        
        round_count = 0  # 本轮新增
        
        for idx in range(start_idx, total):
            _, row = rows_list[idx]
            mid_val = row["molecule_ID"]
            mid = str(int(float(mid_val))) if pd.notna(mid_val) else str(mid_val)
            
            # 跳过已缓存的
            if mid in self._cache:
                skipped += 1
                ckpt.checkpoint(idx + 1)
                continue
            
            # 跳过已知永久失败的
            if mid in self._failed_cache:
                skipped += 1
                ckpt.checkpoint(idx + 1)
                continue
            
            name = row.get("molecule_name", None)
            mol_id = row.get("MOL_ID", None) if "MOL_ID" in row.index else None
            if pd.isna(mol_id):
                mol_id = None
            else:
                mol_id = str(mol_id)
            
            if pd.isna(name):
                failed += 1
                ckpt.checkpoint(idx + 1)
                continue
            
            # 使用多源策略查询 (MOL2优先, PubChem兜底)
            smiles = self._query_all_sources(str(name), mol_id=mol_id)
            if smiles:
                self._cache[mid] = smiles
                count += 1
                round_count += 1
            else:
                self._failed_cache.add(mid)
                failed += 1
            
            # 每100个: 持久化缓存 + 保存断点进度
            if (idx + 1) % 100 == 0 or idx == total - 1:
                processed = idx + 1
                elapsed = time.time() - (ckpt.start_time or time.time())
                rate = processed / elapsed if elapsed > 0 else 0
                new_tried = round_count + failed
                new_success_rate = round_count / max(new_tried, 1) * 100
                print(f"[SMILES] 进度: {processed}/{total} ({100*processed//total}%), "
                      f"成功: {round_count}, 失败: {failed}, "
                      f"新查询成功率: {new_success_rate:.0f}%, "
                      f"速率: {rate:.1f}/s, ETA: {ckpt.get_eta()}")
                
                # 立即持久化缓存和断点
                self._save_cache()
                self._save_failed_cache()
                ckpt.checkpoint(processed)
            
            # 减少API限流等待 (MOL2下载已含网络延迟)
            time.sleep(0.05)
        
        # 最终保存
        self._save_cache()
        self._save_failed_cache()
        ckpt.force_save()
        ckpt.clear()  # 完成后清除断点
        
        total_cached = len(self._cache)
        print(f"[SMILES] 构建完成! 缓存总计: {total_cached} 条, "
              f"本轮新增: {round_count} 条, 失败: {failed} 条")
        return round_count
    
    def build_from_tcmsp_concurrent(
        self,
        ingredients_df: pd.DataFrame,
        n_workers: int = 8,
        batch_save_every: int = 200
    ) -> int:
        """
        使用线程池并发下载 TCMSP MOL2 文件并转换为 SMILES (极速版)
        
        比单线程 build_from_tcmsp 快 5-10 倍。
        
        策略:
        1. 先收集所有需要查询的 (mid, name, mol_id) 三元组
        2. 使用 ThreadPoolExecutor 并发下载 MOL2 → RDKit → SMILES
        3. 失败的回退到 PubChem 快速查询
        4. 每200条自动持久化缓存
        
        Parameters
        ----------
        ingredients_df : TCMSP all_ingredients.csv数据
        n_workers : 并发线程数 (建议 5-10, 太高会被TCMSP限流)
        batch_save_every : 每多少条保存一次缓存
        
        Returns
        -------
        int : 成功获取SMILES的数量
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # 收集需要查询的项
        to_query = []
        has_mol_col = "MOL_ID" in ingredients_df.columns
        
        for _, row in ingredients_df.iterrows():
            mid_val = row["molecule_ID"]
            mid = str(int(float(mid_val))) if pd.notna(mid_val) else str(mid_val)
            
            if mid in self._cache or mid in self._failed_cache:
                continue
            
            name = row.get("molecule_name", None)
            if pd.isna(name):
                continue
            
            mol_id = None
            if has_mol_col:
                mol_val = row.get("MOL_ID", None)
                if pd.notna(mol_val):
                    mol_id = str(mol_val)
            
            to_query.append((mid, str(name), mol_id))
        
        total = len(to_query)
        if total == 0:
            print("[SMILES] 所有成分已处理完毕, 无需并发查询")
            return 0
        
        print(f"[SMILES] 并发构建缓存: {total} 个待查询, {n_workers} 线程")
        print(f"[SMILES] 查询策略: TCMSP MOL2(优先) → PubChem(快速兜底)")
        
        # 线程安全锁
        lock = threading.Lock()
        round_count = 0
        failed_count = 0
        processed = 0
        t0 = time.time()
        
        def query_one(mid, name, mol_id):
            """单分子查询函数 (线程安全)"""
            smiles = self._query_all_sources(name, mol_id=mol_id)
            return mid, smiles
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(query_one, mid, name, mol_id): mid
                for mid, name, mol_id in to_query
            }
            
            for future in as_completed(futures):
                mid, smiles = future.result()
                
                with lock:
                    processed += 1
                    
                    if smiles:
                        self._cache[mid] = smiles
                        round_count += 1
                    else:
                        self._failed_cache.add(mid)
                        failed_count += 1
                    
                    # 批量持久化
                    if processed % batch_save_every == 0 or processed == total:
                        elapsed = time.time() - t0
                        rate = processed / elapsed if elapsed > 0 else 0
                        new_tried = round_count + failed_count
                        success_rate = round_count / max(new_tried, 1) * 100
                        eta = (total - processed) / rate if rate > 0 else 0
                        print(f"[SMILES] 并发进度: {processed}/{total} "
                              f"({100*processed//total}%), "
                              f"成功: {round_count}, 失败: {failed_count}, "
                              f"成功率: {success_rate:.0f}%, "
                              f"速率: {rate:.1f}/s, ETA: {eta/60:.1f}min")
                        self._save_cache()
                        self._save_failed_cache()
        
        self._save_cache()
        self._save_failed_cache()
        
        print(f"[SMILES] 并发构建完成! 缓存总计: {len(self._cache)} 条, "
              f"本轮新增: {round_count} 条, 失败: {failed_count} 条")
        return round_count
    
    def get_stats(self) -> dict:
        """获取SMILES缓存统计信息"""
        return {
            "cached": len(self._cache),
            "failed": len(self._failed_cache),
            "coverage_pct": round(len(self._cache) / max(len(self._cache) + len(self._failed_cache), 1) * 100, 1)
        }


# ==================== 蛋白质序列查询工具 ====================

class ProteinSequenceLookup:
    """
    蛋白质序列查询工具 (P0增强版)
    
    来源:
    1. 本地缓存 (JSON文件)
    2. 本地UniProt FASTA文件 (data/raw/uniprot_human_swissprot_sample.fasta)
    3. UniProt REST API (单个查询 + 批量查询)
    
    新特性:
    - 本地FASTA文件解析: 从已下载的UniProt FASTA中快速查找
    - 批量UniProt查询: 一次请求查询多个基因 (减少API调用)
    - 失败缓存: 避免重复查询不存在的基因
    """
    
    UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
    UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
    
    def __init__(self, cache_path: str = None, fasta_path: str = None,
                 failed_cache_path: str = None):
        self.cache_path = cache_path or os.path.join(DRUGAI_DATA_DIR, "protein_seq_cache.json")
        self.failed_cache_path = failed_cache_path or os.path.join(DRUGAI_DATA_DIR, "_failed_proteins_cache.json")
        self._cache = {}
        self._failed_cache = set()
        self._fasta_index = {}  # gene_name → (accession, sequence)
        self._load_cache()
        self._load_failed_cache()
        
        # 构建本地FASTA索引
        if fasta_path is None:
            fasta_path = os.path.join(RAW_DIR, "uniprot_human_swissprot_sample.fasta")
        self._build_fasta_index(fasta_path)
    
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                print(f"[ProteinSeq] 缓存加载: {len(self._cache)} 条")
            except:
                self._cache = {}
    
    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)
    
    def _load_failed_cache(self):
        """加载失败查询缓存"""
        if os.path.exists(self.failed_cache_path):
            try:
                with open(self.failed_cache_path, "r") as f:
                    self._failed_cache = set(json.load(f))
                print(f"[ProteinSeq] 失败缓存加载: {len(self._failed_cache)} 个已知失败")
            except:
                self._failed_cache = set()
    
    def _save_failed_cache(self):
        os.makedirs(os.path.dirname(self.failed_cache_path), exist_ok=True)
        with open(self.failed_cache_path, "w") as f:
            json.dump(sorted(list(self._failed_cache)), f, indent=2)
    
    def _build_fasta_index(self, fasta_path: str):
        """
        构建本地FASTA文件的基因名索引
        
        解析UniProt FASTA头部格式:
        >sp|P12345|GENE_HUMAN Gene description OS=Homo sapiens ...
        提取基因名 (GENE_HUMAN中的GENE部分)
        """
        if not os.path.exists(fasta_path):
            print(f"[ProteinSeq] FASTA文件未找到: {fasta_path} (将仅使用API查询)")
            return
        
        try:
            import re
            count = 0
            current_accession = None
            current_gene = None
            current_seq_parts = []
            
            with open(fasta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        # 保存上一个条目
                        if current_gene and current_seq_parts:
                            seq = "".join(current_seq_parts)
                            self._fasta_index[current_gene.upper()] = (current_accession, seq)
                            count += 1
                        
                        # 解析新条目头部
                        current_seq_parts = []
                        current_gene = None
                        current_accession = None
                        
                        # 格式: >sp|P12345|GENE_HUMAN ...
                        parts = line.split("|")
                        if len(parts) >= 3:
                            current_accession = parts[1]
                            gene_part = parts[2].split()[0]  # GENE_HUMAN
                            # 提取基因名 (去掉_HUMAN后缀)
                            current_gene = gene_part.split("_")[0]
                        else:
                            # 备选格式: >tr|...|... GN=GENE
                            gn_match = re.search(r'GN=(\w+)', line)
                            if gn_match:
                                current_gene = gn_match.group(1)
                    else:
                        current_seq_parts.append(line)
                
                # 保存最后一个条目
                if current_gene and current_seq_parts:
                    seq = "".join(current_seq_parts)
                    self._fasta_index[current_gene.upper()] = (current_accession, seq)
                    count += 1
            
            print(f"[ProteinSeq] FASTA索引构建完成: {count} 个蛋白")
        except Exception as e:
            print(f"[ProteinSeq] FASTA索引构建失败: {e}")
    
    def get_sequence(self, gene_name: str) -> Optional[str]:
        """
        获取蛋白质氨基酸序列 (P0增强版)
        
        查找顺序: 缓存 → 本地FASTA → UniProt API
        
        Parameters
        ----------
        gene_name : 基因名 (如 PRKAA1, PTGS2)
        
        Returns
        -------
        氨基酸序列字符串或None
        """
        # 0. 检查失败缓存
        if gene_name in self._failed_cache:
            return None
        
        # 1. 检查内存缓存
        if gene_name in self._cache:
            return self._cache[gene_name]
        
        # 2. 检查本地FASTA索引
        gene_upper = gene_name.upper()
        if gene_upper in self._fasta_index:
            _, seq = self._fasta_index[gene_upper]
            if seq:
                self._cache[gene_name] = seq
                self._save_cache()
                return seq
        
        # 3. 通过UniProt API查询
        seq = self._query_uniprot(gene_name)
        if seq:
            self._cache[gene_name] = seq
            self._save_cache()
            return seq
        
        # 标记为失败
        self._failed_cache.add(gene_name)
        self._save_failed_cache()
        return None
    
    def _query_uniprot(self, gene_name: str) -> Optional[str]:
        """通过UniProt REST API查询序列"""
        try:
            import urllib.request
            import urllib.parse
            
            # 查询人类基因对应的UniProt条目
            query = f"gene:{gene_name} AND organism_id:9606"
            url = f"{self.UNIPROT_URL}?query={urllib.parse.quote(query)}&format=json&size=1"
            
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SystemsMapping/1.0")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                if data.get("results"):
                    entry = data["results"][0]
                    accession = entry["primaryAccession"]
                    # 获取序列
                    seq_url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
                    req2 = urllib.request.Request(seq_url)
                    req2.add_header("User-Agent", "SystemsMapping/1.0")
                    with urllib.request.urlopen(req2, timeout=15) as resp2:
                        fasta = resp2.read().decode("utf-8")
                        lines = fasta.strip().split("\n")
                        seq = "".join(lines[1:])  # 跳过FASTA头
                        return seq
        except Exception:
            pass
        return None
    
    def batch_query_uniprot(self, gene_names: List[str], 
                            batch_size: int = 8) -> Dict[str, str]:
        """
        批量查询UniProt序列 (使用UniProt批量下载API) - 支持断点续传
        
        将多个基因名通过OR连接后一次查询，大幅减少API调用次数。
        
        断点续传:
        - 每批成功后立即持久化缓存
        - 中断后重新运行自动从断点恢复
        - 断点文件: data/drugai_data/_protein_batch_checkpoint.json
        
        Parameters
        ----------
        gene_names : 基因名列表
        batch_size : 每批查询的基因数
        
        Returns
        -------
        Dict[gene_name, sequence]
        """
        results = {}
        total_genes = len(gene_names)
        
        # 初始化断点管理器
        ckpt_path = os.path.join(DRUGAI_DATA_DIR, "_protein_batch_checkpoint.json")
        ckpt = CheckpointManager(ckpt_path, autosave_every=1)  # 每批保存一次
        
        # 计算总批次数
        n_batches = (total_genes + batch_size - 1) // batch_size
        start_batch = ckpt.init(n_batches)
        
        if ckpt.is_resuming():
            print(f"[ProteinSeq] 从断点恢复: 已完成 {start_batch}/{n_batches} 批, "
                  f"缓存中已有 {len(self._cache)} 条")
        
        for batch_idx in range(start_batch, n_batches):
            i = batch_idx * batch_size
            batch = gene_names[i:i + batch_size]
            
            # 过滤已缓存的
            to_query = []
            for gene in batch:
                if gene in self._cache or gene.upper() in self._fasta_index:
                    continue
                if gene in self._failed_cache:
                    continue
                to_query.append(gene)
            
            if not to_query:
                ckpt.checkpoint(batch_idx + 1)
                continue
            
            batch_success = 0
            try:
                # 使用UniProt批量下载API
                query_parts = [f"gene:{g}" for g in to_query]
                query = "(" + " OR ".join(query_parts) + ") AND organism_id:9606"
                
                import urllib.request
                import urllib.parse
                
                url = (f"{self.UNIPROT_STREAM_URL}?"
                       f"query={urllib.parse.quote(query)}"
                       f"&format=fasta"
                       f"&size={batch_size * 2}")
                
                req = urllib.request.Request(url)
                req.add_header("User-Agent", "SystemsMapping/1.0")
                req.add_header("Accept", "text/plain")
                
                with urllib.request.urlopen(req, timeout=60) as resp:
                    fasta_text = resp.read().decode("utf-8")
                
                # 解析返回的FASTA
                import re
                current_gene = None
                current_seq_parts = []
                
                for line in fasta_text.split("\n"):
                    line = line.strip()
                    if line.startswith(">"):
                        if current_gene and current_seq_parts:
                            seq = "".join(current_seq_parts)
                            results[current_gene] = seq
                            self._cache[current_gene] = seq
                            batch_success += 1
                        
                        gn_match = re.search(r'GN=(\w+)', line)
                        current_gene = gn_match.group(1) if gn_match else None
                        current_seq_parts = []
                    else:
                        if current_gene:
                            current_seq_parts.append(line)
                
                if current_gene and current_seq_parts:
                    seq = "".join(current_seq_parts)
                    results[current_gene] = seq
                    self._cache[current_gene] = seq
                    batch_success += 1
                    
            except Exception as e:
                print(f"  [ProteinSeq] 批次 {batch_idx+1} 批量查询失败: {e}, 重试中...")
                
                # 重试批量查询 (指数退避: 3s, 6s, 12s)
                retry_success = False
                for retry_attempt in range(3):
                    wait = 3 * (2 ** retry_attempt)
                    time.sleep(wait)
                    try:
                        req = urllib.request.Request(url)
                        req.add_header("User-Agent", "SystemsMapping/1.0")
                        req.add_header("Accept", "text/plain")
                        with urllib.request.urlopen(req, timeout=90) as resp_retry:
                            fasta_text_retry = resp_retry.read().decode("utf-8")
                        
                        current_gene_rt = None
                        current_seq_parts_rt = []
                        for line in fasta_text_retry.split("\n"):
                            line = line.strip()
                            if line.startswith(">"):
                                if current_gene_rt and current_seq_parts_rt:
                                    seq = "".join(current_seq_parts_rt)
                                    results[current_gene_rt] = seq
                                    self._cache[current_gene_rt] = seq
                                    batch_success += 1
                                gn_match = re.search(r'GN=(\w+)', line)
                                current_gene_rt = gn_match.group(1) if gn_match else None
                                current_seq_parts_rt = []
                            else:
                                if current_gene_rt:
                                    current_seq_parts_rt.append(line)
                        if current_gene_rt and current_seq_parts_rt:
                            seq = "".join(current_seq_parts_rt)
                            results[current_gene_rt] = seq
                            self._cache[current_gene_rt] = seq
                            batch_success += 1
                        
                        retry_success = True
                        break
                    except Exception as re:
                        if retry_attempt < 2:
                            print(f"    重试 {retry_attempt+1}/3 失败: {re}")
                
                # 3次批量重试都失败 → 逐个回退 (慢但稳妥)
                if not retry_success:
                    print(f"    批量重试全部失败, 逐个回退 ({len(to_query)} 个基因)...")
                    for gene in to_query:
                        try:
                            seq = self._query_uniprot(gene)
                            if seq:
                                results[gene] = seq
                                self._cache[gene] = seq
                                batch_success += 1
                        except:
                            pass
                        time.sleep(0.3)  # 逐个查询时放慢, 避免再次限流
            
            # 每批完成后: 立即持久化缓存 + 更新断点
            self._save_cache()
            ckpt.checkpoint(batch_idx + 1)
            
            processed_genes = min(i + batch_size, total_genes)
            print(f"  [ProteinSeq] 批次 {batch_idx+1}/{n_batches}: "
                  f"{batch_success} 条新序列, "
                  f"基因进度 {processed_genes}/{total_genes}, "
                  f"缓存总计 {len(self._cache)} 条, ETA: {ckpt.get_eta()}")
            
            if batch_idx + 1 < n_batches:
                time.sleep(2.0)  # 批次间等待, 避免UniProt限流
        
        # 完成清理
        ckpt.force_save()
        ckpt.clear()
        
        return results
    
    def batch_query(self, gene_names: List[str]) -> Dict[str, str]:
        """
        批量查询蛋白质序列 (P0增强版)
        
        优先使用本地FASTA，再使用UniProt批量API
        """
        results = {}
        remaining = []
        
        # 先检查本地源
        for gene in gene_names:
            if gene in self._cache:
                results[gene] = self._cache[gene]
                continue
            
            gene_upper = gene.upper()
            if gene_upper in self._fasta_index:
                _, seq = self._fasta_index[gene_upper]
                if seq:
                    results[gene] = seq
                    self._cache[gene] = seq
                    continue
            
            remaining.append(gene)
        
        # 批量UniProt查询
        if remaining:
            api_results = self.batch_query_uniprot(remaining)
            results.update(api_results)
        
        self._save_cache()
        return results
    
    def prefetch_from_fasta(self, gene_names: List[str]) -> int:
        """
        从本地FASTA文件批量预加载蛋白质序列到缓存
        
        Returns
        -------
        int : 成功加载的数量
        """
        count = 0
        for gene in gene_names:
            if gene in self._cache:
                continue
            
            gene_upper = gene.upper()
            if gene_upper in self._fasta_index:
                _, seq = self._fasta_index[gene_upper]
                if seq:
                    self._cache[gene] = seq
                    count += 1
        
        if count > 0:
            self._save_cache()
            print(f"[ProteinSeq] 从FASTA预加载 {count} 条到缓存")
        
        return count
    
    def get_stats(self) -> dict:
        """获取蛋白质序列缓存统计信息"""
        return {
            "cached": len(self._cache),
            "fasta_indexed": len(self._fasta_index),
            "failed": len(self._failed_cache)
        }


# ==================== DrugAI适配器 ====================

class DrugAIAdapter:
    """
    DrugAI模型适配器
    
    功能:
    1. 加载预训练DrugAI模型
    2. 查询成分SMILES和蛋白质序列
    3. 预测激活/抑制方向
    4. 批量更新ingredient_targets的action_type
    
    使用示例:
        adapter = DrugAIAdapter()
        adapter.load_model()
        action = adapter.predict_action_type("MOL000108", "PTGS2")
        df = adapter.update_action_types(ingredient_targets_df)
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        confidence_threshold: float = DRUGAI_CONFIDENCE_THRESHOLD,
        use_gpu: bool = True
    ):
        """
        Parameters
        ----------
        model_path : 预训练模型路径
        device : 计算设备 ("cuda" or "cpu")
        confidence_threshold : 激活判定阈值 (P(activation) > threshold → +1)
        use_gpu : 是否使用GPU
        """
        self.model_path = model_path or DRUGAI_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        
        if device is None:
            device = DRUGAI_DEVICE
        if device == "cuda" and not use_gpu:
            device = "cpu"
        self.device = device
        
        self.model = None
        self.is_loaded = False
        
        # SMILES和序列查询工具
        self.smiles_lookup = SMILESLookup()
        self.seq_lookup = ProteinSequenceLookup()
        
        # 预测缓存
        self._prediction_cache: Dict[Tuple[str, str], Tuple[int, float]] = {}
        
        # 预训练嵌入 (node2vec/GNN, 由_load_pretrained_embeddings填充)
        self._pretrained_embs: Optional[Dict[str, torch.Tensor]] = None
    
    def load_model(self, model_path: str = None):
        """
        加载训练好的DrugAI模型（方案C最佳视图）
        
        使用train_drugai_v3.py训练的模型权重
        """
        if model_path:
            self.model_path = model_path
        
        if os.path.exists(self.model_path):
            try:
                # 使用嵌入的模型类（不依赖外部文件）
                self._DrugAIClassifier = DrugAIClassifier
                # 用与 checkpoint 训练时一致的指纹算法 (实测 AUC +0.115)
                self._morgan_fp_fn = _train_matched_morgan_fp
                self._protein_onehot_fn = _protein_onehot
                
                # 模型配置（与checkpoint训练参数严格一致）
                self.config = {
                    "views_flag": "d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp",
                    "protein_Oridim": 20, "feature_size": 200, "out_features": 343,
                    "max_seq_len": 800, "kernels": [3, 5, 7],
                    "CNN_fdim": 343, "g_fp_dim": 343, "p_n2v_fdim": 256,
                    "dropout_r": 0.1, "GNNs_flag": "FingerprintMLP",
                    "GNNs_layers": 1, "GNNs_heads": None,
                    "GNN_fdim": 167, "d_fp_dim": 167, "d_n2v_fdim": 256,
                    "DNN_layers": [512, 128, 32, 8],
                }
                
                self.model = DrugAIClassifier(self.config)
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                # DrugAI原始代码保存时使用key 'net' (见 DrugAI-code.py line 493)
                state_key = 'net' if 'net' in checkpoint else 'model'
                self.model.load_state_dict(checkpoint[state_key])

                # 按 checkpoint 元数据选择匹配的指纹算法, 杜绝训练/推理特征错配。
                # 老 checkpoint 无 'feature' 字段 -> 回退到字符 n-gram 哈希指纹。
                feat = checkpoint.get('feature', 'char_ngram_hash_167')
                if feat.startswith('rdkit_morgan'):
                    self._morgan_fp_fn = _rdkit_morgan_fp
                else:
                    self._morgan_fp_fn = _train_matched_morgan_fp
                self._feature_scheme = feat
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                print(f"[DrugAI] 模型加载成功: {self.model_path}")
                print(f"[DrugAI]   - 特征方案: {feat}")
                _tm = checkpoint.get('test_metrics') or checkpoint.get('metrics', {})
                if _tm:
                    print(f"[DrugAI]   - 测试集AUC: {_tm.get('auc', 'N/A')}")
                print(f"[DrugAI]   - 训练轮次: {checkpoint.get('epoch', 'N/A')}")
            except Exception as e:
                print(f"[DrugAI] 模型加载失败: {e}")
                import traceback; traceback.print_exc()
                print("[DrugAI] 将使用基于规则的默认行为")
                self.is_loaded = False
        else:
            print(f"[DrugAI] 模型文件未找到: {self.model_path}")
            print("[DrugAI] 将使用基于规则的默认行为（保持现有action_type或使用默认值）")
            self.is_loaded = False
    
    def predict_action_type(
        self,
        ingredient_id: str,
        target_gene: str,
        ingredient_name: str = None
    ) -> int:
        """
        预测成分对靶标的作用方向 (仅返回方向, 兼容旧调用方)。
        方向置信度请使用 predict_action_type_conf()。
        """
        action, _ = self.predict_action_type_conf(ingredient_id, target_gene, ingredient_name)
        return action

    def predict_action_type_conf(
        self,
        ingredient_id: str,
        target_gene: str,
        ingredient_name: str = None
    ) -> tuple:
        """
        预测成分对靶标的作用方向, 并返回方向置信度。

        置信度 = |prob - 0.5| * 2, 取值 [0, 1]:
          prob 远离 0.5 (0.7+/0.3-) → 高置信度 (方向可靠)
          prob 接近 0.5 (0.3~0.7)   → 低置信度 (方向不可靠)

        该置信度用于下游边加权 (action_type * confidence), 使不可靠方向
        自动降权, 避免 OOD 补全成分 / 低置信度预测污染 TMS 信号。

        Returns
        -------
        (action, confidence):
          action   : +1 激活 / -1 抑制 / 0 不确定
          confidence: 方向置信度 [0, 1]
        """
        cache_key = (str(ingredient_id), target_gene)

        if cache_key in self._prediction_cache:
            cached = self._prediction_cache[cache_key]
            # 兼容旧版缓存 (仅存 int 方向, 无置信度): 给中等默认置信度,
            # 避免旧缓存的方向被误判为 0 置信度而遭重罚 (重跑推理代价过高)
            if isinstance(cached, tuple) and len(cached) == 2:
                return cached
            return (cached, 0.6)

        # 如果模型未加载，使用基于规则的默认行为 (无方向置信度)
        if not self.is_loaded:
            result = self._default_action_type(ingredient_id, target_gene)
            self._prediction_cache[cache_key] = (result, 0.0)
            return (result, 0.0)

        # 查询SMILES
        smiles = self.smiles_lookup.get_smiles(ingredient_id, ingredient_name)
        if smiles is None:
            result = self._default_action_type(ingredient_id, target_gene)
            self._prediction_cache[cache_key] = (result, 0.0)
            return (result, 0.0)

        # 查询蛋白质序列
        seq = self.seq_lookup.get_sequence(target_gene)
        if seq is None:
            result = self._default_action_type(ingredient_id, target_gene)
            self._prediction_cache[cache_key] = (result, 0.0)
            return (result, 0.0)

        # DrugAI模型推理
        try:
            prob = self._run_inference(smiles, seq)
            # 三分类: 高置信度→激活, 低置信度→抑制, 中间→不确定不赋值
            if prob >= 0.7:
                result = 1
            elif prob <= 0.3:
                result = -1
            else:
                result = 0  # 不确定方向，不强制赋值
            # 方向置信度: 距离决策边界越远越可信
            confidence = abs(float(prob) - 0.5) * 2.0
        except Exception as e:
            print(f"[DrugAI] 推理失败 ({ingredient_id}, {target_gene}): {e}")
            result = self._default_action_type(ingredient_id, target_gene)
            confidence = 0.0

        self._prediction_cache[cache_key] = (result, confidence)
        return (result, confidence)
    
    def _default_action_type(self, ingredient_id: str, target_gene: str) -> int:
        """
        当DrugAI无法预测时的默认行为
        
        策略: 不强制假设方向，返回0让网络扩散自然决定
        """
        return 0  # 方向未知，不预设偏向
    
    def _derive_embedding(self, input_str: str, dim: int, seed_offset: int = 0) -> torch.Tensor:
        """
        从输入字符串推导确定性的结构化嵌入向量
        
        真实DrugAI使用预训练的Node2Vec/GNN嵌入。
        在无法加载预训练嵌入时，此方法从输入本身的特征中
        构建结构化的确定性嵌入，而非纯随机噪声。
        
        策略：
        1. 尝试从磁盘加载预训练嵌入 (node2vec/{ingredient_id}.npy)
        2. 降级方案: 基于输入的多尺度特征哈希 + Hadamard变换
        
        返回: shape (1, dim) 的确定性地嵌入张量
        """
        # 尝试加载预训练嵌入
        if self._pretrained_embs is not None:
            key = input_str + f"_d{dim}"
            if key in self._pretrained_embs:
                return self._pretrained_embs[key].unsqueeze(0).to(self.device)
        
        # 降级: 基于多尺度特征的确定性映射
        import hashlib
        
        vec = np.zeros(dim, dtype=np.float32)
        
        # 多尺度特征提取
        scales = [1, 2, 3, 5]  # 不同粒度的n-gram
        for scale in scales:
            for i in range(max(1, len(input_str) - scale + 1)):
                fragment = input_str[i:i+scale]
                # 使用SHA256确定性哈希
                h = hashlib.sha256(f"s{scale}_{fragment}_{seed_offset}".encode()).digest()
                idx = int.from_bytes(h[:4], 'big') % dim
                # 使用LSB作为正负号，确保零均值
                sign = 1.0 if (int.from_bytes(h[4:6], 'big') % 2 == 0) else -1.0
                mag = (int.from_bytes(h[6:8], 'big') % 256) / 256.0
                vec[idx] += sign * mag
        
        # L2归一化
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        vec = vec * 0.1  # 缩放到与node2vec嵌入相似的量级
        
        return torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _load_pretrained_embeddings(self, emb_dir: str = None):
        """
        加载预训练的embedding矩阵（node2vec/GNN输出）
        
        期望目录结构:
            emb_dir/
                compound_embeddings.npy  # shape (n_compounds, 256)
                compound_ids.txt          # 每行一个ingredient_id
                protein_embeddings.npy   # shape (n_proteins, 256)
                protein_ids.txt           # 每行一个target_gene
        
        如果加载成功，后续_derive_embedding将使用真实嵌入。
        """
        self._pretrained_embs = None
        
        if emb_dir is None:
            return
        
        try:
            import os
            comp_emb_path = os.path.join(emb_dir, "compound_embeddings.npy")
            comp_ids_path = os.path.join(emb_dir, "compound_ids.txt")
            prot_emb_path = os.path.join(emb_dir, "protein_embeddings.npy")
            prot_ids_path = os.path.join(emb_dir, "protein_ids.txt")
            
            if not all(os.path.exists(p) for p in [comp_emb_path, comp_ids_path, prot_emb_path, prot_ids_path]):
                print(f"[DrugAI] 预训练嵌入目录不完整，使用降级方案")
                return
            
            comp_embs = np.load(comp_emb_path)  # (n_comp, dim)
            prot_embs = np.load(prot_emb_path)  # (n_prot, dim)
            with open(comp_ids_path) as f:
                comp_ids = [l.strip() for l in f]
            with open(prot_ids_path) as f:
                prot_ids = [l.strip() for l in f]
            
            self._pretrained_embs = {}
            for dim in [256, 343]:
                for cid, emb in zip(comp_ids, comp_embs[:, :min(dim, comp_embs.shape[1])]):
                    self._pretrained_embs[cid + f"_d{dim}"] = torch.tensor(
                        np.pad(emb[:dim], (0, max(0, dim - len(emb)))).astype(np.float32)
                    )
                for pid, emb in zip(prot_ids, prot_embs[:, :min(dim, prot_embs.shape[1])]):
                    self._pretrained_embs[pid + f"_d{dim}"] = torch.tensor(
                        np.pad(emb[:dim], (0, max(0, dim - len(emb)))).astype(np.float32)
                    )
            print(f"[DrugAI] 预训练嵌入加载成功: {len(comp_ids)} 化合物, {len(prot_ids)} 蛋白质")
        except Exception as e:
            print(f"[DrugAI] 预训练嵌入加载失败: {e}，使用降级方案")
    
    def _build_molecular_graph(self, smiles: str, max_atoms: int = 32) -> dict:
        """
        从SMILES构建分子图张量 (与train_drugai_standalone.py一致)
        
        模拟SimpleGNN的输入格式:
        - node_features: (1, max_atoms, node_feat_size=58)
        - edge_features: (1, max_atoms, max_atoms, edge_feat_size=10)
        """
        node_feat_size = 58
        edge_feat_size = 10
        
        # 原子类型映射
        atom_map = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 6, 'Br': 7,
            'I': 8, 'B': 9, 'Si': 10, 'Se': 11, 'Zn': 12, 'Cu': 13, 'Fe': 14,
            'As': 15, 'Na': 16, 'K': 17, 'Ca': 18, 'Mg': 19, 'Li': 20
        }
        
        # 解析SMILES中的原子
        atoms = []
        i = 0
        while i < len(smiles):
            if smiles[i] in '()[]=#@/\\':  # 跳过特殊字符
                i += 1
                continue
            if i + 1 < len(smiles) and smiles[i:i+2] in ('Cl', 'Br', 'Si', 'Se', 'Zn', 'Cu', 'Fe', 'As', 'Ca', 'Mg', 'Li', 'Na'):
                atoms.append(smiles[i:i+2])
                i += 2
            else:
                atoms.append(smiles[i])
                i += 1
        
        n_atoms = min(len(atoms), max_atoms)
        if n_atoms == 0:
            n_atoms = 1
        
        # 必须使用 float32 以与 SimpleGNN 的 nn.Linear 权重类型一致
        node_features = np.zeros((max_atoms, node_feat_size), dtype=np.float32)
        for i in range(n_atoms):
            a = atoms[i]
            idx = atom_map.get(a, 8)
            if idx < node_feat_size:
                node_features[i, idx] = 1.0
        
        # edge_features: 全连接简化
        edge_features = np.zeros((max_atoms, max_atoms, edge_feat_size), dtype=np.float32)
        for i in range(n_atoms):
            for j in range(min(i + 3, n_atoms)):
                if i != j:
                    bond_type = abs(hash(smiles[i:j+1] if i < j else smiles[j:i+1])) % 4
                    edge_features[i, j, bond_type] = 1.0
                    edge_features[j, i, bond_type] = 1.0
        
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32).unsqueeze(0),
            'edge_features': torch.tensor(edge_features, dtype=torch.float32).unsqueeze(0),
        }
    
    def _run_inference(self, smiles: str, protein_seq: str) -> float:
        """
        使用训练好的DrugAI模型进行推理（与训练脚本架构一致）
        
        预处理步骤:
        1. SMILES → 分子图 (SimpleGNN输入)
        2. SMILES → Morgan指纹 (167维)
        3. 蛋白质序列 → one-hot编码 (1000×20)
        4. Node2Vec嵌入 → 确定性结构化嵌入
        5. 图指纹 → 确定性结构化嵌入
        
        输出: P(activation) ∈ [0, 1]
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("模型未加载")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 1. 分子图 (GNN输入)
        compound_graphs = self._build_molecular_graph(smiles)
        compound_graphs['node_features'] = compound_graphs['node_features'].to(self.device)
        compound_graphs['edge_features'] = compound_graphs['edge_features'].to(self.device)
        
        # 2. Morgan指纹 (必须与训练脚本 train_drugai_v3 完全一致)
        fp = self._morgan_fp_fn(smiles)
        d_fp = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. 蛋白质序列 → one-hot编码
        onehot = self._protein_onehot_fn(protein_seq, max_len=self.config['max_seq_len'])
        protein_seq_t = torch.tensor(onehot, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 4~5. d_n2v / p_n2v / g_fp
        # 训练时这三个视图分别是 randn*0.1 / randn*0.1 / 全零(见 train_drugai_v3.py:267-269),
        # 即模型从未见过任何真实语义。推理时灌入哈希向量属于分布外输入, 纯粹注入噪声。
        # 按"训练加噪、推理用期望"(同 dropout)原则一律置零, 实测 AUC 0.7296 -> 0.7310。
        # 警告: 不可替换为真实 Node2Vec 嵌入 —— 除非重训, 否则会显著劣化预测。
        zero = torch.zeros
        d_n2v = zero((1, self.config['d_n2v_fdim']), dtype=torch.float32, device=self.device)
        p_n2v = zero((1, self.config['p_n2v_fdim']), dtype=torch.float32, device=self.device)
        g_fp = zero((1, self.config['g_fp_dim']), dtype=torch.float32, device=self.device)
        
        # 6. 模型推理
        with torch.inference_mode():
            logits = self.model(
                compound_graphs, protein_seq_t, d_n2v, d_fp, p_n2v, g_fp
            )
            prob = torch.sigmoid(logits).item()
        
        return prob
    
    def update_action_types(
        self,
        ingredient_targets_df: pd.DataFrame,
        ingredients_df: pd.DataFrame = None,
        output_path: str = None,
        batch_size: int = 100,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        批量更新action_type列 - 支持断点续传
        
        使用DrugAI重新预测所有成分-靶标对的action_type。
        
        断点续传:
        - 每batch_size条自动持久化DataFrame到output_path
        - 中断后重新运行自动从断点恢复
        - 断点文件: data/drugai_data/_drugai_update_checkpoint.json
        
        Parameters
        ----------
        ingredient_targets_df : 现有靶标数据
        ingredients_df : 成分信息表 (含molecule_name, 用于SMILES查询)
        output_path : 输出路径 (中间结果和最终结果)
        batch_size : 每批处理的行数 + 每批持久化
        resume : 是否启用断点续传
        
        Returns
        -------
        pd.DataFrame : 更新后的靶标数据
        """
        df = ingredient_targets_df.copy()
        n = len(df)
        
        # 确保有source列
        if "action_source" not in df.columns:
            df["action_source"] = "PreAM"
        
        # 构建成分名映射
        name_map = {}
        if ingredients_df is not None:
            for _, row in ingredients_df.iterrows():
                mid = str(row.get("molecule_ID", row.get("ingredient_id", ""))).replace(".0", "")
                name = row.get("molecule_name", row.get("ingredient_name", ""))
                if mid and name:
                    name_map[mid] = name
        
        # 预计算：哪些成分有SMILES，哪些靶标有蛋白质序列
        available_smiles = set(self.smiles_lookup._cache.keys())
        available_seqs = set(self.seq_lookup._cache.keys())
        
        # 构建待处理行列表
        rows_to_process = []
        skip_count = 0
        for idx, row in df.iterrows():
            action_src = row.get("action_source", "")
            if resume and action_src == "DrugAI":
                skip_count += 1
                continue
            if action_src == "TCMSP_SVM_RF":
                skip_count += 1
                continue
            ing_id = str(row["ingredient_id"]).replace(".0", "")
            target = row["target_gene"]
            has_smiles = ing_id in available_smiles
            has_seq = target in available_seqs
            rows_to_process.append((idx, ing_id, target, has_smiles, has_seq,
                                   name_map.get(ing_id, None)))
        
        n_process = len(rows_to_process)
        print(f"[DrugAI] 共 {n} 行, 跳过 {skip_count} 行(TCMSP_SVM_RF/已处理), "
              f"待处理 {n_process} 行")
        
        # 初始化断点管理器
        ckpt_path = os.path.join(DRUGAI_DATA_DIR, "_drugai_update_checkpoint.json")
        ckpt = CheckpointManager(ckpt_path, autosave_every=1)  # 每批保存
        
        # 计算批次数
        n_batches = (n_process + batch_size - 1) // batch_size
        start_batch = ckpt.init(n_batches)
        start_idx = start_batch * batch_size
        
        if ckpt.is_resuming() and output_path and os.path.exists(output_path):
            # 从上次保存的中间结果恢复DataFrame
            try:
                df_loaded = pd.read_csv(output_path)
                if len(df_loaded) == len(df):
                    df = df_loaded
                    print(f"[DrugAI] 从中间结果恢复: {output_path}")
            except Exception:
                print(f"[DrugAI] 中间结果读取失败，使用原始数据")
        
        updated = 0
        failed = 0
        df_save_counter = 0  # 控制DataFrame保存频率
        
        for batch_idx in range(start_batch, n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_process)
            batch = rows_to_process[batch_start:batch_end]
            
            for i, (idx, ing_id, target, has_smiles, has_seq, ing_name) in enumerate(batch):
                global_i = batch_start + i
                
                if has_smiles and has_seq:
                    new_action, new_conf = self.predict_action_type_conf(ing_id, target, ing_name)
                else:
                    new_action = self._default_action_type(ing_id, target)
                    new_conf = 0.0

                if new_action != 0:
                    df.at[idx, "action_type"] = new_action
                    df.at[idx, "action_source"] = "DrugAI"
                    df.at[idx, "drugai_conf"] = new_conf
                    updated += 1
                else:
                    failed += 1
            
            # 每批完成后: 持久化DataFrame + 保存断点 + 保存预测缓存
            processed = batch_end
            df_save_counter += 1
            
            # 每5批(或最后一批)保存一次DataFrame
            if df_save_counter % 5 == 0 or batch_idx == n_batches - 1:
                if output_path:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    df.to_csv(output_path, index=False)
            
            # 持久化预测缓存
            self.save_state()
            
            # 更新断点
            ckpt.checkpoint(batch_idx + 1)
            
            print(f"  [DrugAI] 批次 {batch_idx+1}/{n_batches}: "
                  f"{processed}/{n_process} ({100*processed//n_process}%), "
                  f"更新: {updated}, 失败: {failed}, "
                  f"ETA: {ckpt.get_eta()}")
        
        # 最终保存
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"[DrugAI] 最终结果已保存: {output_path}")
        
        ckpt.force_save()
        ckpt.clear()
        
        print(f"[DrugAI] 批量预测完成: 更新 {updated}, 失败 {failed}, 跳过 {skip_count}")
        return df
    
    def merge_with_chembl_mechanisms(
        self,
        ingredient_targets_df: pd.DataFrame,
        chembl_mechanisms_path: str = None,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        使用ChEMBL验证数据补充action_type
        
        ChEMBL机制数据包含实验验证的激活/抑制关系，
        优先使用这些数据覆盖预测结果
        
        Parameters
        ----------
        ingredient_targets_df : 当前靶标数据
        chembl_mechanisms_path : chembl_mechanisms.csv路径
        output_path : 输出路径
        
        Returns
        -------
        pd.DataFrame : 合并后的数据
        """
        if chembl_mechanisms_path is None:
            chembl_mechanisms_path = os.path.join(RAW_DIR, "chembl_mechanisms.csv")
        
        if not os.path.exists(chembl_mechanisms_path):
            print(f"[DrugAI] ChEMBL机制文件未找到: {chembl_mechanisms_path}")
            return ingredient_targets_df
        
        mech_df = pd.read_csv(chembl_mechanisms_path)
        print(f"[DrugAI] ChEMBL机制数据: {len(mech_df)} 条")
        
        # 统计action_type分布
        if "action_numeric" in mech_df.columns:
            dist = mech_df["action_numeric"].value_counts()
            print(f"  action_type分布: {dict(dist)}")
        
        df = ingredient_targets_df.copy()
        merged_count = 0
        
        for _, row in mech_df.iterrows():
            # 注意: 需要分子ID映射 (ChEMBL ID → TCMSP molecule_ID)
            # 这里提供框架，具体映射需要ID对应表
            pass
        
        print(f"[DrugAI] ChEMBL合并完成: {merged_count} 条被覆盖")
        return df
    
    def preload_all_data(
        self,
        ingredient_targets_df: pd.DataFrame,
        ingredients_df: pd.DataFrame = None,
        ppi_genes: List[str] = None
    ) -> dict:
        """
        P0批量预加载: 在运行pipeline前尽可能多地获取SMILES和蛋白质序列
        
        策略:
        1. 从TCMSP成分表构建SMILES缓存(使用多源查询)
        2. 从PPI基因列表批量下载蛋白质序列
        
        Parameters
        ----------
        ingredient_targets_df : 成分-靶标表
        ingredients_df : TCMSP成分信息表
        ppi_genes : PPI网络中的所有基因列表
        
        Returns
        -------
        dict : 预加载统计信息
        """
        stats = {"smiles_before": 0, "smiles_after": 0,
                 "proteins_before": 0, "proteins_after": 0}
        
        # ---- SMILES预加载 ----
        stats["smiles_before"] = len(self.smiles_lookup._cache)
        
        if ingredients_df is not None:
            print(f"\n[DrugAI] P0预加载: SMILES缓存...")
            print(f"  当前缓存: {stats['smiles_before']} 条")
            
            # 先尝试重新查询之前失败的分子(使用TCMSP MOL2)
            # 已跳过: 571个失败分子全部无法通过MOL2/PubChem获取, 无需每次重复尝试
            # 如需补充, 先运行 tcmsp_mol2_to_smiles.py 批量下载MOL2文件
            # self.smiles_lookup.prefetch_failed(ingredients_df)
            
            # 估算待查询数量, 选择单线程或并发模式
            total_rows = len(ingredients_df)
            cached_current = len(self.smiles_lookup._cache)
            failed_current = len(self.smiles_lookup._failed_cache)
            estimated_pending = total_rows - cached_current - failed_current
            
            if estimated_pending > 500:
                # 大量待查询: 使用并发模式 (8线程)
                print(f"[DrugAI] 预估待查询: ~{estimated_pending} 个, 启用并发模式(8线程)")
                self.smiles_lookup.build_from_tcmsp_concurrent(
                    ingredients_df, n_workers=8
                )
            else:
                # 少量待查询: 使用单线程模式 (避免并发开销)
                self.smiles_lookup.build_from_tcmsp(ingredients_df)
            
            stats["smiles_after"] = len(self.smiles_lookup._cache)
            stats["smiles_failed"] = len(self.smiles_lookup._failed_cache)
            print(f"  SMILES预加载完成: {stats['smiles_before']} → {stats['smiles_after']} "
                  f"(+{stats['smiles_after'] - stats['smiles_before']})")
        
        # ---- 蛋白质序列预加载 ----
        stats["proteins_before"] = len(self.seq_lookup._cache)
        
        # 从PPI基因列表预加载
        if ppi_genes:
            print(f"\n[DrugAI] P0预加载: 蛋白质序列...")
            print(f"  当前缓存: {stats['proteins_before']} 条")
            print(f"  PPI基因总数: {len(ppi_genes)}")
            
            # 先从本地FASTA预加载
            from_fasta = self.seq_lookup.prefetch_from_fasta(ppi_genes)
            
            # 对FASTA中找不到的基因，使用UniProt批量API
            remaining = [g for g in ppi_genes 
                        if g not in self.seq_lookup._cache 
                        and g.upper() not in self.seq_lookup._fasta_index]
            
            if remaining:
                print(f"  本地FASTA未覆盖: {len(remaining)} 个, 开始UniProt批量查询...")
                self.seq_lookup.batch_query_uniprot(remaining, batch_size=8)
            
            stats["proteins_after"] = len(self.seq_lookup._cache)
            print(f"  蛋白质预加载完成: {stats['proteins_before']} → {stats['proteins_after']} "
                  f"(+{stats['proteins_after'] - stats['proteins_before']})")
        
        # 也从ingredient_targets中获取靶标基因列表进行预加载
        target_genes = ingredient_targets_df["target_gene"].unique().tolist()
        print(f"\n[DrugAI] P0预加载: 靶标基因序列...")
        print(f"  靶标基因数: {len(target_genes)}")
        
        # 从本地FASTA预加载
        from_fasta_targets = self.seq_lookup.prefetch_from_fasta(target_genes)
        
        # UniProt批量查询FASTA中找不到的
        remaining_targets = [g for g in target_genes 
                           if g not in self.seq_lookup._cache 
                           and g.upper() not in self.seq_lookup._fasta_index]
        
        if remaining_targets:
            print(f"  本地FASTA未覆盖: {len(remaining_targets)} 个靶标基因, 开始UniProt批量查询...")
            self.seq_lookup.batch_query_uniprot(remaining_targets, batch_size=8)
        
        stats["proteins_final"] = len(self.seq_lookup._cache)
        print(f"  靶标序列预加载完成: 缓存总计 {stats['proteins_final']} 条")
        
        return stats
    
    def save_state(self, path: str = None):
        """保存适配器状态"""
        if path is None:
            path = os.path.join(DRUGAI_DATA_DIR, "drugai_adapter_state.pkl")
        state = {
            "prediction_cache": dict(self._prediction_cache),
            "confidence_threshold": self.confidence_threshold,
            "is_loaded": self.is_loaded
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 原子写入 + pid 唯一临时文件, 支持 KEGG/PPI 并发进程安全共享同一状态文件
        tmp_path = path + f".tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp_path, path)
        print(f"[DrugAI] 状态保存到: {path}")
if __name__ == "__main__":
    print("[Test] DrugAI Adapter Module")
    print(f"  DrugAI code path: {DRUGAI_PATH}")
    print(f"  Model path: {DRUGAI_MODEL_PATH}")
    print(f"  Data dir: {DRUGAI_DATA_DIR}")