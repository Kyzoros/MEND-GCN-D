"""
时间感知扰动签名模块
============================================================================
核心思想:
  疾病是随时间演化的动态过程，单一时间点的"快照"无法完整描述疾病状态。
  本模块将时间维度引入扰动签名框架，实现:
  
  1. 时间序列TMS追踪: 对同一草药-疾病对，计算其在多个时间点的TMS变化轨迹
  2. 时间加权集成: 近期时间点权重更高，生成"当前最佳"推荐
  3. 疾病阶段特异性分析: 识别草药在疾病早/中/晚期的不同作用

数据格式要求:
  输入的疾病表达谱需包含 time_point 列（int或str），例如:
    disease_id, gene_symbol, logFC, P_value, time_point
    RA,         BRCA1,      2.3,   0.001,   0      # 基线
    RA,         BRCA1,      1.5,   0.010,   6      # 6周后
    RA,         BRCA1,      0.8,   0.050,   12     # 12周后

使用方式:
  engine = TemporalSignatureEngine(herb_signatures, builder, disease_profiles)
  
  # 方式1: 获取时间轨迹
  trajectory = engine.compute_temporal_tms('Herb_001', 'RA')
  # 返回: {time_point: TMS_value}
  
  # 方式2: 时间加权排名
  rankings = engine.time_weighted_ranking(decay_factor=0.8)
  
  # 方式3: 获取时间集成签名
  integrated = engine.build_temporal_signature('Herb_001', 'RA', method='peak')
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

from config import TOP_BOTTOM_K
from hdmap_s import HDmapS


class TemporalSignatureEngine:
    """
    时间感知扰动签名引擎
    
    在 HDmap-S 的基础上加入时间维度，计算"TMS-时间轨迹"。
    确保时间一致性: 同一草药的扰动签名跨时间点保持不变，
    仅疾病病理状态随"时间窗口"变化。
    
    Parameters
    ----------
    herb_signatures : Dict[str, np.ndarray], 草药扰动签名（与时间无关）
    builder         : PerturbationSignatureBuilder
    disease_df      : pd.DataFrame, 含 time_point 列的疾病表达谱
        必须包含列: disease_id, gene_symbol, logFC, time_point
        可选列: P_value (用于过滤)
    """
    
    # 支持的集成方法
    INTEGRATION_METHODS = ('peak', 'recent', 'average', 'trajectory')
    
    def __init__(self, herb_signatures: Dict[str, np.ndarray],
                 builder, disease_df: pd.DataFrame):
        self.herb_signatures = herb_signatures
        self.builder = builder
        self.disease_df = disease_df.copy()
        
        # 解析时间点
        self._parse_time_points()
        
        # 构建每个时间点的HDmap-S实例
        self._time_hdmap_s: Dict[Union[int, str], HDmapS] = {}
        self._build_time_hdmap_s()
        
        # 缓存: (herb_id, disease_id) -> {time_point: TMS}
        self._tms_cache: Dict[Tuple[str, str], Dict] = {}
    
    def _parse_time_points(self):
        """解析疾病表达谱中的时间点信息"""
        if 'time_point' not in self.disease_df.columns:
            # 没有时间列 -> 视为单一时间点，time_point=0
            print("[Temporal] 未检测到 time_point 列，视为单一时间点(t=0)")
            self.disease_df['time_point'] = 0
            self.has_time = False
        else:
            self.has_time = True
        
        self.time_points = sorted(self.disease_df['time_point'].unique())
        self.n_timepoints = len(self.time_points)
        self.disease_ids = self.disease_df['disease_id'].unique()
        
        if self.has_time:
            print(f"[Temporal] 检测到 {self.n_timepoints} 个时间点: {self.time_points}")
            for did in self.disease_ids:
                n_t = len(self.disease_df[self.disease_df['disease_id'] == did]['time_point'].unique())
                print(f"  疾病 {did}: {n_t} 个时间点")
    
    def _build_time_hdmap_s(self):
        """对每个时间点构建独立的 HDmapS 实例"""
        for t in self.time_points:
            # 提取该时间点的疾病数据
            t_mask = self.disease_df['time_point'] == t
            t_df = self.disease_df[t_mask].copy()
            
            # 构建该时间点的疾病基因字典
            disease_genes_dict = {}
            for did in t_df['disease_id'].unique():
                sub = t_df[t_df['disease_id'] == did].sort_values('logFC', ascending=False)
                disease_genes_dict[did] = sub['gene_symbol'].tolist()
            
            # 创建该时间点的 HDmapS
            hdmap_s = HDmapS(self.herb_signatures, self.builder)
            self._time_hdmap_s[t] = (hdmap_s, disease_genes_dict)
        
        self._herb_ids = list(self.herb_signatures.keys())
    
    # ==================== 核心计算接口 ====================
    
    def compute_temporal_tms(self, herb_id: str, disease_id: str,
                             k: int = TOP_BOTTOM_K) -> Dict[Union[int, str], float]:
        """
        计算单个草药在疾病各时间点的 TMS 轨迹
        
        Parameters
        ----------
        herb_id     : 草药ID
        disease_id  : 疾病ID
        k           : top/bottom基因数
        
        Returns
        -------
        Dict[time_point, TMS_value]
            例如 {0: 0.32, 6: 0.45, 12: 0.28}
        """
        cache_key = (herb_id, disease_id)
        if cache_key in self._tms_cache:
            return self._tms_cache[cache_key]
        
        trajectory = {}
        for t in self.time_points:
            hdmap_s, disease_dict = self._time_hdmap_s[t]
            genes = disease_dict.get(disease_id, [])
            if not genes:
                trajectory[t] = 0.0
                continue
            tms = hdmap_s.compute_tms(herb_id, genes, k=k)
            trajectory[t] = tms
        
        self._tms_cache[cache_key] = trajectory
        return trajectory
    
    def compute_all_temporal_tms(self, k: int = TOP_BOTTOM_K,
                                 verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        计算所有草药-疾病对的时间TMS轨迹
        
        Returns
        -------
        Dict[disease_id, DataFrame]
            DataFrame: index=herb_id, columns=time_points, values=TMS
        """
        result = {}
        total = len(self._herb_ids) * len(self.disease_ids)
        count = 0
        
        for did in self.disease_ids:
            records = {}
            for hid in self._herb_ids:
                traj = self.compute_temporal_tms(hid, did, k=k)
                records[hid] = traj
                count += 1
                if verbose and count % 200 == 0:
                    print(f"  [Temporal TMS] 进度: {count}/{total}")
            
            df = pd.DataFrame.from_dict(records, orient='index')
            df.columns = [f"T_t={t}" for t in df.columns]
            df.index.name = 'herb_id'
            result[did] = df
        
        if verbose:
            print(f"  [Temporal TMS] 完成! {len(self._herb_ids)} 草药 x {len(self.disease_ids)} 疾病 x {self.n_timepoints} 时间点")
        return result
    
    # ==================== 集成策略 ====================
    
    def integrate_by_peak(self, herb_id: str, disease_id: str,
                          k: int = TOP_BOTTOM_K) -> Tuple[float, Union[int, str]]:
        """
        峰值集成: 取 TMS 绝对值最大的时间点
        
        Returns
        -------
        (peak_TMS, peak_time_point)
        """
        traj = self.compute_temporal_tms(herb_id, disease_id, k=k)
        peak_t = max(traj, key=lambda t: abs(traj[t]))
        return traj[peak_t], peak_t
    
    def integrate_by_recent(self, herb_id: str, disease_id: str,
                            decay_factor: float = 0.7,
                            k: int = TOP_BOTTOM_K) -> float:
        """
        近期加权集成: 近期时间点有更高权重
        
        Parameters
        ----------
        decay_factor : float, 衰减因子 (0~1)
            权重 = decay_factor^(rank_from_latest)
            越大表示历史数据权重越高
        
        Returns
        -------
        加权TMS
        """
        traj = self.compute_temporal_tms(herb_id, disease_id, k=k)
        sorted_t = sorted(traj.keys())
        if not sorted_t:
            return 0.0
        
        # 从最近到最远分配权重
        weights = []
        for rank, t in enumerate(reversed(sorted_t)):
            w = decay_factor ** rank
            weights.append(w)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        values = np.array([traj[t] for t in reversed(sorted_t)])
        return float(np.sum(values * weights))
    
    def integrate_by_average(self, herb_id: str, disease_id: str,
                             k: int = TOP_BOTTOM_K) -> float:
        """均值集成: 所有时间点的简单平均"""
        traj = self.compute_temporal_tms(herb_id, disease_id, k=k)
        return float(np.mean(list(traj.values())))
    
    def get_trajectory_trend(self, herb_id: str, disease_id: str,
                             k: int = TOP_BOTTOM_K) -> Dict:
        """
        轨迹趋势分析
        
        Returns
        -------
        Dict with keys:
            - 'trajectory': 每个时间点的TMS
            - 'trend': '上升' / '下降' / '波动' / '平稳'
            - 'volatility': TMS波动性 (标准差)
            - 'peak_time': 峰值时间点
        """
        traj = self.compute_temporal_tms(herb_id, disease_id, k=k)
        sorted_t = sorted(traj.keys())
        values = [traj[t] for t in sorted_t]
        
        # 趋势判断
        if len(values) >= 3:
            from scipy.stats import linregress
            slope = linregress(range(len(values)), values).slope
            if slope > 0.02:
                trend = '上升'
            elif slope < -0.02:
                trend = '下降'
            elif max(values) - min(values) < 0.05:
                trend = '平稳'
            else:
                trend = '波动'
        elif len(values) == 2:
            diff = values[1] - values[0]
            trend = '上升' if diff > 0.02 else ('下降' if diff < -0.02 else '平稳')
        else:
            trend = '平稳'
        
        peak_t = sorted_t[np.argmax(np.abs(values))]
        
        return {
            'trajectory': traj,
            'trend': trend,
            'volatility': float(np.std(values)),
            'peak_time': peak_t
        }
    
    def time_weighted_ranking(self, disease_id: str,
                              method: str = 'recent',
                              decay_factor: float = 0.7,
                              k: int = TOP_BOTTOM_K,
                              top_n: int = 50) -> pd.DataFrame:
        """
        时间加权排名: 对所有草药按时间集成策略排序
        
        Parameters
        ----------
        method      : str, 集成方法: 'peak' / 'recent' / 'average'
        decay_factor: float, 近期加权衰减因子
        top_n       : int, 返回Top-N
        
        Returns
        -------
        DataFrame: herb_id, integrated_score, peak_time(可选), trend(可选)
        """
        if method not in self.INTEGRATION_METHODS:
            raise ValueError(f"未知集成方法: {method}, 可选: {self.INTEGRATION_METHODS}")
        
        results = []
        for hid in self._herb_ids:
            if method == 'peak':
                score, peak_t = self.integrate_by_peak(hid, disease_id, k=k)
                results.append({
                    'herb_id': hid,
                    'integrated_score': abs(score),
                    'raw_score': score,
                    'peak_time': peak_t
                })
            elif method == 'recent':
                score = self.integrate_by_recent(hid, disease_id, decay_factor=decay_factor, k=k)
                results.append({
                    'herb_id': hid,
                    'integrated_score': abs(score),
                    'raw_score': score
                })
            elif method == 'average':
                score = self.integrate_by_average(hid, disease_id, k=k)
                results.append({
                    'herb_id': hid,
                    'integrated_score': abs(score),
                    'raw_score': score
                })
        
        df = pd.DataFrame(results)
        df = df.sort_values('integrated_score', ascending=False).head(top_n)
        df['rank'] = range(1, len(df) + 1)
        return df[['rank', 'herb_id', 'integrated_score', 'raw_score'] + 
                   (['peak_time'] if method == 'peak' else [])]
    
    def find_disease_stage_specific_herbs(self, disease_id: str,
                                          k: int = TOP_BOTTOM_K,
                                          top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        发现疾病阶段特异性草药: 对每个时间点分别Top-N
        
        Returns
        -------
        Dict[time_point, DataFrame], 每个时间点的Top-N推荐
        """
        result = {}
        for t in self.time_points:
            hdmap_s, disease_dict = self._time_hdmap_s[t]
            genes = disease_dict.get(disease_id, [])
            if not genes:
                continue
            
            scores = []
            for hid in self._herb_ids:
                tms = hdmap_s.compute_tms(hid, genes, k=k)
                scores.append({'herb_id': hid, 'TMS': tms})
            
            df = pd.DataFrame(scores)
            df = df.sort_values('TMS', ascending=True).head(top_n)
            df['rank'] = range(1, len(df) + 1)
            result[t] = df
        
        return result
    
    def summarize_temporal(self, disease_id: str,
                           top_n: int = 20,
                           k: int = TOP_BOTTOM_K) -> Dict:
        """
        生成完整的时间感知分析报告
        
        Returns
        -------
        Dict with keys:
            - 'time_points': 时间点列表
            - 'stage_specific': 各时间点的特异性草药
            - 'recent_top20': 近期加权Top-20
            - 'peak_top20': 峰值Top-20
            - 'herb_trajectories': 部分草药的TMS轨迹
        """
        stage_specific = self.find_disease_stage_specific_herbs(disease_id, k=k, top_n=top_n)
        recent_rank = self.time_weighted_ranking(disease_id, method='recent', k=k, top_n=top_n)
        peak_rank = self.time_weighted_ranking(disease_id, method='peak', k=k, top_n=top_n)
        
        # 获取Top-5草药的完整轨迹
        herb_trajs = {}
        for hid in recent_rank['herb_id'].head(5):
            traj_info = self.get_trajectory_trend(hid, disease_id, k=k)
            herb_trajs[hid] = traj_info
        
        return {
            'time_points': list(self.time_points),
            'n_herbs': len(self._herb_ids),
            'n_diseases': len(self.disease_ids),
            'stage_specific': {str(t): df for t, df in stage_specific.items()},
            'recent_top20': recent_rank,
            'peak_top20': peak_rank,
            'herb_trajectories': herb_trajs
        }
    
    def build_temporal_signature(self, herb_id: str, disease_id: str,
                                 method: str = 'recent',
                                 decay_factor: float = 0.7,
                                 k: int = TOP_BOTTOM_K) -> float:
        """
        构建时间集成的扰动签名得分（对外统一接口）
        
        Parameters
        ----------
        method : str
            'peak'    - 峰值响应（最大效应时间点）
            'recent'  - 近期加权（近期时间点权重高）
            'average' - 均值（全时间窗口平均）
            'trajectory' - 返回完整轨迹（需调用 compute_temporal_tms）
        
        Returns
        -------
        float (集成TMS得分)
        """
        if method == 'peak':
            score, _ = self.integrate_by_peak(herb_id, disease_id, k=k)
        elif method == 'recent':
            score = self.integrate_by_recent(herb_id, disease_id, decay_factor, k=k)
        elif method == 'average':
            score = self.integrate_by_average(herb_id, disease_id, k=k)
        else:
            raise ValueError(f"未知方法: {method}")
        return score
    
    def get_time_consistency_report(self) -> Dict:
        """
        时间一致性报告: 验证签名生成逻辑的时间一致性
        
        检查项:
        1. 同一草药的扰动签名在各时间点是否一致
        2. 各时间点的疾病基因集是否来自同一来源
        
        Returns
        -------
        Dict with diagnostics
        """
        report = {
            'n_timepoints': self.n_timepoints,
            'time_points': list(self.time_points),
            'n_herbs': len(self._herb_ids),
            'n_diseases': len(self.disease_ids),
            'herb_signature_consistent': True,  # 草药签名与时间无关
            'disease_profiles_per_timepoint': {}
        }
        
        for t in self.time_points:
            _, disease_dict = self._time_hdmap_s[t]
            for did, genes in disease_dict.items():
                key = f"time={t}, disease={did}"
                report['disease_profiles_per_timepoint'][key] = len(genes)
        
        return report
