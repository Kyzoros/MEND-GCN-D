#!/usr/bin/env python3
"""
扩展评估指标模块 — AUROC, AUPRC, F1-score, 自动报告生成
============================================================

提供:
    1. TMS 排名评估: AUROC, AUPRC, F1-score, Precision@K, NDCG
    2. 显著性分析: Z-score, P-value, 效应量 (Cohen's d)
    3. 新颖性评分: 草药推荐的文献新颖度
    4. 自动化报告生成: 包含对比表格和统计摘要的 Markdown 报告
    5. 图表数据导出: 可直接用于 matplotlib/seaborn 的对比数据

用法:
    from evaluation_metrics import *
    metrics = compute_tms_ranking_metrics(tms_scores, top_n=50)
    report = generate_evaluation_report(all_results, comparison_df, output_dir)
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import sys
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)

try:
    from scipy import stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("scipy 未安装，部分统计检验将跳过")


# =====================================================================
# 核心评估指标
# =====================================================================

def compute_tms_ranking_metrics(
    tms_scores: Union[pd.Series, np.ndarray],
    top_n: int = 50,
    disease_name: str = "",
    known_positive_herbs: Optional[List[str]] = None,
    known_negative_herbs: Optional[List[str]] = None,
    higher_is_better: bool = False,
) -> Dict[str, float]:
    """
    计算 TMS 排名的多项评估指标
    
    Parameters
    ----------
    tms_scores : Series or ndarray
        TMS 得分。方向约定由 higher_is_better 决定:
          higher_is_better=False (weighted/ndcg): 负TMS=治疗方向, 越小越好
          higher_is_better=True  (dual 合并通道)  : 正TMS=治疗方向, 越大越好
    top_n : int
        Top-N 用于计算 Precision@K 等指标
    disease_name : str
        疾病名 (仅用于日志)
    known_positive_herbs : list, optional
        已知有效的草药列表 (用于计算 AUROC/AUPRC)
    higher_is_better : bool
        分值方向 (见上方约定); dual 模式应传 True
    
    Returns
    -------
    dict:
        - tms_mean, tms_std, tms_min, tms_max, tms_range
        - n_negative: 负 TMS 比例 (治疗方向)
        - tms_skewness, tms_kurtosis
        - discrimination_ratio: top/bottom 分位数比值
        - cv: 变异系数
        - z_score_top1: Top-1 的 Z-score
        - effect_size: Cohen's d (负值/正值群体)
    """
    if isinstance(tms_scores, pd.Series):
        values = tms_scores.values
    else:
        values = np.asarray(tms_scores, dtype=np.float64)
    
    n_total = len(values)
    n_negative = int((values < 0).sum())
    
    metrics = {
        "disease": disease_name,
        "n_total_herbs": n_total,
        "n_negative_tms": n_negative,
        "pct_negative": round(100.0 * n_negative / n_total, 1) if n_total > 0 else 0,
        "tms_mean": float(np.mean(values)),
        "tms_std": float(np.std(values, ddof=1)),
        "tms_median": float(np.median(values)),
        "tms_min": float(np.min(values)),
        "tms_max": float(np.max(values)),
        "tms_range": float(np.max(values) - np.min(values)),
    }
    
    # 偏度/峰度
    if n_total > 3:
        metrics["tms_skewness"] = float(stats.skew(values)) if _HAS_SCIPY else 0.0
        metrics["tms_kurtosis"] = float(stats.kurtosis(values)) if _HAS_SCIPY else 0.0
    
    # 变异系数
    if abs(metrics["tms_mean"]) > 1e-10:
        metrics["cv"] = float(metrics["tms_std"] / abs(metrics["tms_mean"]))
    else:
        metrics["cv"] = float("inf")
    
    # 区分度: top 10% vs bottom 10% 均值比
    sorted_vals = np.sort(values)
    n_10pct = max(1, n_total // 10)
    top_mean = float(np.mean(sorted_vals[:n_10pct]))
    bottom_mean = float(np.mean(sorted_vals[-n_10pct:]))
    if abs(bottom_mean) > 1e-10:
        metrics["discrimination_ratio"] = float(top_mean / bottom_mean)
    else:
        metrics["discrimination_ratio"] = float("inf")
    metrics["top10pct_mean"] = top_mean
    metrics["bottom10pct_mean"] = bottom_mean
    
    # Z-score for Top-1
    if metrics["tms_std"] > 1e-10:
        metrics["z_score_top1"] = float(
            (metrics["tms_min"] - metrics["tms_mean"]) / metrics["tms_std"]
        )
    else:
        metrics["z_score_top1"] = 0.0
    
    # Cohen's d: 负值组 vs 正值组
    neg_vals = values[values < 0]
    pos_vals = values[values > 0]
    if len(neg_vals) > 1 and len(pos_vals) > 1 and _HAS_SCIPY:
        n1, n2 = len(neg_vals), len(pos_vals)
        s1, s2 = np.std(neg_vals, ddof=1), np.std(pos_vals, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        if pooled_std > 1e-10:
            metrics["cohens_d"] = float(
                (np.mean(neg_vals) - np.mean(pos_vals)) / pooled_std
            )
    
    # AUROC / AUPRC (需已知正例/负例列表，仅在有标签的草药上评估)
    if known_positive_herbs and len(known_positive_herbs) > 0:
        if isinstance(tms_scores, pd.Series):
            herb_ids = tms_scores.index.tolist()
            scores_arr = tms_scores.values
        else:
            herb_ids = [str(i) for i in range(len(values))]
            scores_arr = values
        
        pos_set = set(known_positive_herbs)
        neg_set = set(known_negative_herbs) if known_negative_herbs else set()
        
        # 仅在有标签的草药上评估 (避免把未知草药当负样本)
        labeled_mask = []
        y_true_list = []
        y_pred_list = []
        for i, hid in enumerate(herb_ids):
            hid_str = str(hid)
            if hid_str in pos_set:
                labeled_mask.append(True)
                y_true_list.append(1)
                y_pred_list.append(float(scores_arr[i]) if higher_is_better else -float(scores_arr[i]))
            elif neg_set and hid_str in neg_set:
                labeled_mask.append(True)
                y_true_list.append(0)
                y_pred_list.append(float(scores_arr[i]) if higher_is_better else -float(scores_arr[i]))
            else:
                labeled_mask.append(False)
        
        if len(y_true_list) >= 6 and sum(y_true_list) > 0 and (len(y_true_list) - sum(y_true_list)) > 0:
            y_true = np.array(y_true_list)
            y_pred = np.array(y_pred_list)
            try:
                auroc, auprc, f1, recall, precision, accuracy = _compute_classification_metrics(y_true, y_pred)
                metrics["auroc"] = auroc
                metrics["auprc"] = auprc
                metrics["f1_score"] = f1
                metrics["recall"] = recall
                metrics["precision"] = precision
                metrics["accuracy"] = accuracy
                metrics["n_labeled"] = len(y_true_list)
                metrics["n_positive"] = int(sum(y_true_list))
                metrics["n_negative"] = int(len(y_true_list) - sum(y_true_list))
            except Exception as e:
                metrics["auroc_error"] = str(e)
    
    # Precision@K / NDCG@K (无金标准时用排名自身评估, 方向随 higher_is_better)
    metrics.update(_compute_ranking_quality(values, top_n, higher_is_better))
    
    return metrics


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    """计算 AUROC, AUPRC, F1-score, Recall, Precision, Accuracy"""
    nan6 = (float('nan'),) * 6
    if not _HAS_SCIPY:
        return nan6
    
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return nan6
    
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, accuracy_score
        auroc = float(roc_auc_score(y_true, y_pred))
        auprc = float(average_precision_score(y_true, y_pred))
        threshold = np.median(y_pred)
        y_pred_binary = (y_pred >= threshold).astype(int)
        f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
        recall = float(recall_score(y_true, y_pred_binary, zero_division=0))
        precision = float(precision_score(y_true, y_pred_binary, zero_division=0))
        accuracy = float(accuracy_score(y_true, y_pred_binary))
        return auroc, auprc, f1, recall, precision, accuracy
    except ImportError:
        # 使用 scipy 手动计算
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]
        
        # AUROC = Mann-Whitney U / (n_pos * n_neg)
        u_stat = 0
        for ps in pos_scores:
            u_stat += np.sum(ps > neg_scores) + 0.5 * np.sum(ps == neg_scores)
        auroc = float(u_stat / (len(pos_scores) * len(neg_scores)))
        
        # AUPRC 简化: 平均精度
        order = np.argsort(-y_pred)
        y_true_sorted = y_true[order]
        precisions = np.cumsum(y_true_sorted) / np.arange(1, len(y_true_sorted) + 1)
        recalls = np.cumsum(y_true_sorted) / n_pos
        # numpy 2.0+ 移除了 np.trapz，改用 np.trapezoid
        try:
            _trapz_fn = np.trapezoid
        except AttributeError:
            _trapz_fn = np.trapz  # numpy < 2.0 回退
        auprc = float(_trapz_fn(np.concatenate([[0], precisions]),
                                 np.concatenate([[0], recalls])))
        
        f1 = float('nan')
    
    return auroc, auprc, f1, float('nan'), float('nan'), float('nan')


def _auroc_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """无 sklearn 时用手动 Mann-Whitney 计算 AUROC"""
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    u = float(np.sum([np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos]))
    return u / (len(pos) * len(neg))


_EVIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def load_gold_standard(
    path: str,
    herb_ingredients_df: Optional[pd.DataFrame] = None,
    min_evidence: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    """
    加载草药-疾病金标准表
    (CSV 列: herb_id, herb_name, disease_id, evidence_level, mechanism, source, label, reason)

    label 列约定:
        label=1 (或缺省): 阳性 / 已知有效
        label=0        : 阴性 / 已知无效 (labeled negative)
    无 label 列时, 所有条目均当阳性 (向后兼容旧 gold_standard.csv)。

    返回:
        gold_positive: {disease_id: [herb_id,...]}  已知有效
        gold_negative: {disease_id: [herb_id,...]}  已知无效 (可能为空)
        warnings:      未成功映射的条目提示
    """
    gold: Dict[str, List[str]] = {}
    gold_neg: Dict[str, List[str]] = {}
    warns: List[str] = []
    if not os.path.exists(path):
        warns.append(f"金标准文件不存在: {path}")
        return gold, gold_neg, warns

    df = pd.read_csv(path)
    name_map = {}
    if herb_ingredients_df is not None and 'herb_name' in herb_ingredients_df.columns:
        name_map = dict(zip(
            herb_ingredients_df['herb_name'].astype(str),
            herb_ingredients_df['herb_id'].astype(str),
        ))

    min_rank = _EVIDENCE_RANK.get(str(min_evidence).lower(), -1) if min_evidence else -1
    has_label = 'label' in df.columns

    for _, row in df.iterrows():
        disease = str(row.get('disease_id', '')).strip()
        if not disease or disease.lower() == 'nan':
            continue
        # 证据等级过滤
        if min_rank >= 0:
            lvl = str(row.get('evidence_level', 'low')).strip().lower()
            if _EVIDENCE_RANK.get(lvl, -1) < min_rank:
                continue
        hid = str(row['herb_id']).strip() if pd.notna(row.get('herb_id')) else ''
        if (not hid or hid.lower() == 'nan') and 'herb_name' in df.columns:
            hname = str(row.get('herb_name', '')).strip()
            hid = name_map.get(hname, '')
        if not hid:
            warns.append(f"无法解析草药 (disease={disease}, name={row.get('herb_name')})")
            continue
        # label 判定 (无 label 列 → 全部当阳性)
        label = str(row.get('label', '')).strip().lower() if has_label else ''
        is_neg = (label == '0')
        target = gold_neg if is_neg else gold
        target.setdefault(disease, [])
        if hid not in target[disease]:
            target[disease].append(hid)
    return gold, gold_neg, warns


def load_gold_standard_mechanism(
    path: str,
    herb_ingredients_df: Optional[pd.DataFrame] = None,
    min_evidence: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]], Dict[str, List[str]], List[str]]:
    """
    同 load_gold_standard, 但额外按 mechanism (antagonist/engagement) 拆分金标草药。
    同时返回 labeled negatives (label=0)。

    返回:
        gold_standard:        {disease_id: [herb_id, ...]}           已知有效 (阳性)
        mechanism_map:        {disease_id: {"antagonist": [...], "engagement": [...]}}
        gold_negative:        {disease_id: [herb_id, ...]}           已知无效 (阴性)
        warnings:             ...
    """
    gold: Dict[str, List[str]] = {}
    mech: Dict[str, Dict[str, List[str]]] = {}
    gold_neg: Dict[str, List[str]] = {}
    warns: List[str] = []
    if not os.path.exists(path):
        warns.append(f"金标准文件不存在: {path}")
        return gold, mech, gold_neg, warns

    df = pd.read_csv(path)
    name_map = {}
    if herb_ingredients_df is not None and 'herb_name' in herb_ingredients_df.columns:
        name_map = dict(zip(
            herb_ingredients_df['herb_name'].astype(str),
            herb_ingredients_df['herb_id'].astype(str),
        ))
    min_rank = _EVIDENCE_RANK.get(str(min_evidence).lower(), -1) if min_evidence else -1
    has_mech = 'mechanism' in df.columns
    has_label = 'label' in df.columns

    for _, row in df.iterrows():
        disease = str(row.get('disease_id', '')).strip()
        if not disease or disease.lower() == 'nan':
            continue
        if min_rank >= 0:
            lvl = str(row.get('evidence_level', 'low')).strip().lower()
            if _EVIDENCE_RANK.get(lvl, -1) < min_rank:
                continue
        hid = str(row['herb_id']).strip() if pd.notna(row.get('herb_id')) else ''
        if (not hid or hid.lower() == 'nan') and 'herb_name' in df.columns:
            hname = str(row.get('herb_name', '')).strip()
            hid = name_map.get(hname, '')
        if not hid:
            warns.append(f"无法解析草药 (disease={disease}, name={row.get('herb_name')})")
            continue
        label = str(row.get('label', '')).strip().lower() if has_label else ''
        is_neg = (label == '0')
        if is_neg:
            gold_neg.setdefault(disease, [])
            if hid not in gold_neg[disease]:
                gold_neg[disease].append(hid)
            continue
        gold.setdefault(disease, [])
        if hid not in gold[disease]:
            gold[disease].append(hid)
        if has_mech:
            m = str(row.get('mechanism', 'engagement')).strip().lower()
            mech.setdefault(disease, {"antagonist": [], "engagement": []})
            if hid not in mech[disease][m]:
                mech[disease][m].append(hid)
    return gold, mech, gold_neg, warns


def evaluate_gold_standard(
    tms_df: pd.DataFrame,
    gold_standard: Dict[str, List[str]],
    top_k: int = 10,
    verbose: bool = True,
    higher_is_better: bool = False,
    negative_standard: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    对 TMS 得分矩阵逐疾病计算金标准 AUROC / AUPRC / F1 / Precision@K

    约定: y_true=1 表示金标准已知有效草药。
    得分方向:
        higher_is_better=False (默认, weighted/ndcg): y_pred = -TMS (负TMS=更匹配=高分)
        higher_is_better=True  (dual 合并通道)        : y_pred = +TMS (越高=越治疗)

    评估子集:
        当 negative_standard 为 None (默认) 时, 沿用原约定: 把所有"非正例"草药
        当作阴性 (标准药物重定位做法, 对未收录有效草药保守偏严)。此时 n_total_herbs
        为 TMS 矩阵全部草药数。
        当提供 negative_standard (label=0 的 labeled negatives) 时, 仅在该疾病的
        "已知阳性 ∪ 已知阴性" 子集上计算 AUROC/AUPRC, 显著避免把 500+ 味"未知"
        草药误当阴性带来的方差污染。此时 n_total_herbs 为子集大小。

    Parameters
    ----------
    tms_df : DataFrame, 含 herb_id 列 + 每疾病一列 TMS 分值
    gold_standard : {disease_id: [herb_id,...]}  已知有效(阳性)
    top_k : 计算 Precision@K / F1@K 的 K
    higher_is_better : TMS 分值方向
    negative_standard : {disease_id: [herb_id,...]} 已知无效(阴性), 可选

    Returns
    -------
    DataFrame: 每行一个疾病
    """
    rows = []
    disease_cols = [c for c in tms_df.columns if c != 'herb_id']
    for col in disease_cols:
        if col not in gold_standard:
            continue
        positives = set(str(h) for h in gold_standard[col])
        if not positives:
            continue
        series = tms_df.set_index('herb_id')[col]
        idx = [str(i) for i in series.index.tolist()]
        present_pos = positives & set(idx)
        if len(present_pos) == 0:
            if verbose:
                print(f"  [跳过] {col}: 金标准草药均未出现在 TMS 矩阵中")
            continue

        # 子集选择: labeled negatives 模式 vs 全库模式
        if negative_standard is not None:
            neg_set = set(str(h) for h in negative_standard.get(col, []))
            present_neg = neg_set & set(idx)
            if len(present_neg) == 0:
                if verbose:
                    print(f"  [跳过] {col}: 提供了阴性但均无匹配 TMS 矩阵")
                continue
            subset = list(present_pos) + list(present_neg)
            y_true = np.array([1 if h in present_pos else 0 for h in subset], dtype=int)
            y_pred = (series.loc[subset].values.astype(float)
                      if higher_is_better else -series.loc[subset].values.astype(float))
            n_total = len(subset)
        else:
            y_true = np.array([1 if h in positives else 0 for h in idx], dtype=int)
            y_pred = (series.values.astype(float)
                      if higher_is_better else -series.values.astype(float))
            n_total = len(idx)

        auroc = auprc = float('nan')
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            if 0 < int(y_true.sum()) < len(y_true):
                auroc = float(roc_auc_score(y_true, y_pred))
                auprc = float(average_precision_score(y_true, y_pred))
        except ImportError:
            auroc = _auroc_manual(y_true, y_pred)

        # Top-K (按 y_pred 降序) 指标 — 在全库范围内排名 (排名质量不依赖评估子集)
        order = np.argsort(y_pred)[::-1]
        topk = [idx[i] for i in order[:top_k]]
        hits = len(set(topk) & positives)
        prec_k = hits / top_k if top_k else 0.0
        rec_k = hits / len(present_pos) if present_pos else 0.0
        f1_k = (2 * prec_k * rec_k / (prec_k + rec_k)) if (prec_k + rec_k) > 0 else 0.0

        rows.append({
            "disease_id": col,
            "n_positive": len(present_pos),
            "n_negative": (len(present_neg) if negative_standard is not None else int(n_total - len(present_pos))),
            "n_total_herbs": n_total,
            "eval_mode": ("labeled" if negative_standard is not None else "all_as_neg"),
            "AUROC": round(auroc, 4) if auroc == auroc else float('nan'),
            "AUPRC": round(auprc, 4) if auprc == auprc else float('nan'),
            f"precision@{top_k}": round(prec_k, 4),
            f"recall@{top_k}": round(rec_k, 4),
            f"f1@{top_k}": round(f1_k, 4),
            "n_hits_in_topK": hits,
            "mean_tms_positive": round(float(series.loc[list(present_pos)].mean()), 4),
        })
        if verbose:
            mode = "labeled" if negative_standard is not None else "all_as_neg"
            print(f"  {col}[{mode}]: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                  f"P@{top_k}={prec_k:.4f}  (命中 {hits}/{top_k}, 阳性 {len(present_pos)}, "
                  f"阴性 {len(present_neg) if negative_standard is not None else 'NA'})")
    return pd.DataFrame(rows)


def _compute_ranking_quality(values: np.ndarray, top_n: int,
                             higher_is_better: bool = False) -> Dict[str, float]:
    """基于排名统计学计算排名质量指标。

    方向随 higher_is_better 切换:
      higher_is_better=False (weighted/ndcg): 负TMS=治疗方向 (越小越好)
      higher_is_better=True  (dual 合并通道)  : 正TMS=治疗方向 (越大越好)
    返回的 precision/ndcg/treatment_ratio 始终表示"治疗方向草药"在 Top-N 中的占比/排序增益。
    """
    n = len(values)
    # 治疗方向分值: 越大越治疗
    treat = values if higher_is_better else -values
    # 按治疗分值降序取 Top-N
    sorted_idx = np.argsort(treat)[::-1]
    top_n_indices = sorted_idx[:top_n]
    n_treat_in_top = int((treat[top_n_indices] > 0).sum())

    # NDCG-like: 治疗分值越高增益越大
    gains = np.maximum(treat, 0.0)
    ideal_gains = np.sort(gains)[::-1]

    dcg = np.sum(gains[top_n_indices] / np.log2(np.arange(2, min(top_n, n) + 2)))
    idcg = np.sum(ideal_gains[:top_n] / np.log2(np.arange(2, min(top_n, n) + 2)))

    return {
        f"precision@{top_n}": round(n_treat_in_top / top_n, 4) if top_n > 0 else 0,
        f"ndcg@{top_n}": round(float(dcg / idcg), 4) if idcg > 1e-10 else 0,
        f"treatment_ratio@{top_n}": round(n_treat_in_top / top_n, 4) if top_n > 0 else 0,
    }


def compute_novelty_score(
    herb_names: List[str],
    disease_name: str,
    literature_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    计算草药推荐的新颖性得分
    
    新颖性 = 1 - normalized(co-occurrence_count)
    从未被文献报道的草药 = 高新颖性
    
    Parameters
    ----------
    herb_names : list
        推荐的草药名称列表
    disease_name : str
        疾病名
    literature_counts : dict, optional
        {herb_name: co_occurrence_count} 文献共现计数
    
    Returns
    -------
    dict: novelty scores
    """
    if literature_counts is None:
        return {"novelty_mean": float('nan'), "novelty_scores": {}}
    
    counts = [literature_counts.get(h, 0) for h in herb_names]
    max_count = max(counts) if counts and max(counts) > 0 else 1
    novelty_scores = {h: 1.0 - c / max_count for h, c in zip(herb_names, counts)}
    
    return {
        "novelty_mean": float(np.mean(list(novelty_scores.values()))),
        "novelty_median": float(np.median(list(novelty_scores.values()))),
        "novelty_scores": novelty_scores,
        "n_zero_literature": int(sum(1 for c in counts if c == 0)),
    }


def evaluate_herb_prediction(
    predictions: pd.DataFrame,
    herb_signatures: Dict,
    disease_genes: List[str],
    builder,
    known_positive: Optional[List[str]] = None,
) -> Dict:
    """
    综合评估草药预测结果
    
    Parameters
    ----------
    predictions : DataFrame
        含 herb_id, TMS_score 列的预测结果
    herb_signatures : dict
        草药扰动签名
    disease_genes : list
        疾病基因列表
    builder : PerturbationSignatureBuilder
        签名构建器
    known_positive : list, optional
        已知有效草药
    
    Returns
    -------
    dict: 综合评估结果
    """
    results = {}
    
    # TMS 排名指标
    if 'TMS_score' in predictions.columns:
        tms = predictions.set_index('herb_id')['TMS_score']
    elif predictions.shape[1] >= 2:
        tms = predictions.set_index(predictions.columns[0])[predictions.columns[1]]
    else:
        return {"error": "无法提取 TMS 得分列"}
    
    results["ranking_metrics"] = compute_tms_ranking_metrics(
        tms, top_n=50, known_positive_herbs=known_positive
    )
    
    # 基因覆盖率
    all_disease_set = set(disease_genes)
    gene_coverages = {}
    for herb in predictions['herb_id'].head(10):
        if herb in herb_signatures:
            sig = herb_signatures[herb]
            top_genes, bottom_genes = builder.extract_top_bottom_genes(
                sig, k=250
            )
            perturbed = set(top_genes) | set(bottom_genes)
            coverage = len(perturbed & all_disease_set) / max(len(all_disease_set), 1)
            gene_coverages[herb] = round(coverage, 4)
    
    results["gene_coverages"] = gene_coverages
    if gene_coverages:
        results["mean_coverage_top10"] = round(
            np.mean(list(gene_coverages.values())), 4
        )
    
    return results


# =====================================================================
# 消融对比分析
# =====================================================================

def compare_ablation_scenarios(
    all_results: Dict[str, dict],
) -> pd.DataFrame:
    """
    对比消融场景
    
    返回包含各场景关键指标的对比表，按 TMS 均值排名
    """
    rows = []
    for scenario_id, result in all_results.items():
        row = {
            "场景": result.get("scenario_name", scenario_id),
            "场景ID": scenario_id,
            "TMS_均值": result.get("tms_mean"),
            "TMS_标准差": result.get("tms_std"),
            "Top1_TMS": result.get("top1_score"),
            "负TMS比例": result.get("pct_negative") if "pct_negative" in result
                        else (result.get("n_negative_tms", 0) / result.get("n_total_herbs", 500)
                              if result.get("n_total_herbs") else None),
            "GA最优得分": result.get("ga_best_score"),
            "HoF最优得分": result.get("hof_best_score"),
            "GA代数": result.get("ga_generations"),
            "耗时(秒)": round(result.get("runtime_seconds", 0), 1),
        }
        # 扩展指标
        metrics = result.get("metrics", {})
        if metrics:
            row["区分度"] = metrics.get("discrimination_ratio")
            row["Top1_Zscore"] = metrics.get("z_score_top1")
            row["CV"] = metrics.get("cv")
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if "TMS_均值" in df.columns:
        df = df.sort_values("TMS_均值")
    
    return df


# =====================================================================
# 自动报告生成
# =====================================================================

def generate_evaluation_report(
    all_results: Dict[str, dict],
    comparison_df: pd.DataFrame,
    output_dir: str,
    title: str = "Systems-Mapping 评估报告",
) -> str:
    """
    生成结构化的评估报告 (Markdown 格式)
    
    Parameters
    ----------
    all_results : dict
        各场景的结果字典
    comparison_df : DataFrame
        消融对比表
    output_dir : str
        报告输出目录
    title : str
        报告标题
    
    Returns
    -------
    str: 报告文件路径
    """
    report_path = os.path.join(output_dir, "evaluation_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # ---- 摘要 ----
        f.write("## 1. 运行摘要\n\n")
        n_scenarios = len(all_results)
        full_result = all_results.get("full", list(all_results.values())[0] if all_results else {})
        
        f.write(f"| 指标 | 值 |\n")
        f.write(f"|------|----|\n")
        f.write(f"| 场景数 | {n_scenarios} |\n")
        if full_result:
            f.write(f"| 疾病 | {full_result.get('disease_id', 'N/A')} |\n")
            f.write(f"| 疾病基因数 | {full_result.get('n_disease_genes', 'N/A')} |\n")
            f.write(f"| 完整模型 Top1 TMS | {full_result.get('top1_score', 'N/A')} |\n")
            f.write(f"| 完整模型 负TMS比例 | {full_result.get('n_negative_tms', 0)/500*100 if full_result.get('n_negative_tms') else 'N/A'}% |\n")
        f.write("\n---\n\n")
        
        # ---- 对比表 ----
        f.write("## 2. 消融对比表\n\n")
        if not comparison_df.empty:
            try:
                f.write(comparison_df.to_markdown(index=False))
            except ImportError:
                f.write("```\n")
                f.write(comparison_df.to_string(index=False))
                f.write("\n```\n")
        else:
            f.write("(无对比数据)\n")
        f.write("\n\n---\n\n")
        
        # ---- 各场景详情 ----
        f.write("## 3. 各场景详细结果\n\n")
        for scenario_id, result in all_results.items():
            f.write(f"### 3.{list(all_results.keys()).index(scenario_id)+1} "
                    f"{result.get('scenario_name', scenario_id)}\n\n")
            
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|----|\n")
            f.write(f"| TMS 均值 | {result.get('tms_mean', 'N/A')} |\n")
            f.write(f"| TMS 标准差 | {result.get('tms_std', 'N/A')} |\n")
            f.write(f"| Top1 草药 | {result.get('top1_id', 'N/A')} |\n")
            f.write(f"| Top1 TMS | {result.get('top1_score', 'N/A')} |\n")
            
            if result.get('ga_best_combo'):
                f.write(f"| GA 最优组合 | {', '.join(result['ga_best_combo'])} |\n")
                f.write(f"| GA 得分 | {result.get('ga_best_score', 'N/A')} |\n")
            
            f.write(f"| 运行时 | {result.get('runtime_seconds', 'N/A')}s |\n")
            
            if result.get('overrides'):
                f.write(f"\n**配置覆盖**: {result['overrides']}\n")
            
            f.write("\n")
        
        f.write("\n---\n\n")
        
        # ---- 结论与建议 ----
        f.write("## 4. 结论与建议\n\n")
        
        # 找出最佳场景（优先用AUROC，缺则用TMS均值）
        if not comparison_df.empty:
            if 'AUROC' in comparison_df.columns and comparison_df['AUROC'].notna().any():
                best = comparison_df.loc[comparison_df['AUROC'].idxmax()]
                f.write(f"- **最佳 AUROC**: {best['场景']} "
                        f"(AUROC={best['AUROC']:.4f})\n")
            if 'AUPR' in comparison_df.columns and comparison_df['AUPR'].notna().any():
                best_aupr = comparison_df.loc[comparison_df['AUPR'].idxmax()]
                f.write(f"- **最佳 AUPR**: {best_aupr['场景']} "
                        f"(AUPR={best_aupr['AUPR']:.4f})\n")
            if 'TMS_均值' in comparison_df.columns:
                best_tms = comparison_df.loc[comparison_df['TMS_均值'].idxmin()]
                f.write(f"- **最佳 TMS 均值**（仅参考）: {best_tms['场景']} "
                        f"(均值={best_tms['TMS_均值']:.4f})\n")
            
            # 消融影响分析
            if "full" in all_results:
                full_metrics = all_results["full"]
                full_auroc = full_metrics.get("metrics", {}).get("auroc", None)
                f.write("\n### 模块消融影响\n\n")
                if full_auroc is not None:
                    f.write("| 移除模块 | AUROC | AUROC变化 | 影响 |\n")
                    f.write("|---------|-------|----------|------|\n")
                    for sid, result in all_results.items():
                        if sid == "full":
                            f.write(f"| 完整模型 | {full_auroc:.4f} | — | — |\n")
                            continue
                        auroc = result.get("metrics", {}).get("auroc", float('nan'))
                        if auroc is not None and not (isinstance(auroc, float) and np.isnan(auroc)):
                            delta = auroc - full_auroc
                            impact = "🔴 严重下降" if delta < -0.05 else \
                                     "🟡 轻度下降" if delta < -0.01 else \
                                     "🟢 无影响" if delta < 0.01 else "↑ 提升"
                            f.write(f"| {result.get('scenario_name', sid)} "
                                    f"| {auroc:.4f} | {delta:+.4f} | {impact} |\n")
                else:
                    f.write("| 移除模块 | TMS 均值变化 | 影响 |\n")
                    f.write("|---------|-------------|------|\n")
                    full_tms = full_metrics.get("tms_mean", 0)
                    for sid, result in all_results.items():
                        if sid == "full":
                            continue
                        tms_diff = result.get("tms_mean", 0) - full_tms
                        impact = "⚠️ 显著下降" if tms_diff > 0.005 else \
                                 "↓ 轻微下降" if tms_diff > 0.001 else \
                                 "↑ 提升" if tms_diff < -0.001 else "→ 无显著影响"
                        f.write(f"| {result.get('scenario_name', sid)} "
                                f"| {tms_diff:+.4f} | {impact} |\n")
        
        f.write("\n---\n")
        f.write(f"*报告由 Systems-Mapping 自动生成*\n")
    
    print(f"  评估报告已保存至: {report_path}")
    return report_path


def export_chart_data(
    tms_scores: pd.Series,
    output_dir: str,
    prefix: str = "tms",
) -> Dict[str, str]:
    """
    导出可用于可视化的图表数据
    
    Returns
    -------
    dict: {chart_type: filepath} 各图表数据文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    files = {}
    
    # 1. TMS 分布直方图数据
    hist, bins = np.histogram(tms_scores.values, bins=50)
    hist_df = pd.DataFrame({
        "bin_start": bins[:-1],
        "bin_end": bins[1:],
        "count": hist,
    })
    hist_path = os.path.join(output_dir, f"{prefix}_distribution.csv")
    hist_df.to_csv(hist_path, index=False)
    files["distribution"] = hist_path
    
    # 2. TMS 排名折线图数据
    sorted_scores = np.sort(tms_scores.values)
    rank_df = pd.DataFrame({
        "rank": range(1, len(sorted_scores) + 1),
        "tms_score": sorted_scores,
        "is_negative": sorted_scores < 0,
    })
    rank_path = os.path.join(output_dir, f"{prefix}_ranking.csv")
    rank_df.to_csv(rank_path, index=False)
    files["ranking"] = rank_path
    
    # 3. Top-N 条形图数据
    top_n = tms_scores.nsmallest(20)
    bar_df = pd.DataFrame({
        "herb": top_n.index,
        "tms_score": top_n.values,
    })
    bar_path = os.path.join(output_dir, f"{prefix}_top20.csv")
    bar_df.to_csv(bar_path, index=False)
    files["top20"] = bar_path
    
    return files
