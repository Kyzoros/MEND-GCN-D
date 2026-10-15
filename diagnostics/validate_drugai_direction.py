#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugAI 方向预测准确率验证 (路线 1) —— 加固版
================================================================================
策略: 用 DrugBank 已知作用方向的金标准 (experimental_mechanisms_gold.csv,
      11,038 条, 全部 DrugBank) 匹配你的成分, 再让 DrugAI 模型对匹配上的
      (成分, 靶点) 对做原始方向预测, 与金标准比对准确率。

为何要重跑模型: 现有 ingredient_targets_enhanced_kegg.csv 里 exp_verified=1 的
144 条和额外匹配的 37 条, action_type 都已被"实验真值覆盖", DrugAI 原始
预测已丢失。所以必须重新调 DrugAI 模型做预测。

依赖: torch, dgl, rdkit (你之前跑 DrugAI 增强时已经装过)
预计耗时: 597 条 × GNN 推理, 几分钟
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RAW = os.path.join(BASE, "data", "raw")
sys.path.insert(0, BASE)

# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
from drugai_adapter import DrugAIAdapter
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

OUT = os.path.join(PROC, "drugai_direction_validation.csv")


def canon(s):
    """RDKit 规范化 SMILES; 失败返回 None"""
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True) if m else None


def main():
    # ----1) 加载 DrugBank 金标准
    gold = pd.read_csv(os.path.join(PROC, "experimental_mechanisms_gold.csv"), low_memory=False)
    print(f"[Gold] {len(gold)} 行, action∈{{-1(抑制),+1(激活)}}, {gold['gene'].nunique()} 基因")

    # ----2) 加载成分 SMILES 并规范化
    ing_map = pd.read_csv(os.path.join(RAW, "ingredient_chembl_mapping.csv"), low_memory=False)
    ing_map = ing_map.copy()
    ing_map["canon"] = ing_map["smiles"].astype(str).apply(canon)
    gold = gold.copy()
    gold["canon"] = gold["canon"].astype(str).apply(canon)

    # ----3) RDKit canonical 规范化匹配
    g = (gold.dropna(subset=["canon"])[["canon", "gene", "action"]]
              .drop_duplicates(["canon", "gene"]))
    i = (ing_map.dropna(subset=["canon"])[["ingredient_id", "canon"]]
              .drop_duplicates(["ingredient_id", "canon"]))
    m = g.merge(i, on="canon", how="inner")
    print(f"[匹配] {len(m)} 条 (ingredient_id, gene) 唯一对, "
          f"涉及 {m['ingredient_id'].nunique()} 成分 / {m['gene'].nunique()} 靶点")

    pairs = m[["ingredient_id", "gene", "action"]].copy()
    pairs["ingredient_id"] = pairs["ingredient_id"].astype(int).astype(str)
    pairs["gene"] = pairs["gene"].astype(str)
    pairs = pairs.reset_index(drop=True)

    # ----4) 加载 DrugAI 模型
    print("[DrugAI] 加载模型 best_model_rdkit.pkl ...")
    adapter = DrugAIAdapter(use_gpu=False)
    adapter.load_model()
    print(f"[DrugAI] 加载: {adapter.is_loaded}, 特征: {getattr(adapter, '_feature_scheme', 'N/A')}")

    if not adapter.is_loaded:
        print("[DrugAI] 模型未加载, 无法继续。请检查 Drug_AI_best_view/best_model_rdkit.pkl")
        return

    # ----5) 遍历预测 (含容错)
    n = len(pairs)
    rows = []
    for i, r in pairs.iterrows():
        iid, gene, true_act = r["ingredient_id"], r["gene"], int(r["action"])
        try:
            pred_act, conf = adapter.predict_action_type_conf(iid, gene)
            pred_act = int(pred_act) if pred_act is not None else None
            conf = float(conf) if conf is not None and conf == conf else float("nan")  # NaN 检查
        except Exception as e:
            print(f"  [warn] 预测失败 ({iid}, {gene}): {e}")
            pred_act, conf = None, float("nan")
        rows.append({"ingredient_id": iid, "gene": gene,
                     "action_true": true_act, "action_pred": pred_act,
                     "confidence": conf})
        if (i + 1) % 100 == 0:
            print(f"  ... 已预测 {i+1}/{n}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\n[写入] {OUT}")

    # ----6) 整体准确率 (pred∈{-1,+1} 才计为确定预测, =0 视为弃权)
    df_eval = df[df["action_pred"].isin([-1, 1])].copy()
    if len(df_eval) == 0:
        print("[结果] 没有可评估的预测 (全部弃权或失败)")
        return

    df_eval["correct"] = (df_eval["action_pred"] == df_eval["action_true"]).astype(int)

    print(f"\n=== 整体准确率 (剔除 DrugAI 弃权 pred=0 / 失败) ===")
    print(f"总可评估: {len(df_eval)} / 总 {len(df)}")
    print(f"准确率:   {df_eval['correct'].mean():.4f}")
    print(f"预测分布: {df_eval['action_pred'].value_counts().to_dict()}")
    print(f"真值分布: {df_eval['action_true'].value_counts().to_dict()}")
    cm = pd.crosstab(df_eval["action_true"], df_eval["action_pred"],
                     rownames=["true"], colnames=["pred"])
    print("混淆矩阵 (true \\ pred):")
    print(cm.to_string())

    # ----7) 按置信度分层 (校准曲线) —— 用预计算列避免 lambda + dropna=False 包含空组
    if len(df_eval) >= 50:
        df_eval["conf_bin"] = pd.cut(df_eval["confidence"],
                                     bins=[0, 0.3, 0.5, 0.7, 0.9, 1.01],
                                     right=False)
        df_eval["pred_is_1"] = (df_eval["action_pred"] == 1).astype(int)
        df_eval["pred_is_neg1"] = (df_eval["action_pred"] == -1).astype(int)

        g = df_eval.groupby("conf_bin", observed=False, dropna=False).agg(
            n=("correct", "size"),
            acc=("correct", "mean"),
            conf_mean=("confidence", "mean"),
            rate_pos=("pred_is_1", "mean"),
            rate_neg=("pred_is_neg1", "mean"),
        )
        print("\n=== 按 confidence 分层 (校准) ===")
        print(g.to_string())
    else:
        print(f"\n[跳过分层] 可评估样本 {len(df_eval)} < 50")

    # ----8) 严格准确率 (把弃权 pred=0 / 失败 NaN 都当错)
    df_strict = df.copy()
    df_strict["action_pred_filled"] = df_strict["action_pred"].fillna(0)
    df_strict["correct_strict"] = (df_strict["action_pred_filled"] == df_strict["action_true"]).astype(int)
    print("\n=== 严格准确率 (pred=0 或 NaN 算错) ===")
    print(f"准确率: {df_strict['correct_strict'].mean():.4f}")

    # ----9) 低置信度 (confidence<0.5) 准确率 vs 高置信度
    if df_eval["confidence"].notna().sum() > 0:
        low = df_eval[df_eval["confidence"] < 0.5]
        high = df_eval[df_eval["confidence"] >= 0.5]
        print("\n=== 按 confidence 阈值分块 ===")
        if len(low):
            print(f"低置信 (conf<0.5): n={len(low)}, acc={low['correct'].mean():.4f}")
        if len(high):
            print(f"高置信 (conf>=0.5): n={len(high)}, acc={high['correct'].mean():.4f}")


if __name__ == "__main__":
    main()