#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugAI 方向准确率验证: 用项目内 DrugBank 金标准(11,038 条)对照 DrugAI 预测
================================================================================
  gold:       data/processed/experimental_mechanisms_gold.csv  (canon SMILES, gene, action∈-1/+1, DrugBank)
  ingredient SMILES: data/raw/ingredient_chembl_mapping.csv  (ingredient_id, smiles)
  DrugAI 预测: data/processed/ingredient_targets_enhanced_kegg.csv  (ingredient_id, target_gene, action_type, drugai_conf)

匹配: 用 RDKit 把两边的 SMILES 规范化成 canonical 再 join, 避免字符串形式差异。
"""
import os
import sys

import pandas as pd
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RAW = os.path.join(BASE, "data", "raw")

# --- 1. Gold (DrugBank 11,038) ---
gold = pd.read_csv(os.path.join(PROC, "experimental_mechanisms_gold.csv"), low_memory=False)
print(f"gold rows: {len(gold)}  action dist: {gold['action'].value_counts().to_dict()}")
print(f"gold unique canon: {gold['canon'].nunique()}  unique gene: {gold['gene'].nunique()}")

# --- 2. 成分 SMILES (ingredient_chembl_mapping.csv) ---
ing_map = pd.read_csv(os.path.join(RAW, "ingredient_chembl_mapping.csv"), low_memory=False)
print(f"ing_map rows: {len(ing_map)}  cols: {list(ing_map.columns)}  "
      f"smiles 非空: {ing_map['smiles'].notna().sum()}")

# --- 3. RDKit 规范化两边 SMILES ---
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # 静音
    def canon(s):
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m, canonical=True) if m else None
    gold["canon_rdk"] = gold["canon"].astype(str).apply(canon)
    ing_map["canon_rdk"] = ing_map["smiles"].astype(str).apply(canon)
    use_rdkit = True
    print("RDKit OK, 已做 canonical 规范化")
except Exception as e:
    print(f"RDKit 不可用 ({e}); 退化为简单规范化 (low+strip)")
    gold["canon_rdk"] = gold["canon"].astype(str).str.strip().str.lower()
    ing_map["canon_rdk"] = ing_map["smiles"].astype(str).str.strip().str.lower()
    use_rdkit = False

# 去重 (gold 同 canon+gene 可能多 evidence)
g = gold.dropna(subset=["canon_rdk"])[["canon_rdk", "gene", "action"]].drop_duplicates(["canon_rdk", "gene"])
i = ing_map.dropna(subset=["canon_rdk"])[["ingredient_id", "canon_rdk"]].drop_duplicates(["ingredient_id", "canon_rdk"])
print(f"\n=== 规范化后去重 ===")
print(f"gold (canon, gene) 唯一: {len(g)}")
print(f"ing_map (ingredient, canon) 唯一: {len(i)}")

m = g.merge(i, on="canon_rdk", how="inner")
print(f"\n=== SMILES 匹配 ===")
matched_smis = len(m)
print(f"通过 SMILES 匹配上的 (ingredient_id, canon, gene) 行: {matched_smis}")
print(f"涉及独特成分: {m['ingredient_id'].nunique() if matched_smis else 0}")
print(f"涉及独特靶点: {m['gene'].nunique() if matched_smis else 0}")

# --- 4. 在匹配上的边上, DrugAI 预测了什么 ---
enh = pd.read_csv(os.path.join(BASE, "data", "processed", "ingredient_targets_enhanced_kegg.csv"),
                  low_memory=False)
m["ingredient_id"] = m["ingredient_id"].astype(int)
mg = m.merge(
    enh[["ingredient_id", "target_gene", "action_type", "action_source", "drugai_conf"]],
    left_on=["ingredient_id", "gene"], right_on=["ingredient_id", "target_gene"], how="inner")
print(f"\n=== gold × DrugAI 可比对边 ===")
print(f"在 enh 中找到对应 (成分, 靶点): {len(mg)}")
print(f"action_source 分布: {mg['action_source'].value_counts().to_dict()}")
print(f"drugai_conf 非空 (DrugAI 实际做了预测): {mg['drugai_conf'].notna().sum()}")

# --- 5. DrugAI 方向准确率 (用 DrugAI 真正预测过的边, 即 drugai_conf 非空) ---
da = mg.dropna(subset=["drugai_conf"]).copy()
da["action_pred"] = da["action_type"]
da["action_true"] = da["action"]   # gold 已是 -1/+1
da["correct"] = (da["action_pred"] == da["action_true"])
print(f"\n=== DrugAI 方向准确率 ===")
print(f"可评估样本: {len(da)}")
if len(da) > 0:
    print(f"整体准确率: {da['correct'].mean():.4f}")
    print(f"预测分布 pred: {da['action_pred'].value_counts().to_dict()}")
    print(f"真值分布 true: {da['action_true'].value_counts().to_dict()}")
    # 混淆矩阵
    cm = pd.crosstab(da["action_true"], da["action_pred"],
                     rownames=["true"], colnames=["pred"])
    print("混淆矩阵 (true \\ pred):")
    print(cm.to_string())

# --- 6. 按置信度分层 (校准曲线) ---
if len(da) >= 50:
    print(f"\n=== 按 drugai_conf 分层 ===")
    da["conf_bin"] = pd.cut(da["drugai_conf"], bins=[0, 0.5, 0.8, 0.9, 0.95, 1.01],
                            right=False)
    g2 = da.groupby("conf_bin", observed=False).agg(
        n=("correct", "size"), acc=("correct", "mean"),
        conf_mean=("drugai_conf", "mean"))
    print(g2.to_string())