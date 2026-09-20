"""
数据加载模块 - 加载疾病表达谱、草药成分-靶标数据、PPI网络
=================================================================
支持两种模式:
1. 原始模式: 使用PreAM+WES的数据 (ingredient_targets.csv)
2. 增强模式: 使用DrugAI+HDCTI替代PreAM+WES
   - DrugAI替代PreAM: 重新预测action_type (激活/抑制)
   - HDCTI替代WES: 预测新靶标 + 补充TCMSP数据

多源PPI融合:
  - 支持从STRING、BioGRID等多个PPI数据库同时加载
  - 融合策略: 证据计数加权 / 置信度加权 / 简单并集
  - 输出加权邻接矩阵，自动适配热扩散和RWR
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from scipy import sparse
import config as _cfg

from config import (
    RAW_DIR, PROCESSED_DIR,
    PPI_N_PROTEINS, PPI_N_INTERACTIONS,
    # HDCTI & DrugAI 配置
    HDCTI_USE, HDCTI_CONFIDENCE_THRESHOLD, HDCTI_TOP_K,
    HDCTI_USE_TCMSP_TARGETS, HDCTI_TCMSP_SVM_THRESHOLD, HDCTI_TCMSP_RF_THRESHOLD,
    HDCTI_USE_GCN_PREDICTION, HDCTI_SIMILARITY_METHOD, HDCTI_ZSCORE_THRESHOLD,
    DRUGAI_USE, DRUGAI_MODEL_PATH, DRUGAI_CONFIDENCE_THRESHOLD,
    DRUGAI_DEVICE, DRUGAI_USE_GPU, DRUGAI_DISEASE_VALIDATE,
    # 实验验证机制覆盖 (办法B)
    EXP_MECH_ENABLE, EXP_MECH_MATCH_MODE,
    TCMSP_DATA_DIR, DRUGAI_DATA_DIR,
    # 边置信度加权 (解决"覆盖率提高反而降 AUROC")
    DIRECTION_CONFIDENCE_WEIGHTING, OOD_COMPLETION_PENALTY,
    KEGG_DIRECTION_RELIABILITY, PPI_DIRECTION_RELIABILITY,
    # 多源PPI融合配置
    PPI_MULTI_SOURCE, PPI_SOURCE_DIR, PPI_FUSION_STRATEGY, PPI_MIN_SOURCES,
)


# ==================== 数据格式说明 ====================
# 用户需要自行准备的数据文件（放入 data/raw/ 目录）：
#
# 1. disease_profiles.csv
#    - 列：disease_id, gene_symbol, logFC, P_value
#    - 论文来源：GEO数据库, 460个疾病表达谱
#    - 需要用户从GEO下载并差异分析
#
# 2. herb_ingredients.csv
#    - 列：herb_id, herb_name, ingredient_id, ingredient_name, content_ratio
#    - 论文来源：TCMSP数据库 (502 herbs, 10,329 ingredients)
#
# 3. ingredient_targets.csv
#    - 列：ingredient_id, target_gene, action_type (+1=激活, -1=抑制)
#    - 论文来源：ChEMBL v23 + BindingDB + WES预测
#
# 4. ppi_network.csv
#    - 列：protein_A, protein_B
#    - 论文来源：整合PPI网络 (13,460 proteins, 141,296 interactions)
#    - 参考：Menche et al., Science 2015
#
# 5. herb_metadata.csv (可选)
#    - 列：herb_id, herb_name, herb_pinyin, herb_latin, herb_category


def load_disease_profiles(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    加载疾病病理状态表达谱
    
    Returns
    -------
    DataFrame columns: disease_id, gene_symbol, logFC, P_value
    """
    if filepath is None:
        filepath = os.path.join(PROCESSED_DIR, "disease_profiles_processed.csv")
    
    if not os.path.exists(filepath):
        # 尝试从 raw 加载
        raw_path = os.path.join(RAW_DIR, "disease_profiles.csv")
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            _preprocess_disease_profiles(df, filepath)
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(
                f"疾病表达谱文件未找到: {raw_path}\n"
                "请从GEO数据库下载疾病表达数据并放入 data/raw/disease_profiles.csv\n"
                "格式: disease_id, gene_symbol, logFC, P_value"
            )
    return pd.read_csv(filepath)


def _preprocess_disease_profiles(raw: pd.DataFrame, outpath: str) -> None:
    """预处理：过滤无效基因标识、去重"""
    # 1. 过滤数字探针ID（无法与PPI匹配）
    numeric_mask = raw["gene_symbol"].astype(str).str.match(r'^\d+$')
    n_numeric = numeric_mask.sum()
    if n_numeric > 0:
        raw = raw[~numeric_mask]
        print(f"[数据] 过滤数字探针ID: {n_numeric} 行移除 (无法匹配PPI)")
    
    # 2. 保留每个基因P值最小的探针
    df = raw.sort_values("P_value").drop_duplicates(
        subset=["disease_id", "gene_symbol"], keep="first"
    )
    # 按logFC排序
    df = df.sort_values(["disease_id", "logFC"], ascending=[True, False])
    df.to_csv(outpath, index=False)
    print(f"[数据] 疾病表达谱预处理完成 -> {outpath} ({len(df)} 条)")


def get_disease_gene_ranks(disease_id: str, profiles_df: pd.DataFrame) -> pd.Series:
    """
    获取单个疾病的基因排序（按logFC降序）
    返回 Series: index=gene, value=rank (1-based)
    """
    sub = profiles_df[profiles_df["disease_id"] == disease_id].copy()
    sub = sub.sort_values("logFC", ascending=False)
    return pd.Series(data=range(1, len(sub) + 1), index=sub["gene_symbol"].values)


def load_herb_ingredients(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    加载草药-成分关系表
    
    Returns
    -------
    DataFrame columns: herb_id, herb_name, ingredient_id, ingredient_name, content_ratio
    """
    if filepath is None:
        filepath = os.path.join(PROCESSED_DIR, "herb_ingredients_processed.csv")
    
    if not os.path.exists(filepath):
        raw_path = os.path.join(RAW_DIR, "herb_ingredients.csv")
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            df.to_csv(filepath, index=False)
            return df
        else:
            raise FileNotFoundError(
                f"草药-成分数据未找到: {raw_path}\n"
                "请从TCMSP数据库下载 https://tcmsp-e.com\n"
                "格式: herb_id, herb_name, ingredient_id, ingredient_name, content_ratio"
            )
    return pd.read_csv(filepath)


def _get_pubchem_completed_ids():
    """识别补全(PubChem)来源成分: chembl_id为空者(smiles_batch_complete.py:148)。
    补全SMILES多为OOD天然产物, DrugAI方向不可信, 需强制中性(action_type=0)。"""
    ids = set()
    try:
        mp_path = os.path.join(RAW_DIR, "ingredient_chembl_mapping.csv")
        if os.path.exists(mp_path):
            mp = pd.read_csv(mp_path)
            if "chembl_id" in mp.columns and "ingredient_id" in mp.columns:
                empty_ck = mp["chembl_id"].isna() | (
                    mp["chembl_id"].astype(str).str.strip().isin(["", "nan", "None"])
                )
                ids = set(str(i).replace(".0", "") for i in mp.loc[empty_ck, "ingredient_id"])
    except Exception as e:
        print(f"[DataLoader] 读取SMILES来源失败: {e}")
    return ids


def _get_chembl_ingredient_ids():
    """返回有 ChEMBL ID 的成分ID集合（可靠 SMILES 来源，非 PubChem 补全 OOD）。

    与 _get_pubchem_completed_ids() 互补: 有 chembl_id → ChEMBL 验证过 → 可靠；
    无 chembl_id → PubChem 补全 → OOD 天然产物 → 不可信。
    """
    ids = set()
    try:
        mp_path = os.path.join(RAW_DIR, "ingredient_chembl_mapping.csv")
        if os.path.exists(mp_path):
            mp = pd.read_csv(mp_path)
            if "chembl_id" in mp.columns and "ingredient_id" in mp.columns:
                valid = mp["chembl_id"].notna() & (
                    ~mp["chembl_id"].astype(str).str.strip().isin(["", "nan", "None"])
                )
                ids = set(str(i).replace(".0", "") for i in mp.loc[valid, "ingredient_id"])
    except Exception as e:
        print(f"[DataLoader] 读取ChEMBL映射失败: {e}")
    return ids


def filter_chembl_only(df: pd.DataFrame) -> pd.DataFrame:
    """过滤：仅保留 ChEMBL 来源成分的靶标，排除 PubChem 补全 OOD 成分。

    PubChem 补全成分（62%的成分, SMILES 为 OOD 天然产物）的 DrugAI 方向预测不可靠、
    HDCTI 结构相似度匹配易误报，直接靶标作为噪声源拖低 AUROC。
    此过滤独立于靶标来源(source 列)，仅按成分的 SMILES 来源区分。
    """
    chembl_ids = _get_chembl_ingredient_ids()
    if not chembl_ids:
        print("[ChEMBL过滤] 未找到ChEMBL映射，跳过过滤")
        return df
    id_col = df["ingredient_id"].astype(str).str.replace(".0", "", regex=False)
    mask = id_col.isin(chembl_ids)
    n_before = len(df)
    df = df[mask].copy()
    n_after = len(df)
    n_removed = n_before - n_after
    print(f"[ChEMBL过滤] 仅保留ChEMBL来源成分靶标: {n_before} → {n_after} "
          f"({n_after*100/n_before:.0f}%), 移除PubChem补全 {n_removed} 行")
    return df


def _neutralize_completed_directions(df: pd.DataFrame) -> pd.DataFrame:
    """将补全(PubChem)来源成分的 action_type 强制中性(0), 仅保留靶标存在性(Path1桥梁),
    避免OOD天然产物的不可信DrugAI方向污染TMS。无论走缓存还是重建均生效。"""
    ids = _get_pubchem_completed_ids()
    if not ids:
        return df
    id_col = df["ingredient_id"].astype(str).str.replace(".0", "", regex=False)
    mask = id_col.isin(ids)
    n_neu = int(mask.sum())
    if n_neu == 0:
        return df
    df = df.copy()
    df.loc[mask, "action_type"] = 0
    df.loc[mask, "action_source"] = "Neutralized_Completed"
    print(f"[DataLoader] 识别补全(PubChem)来源成分: {len(ids)} 条 "
          f"→ 其中 {n_neu} 行 action_type 强制中性(0), 仅保留靶标存在性(Path1桥梁)")
    return df


def _apply_edge_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """边置信度加权 — 解决"覆盖率提高反而降 AUROC"的核心机制。

    签名权重 = action_type * confidence。原先 confidence 是成分级(大量0.5默认),
    且 DrugAI 的每对方向置信度被丢弃, 导致 18.6 万条 HDCTI 靶标(含大量 OOD 补全
    天然产物)被等权、且大概率错的方向加入, 在 KEGG 通路拓扑里变成噪声。

    本函数将 confidence 重算为质量感知的边权重:
        confidence = 基础置信度
                   × DrugAI方向置信度(drugai_conf, 不可靠方向自动降权)
                   × OOD补全惩罚(补全SMILES为OOD天然产物, 方向不可信)
                   × 网络方向可靠性折扣(KEGG下生化方向噪声更大)

    使"覆盖率提高"转化为"正确信号增加"而非"噪声增加"。无论走缓存还是重建均生效。
    """
    if not DIRECTION_CONFIDENCE_WEIGHTING:
        return df

    df = df.copy()

    # 1) 基础置信度: 始终使用 RAW 基置信度(base_confidence), 而非已加权的 confidence,
    #    避免对「上一次加权结果」重复乘惩罚(每次运行权重被叠加压低 → 越跑越差)。
    if "base_confidence" in df.columns:
        base = df["base_confidence"].fillna(0.5).astype(float)
    else:
        base = df["confidence"].fillna(0.5).astype(float) if "confidence" in df.columns else pd.Series(0.5, index=df.index)
        df["base_confidence"] = base  # 首次: 固化原始基置信度, 供后续幂等重算

    # 2) DrugAI 每对方向置信度 (predict_action_type_conf 捕获的 |prob-0.5|*2)
    #    缺失列(旧缓存未捕获)→ 给 DrugAI 行一个合理的中等默认置信度 0.6,
    #    避免重罚; 非DrugAI行不影响(下方仅对 drugai_active 应用)
    if "drugai_conf" in df.columns:
        drugai_conf = df["drugai_conf"].fillna(0.0).astype(float)
    else:
        drugai_conf = pd.Series(0.6, index=df.index)
    drugai_active = (df.get("action_source") == "DrugAI")
    active_mask = drugai_active.to_numpy()
    dir_component = pd.Series(1.0, index=df.index)
    dir_component.loc[active_mask] = drugai_conf.loc[active_mask].clip(lower=0.05)

    # 3) OOD补全(PubChem)靶标惩罚: SMILES 为 OOD 天然产物, DrugAI 方向不可信
    completed = _get_pubchem_completed_ids()
    id_col = df["ingredient_id"].astype(str).str.replace(".0", "", regex=False)
    ood_mask = id_col.isin(completed)
    ood_pen = pd.Series(1.0, index=df.index)
    ood_pen.loc[ood_mask.to_numpy()] = OOD_COMPLETION_PENALTY

    # 4) 网络方向可靠性折扣: KEGG 通路共成员关系使生化方向噪声更大
    dir_rel = KEGG_DIRECTION_RELIABILITY if _cfg.NETWORK_TYPE == "kegg" else PPI_DIRECTION_RELIABILITY
    rel_factor = pd.Series(1.0, index=df.index)
    rel_factor.loc[active_mask] = dir_rel

    # 合成最终边置信度
    df["confidence"] = (base * dir_component * ood_pen * rel_factor).clip(upper=1.0)

    n_ood = int(ood_mask.sum())
    n_drugai = int(drugai_active.sum())
    avg_dc = drugai_conf.loc[active_mask].mean() if active_mask.any() else 0.0
    print(f"[DataLoader] 边置信度加权完成: "
          f"OOD补全惩罚 {n_ood} 行 ×{OOD_COMPLETION_PENALTY}, "
          f"DrugAI方向 {n_drugai} 行 ×drugai_conf(平均 {avg_dc:.3f}), "
          f"网络方向可靠性折扣 ×{dir_rel}({_cfg.NETWORK_TYPE})")
    return df


def load_ingredient_targets(
    filepath: Optional[str] = None,
    use_hdcti: bool = HDCTI_USE,
    use_drugai: bool = DRUGAI_USE,
    herb_ingredients_df: Optional[pd.DataFrame] = None,
    disease_profiles_df: Optional[pd.DataFrame] = None,
    ppi_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    加载成分-靶标关系表（增强版）
    
    支持两种模式:
    1. 原始模式 (use_hdcti=False, use_drugai=False): 
       直接加载 ingredient_targets.csv (PreAM + WES)
    2. 增强模式:
       - use_hdcti=True: 使用HDCTI预测新靶标 + TCMSP数据补充
       - use_drugai=True: 使用DrugAI重新预测action_type
    
    Parameters
    ----------
    herb_ingredients_df : 草药-成分表（HDCTI GCN预测需要）
    disease_profiles_df : 疾病表达谱（HDCTI GCN预测需要）
    ppi_df : PPI网络（HDCTI GCN预测需要）
    
    Returns
    -------
    DataFrame columns: ingredient_id, target_gene, action_type (+1/-1), source
    """
    if filepath is None:
        # 缓存优先：如果增强版数据已存在且启用了增强模式，直接加载
        if (use_hdcti or use_drugai):
            enhanced_path = os.path.join(PROCESSED_DIR, f"ingredient_targets_enhanced_{_cfg.NETWORK_TYPE}.csv")
            if os.path.exists(enhanced_path):
                print(f"[DataLoader] 加载已缓存的增强版靶标数据: {enhanced_path}")
                df = pd.read_csv(enhanced_path, low_memory=False)
                if "source" not in df.columns:
                    df["source"] = "enhanced"
                result = _merge_confidence(df, PROCESSED_DIR)
                df = result if result is not None else df
                # 补全(PubChem)来源成分方向强制中性(无论走缓存还是重建均生效)
                df = _neutralize_completed_directions(df)
                # 边置信度加权(质量感知): OOD补全惩罚 + DrugAI方向置信度 + KEGG可靠性
                df = _apply_edge_confidence(df)
                df.to_csv(enhanced_path, index=False)
                print(f"[DataLoader] 已写入加权后的增强版靶标数据: {enhanced_path}")
                return df
        
        filepath = os.path.join(PROCESSED_DIR, "ingredient_targets_processed.csv")
    
    # 加载原始数据
    df = None
    if not os.path.exists(filepath):
        raw_path = os.path.join(RAW_DIR, "ingredient_targets.csv")
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
        else:
            raise FileNotFoundError(
                f"成分-靶标数据未找到: {raw_path}\n"
                "请从ChEMBL v23 + BindingDB 获取，或使用WES算法预测\n"
                "格式: ingredient_id, target_gene, action_type (1=激活, -1=抑制)"
            )
    else:
        df = pd.read_csv(filepath)
    
    # 标记数据来源
    if "source" not in df.columns:
        df["source"] = "original"
    
    # --- 步骤1: HDCTI替代WES ---
    if use_hdcti:
        df = _apply_hdcti_enhancement(
            df, herb_ingredients_df, disease_profiles_df, ppi_df
        )
    
    # --- 步骤2: DrugAI替代PreAM ---
    if use_drugai:
        df = _apply_drugai_enhancement(df)
        df = _neutralize_completed_directions(df)
    
    # ---- 自动合并靶标置信度权重 (SVM/RF) ----
    result = _merge_confidence(df, PROCESSED_DIR)
    df = result if result is not None else df

    # ---- 边置信度加权(质量感知): OOD补全惩罚 + DrugAI方向置信度 + KEGG可靠性 ----
    df = _apply_edge_confidence(df)

    # 保存增强后的数据 (已含最终权重, 供下次缓存直接加载)
    if use_hdcti or use_drugai:
        enhanced_path = os.path.join(PROCESSED_DIR, f"ingredient_targets_enhanced_{_cfg.NETWORK_TYPE}.csv")
        df.to_csv(enhanced_path, index=False)
        print(f"[DataLoader] 增强版靶标数据已保存(含边置信度加权, {_cfg.NETWORK_TYPE}): {enhanced_path}")

    return df


def _merge_confidence(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """将独立置信度文件合并为「原始基置信度」列 base_confidence（RAW，不参与加权叠加）。

    关键修复(解决"越跑越差"): 原先直接覆盖 working 列 confidence, 且命中缓存时
    直接 return, 导致 _apply_edge_confidence 在「已加权」的 confidence 上重复乘惩罚,
    每次运行权重被叠加压低。现改为仅写入 base_confidence(原始), 保留 confidence 作为
    工作列, 使加权可从原始值幂等重算, 多次运行结果一致。
    """
    confidence_path = os.path.join(data_dir, "ingredient_confidence.csv")
    df = df.copy()
    # 如果已从增强版缓存加载(含 base_confidence 列)，直接返回，避免 merge 时列名冲突
    if "base_confidence" in df.columns:
        return df
    if not os.path.exists(confidence_path):
        if "base_confidence" not in df.columns:
            df["base_confidence"] = df["confidence"].fillna(0.5) if "confidence" in df.columns else 0.5
        return df
    if "ingredient_id" not in df.columns or "target_gene" not in df.columns:
        return df
    df["ingredient_id"] = df["ingredient_id"].astype(str).str.strip()
    df["target_gene"] = df["target_gene"].astype(str).str.strip()
    df_conf = pd.read_csv(confidence_path)
    df_conf["ingredient_id"] = df_conf["ingredient_id"].astype(str).str.strip()
    df_conf["target_gene"] = df_conf["target_gene"].astype(str).str.strip()
    # 重命名为 base_confidence, 避免覆盖 working 列 confidence
    if "confidence" in df_conf.columns:
        df_conf = df_conf.rename(columns={"confidence": "base_confidence"})
    if "base_confidence" not in df_conf.columns:
        # 防御: rename 失败(如列名大小写/空格差异) → 设默认 0.5
        df_conf["base_confidence"] = 0.5
    n_before = len(df)
    merged = df.merge(
        df_conf[["ingredient_id", "target_gene", "base_confidence"]],
        on=["ingredient_id", "target_gene"], how="left",
    )
    if "base_confidence" in merged.columns:
        merged["base_confidence"] = merged["base_confidence"].fillna(0.5)
    else:
        # 防御: merge 未带入 base_confidence → fallback
        merged["base_confidence"] = 0.5
    # working 列 confidence 不在此处覆盖(由 _apply_edge_confidence 基于 base_confidence 计算)
    if "confidence" not in merged.columns:
        merged["confidence"] = merged["base_confidence"]
    n_with_conf = int((merged["base_confidence"] > 0.5).sum())
    print(f"[DataLoader] 已合并原始基置信度(base_confidence): {len(df_conf)} 条, "
          f"高置信(>0.5) {n_with_conf}/{n_before} ({100*n_with_conf/n_before:.0f}%)")
    return merged


def _apply_hdcti_enhancement(
    df: pd.DataFrame,
    herb_ingredients_df: Optional[pd.DataFrame] = None,
    disease_profiles_df: Optional[pd.DataFrame] = None,
    ppi_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    使用HDCTI增强靶标数据 (替代WES)
    
    步骤:
    1. 从TCMSP爬取数据补充靶标
    2. 使用HDCTI图传播预测新靶标 (GCN异构图)
    3. 合并到现有数据中
    """
    from hdcti_adapter import HDCTIAdapter
    
    df = df.copy()
    original_count = len(df)
    # 标记原始数据来源
    if "source" not in df.columns:
        df["source"] = "original"
    
    # 步骤0: 从疾病表达谱构建基因方向字典
    # 逻辑：疾病中上调的基因 → 需要抑制(-1)；疾病中下调的基因 → 需要激活(+1)
    disease_dir = None
    if disease_profiles_df is not None and not disease_profiles_df.empty:
        disease_dir = {}
        for _, row in disease_profiles_df.iterrows():
            gene = row["gene_symbol"]
            log_fc = row["logFC"]
            if log_fc > 0:
                disease_dir[gene] = -1   # 疾病中上调 → 草药应抑制
            else:
                disease_dir[gene] = +1   # 疾病中下调 → 草药应激活
        # 如果有多个疾病，取第一个（通常只有一个）
        print(f"[DataLoader] 疾病方向字典: {len(disease_dir)} 个基因 "
              f"(上调={(pd.Series(disease_dir.values())<0).sum()}, "
              f"下调={(pd.Series(disease_dir.values())>0).sum()})")
    
    # 步骤1: 从TCMSP数据补充靶标
    if HDCTI_USE_TCMSP_TARGETS:
        tcmsp_targets = HDCTIAdapter.load_tcmsp_targets(
            min_svm_score=HDCTI_TCMSP_SVM_THRESHOLD,
            min_rf_score=HDCTI_TCMSP_RF_THRESHOLD
        )
        if not tcmsp_targets.empty:
            adapter = HDCTIAdapter()
            df = adapter.merge_targets(df, tcmsp_targets, disease_dir=disease_dir)
    
    # 步骤2: 使用HDCTI GCN图传播预测新靶标
    if HDCTI_USE_GCN_PREDICTION:
        # 只传入目标疾病的基因到GCN图中，避免多疾病基因污染图结构
        # (7月12日版本只有单疾病数据，GCN图更精准)
        gcn_disease_df = disease_profiles_df
        if disease_profiles_df is not None and 'disease_id' in disease_profiles_df.columns:
            diseases = disease_profiles_df['disease_id'].unique()
            if len(diseases) > 1:
                gcn_disease_df = disease_profiles_df[disease_profiles_df['disease_id'] == diseases[0]]
        hdcti_predictions = _run_hdcti_gcn_prediction(
            df, herb_ingredients_df, gcn_disease_df, ppi_df
        )
        if hdcti_predictions is not None and not hdcti_predictions.empty:
            adapter = HDCTIAdapter()
            df = adapter.merge_targets(df, hdcti_predictions, disease_dir=disease_dir)
    
    new_count = len(df)
    print(f"[DataLoader] HDCTI增强: {original_count} → {new_count} 条靶标")
    return df


def _run_hdcti_gcn_prediction(
    ingredient_targets_df: pd.DataFrame,
    herb_ingredients_df: Optional[pd.DataFrame] = None,
    disease_profiles_df: Optional[pd.DataFrame] = None,
    ppi_df: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    """
    运行HDCTI GCN图传播预测新靶标（带缓存）
    
    通过H-C-P-D异构图上的Jaccard相似性传播，
    为每个成分预测top_k个新靶标蛋白。
    
    Returns
    -------
    DataFrame columns: ingredient_id, target_gene, confidence, source
    或 None（如果跳过/失败）
    """
    from hdcti_adapter import HDCTIAdapter
    
    # 缓存路径
    cache_dir = os.path.join(PROCESSED_DIR, "hdcti_cache")
    os.makedirs(cache_dir, exist_ok=True)
    predictions_cache = os.path.join(cache_dir, "gcn_predictions.csv")
    
    # 尝试加载缓存
    if os.path.exists(predictions_cache):
        print("[HDCTI] 加载缓存的GCN预测结果...")
        cached = pd.read_csv(predictions_cache)
        print(f"[HDCTI] 缓存命中: {len(cached)} 条预测靶标")
        return cached
    
    # 加载所需数据（如果未传入则自动加载）
    if herb_ingredients_df is None:
        print("[HDCTI] 自动加载草药-成分数据...")
        herb_ingredients_df = load_herb_ingredients()
    if disease_profiles_df is None:
        print("[HDCTI] 自动加载疾病表达谱...")
        disease_profiles_df = load_disease_profiles()
    if ppi_df is None:
        print("[HDCTI] 自动加载PPI网络...")
        ppi_df = load_ppi_network()
    
    # 获取PPI基因列表（用于过滤蛋白质节点）
    ppi_genes = sorted(
        set(ppi_df['protein_A'].tolist()) | set(ppi_df['protein_B'].tolist())
    )
    print(f"[HDCTI] PPI基因列表: {len(ppi_genes)} 个蛋白")
    
    # 构建适配器并运行图传播预测
    print("[HDCTI] 构建GCN异构图...")
    adapter = HDCTIAdapter(
        confidence_threshold=HDCTI_CONFIDENCE_THRESHOLD,
        top_k=HDCTI_TOP_K,
        similarity_method=HDCTI_SIMILARITY_METHOD
    )
    
    adapter.build_graph(
        herb_ingredients_df=herb_ingredients_df,
        ingredient_targets_df=ingredient_targets_df,
        disease_profiles_df=disease_profiles_df,
        ppi_genes=ppi_genes
    )
    
    # 批量预测所有成分的新靶标
    hdcti_predictions = adapter.predict_all_new_targets(
        ingredient_targets_df=ingredient_targets_df,
        herb_ingredients_df=herb_ingredients_df
    )
    
    print(f"[HDCTI] GCN预测完成: {len(hdcti_predictions)} 条新靶标")
    
    # ---- z-score 全局质量过滤 (理论: 所有成分在同一张图上产生分数,绝对可比) ----
    if HDCTI_ZSCORE_THRESHOLD is not None and not hdcti_predictions.empty:
        conf_col = hdcti_predictions["confidence"].astype(float)
        mean_c, std_c = conf_col.mean(), conf_col.std()
        if std_c > 0:
            z_scores = (conf_col - mean_c) / std_c
            mask = z_scores >= HDCTI_ZSCORE_THRESHOLD
            n_before = len(hdcti_predictions)
            hdcti_predictions = hdcti_predictions[mask].copy()
            n_after = len(hdcti_predictions)
            print(f"[HDCTI] z-score过滤(z>={HDCTI_ZSCORE_THRESHOLD}): "
                  f"{n_before} → {n_after} ({n_after*100/max(n_before,1):.1f}%), "
                  f"全局置信度均值={mean_c:.2f} σ={std_c:.2f}")
        else:
            print(f"[HDCTI] z-score过滤跳过(σ=0, 所有分数相同)")
    
    # ---- KEGG通路过滤 ----
    from config import KEGG_FILTER_ENABLED
    if KEGG_FILTER_ENABLED and not hdcti_predictions.empty:
        from kegg_pathway_filter import KEGGPathwayFilter
        # 使用当前疾病(AD)的表达谱基因作为种子
        ad_genes = disease_profiles_df[
            disease_profiles_df['disease_id'].astype(str).str.upper() == 'AD'
        ]['gene_symbol'].unique().tolist() if 'disease_id' in disease_profiles_df.columns \
                   else disease_profiles_df['gene_symbol'].unique().tolist()
        kegg_filter = KEGGPathwayFilter()
        kegg_filter.build_filter(ad_genes, max_hops=2)
        
        n_before = len(hdcti_predictions)
        hdcti_predictions = hdcti_predictions[
            hdcti_predictions['target_gene'].apply(kegg_filter.is_reachable)
        ]
        print(f"[KEGG Filter] 过滤: {n_before} → {len(hdcti_predictions)} 条靶标 "
              f"(-{n_before - len(hdcti_predictions)})")
    
    # 保存缓存
    if not hdcti_predictions.empty:
        hdcti_predictions.to_csv(predictions_cache, index=False)
        print(f"[HDCTI] GCN预测结果已缓存: {predictions_cache}")
    
    return hdcti_predictions


def _apply_drugai_enhancement(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用DrugAI增强action_type预测 (替代PreAM) - P0增强版
    
    处理原始PreAM数据和HDCTI预测数据行（source="original"或"HDCTI_predicted"），跳过TCMSP_SVM_RF行
    
    步骤:
    1. 加载DrugAI适配器
    2. P0预加载: 批量获取SMILES和蛋白质序列
    3. 仅对原始数据行批量重新预测action_type
    4. 优先使用ChEMBL验证数据
    """
    try:
        from drugai_adapter import DrugAIAdapter
    except ImportError as e:
        print(f"[DataLoader] DrugAI适配器不可用 (缺少依赖: {e})，跳过DrugAI增强")
        return df
    
    df = df.copy()
    
    # 分离需要DrugAI处理的数据和跳过的数据
    # DrugAI处理: original(PreAM) + HDCTI_predicted(GCN图传播) 
    # 跳过: TCMSP_SVM_RF(低质量ML预测)
    if "source" in df.columns:
        original_mask = df["source"].isin(["original", "HDCTI_predicted"])
        df_original = df[original_mask].copy()
        df_tcmsp = df[~original_mask].copy()
        n_skip = (~original_mask).sum()
        print(f"[DataLoader] DrugAI处理: {len(df_original)} 行 (original+HDCTI), "
              f"跳过 {n_skip} 行 (TCMSP_SVM_RF等)")
    else:
        df_original = df
        df_tcmsp = pd.DataFrame()
    
    # 确保有action_source列
    if "action_source" not in df_original.columns:
        df_original["action_source"] = "PreAM"
    
    # 步骤1: 使用DrugAI重新预测（仅对有SMILES和蛋白质序列的少数行进行模型推理）
    try:
        adapter = DrugAIAdapter(
            model_path=DRUGAI_MODEL_PATH,
            device=DRUGAI_DEVICE,
            confidence_threshold=DRUGAI_CONFIDENCE_THRESHOLD,
            use_gpu=DRUGAI_USE_GPU
        )
        adapter.load_model()
    except Exception as e:
        print(f"[DataLoader] DrugAI初始化失败: {e}，跳过DrugAI增强")
        return df
    
    # 尝试加载成分信息(用于SMILES查询)
    ingredients_df = _load_tcmsp_ingredients()

    # ===== P0预加载: 批量获取SMILES和蛋白质序列 =====
    # 在开始DrugAI推理前，尽可能多地填充SMILES和序列缓存
    print(f"\n[DataLoader] ===== P0数据预加载 =====")
    
    # 显示预加载前状态
    smiles_stats_before = adapter.smiles_lookup.get_stats()
    protein_stats_before = adapter.seq_lookup.get_stats()
    print(f"  SMILES 预加载前: {smiles_stats_before['cached']} 条 "
          f"(覆盖率 {smiles_stats_before['coverage_pct']}%)")
    print(f"  蛋白质 预加载前: {protein_stats_before['cached']} 条")
    
    # 从PPI网络获取所有基因列表
    ppi_genes = None
    try:
        ppi_df = load_ppi_network()
        ppi_genes = sorted(set(ppi_df['protein_A'].tolist() + ppi_df['protein_B'].tolist()))
    except:
        pass
    
    # 执行预加载
    preload_stats = adapter.preload_all_data(
        ingredient_targets_df=df_original,
        ingredients_df=ingredients_df,
        ppi_genes=ppi_genes
    )
    
    # 显示预加载后状态
    smiles_stats_after = adapter.smiles_lookup.get_stats()
    protein_stats_after = adapter.seq_lookup.get_stats()
    print(f"  SMILES 预加载后: {smiles_stats_after['cached']} 条 "
          f"(覆盖率 {smiles_stats_after['coverage_pct']}%)")
    print(f"  蛋白质 预加载后: {protein_stats_after['cached']} 条")
    print(f"[DataLoader] ===== P0预加载完成 =====")
    
    # 使用DrugAI模型推理替换原始action_type
    # DrugAI模型推理较慢（~15对/秒），大样本量时请耐心等待
    if DRUGAI_USE:
        print(f"\n[DataLoader] DrugAI启用模型推理，开始更新action_type...")
        df_original = adapter.update_action_types(
            df_original,
            ingredients_df=ingredients_df,
            batch_size=100,
            resume=True  # 断点续传：跳过已标记DrugAI的行
        )
        print(f"[DataLoader] DrugAI推理完成")

        # ---- 疾病上下文方向修正 ----
        # DrugAI预测的激活/抑制方向可能不适用于AD病理
        # 用AD疾病表达谱验证：激活AD中已高表达的基因 → 错误 → 修正为0
        if DRUGAI_DISEASE_VALIDATE:
            from config import PROCESSED_DIR
            disease_path = os.path.join(PROCESSED_DIR, 'disease_profiles_AD.csv')
            if os.path.exists(disease_path):
                disease_df = pd.read_csv(disease_path)
                disease_dir = {}
                for _, row in disease_df.iterrows():
                    gene = row['gene_symbol']
                    if gene not in disease_dir:
                        disease_dir[gene] = row['logFC']  # + = up in AD, - = down in AD
                
                n_corrected = 0
                for idx in df_original.index:
                    gene = df_original.at[idx, 'target_gene']
                    action = df_original.at[idx, 'action_type']
                    if gene in disease_dir and action != 0:
                        logFC = disease_dir[gene]
                        # 治疗逻辑: AD高表达→需抑制, AD低表达→需激活
                        # DrugAI预测的是生化方向,不能篡改
                        # 当生化方向与治疗需求矛盾时, 该靶标对治疗无益 → 排除
                        if (action > 0 and logFC > 0) or (action < 0 and logFC < 0):
                            df_original.at[idx, 'action_type'] = 0
                            n_corrected += 1
                
                print(f"[DiseaseValidate] 疾病方向修正: {n_corrected} 条预测与AD病理矛盾 → action_type=0")
            else:
                print(f"[DiseaseValidate] 警告: 未找到 {disease_path}")
    else:
        print("[DataLoader] DrugAI已禁用(DRUGAI_USE=False)，使用原始action_type")
    
    # 合并回完整数据
    if not df_tcmsp.empty:
        df = pd.concat([df_original, df_tcmsp], ignore_index=True)
    else:
        df = df_original
    
    # 步骤2: 实验验证机制覆盖 (真值优先于模型预测)
    # 汇总 DrugBank(11229条) + ChEMBL 的"激活/抑制"实验结论, 命中即覆盖 DrugAI 预测。
    # 实测覆盖率很低(骨架匹配 0.078%), 因为草药成分多为天然产物、与已批准药物重叠极小,
    # 所以它保证的是"已知处不瞎猜", 而非整体准确率的主要来源。
    if EXP_MECH_ENABLE:
        try:
            from experimental_mechanisms import apply_experimental_override
            from config import PROCESSED_DIR
            df = apply_experimental_override(
                df, processed_dir=PROCESSED_DIR, raw_dir=RAW_DIR,
                match_mode=EXP_MECH_MATCH_MODE,
            )
        except Exception as e:
            print(f"[DataLoader] 实验机制覆盖失败(不影响主流程): {e}")

    return df


def _load_tcmsp_ingredients() -> Optional[pd.DataFrame]:
    """加载TCMSP成分数据 (用于SMILES查询)"""
    tcmsp_path = os.path.join(TCMSP_DATA_DIR, "all_ingredients.csv")
    if os.path.exists(tcmsp_path):
        return pd.read_csv(tcmsp_path)
    return None


def load_tcmsp_data(data_type: str = "all") -> pd.DataFrame:
    """
    加载TCMSP爬取数据
    
    Parameters
    ----------
    data_type : 'ingredients', 'targets', 'diseases', 或 'all'
    
    Returns
    -------
    pd.DataFrame
    """
    file_map = {
        "ingredients": "all_ingredients.csv",
        "targets": "all_targets.csv",
        "diseases": "all_diseases.csv"
    }
    
    if data_type == "all":
        results = {}
        for key, fname in file_map.items():
            path = os.path.join(TCMSP_DATA_DIR, fname)
            if os.path.exists(path):
                results[key] = pd.read_csv(path)
                print(f"[DataLoader] TCMSP {key}: {len(results[key])} 条")
            else:
                results[key] = pd.DataFrame()
        return results
    else:
        path = os.path.join(TCMSP_DATA_DIR, file_map.get(data_type, ""))
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"[DataLoader] TCMSP {data_type}: {len(df)} 条")
            return df
        return pd.DataFrame()


def load_ppi_network(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    加载PPI蛋白互作网络
    
    Returns
    -------
    DataFrame columns: protein_A, protein_B
    """
    if filepath is None:
        filepath = os.path.join(PROCESSED_DIR, "ppi_network_processed.csv")
    
    if not os.path.exists(filepath):
        raw_path = os.path.join(RAW_DIR, "ppi_network.csv")
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            df.to_csv(filepath, index=False)
            return df
        else:
            raise FileNotFoundError(
                f"PPI网络数据未找到: {raw_path}\n"
                "论文来源: Menche et al., Science 2015\n"
                "包含 13,460 个蛋白, 141,296 条相互作用\n"
                "格式: protein_A, protein_B"
            )
    return pd.read_csv(filepath)


def build_ppi_adjacency(ppi_df: pd.DataFrame = None,
                        use_multi_source: bool = None) -> Tuple[Any, Dict[str, int]]:
    """
    构建PPI邻接矩阵（稀疏格式）
    
    支持两种模式:
      - 单源模式 (默认): 使用传入的 ppi_df 构建邻接矩阵
      - 多源融合模式 (PPI_MULTI_SOURCE=True): 
        自动加载 data/raw/ppi_sources/ 中的多个PPI源文件并融合

    Parameters
    ----------
    ppi_df : DataFrame with columns [protein_A, protein_B], 可选
        单源PPI数据。多源模式下可传None。
    use_multi_source : bool, 可选
        是否使用多源融合。默认跟随 config.PPI_MULTI_SOURCE。
    
    Returns
    -------
    adj_matrix : scipy.sparse.csr_matrix, shape (n_proteins, n_proteins), 对称邻接矩阵
    gene2idx   : dict, 基因名到索引的映射
    """
    from scipy.sparse import csr_matrix
    
    # ---- 多源PPI融合模式 ----
    if use_multi_source is None:
        use_multi_source = PPI_MULTI_SOURCE
    
    if use_multi_source:
        return _build_multi_source_adjacency()
    
    # ---- 默认单源模式 ----
    if ppi_df is None:
        ppi_df = load_ppi_network()
    
    adj, gene2idx = _build_single_source_adjacency(ppi_df)
    print(f"[PPI] 邻接矩阵构建完成: {len(gene2idx)} x {len(gene2idx)}, "
          f"{ppi_df.shape[0]} 条边")
    return adj, gene2idx


def load_herb_metadata(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    加载草药元数据（可选）
    """
    if filepath is None:
        filepath = os.path.join(PROCESSED_DIR, "herb_metadata.csv")
    
    default_path = os.path.join(RAW_DIR, "herb_metadata.csv")
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    elif os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        # 返回空df
        return pd.DataFrame(columns=["herb_id", "herb_name"])


def get_all_protein_list(ppi_df: pd.DataFrame) -> List[str]:
    """获取PPI网络中所有蛋白列表（去重排序）"""
    return sorted(set(ppi_df["protein_A"].tolist() + ppi_df["protein_B"].tolist()))


def _build_multi_source_adjacency() -> Tuple[Any, Dict[str, int]]:
    """
    多源PPI融合邻接矩阵构建（内部辅助函数）
    
    自动检测 data/raw/ppi_sources/ 中的PPI源文件:
      - 存在源文件 → 使用 MultiSourcePPIFuser 融合多源
      - 无源文件 → 回退到默认 ppi_network.csv
    
    Returns
    -------
    adj_matrix, gene2idx
    """
    # 检测源文件
    source_files = []
    if os.path.isdir(PPI_SOURCE_DIR):
        source_files = [f for f in os.listdir(PPI_SOURCE_DIR)
                        if f.endswith('.csv') or f.endswith('.txt') or f.endswith('.tsv')]
    
    if len(source_files) >= 1:
        print(f"[PPI] 多源PPI融合模式: 发现 {len(source_files)} 个源文件")
        for sf in source_files:
            print(f"       - {sf}")
        fuser = MultiSourcePPIFuser(source_dir=PPI_SOURCE_DIR, verbose=True)
        adj, gene2idx = fuser.build_fused_adjacency(
            strategy=PPI_FUSION_STRATEGY,
            min_sources=PPI_MIN_SOURCES
        )
        print(f"[PPI] 融合完成: {len(gene2idx)} 蛋白, 矩阵 {adj.shape}")
        return adj, gene2idx
    else:
        print(f"[PPI] 多源PPI模式已启用，但 {PPI_SOURCE_DIR} 中未找到源文件")
        print(f"      回退到默认 ppi_network.csv")
        ppi_df = load_ppi_network()
        return _build_single_source_adjacency(ppi_df)


def _build_single_source_adjacency(ppi_df: pd.DataFrame) -> Tuple[Any, Dict[str, int]]:
    """从单个PPI DataFrame构建邻接矩阵"""
    from scipy.sparse import csr_matrix
    proteins = sorted(set(ppi_df["protein_A"].tolist() + ppi_df["protein_B"].tolist()))
    gene2idx = {p: i for i, p in enumerate(proteins)}
    n = len(gene2idx)
    rows, cols = [], []
    for _, row in ppi_df.iterrows():
        a, b = row["protein_A"], row["protein_B"]
        if a in gene2idx and b in gene2idx:
            ia, ib = gene2idx[a], gene2idx[b]
            rows.extend([ia, ib])
            cols.extend([ib, ia])
    adj = csr_matrix(([1.0] * len(rows), (rows, cols)), shape=(n, n))
    return adj, gene2idx


# ==================== 多源PPI融合 ====================

class MultiSourcePPIFuser:
    """
    多源PPI网络融合器
    
    从多个PPI数据库加载数据，融合为加权邻接矩阵。
    融合策略:
      - evidence_count: 边权重 = 支持该边的独立数据源数量（推荐）
      - confidence:     边权重 = 各源置信度的加权平均
      - union:          边权重 = 1（仅保留是否存在，丢弃权重信息）
    
    使用方法:
        1. 将各PPI源文件放入 data/raw/ppi_sources/ 目录
        2. 各文件格式: protein_A, protein_B [, confidence]
        3. 调用 fuser = MultiSourcePPIFuser()
        4. adj, gene2idx = fuser.build_fused_adjacency(strategy='evidence_count')
    """
    
    # 预定义的PPI源文件命名规则
    SOURCE_PREFIX = "ppi_source_"
    
    def __init__(self, source_dir: str = None, verbose: bool = True):
        """
        Parameters
        ----------
        source_dir : str, PPI源文件目录，默认 data/raw/ppi_sources/
        verbose    : bool, 是否打印详细信息
        """
        if source_dir is None:
            from config import RAW_DIR
            source_dir = os.path.join(RAW_DIR, "ppi_sources")
        self.source_dir = source_dir
        self.verbose = verbose
        self._loaded_sources: Dict[str, pd.DataFrame] = {}
        self._fused_df: Optional[pd.DataFrame] = None
    
    def discover_sources(self) -> List[str]:
        """发现目录中的所有PPI源文件"""
        if not os.path.isdir(self.source_dir):
            if self.verbose:
                print(f"[多源PPI] 源目录不存在: {self.source_dir}")
            return []
        
        files = sorted(os.listdir(self.source_dir))
        ppi_files = [f for f in files if f.endswith('.csv') and 
                     not f.startswith('_')]
        return [os.path.join(self.source_dir, f) for f in ppi_files]
    
    def load_source(self, filepath: str, source_name: str = None,
                    sep: str = ',') -> pd.DataFrame:
        """
        加载单个PPI源文件
        
        支持格式:
          1. 两列: protein_A, protein_B（每条边权重=1）
          2. 三列: protein_A, protein_B, confidence（权重=confidence值）
          3. STRING格式: 自动检测 combined_score 列
        
        Parameters
        ----------
        filepath    : str, 文件路径
        source_name : str, 源名称（用于日志），默认用文件名
        sep         : str, 分隔符
        
        Returns
        -------
        DataFrame with columns: protein_A, protein_B, weight
        """
        if source_name is None:
            source_name = os.path.splitext(os.path.basename(filepath))[0]
        
        df = pd.read_csv(filepath, sep=sep)
        
        # 标准化列名
        col_map = {}
        for c in df.columns:
            c_lower = c.lower().replace(' ', '_')
            if c_lower in ('protein_a', 'protein1', 'gene1', 'gene_a', 'node1', 'symbol1'):
                col_map[c] = 'protein_A'
            elif c_lower in ('protein_b', 'protein2', 'gene2', 'gene_b', 'node2', 'symbol2'):
                col_map[c] = 'protein_B'
            elif c_lower in ('confidence', 'score', 'combined_score', 'weight', 'coverage'):
                col_map[c] = 'confidence'
        df = df.rename(columns=col_map)
        
        # 验证必要列
        for col in ['protein_A', 'protein_B']:
            if col not in df.columns:
                raise ValueError(f"PPI源 {source_name} 缺少列 '{col}'，可用列: {list(df.columns)}")
        
        # 确定权重
        if 'confidence' in df.columns:
            # 有置信度列：归一化到 [0, 1]
            weights = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.5)
            w_min, w_max = weights.min(), weights.max()
            if w_max > w_min:
                weights = (weights - w_min) / (w_max - w_min)
            df['weight'] = weights.clip(0, 1).astype(np.float32)
        else:
            # 无置信度列：每条边权重=1
            df['weight'] = np.float32(1.0)
        
        result = df[['protein_A', 'protein_B', 'weight']].dropna().copy()
        
        # 去重（同一条边的多条记录取最大权重）
        result = result.groupby(['protein_A', 'protein_B'], as_index=False)['weight'].max()
        
        if self.verbose:
            n_proteins = len(set(result['protein_A']) | set(result['protein_B']))
            print(f"  [{source_name}] {len(result)} 条边, {n_proteins} 个蛋白")
        
        self._loaded_sources[source_name] = result
        return result
    
    def load_all_sources(self, filepaths: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载所有PPI源
        
        Parameters
        ----------
        filepaths : List[str], 源文件路径列表。为None时自动发现
        
        Returns
        -------
        Dict[source_name, DataFrame]
        """
        if filepaths is None:
            filepaths = self.discover_sources()
        
        if not filepaths:
            if self.verbose:
                print("[多源PPI] 未找到多源PPI文件，将使用默认单源PPI")
            return {}
        
        print(f"[多源PPI] 发现 {len(filepaths)} 个PPI源:")
        for fp in filepaths:
            name = os.path.splitext(os.path.basename(fp))[0]
            self.load_source(fp, source_name=name)
        
        return self._loaded_sources
    
    def fuse(self, strategy: str = 'evidence_count',
             min_sources: int = 1) -> pd.DataFrame:
        """
        融合多个PPI源
        
        Parameters
        ----------
        strategy : str
            - 'evidence_count': 边权重=支持的源数量（默认，推荐）
            - 'confidence':     边权重=各源置信度的均值
            - 'union':          边权重=1（简单并集，丢弃权重）
        min_sources : int
            最小支持源数（低于此数的边被过滤）
        
        Returns
        -------
        DataFrame: protein_A, protein_B, weight
        """
        if not self._loaded_sources:
            raise ValueError("请先调用 load_all_sources() 加载PPI源")
        
        n_sources = len(self._loaded_sources)
        
        if strategy == 'union':
            # 简单并集：所有边的权重=1
            parts = []
            for name, df in self._loaded_sources.items():
                df_simple = df[['protein_A', 'protein_B']].drop_duplicates()
                parts.append(df_simple)
            fused = pd.concat(parts).drop_duplicates().reset_index(drop=True)
            fused['weight'] = np.float32(1.0)
            if self.verbose:
                print(f"[多源PPI] 融合策略=union ({n_sources}源): {len(fused)} 条边")
        
        elif strategy == 'confidence':
            # 置信度均值：叠加所有源，同一边取权重均值
            parts = []
            for name, df in self._loaded_sources.items():
                parts.append(df[['protein_A', 'protein_B', 'weight']])
            fused = pd.concat(parts, ignore_index=True)
            fused = fused.groupby(['protein_A', 'protein_B'], as_index=False)['weight'].mean()
            if self.verbose:
                print(f"[多源PPI] 融合策略=confidence ({n_sources}源): {len(fused)} 条边")
        
        elif strategy == 'evidence_count':
            # 证据计数：边权重 = 支持该边的源数量 / 总源数
            parts = []
            for name, df in self._loaded_sources.items():
                _df = df[['protein_A', 'protein_B']].copy()
                _df['source'] = name
                parts.append(_df)
            all_edges = pd.concat(parts, ignore_index=True)
            # 统计每条边被多少个源支持
            edge_counts = all_edges.groupby(
                ['protein_A', 'protein_B'], as_index=False
            ).agg(weight=('source', 'nunique'))
            edge_counts['weight'] = edge_counts['weight'].astype(np.float32) / n_sources
            fused = edge_counts
            if self.verbose:
                n_filtered = len(fused[fused['weight'] * n_sources < min_sources])
                print(f"[多源PPI] 融合策略=evidence_count ({n_sources}源): "
                      f"{len(fused)} 条边, 过滤<{min_sources}源: -{n_filtered}条")
        
        else:
            raise ValueError(f"未知融合策略: {strategy}，可选: evidence_count, confidence, union")
        
        # 按最小源数过滤
        if strategy in ('evidence_count',):
            fused = fused[fused['weight'] * n_sources >= min_sources].copy()
        
        # 对称化（确保 A-B 和 B-A 统一）
        pairs = fused[['protein_A', 'protein_B']].values
        mask = pairs[:, 0] > pairs[:, 1]
        swapped = pairs[mask].copy()
        fused.loc[mask, 'protein_A'] = swapped[:, 1]
        fused.loc[mask, 'protein_B'] = swapped[:, 0]
        fused = fused.groupby(['protein_A', 'protein_B'], as_index=False)['weight'].max()
        
        self._fused_df = fused
        return fused
    
    def build_fused_adjacency(
        self, strategy: str = 'evidence_count', min_sources: int = 1
    ) -> Tuple[Any, Dict[str, int]]:
        """
        一站式构建融合PPI的加权邻接矩阵
        
        Parameters
        ----------
        strategy    : str, 融合策略
        min_sources : int, 最小支持源数
        
        Returns
        -------
        adj_matrix : scipy.sparse.csr_matrix, 加权邻接矩阵 (float32)
        gene2idx   : dict, 基因名→索引
        """
        from scipy.sparse import csr_matrix
        
        # 加载并融合
        self.load_all_sources()
        if not self._loaded_sources:
            # 没有多源数据，回退到默认单源
            if self.verbose:
                print("[多源PPI] 无多源数据，使用默认 ppi_network.csv")
            ppi_df = load_ppi_network()
            return build_ppi_adjacency(ppi_df)
        
        fused = self.fuse(strategy=strategy, min_sources=min_sources)
        
        # 构建加权邻接矩阵
        all_genes = sorted(set(fused['protein_A'].tolist() + fused['protein_B'].tolist()))
        gene2idx = {g: i for i, g in enumerate(all_genes)}
        n = len(all_genes)
        
        row_indices = []
        col_indices = []
        weights = []
        
        for _, row in fused.iterrows():
            i, j = gene2idx[row['protein_A']], gene2idx[row['protein_B']]
            w = float(row['weight'])
            # 对称添加
            row_indices.extend([i, j])
            col_indices.extend([j, i])
            weights.extend([w, w])
        
        adj = csr_matrix((weights, (row_indices, col_indices)), 
                         shape=(n, n), dtype=np.float32)
        
        print(f"[多源PPI] 加权邻接矩阵构建完成: {n} x {n}, {len(fused)} 条边, "
              f"平均权重={fused['weight'].mean():.4f}")
        return adj, gene2idx
    
    def get_source_stats(self) -> pd.DataFrame:
        """获取各PPI源的统计信息"""
        stats = []
        for name, df in self._loaded_sources.items():
            n_prots = len(set(df['protein_A']) | set(df['protein_B']))
            stats.append({
                'source': name,
                'edges': len(df),
                'proteins': n_prots,
                'mean_weight': df['weight'].mean()
            })
        return pd.DataFrame(stats)


# ==================== 便捷函数 ====================

def build_fused_ppi_adjacency(
    strategy: str = 'evidence_count',
    min_sources: int = 1,
    source_dir: str = None,
    verbose: bool = True
) -> Tuple[Any, Dict[str, int]]:
    """
    一键构建融合PPI加权邻接矩阵
    
    如果 data/raw/ppi_sources/ 中有多个PPI源文件则融合，
    否则回退到默认的 ppi_network.csv。
    
    Parameters
    ----------
    strategy    : str, 融合策略
    min_sources : int, 最小支持源数
    source_dir  : str, PPI源目录
    verbose     : bool, 是否打印信息
    
    Returns
    -------
    adj_matrix : scipy.sparse.csr_matrix
    gene2idx   : dict
    """
    fuser = MultiSourcePPIFuser(source_dir=source_dir, verbose=verbose)
    return fuser.build_fused_adjacency(strategy=strategy, min_sources=min_sources)


if __name__ == "__main__":
    # 简单测试
    print("[测试] 数据加载模块就绪")
    print(f"  RAW目录: {RAW_DIR}")
    print(f"  PROCESSED目录: {PROCESSED_DIR}")
    print(f"  TCMSP数据目录: {TCMSP_DATA_DIR}")
    print(f"  DrugAI数据目录: {DRUGAI_DATA_DIR}")
    print()
    print(f"  配置: HDCTI={HDCTI_USE}, DrugAI={DRUGAI_USE}")
    print()
    print("  使用方式:")
    print("  1. 原始模式: df = load_ingredient_targets(use_hdcti=False, use_drugai=False)")
    print("  2. 增强模式: df = load_ingredient_targets(use_hdcti=True, use_drugai=True)")
    print("  3. TCMSP数据: data = load_tcmsp_data('targets')")
