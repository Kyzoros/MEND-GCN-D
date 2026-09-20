"""
prepare_disease_profiles.py
=================================================================
合并 8 个疾病的差异表达基因 (DEG) -> data/processed/disease_profiles_processed.csv

数据来源 (用户从 GEO 下载, 已人工确认分组正确):
  AD       GSE48350_AD.top.table.csv  (Affymetrix, 主集)
  AD(复现)  GSE63060_AD.top.table.csv  (Illumina, 单独保存做复现性检验)
  RA       GSE77298_RA.top.table.csv
  T2D      GSE20966_T2D.top.table.csv
  HCC      GSE14520_HCC.top.table.csv
  CRC      GSE32323_CRC.top.table.csv
  ASTHMA   GSE4302_ASTHMA.top.table.csv
  OA       GSE114007_OA.top.table.csv   (DESeq2 / RNA-seq 格式, 需列名转换)
  PAH      (无原始 GEO, 从旧 disease_profiles_processed.csv 借用)

每病按 P 值取 top-K (= config.disease_deg_top_k, 默认 1500) 最显著 DEG,
保证跨病模块规模一致, 消除旧方法中 PAH 9674 vs AD 3684 的失衡。

输出:
  data/processed/disease_profiles_processed.csv          (8 病, 列: disease_id,gene_symbol,logFC,P_value)
  data/processed/disease_profiles_AD_replicate.csv       (GSE63060, 复现性检验用)
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import PROCESSED_DIR, DISEASE_DEG_TOP_K  # noqa: E402

GEO_DIR = r"C:\Users\alikex\Desktop\Work\差异表达基因"
TOP_K = int(DISEASE_DEG_TOP_K) if DISEASE_DEG_TOP_K and DISEASE_DEG_TOP_K > 0 else 1500

# 主集: disease_id -> GEO2R top.table 文件名 (GEO2R/limma 格式)
LIMMA_FILES = {
    "AD":      "GSE48350_AD.top.table.csv",
    "RA":      "GSE77298_RA.top.table.csv",
    "T2D":     "GSE20966_T2D.top.table.csv",
    "HCC":     "GSE14520_HCC.top.table.csv",
    "CRC":     "GSE32323_CRC.top.table.csv",
    "ASTHMA":  "GSE4302_ASTHMA.top.table.csv",
}
AD_REPLICATE = "GSE63060_AD.top.table.csv"   # Illumina 平台, 复现性集
OA_FILE = "GSE114007_OA.top.table.csv"        # DESeq2 / RNA-seq 格式


def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_limma(path: str) -> pd.DataFrame:
    """GEO2R/limma 标准格式: Gene.symbol, logFC, P.Value (GEO2R 导出为 tab 分隔)"""
    df = pd.read_csv(path, sep="\t")
    return pd.DataFrame({
        "gene_symbol": df["Gene.symbol"],
        "logFC": _coerce_num(df["logFC"]),
        "P_value": _coerce_num(df["P.Value"]),
    })


def load_deseq2(path: str) -> pd.DataFrame:
    """DESeq2 格式: Symbol, log2FoldChange, pvalue (GEO2R 导出为 tab 分隔)"""
    df = pd.read_csv(path, sep="\t")
    return pd.DataFrame({
        "gene_symbol": df["Symbol"],
        "logFC": _coerce_num(df["log2FoldChange"]),
        "P_value": _coerce_num(df["pvalue"]),
    })


def clean_take_top(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """展开多基因符号(///分隔), 清洗, 按 P 值取 top-K"""
    df = df.copy()
    df["gene_symbol"] = df["gene_symbol"].astype(str).str.split("///")
    df = df.explode("gene_symbol")
    df["gene_symbol"] = df["gene_symbol"].str.strip()
    df = df[
        df["gene_symbol"].notna()
        & (df["gene_symbol"] != "")
        & (df["gene_symbol"].str.upper() != "NA")
    ]
    df = df.dropna(subset=["P_value"])
    df = df.drop_duplicates(subset="gene_symbol", keep="first")
    df = df.sort_values("P_value", na_position="last").head(k)
    return df[["gene_symbol", "logFC", "P_value"]].reset_index(drop=True)


frames = []

# ---- limma 格式 6 病 ----
for did, fn in LIMMA_FILES.items():
    p = os.path.join(GEO_DIR, fn)
    if not os.path.exists(p):
        print(f"[WARN] 缺失 {fn}, 跳过 {did}")
        continue
    df = clean_take_top(load_limma(p), TOP_K)
    df.insert(0, "disease_id", did)
    frames.append(df)
    print(f"  {did:7s}: {len(df):4d} genes  <- {fn}")

# ---- OA (DESeq2) ----
poa = os.path.join(GEO_DIR, OA_FILE)
if os.path.exists(poa):
    df = clean_take_top(load_deseq2(poa), TOP_K)
    df.insert(0, "disease_id", "OA")
    frames.append(df)
    print(f"  {'OA':7s}: {len(df):4d} genes  <- {OA_FILE} (DESeq2)")
else:
    print(f"[WARN] 缺失 {OA_FILE}, 跳过 OA")

# ---- AD 复现性集 (单独保存, 不并入主表) ----
pr = os.path.join(GEO_DIR, AD_REPLICATE)
if os.path.exists(pr):
    ad_rep = clean_take_top(load_limma(pr), TOP_K)
    ad_rep.to_csv(os.path.join(PROCESSED_DIR, "disease_profiles_AD_replicate.csv"), index=False)
    print(f"  [复现] AD replicate (GSE63060): {len(ad_rep)} genes "
          f"-> disease_profiles_AD_replicate.csv")
else:
    print(f"[WARN] 缺失 {AD_REPLICATE}, 跳过 AD 复现性集")

# ---- PAH 从旧 processed 借用 (无原始 GEO) ----
old_path = os.path.join(PROCESSED_DIR, "disease_profiles_processed.csv")
if os.path.exists(old_path):
    old = pd.read_csv(old_path)
    pah = old[old["disease_id"] == "PAH"][["disease_id", "gene_symbol", "logFC", "P_value"]].copy()
    if len(pah) > TOP_K:
        pah = pah.sort_values("P_value").head(TOP_K)
    frames.append(pah)
    print(f"  {'PAH':7s}: {len(pah):4d} genes  <- 旧 processed (借用, 无原始 GEO)")
else:
    print("[WARN] 旧 processed 不存在, PAH 缺失")

# ---- 合并写出 ----
result = pd.concat(frames, ignore_index=True)[["disease_id", "gene_symbol", "logFC", "P_value"]]
out_path = os.path.join(PROCESSED_DIR, "disease_profiles_processed.csv")
result.to_csv(out_path, index=False)

print("\n=== 写入完成 ===")
print(f"  文件: {out_path}")
print(f"  总行数: {len(result)}, 病种数: {result['disease_id'].nunique()}")
print("  各病基因数:")
print(result["disease_id"].value_counts().to_string())
