#!/usr/bin/env python3
"""
将 GEO2R 导出的差异表达基因文件转换为流水线标准格式
"""
import pandas as pd
import os


def convert_geo2r_to_standard(input_path: str, output_path: str, disease_id: str,
                              keep_adj_p: bool = True):
    print(f"\n处理: {input_path} -> {disease_id}")
    # 先尝试逗号分隔，失败则用Tab分隔
    try:
        df = pd.read_csv(input_path, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(input_path, sep='\t')

    # 1. 列重命名
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_").replace(".", "_")
        if col_lower in ("gene_symbol", "genesymbol", "gene.symbol"):
            rename_map[col] = "gene_symbol"
        elif col_lower in ("logfc", "log_fc", "log2fc"):
            rename_map[col] = "logFC"
        elif col_lower in ("p_value", "pvalue", "p.val", "p.value"):
            rename_map[col] = "P_value"
        elif col_lower in ("adj_p_val", "adj_p_value", "adj_pvalue", "adj_p_val", "fdr", "q_value"):
            rename_map[col] = "adj_P_value"
    df = df.rename(columns=rename_map)

    print(f"  原始行数: {len(df)}, 列: {list(df.columns)}")

    # 2. 过滤空基因名
    df = df[df["gene_symbol"].notna() & (df["gene_symbol"].astype(str).str.strip() != "")]
    print(f"  过滤空基因名后: {len(df)}")

    # 3. 过滤纯数字探针ID
    numeric_mask = df["gene_symbol"].astype(str).str.match(r'^\d+$')
    df = df[~numeric_mask]
    print(f"  过滤数字ID后: {len(df)}")

    # 4. 类型转换
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df["P_value"] = pd.to_numeric(df["P_value"], errors="coerce")
    if "adj_P_value" in df.columns:
        df["adj_P_value"] = pd.to_numeric(df["adj_P_value"], errors="coerce")
    df = df.dropna(subset=["logFC", "P_value"])

    # 5. 添加 disease_id
    df["disease_id"] = disease_id

    # 6. 保留每个基因P值最小的记录
    df = df.sort_values("P_value").drop_duplicates(
        subset=["disease_id", "gene_symbol"], keep="first"
    )
    # 按logFC排序
    df = df.sort_values(["disease_id", "logFC"], ascending=[True, False])

    # 7. 输出标准列（保留adj_P_value）
    output_cols = ["disease_id", "gene_symbol", "logFC", "P_value"]
    if keep_adj_p and "adj_P_value" in df.columns:
        output_cols.append("adj_P_value")
    result = df[output_cols].reset_index(drop=True)

    result.to_csv(output_path, index=False)
    print(f"  输出: {output_path} ({len(result)} 条)")
    print(f"  上调基因: {(result['logFC'] > 0).sum()}, 下调基因: {(result['logFC'] < 0).sum()}")
    return result


if __name__ == "__main__":
    raw_dir = r"C:\Users\alikex\Desktop\Work\差异表达基因"
    out_dir = "data/processed"

    # PAH (肺动脉高压)
    convert_geo2r_to_standard(
        os.path.join(raw_dir, "tT2.csv"),
        os.path.join(out_dir, "disease_profiles_PAH.csv"),
        "PAH"
    )

    # RA (类风湿性关节炎)
    convert_geo2r_to_standard(
        os.path.join(raw_dir, "GSE77298_RA.top.table.csv"),
        os.path.join(out_dir, "disease_profiles_RA.csv"),
        "RA"
    )

    # T2D (2型糖尿病)
    convert_geo2r_to_standard(
        os.path.join(raw_dir, "GSE20966_T2D.top.table.csv"),
        os.path.join(out_dir, "disease_profiles_T2D.csv"),
        "T2D"
    )

    # AD (阿尔茨海默病)
    convert_geo2r_to_standard(
        os.path.join(raw_dir, "GSE63060_AD.top.table.csv"),
        os.path.join(out_dir, "disease_profiles_AD.csv"),
        "AD"
    )
