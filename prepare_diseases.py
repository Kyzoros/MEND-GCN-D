#!/usr/bin/env python3
"""
P0-3: 多疾病验证数据准备脚本
===============================
从GEO数据库准备多种疾病的差异表达数据，用于验证模型泛化能力。

目标疾病列表 (建议):
    D1: 类风湿关节炎 (RA) - 已有 D20_RA_Demo
    D2: 2型糖尿病 (T2D) - GSE156593 / GSE20966
    D3: 阿尔茨海默病 (AD) - GSE63060 + GSE63061
    D4: 胃癌 (GC) - GSE54129
    D5: 非小细胞肺癌 (NSCLC) - GSE19804

数据来源:
    GEO2R在线工具: https://www.ncbi.nlm.nih.gov/geo/geo2r/
    或使用 GEOquery R包下载

使用方式:
    # 步骤1: 手动从GEO下载各疾病的差异表达数据
    # 步骤2: 将每个疾病的数据保存为CSV文件
    # 步骤3: 运行本脚本进行格式化和合并
    python prepare_diseases.py --input_dir ./geo_data --output combined_diseases.csv

输入文件格式 (每个疾病一个CSV):
    gene_symbol,logFC,P_value,adj_P_value
    TP53,2.5,0.001,0.05
    BRCA1,-1.8,0.003,0.08
    ...

输出格式 (与现有disease_profiles.csv一致):
    gene_symbol,logFC,P_value,adj_P_value,disease_id
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR if os.path.exists(os.path.join(SCRIPT_DIR, "config.py")) else os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    from config import RAW_DIR, PROCESSED_DIR
except ImportError:
    RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


# ==================== 预定义的疾病模板 ====================

DISEASE_TEMPLATES = {
    "D21_T2D_Demo": {
        "name": "Type 2 Diabetes (T2D)",
        "description": "2型糖尿病",
        "geo_id": "GSE156593",  # 胰岛组织, T2D vs 正常
        "tissue": "Pancreatic islets",
        "expected_genes_up": ["IRS1", "INS", "GCK"],       # 预期上调
        "expected_genes_down": ["TNF", "IL6", "NFKB1"],    # 预期下调
    },
    "D22_AD_Demo": {
        "name": "Alzheimer's Disease (AD)",
        "description": "阿尔茨海默病",
        "geo_id": "GSE63060",  # 脑组织, AD vs 对照
        "tissue": "Brain (prefrontal cortex)",
        "expected_genes_up": ["APP", "MAPT", "BACE1"],
        "expected_genes_down": ["BDNF", "SYN1", "DLG4"],
    },
    "D23_GC_Demo": {
        "name": "Gastric Cancer (GC)",
        "description": "胃癌",
        "geo_id": "GSE54129",  # 胃组织, 肿瘤 vs 正常
        "tissue": "Gastric tissue",
        "expected_genes_up": ["MYC", "EGFR", "ERBB2"],
        "expected_genes_down": ["CDH1", "TFF1", "GKN1"],
    },
    "D24_NSCLC_Demo": {
        "name": "Non-Small Cell Lung Cancer (NSCLC)",
        "description": "非小细胞肺癌",
        "geo_id": "GSE19804",  # 肺组织, 肿瘤 vs 正常
        "tissue": "Lung tissue",
        "expected_genes_up": ["KRAS", "TP53", "EGFR"],
        "expected_genes_down": ["CDH1", "CAV1", "AGER"],
    },
    "D25_HTN_Demo": {
        "name": "Hypertension (HTN)",
        "description": "高血压",
        "geo_id": "GSE74144",  # 血管组织, 高血压 vs 正常
        "tissue": "Artery tissue",
        "expected_genes_up": ["AGT", "REN", "ACE"],
        "expected_genes_down": ["NOS3", "EDN1", "ADRB2"],
    },
}


def validate_disease_file(filepath: str) -> tuple:
    """
    验证疾病表达谱文件格式
    
    Returns
    -------
    (is_valid, message, gene_count)
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return False, f"无法读取文件: {e}", 0
    
    # 检查必要列
    required_cols = {"gene_symbol", "logFC", "P_value"}
    missing = required_cols - set(df.columns)
    if missing:
        # 尝试自动识别列名
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower().replace(" ", "_").replace(".", "_")
            if col_lower in ("gene_symbol", "genesymbol", "symbol", "gene", "gene_name", "id"):
                col_mapping[col] = "gene_symbol"
            elif col_lower in ("logfc", "log2fc", "log2_fold_change", "log_fc"):
                col_mapping[col] = "logFC"
            elif col_lower in ("p_value", "pvalue", "p_val", "pval"):
                col_mapping[col] = "P_value"
            elif col_lower in ("adj_p_value", "adj_pvalue", "padj", "fdr", "q_value"):
                col_mapping[col] = "adj_P_value"
        
        if col_mapping:
            df = df.rename(columns=col_mapping)
            still_missing = required_cols - set(df.columns)
            if still_missing:
                return False, f"缺少列: {still_missing}. 可用列: {list(df.columns)}", 0
        else:
            return False, f"缺少列: {missing}. 可用列: {list(df.columns)}", 0
    
    # 数据类型转换
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df["P_value"] = pd.to_numeric(df["P_value"], errors="coerce")
    
    # 过滤无效行
    df = df.dropna(subset=["gene_symbol", "logFC", "P_value"])
    
    if len(df) == 0:
        return False, "过滤后无有效数据", 0
    
    # 确保有adj_P_value列
    if "adj_P_value" not in df.columns:
        df["adj_P_value"] = df["P_value"]  # 用P_value填充
    
    return True, f"有效: {len(df)} 个基因", len(df)


def standardize_disease_file(filepath: str, disease_id: str, 
                              p_threshold: float = 0.05) -> pd.DataFrame:
    """
    标准化疾病表达谱文件
    
    处理步骤:
    1. 列名标准化
    2. 过滤P_value > p_threshold的基因
    3. 如果gene_symbol是探针ID，保留以便后续预处理
    4. 添加disease_id列
    """
    df = pd.read_csv(filepath)
    
    # 列名标准化
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_").replace(".", "_")
        if col_lower in ("gene_symbol", "genesymbol", "symbol", "gene", "gene_name", "id"):
            col_mapping[col] = "gene_symbol"
        elif col_lower in ("logfc", "log2fc", "log2_fold_change", "log_fc", "fold_change"):
            col_mapping[col] = "logFC"
        elif col_lower in ("p_value", "pvalue", "p_val", "pval"):
            col_mapping[col] = "P_value"
        elif col_lower in ("adj_p_value", "adj_pvalue", "padj", "fdr", "q_value"):
            col_mapping[col] = "adj_P_value"
    
    df = df.rename(columns={k: v for k, v in col_mapping.items() 
                            if v not in df.columns})
    
    # 确保必要列存在
    if "adj_P_value" not in df.columns:
        df["adj_P_value"] = df["P_value"]
    
    # 类型转换
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df["P_value"] = pd.to_numeric(df["P_value"], errors="coerce")
    
    # 过滤
    df = df.dropna(subset=["gene_symbol", "logFC", "P_value"])
    df = df[df["P_value"] <= p_threshold]
    
    # 添加disease_id
    df["disease_id"] = disease_id
    
    # 保留必要列
    result = df[["gene_symbol", "logFC", "P_value", "adj_P_value", "disease_id"]]
    
    return result


def merge_disease_profiles(input_dir: str, output_path: str,
                            p_threshold: float = 0.05,
                            exclude_existing: bool = True) -> pd.DataFrame:
    """
    合并多个疾病表达谱文件
    
    Parameters
    ----------
    input_dir : str, 包含各疾病CSV文件的目录
    output_path : str, 输出文件路径
    p_threshold : float, P值阈值
    exclude_existing : bool, 是否排除已有疾病(避免重复)
    
    Returns
    -------
    pd.DataFrame, 合并后的疾病表达谱
    """
    # 获取现有的疾病ID列表
    existing_ids = set()
    if exclude_existing:
        existing_path = os.path.join(PROCESSED_DIR, "disease_profiles_processed.csv")
        if os.path.exists(existing_path):
            existing = pd.read_csv(existing_path)
            existing_ids = set(existing["disease_id"].unique())
            print(f"已有疾病: {existing_ids}")
    
    # 收集所有疾病文件
    dfs = []
    
    # 如果exclude_existing=False，先加载已有数据
    if not exclude_existing and existing_ids:
        existing_path = os.path.join(PROCESSED_DIR, "disease_profiles_processed.csv")
        if os.path.exists(existing_path):
            dfs.append(pd.read_csv(existing_path))
    
    # 遍历输入目录
    if os.path.isdir(input_dir):
        csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
        
        for csv_file in csv_files:
            filepath = os.path.join(input_dir, csv_file)
            disease_id = os.path.splitext(csv_file)[0]
            
            # 检查是否是已有疾病
            if disease_id in existing_ids:
                print(f"  跳过已有疾病: {disease_id}")
                continue
            
            # 验证
            is_valid, msg, count = validate_disease_file(filepath)
            if not is_valid:
                print(f"  [警告] {csv_file}: {msg}")
                continue
            
            print(f"  [处理] {disease_id}: {msg}")
            
            # 标准化
            df = standardize_disease_file(filepath, disease_id, p_threshold)
            dfs.append(df)
    
    if not dfs:
        print("[错误] 没有找到有效的疾病数据文件")
        return pd.DataFrame()
    
    # 合并
    combined = pd.concat(dfs, ignore_index=True)
    
    # 保存
    combined.to_csv(output_path, index=False)
    print(f"\n合并完成!")
    print(f"  疾病数: {combined['disease_id'].nunique()}")
    print(f"  基因条目数: {len(combined)}")
    
    # 统计每种疾病的基因数
    for did in sorted(combined['disease_id'].unique()):
        count = len(combined[combined['disease_id'] == did])
        print(f"    {did}: {count} 个差异基因")
    
    print(f"  输出文件: {output_path}")
    
    return combined


def create_disease_from_geo2r_result(geo2r_csv: str, disease_id: str, 
                                       name: str, output_path: str) -> bool:
    """
    从GEO2R导出结果创建疾病文件
    
    GEO2R通常会导出包含以下列的CSV:
    - ID (探针ID或基因符号)
    - logFC
    - P.Value
    - adj.P.Val
    
    这个函数将其转换为标准格式。
    """
    try:
        df = pd.read_csv(geo2r_csv)
        print(f"GEO2R文件: {len(df)} 行, 列: {list(df.columns)}")
        
        result = standardize_disease_file(geo2r_csv, disease_id)
        
        if len(result) > 0:
            result.to_csv(output_path, index=False)
            print(f"创建成功: {output_path} ({len(result)} 个基因)")
            return True
        else:
            print("标准化后无有效数据")
            return False
            
    except Exception as e:
        print(f"处理失败: {e}")
        return False


def print_geo2r_guide():
    """打印GEO2R使用指南"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              GEO2R 数据下载指南                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  步骤1: 打开GEO2R分析页面                                    ║
║    https://www.ncbi.nlm.nih.gov/geo/geo2r/                  ║
║                                                              ║
║  步骤2: 输入GEO数据集编号                                     ║
║    例如: GSE156593                                            ║
║                                                              ║
║  步骤3: 定义实验组和对照组                                     ║
║    - 选择疾病样本为实验组                                      ║
║    - 选择正常样本为对照组                                      ║
║                                                              ║
║  步骤4: 运行分析并下载结果                                     ║
║    - 点击 "Analyze" 按钮                                      ║
║    - 点击 "Download full table" 导出CSV                       ║
║                                                              ║
║  步骤5: 保存CSV文件到 geo_data/ 目录                          ║
║    命名格式: {disease_id}.csv                                  ║
║    例如: D21_T2D_Demo.csv                                     ║
║                                                              ║
║  步骤6: 运行本脚本进行合并                                     ║
║    python prepare_diseases.py --input_dir ./geo_data          ║
║                                                              ║
║  建议的GEO数据集:                                             ║
║    - GSE156593: T2D胰岛组织 (n=20 vs 20)                     ║
║    - GSE63060: AD前额叶皮层 (n=145 vs 104)                   ║
║    - GSE54129: 胃癌组织 (n=21 vs 21)                          ║
║    - GSE19804: NSCLC肺组织 (n=60 vs 60)                      ║
║    - GSE74144: 高血压动脉组织 (n=15 vs 15)                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="P0-3: 多疾病验证数据准备",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input_dir", type=str, default="./geo_data",
                       help="包含各疾病CSV文件的目录")
    parser.add_argument("--output", type=str, 
                       default=os.path.join(RAW_DIR, "disease_profiles_multi.csv"),
                       help="输出文件路径")
    parser.add_argument("--p_threshold", type=float, default=0.05,
                       help="P值阈值 (默认0.05)")
    parser.add_argument("--merge_with_existing", action="store_true",
                       help="与现有疾病数据合并 (而非仅新疾病)")
    parser.add_argument("--guide", action="store_true",
                       help="显示GEO2R使用指南后退出")
    parser.add_argument("--check", action="store_true",
                       help="检查当前疾病数据概况")
    
    args = parser.parse_args()
    
    if args.guide:
        print_geo2r_guide()
        return
    
    if args.check:
        check_current_diseases()
        return
    
    # 确保输入目录存在
    if not os.path.isdir(args.input_dir):
        print(f"[提示] 输入目录不存在: {args.input_dir}")
        print(f"       请先创建此目录并放入疾病CSV文件")
        print(f"       使用 --guide 查看GEO2R下载指南")
        # 创建目录
        os.makedirs(args.input_dir, exist_ok=True)
        print(f"       已创建目录: {args.input_dir}")
        print_geo2r_guide()
        return
    
    print("=" * 60)
    print("P0-3: 多疾病验证数据准备")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出文件: {args.output}")
    print(f"P值阈值: {args.p_threshold}")
    print()
    
    # 合并疾病数据
    combined = merge_disease_profiles(
        input_dir=args.input_dir,
        output_path=args.output,
        p_threshold=args.p_threshold,
        exclude_existing=not args.merge_with_existing
    )
    
    if len(combined) == 0:
        print("\n请先将GEO2R导出的CSV文件放入输入目录")
        print_geo2r_guide()
    else:
        print(f"\n✅ 多疾病数据准备完成!")
        print(f"   下一步: 将 {args.output} 复制到 data/raw/disease_profiles.csv")
        print(f"   然后运行: python main.py")


def check_current_diseases():
    """检查当前疾病数据概况"""
    existing_path = os.path.join(PROCESSED_DIR, "disease_profiles_processed.csv")
    if not os.path.exists(existing_path):
        existing_path = os.path.join(RAW_DIR, "disease_profiles.csv")
    
    if os.path.exists(existing_path):
        df = pd.read_csv(existing_path)
        print("=" * 60)
        print("当前疾病数据概况")
        print("=" * 60)
        print(f"文件: {existing_path}")
        print(f"总条目: {len(df)}")
        print(f"列: {list(df.columns)}")
        print()
        
        for did in sorted(df['disease_id'].unique()):
            sub = df[df['disease_id'] == did]
            n_up = (sub['logFC'] > 0).sum()
            n_down = (sub['logFC'] < 0).sum()
            print(f"  {did}: {len(sub)} 个基因 (上调:{n_up}, 下调:{n_down})")
    else:
        print("未找到疾病表达谱文件")


if __name__ == "__main__":
    # 如果没有命令行参数，显示交互式指南
    if len(sys.argv) == 1:
        print("=" * 60)
        print("P0-3: 多疾病验证数据准备")
        print("=" * 60)
        print()
        check_current_diseases()
        print()
        print_geo2r_guide()
        print("\n使用方式:")
        print("  python prepare_diseases.py --guide         查看GEO2R指南")
        print("  python prepare_diseases.py --check          查看当前疾病概况")
        print("  python prepare_diseases.py --input_dir DIR  合并疾病数据")
    else:
        main()
