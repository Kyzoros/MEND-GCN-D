"""
统一配置模块 — YAML驱动 + 环境变量覆盖 + 向后兼容
=================================================================
优先级: 命令行参数 > 环境变量 > YAML配置 > 代码默认值

用法:
    from config import PROJECT_ROOT, GA_MAX_GENERATIONS, ...
    # 所有原有变量名保持不变，值来源于 config.yaml

运行时覆盖:
    export SM_HDCTI_PATH=/path/to/hdcti
    export SM_DRUGAI_USE=false
    python main.py --disease AD --output-dir ./my_results
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

# ---- 尝试导入 yaml ----
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# ===========================================================================
# 辅助函数
# ===========================================================================

def _load_yaml_config(yaml_path: str) -> dict:
    """加载 YAML 配置文件"""
    if not _HAS_YAML:
        warnings.warn("PyYAML 未安装。使用代码默认值。pip install pyyaml")
        return {}
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        warnings.warn(f"YAML 配置加载失败 ({yaml_path}): {e}, 使用默认值")
        return {}

def _env_override(env_var: str, default: Any) -> Any:
    """环境变量覆盖"""
    val = os.environ.get(env_var)
    if val is None:
        return default
    # 类型转换
    if isinstance(default, bool):
        return val.lower() in ('true', '1', 'yes', 'on')
    if isinstance(default, int):
        try:
            return int(val)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(val)
        except ValueError:
            return default
    if isinstance(default, list):
        try:
            import json
            return json.loads(val)
        except:
            return val.split(',')
    return val

def _nested_get(d: dict, *keys, default=None):
    """安全获取嵌套字典值"""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
            if d is None:
                return default
        else:
            return default
    return d

# ===========================================================================
# 基础路径 (始终由代码自动检测，不可由 YAML 覆盖)
# ===========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
ABLATION_DIR = os.path.join(RESULTS_DIR, "ablation")

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR,
           FIGURES_DIR, RESULTS_DIR, REPORTS_DIR, ABLATION_DIR]:
    os.makedirs(d, exist_ok=True)

# PPI 源目录
PPI_SOURCE_DIR = os.path.join(RAW_DIR, "ppi_sources")
os.makedirs(PPI_SOURCE_DIR, exist_ok=True)

# ===========================================================================
# 加载 YAML 配置
# ===========================================================================
_YAML_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
_cfg = _load_yaml_config(_YAML_PATH)

# ===========================================================================
# 外部工具路径 (YAML + 环境变量覆盖)
# ===========================================================================
HDCTI_PATH = _env_override(
    "SM_HDCTI_PATH",
    _nested_get(_cfg, "paths", "hdcti_path", default="")
)
DRUGAI_PATH = _env_override(
    "SM_DRUGAI_PATH",
    _nested_get(_cfg, "paths", "drugai_path", default="")
)
TCMSP_DATA_DIR = _env_override(
    "SM_TCMSP_DATA_DIR",
    _nested_get(_cfg, "paths", "tcmsp_data_dir", default="")
)
DRUGAI_DATA_DIR = os.path.join(DATA_DIR, "drugai_data")
os.makedirs(DRUGAI_DATA_DIR, exist_ok=True)

# DrugAI 模型路径
_drugai_model = _nested_get(_cfg, "paths", "drugai_model_path")
if _drugai_model:
    DRUGAI_MODEL_PATH = _drugai_model
else:
    _model_dir = os.path.join(os.path.dirname(DATA_DIR), "Drug_AI_best_view")
    _candidates = [os.path.join(_model_dir, f) for f in os.listdir(_model_dir)
                   if f.endswith('.pkl')] if os.path.isdir(_model_dir) else []
    # 优先使用 RDKit 指纹重训模型 (retrain_drugai_rdkit.py):
    #   drug-disjoint TEST AUC=0.8901, 且特征确定性(不依赖 PYTHONHASHSEED);
    #   旧 best_model_auc_*.pkl 的字符哈希指纹无法跨进程复现, 线上实测 AUC 仅 0.6143。
    _preferred = [p for p in _candidates if os.path.basename(p) == "best_model_rdkit.pkl"]
    DRUGAI_MODEL_PATH = (_preferred or _candidates or [""])[0]

# ===========================================================================
# 数据库URL
# ===========================================================================
TCMSP_URL = "https://tcmsp-e.com/tcmspsearch.php"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

# ===========================================================================
# 文件命名 (保持 config.FILE_XXX 原有风格)
# ===========================================================================
FILE_DISEASE_PROFILES = _nested_get(_cfg, "files", "disease_profiles", default="disease_profiles.csv")
FILE_DISEASE_PROFILES_PROCESSED = _nested_get(_cfg, "files", "disease_profiles_processed", default="disease_profiles_processed.csv")
FILE_HERB_INGREDIENTS = _nested_get(_cfg, "files", "herb_ingredients", default="herb_ingredients.csv")
FILE_HERB_INGREDIENTS_PROCESSED = _nested_get(_cfg, "files", "herb_ingredients_processed", default="herb_ingredients_processed.csv")
FILE_INGREDIENT_TARGETS = _nested_get(_cfg, "files", "ingredient_targets", default="ingredient_targets.csv")
FILE_INGREDIENT_TARGETS_PROCESSED = _nested_get(_cfg, "files", "ingredient_targets_processed", default="ingredient_targets_processed.csv")
FILE_INGREDIENT_TARGETS_ENHANCED = _nested_get(_cfg, "files", "ingredient_targets_enhanced", default="ingredient_targets_enhanced.csv")
FILE_PPI_NETWORK = _nested_get(_cfg, "files", "ppi_network", default="ppi_network.csv")
FILE_PPI_NETWORK_PROCESSED = _nested_get(_cfg, "files", "ppi_network_processed", default="ppi_network_processed.csv")
FILE_HERB_META = _nested_get(_cfg, "files", "herb_metadata", default="herb_metadata.csv")
FILE_HERB_PERTURBATIONS = "herb_perturbation_signatures.csv"
FILE_HDMAP_S_RESULTS = "hdmap_s_scores.csv"
FILE_HDMAP_M_RESULTS = "hdmap_m_results.csv"
FILE_CHEMBL_MECHANISMS = _nested_get(_cfg, "files", "chembl_mechanisms", default="chembl_mechanisms.csv")
FILE_SIGNATURE_CACHE = _nested_get(_cfg, "files", "signature_cache", default="herb_signature_cache.npz")
FILE_SIGNATURE_META = _nested_get(_cfg, "files", "signature_meta", default="herb_signature_meta.json")
FILE_GOLD_STANDARD = _nested_get(_cfg, "files", "gold_standard", default="gold_standard.csv")
GOLD_STANDARD_PATH = os.path.join(DATA_DIR, FILE_GOLD_STANDARD)

# ===========================================================================
# HDmap-S 参数
# ===========================================================================
TOP_BOTTOM_K = _env_override("SM_TOP_BOTTOM_K",
    _nested_get(_cfg, "hdmap_s", "top_bottom_k", default=500))  # 调优: 250→500, 扩大匹配空间, 提升 TMS 范围
PERMUTATION_N = _env_override("SM_PERMUTATION_N",
    _nested_get(_cfg, "hdmap_s", "permutation_n", default=100))

# 疾病基因logFC过滤阈值: 只保留|logFC|大于此值的基因参与TMS计算
# 调优说明: 阈值过高则候选基因过少, K/n_disease 比例失衡, TMS 范围被压缩
#   threshold=0.05: 平衡噪声与候选量
#   threshold=0.1:   严格, 候选约2000 (AD)
#   threshold=0:     保留所有差异基因, 候选约17,000 (噪声大)
DISEASE_LOGFC_THRESHOLD = _env_override("SM_LOGFC_THRESHOLD",
    _nested_get(_cfg, "hdmap_s", "logfc_threshold", default=0.05))

# ---- 疾病基因logFC过滤阈值
DISEASE_PVALUE_THRESHOLD = _env_override("SM_PVALUE_THRESHOLD",
    _nested_get(_cfg, "hdmap_s", "pvalue_threshold", default=0.05))

# 疾病基因集: 按 P 值取每病 top-K 最显著 DEG (各病规模一致, 解决跨病失衡)
#   0 或负 = 停用, 退回 logfc_threshold / pvalue_threshold 绝对阈值模式
DISEASE_DEG_TOP_K = int(_env_override("SM_DISEASE_DEG_TOP_K",
    _nested_get(_cfg, "hdmap_s", "disease_deg_top_k", default=0)))

# TMS计算方法: "ndcg" (原始) | "weighted" (cosine, 全基因参与) | "dual" (双通道, 默认)
TMS_METHOD = _env_override("SM_TMS_METHOD",
    _nested_get(_cfg, "hdmap_s", "tms_method", default="dual"))

# 双通道TMS的疾病机制先验: {disease_id: "antagonist" | "engagement"}
# 决定每个疾病用哪个通道作为"治疗"方向。缺省 engagement。
DISEASE_TMS_MECHANISM = _nested_get(_cfg, "hdmap_s", "disease_tms_mechanism", default={})
if not isinstance(DISEASE_TMS_MECHANISM, dict):
    DISEASE_TMS_MECHANISM = {}

# ---- 疾病关键靶标基因权重 ----
# 文献已知的AD药物靶标/因果基因，在TMS增益计算时获得额外权重
# gain × KNOWN_TARGET_WEIGHT 使命中药草得到更低(更好)的TMS
KNOWN_TARGET_WEIGHT = 2.0
KNOWN_AD_TARGET_GENES = {
    # 因果 / 遗传风险
    "APP", "PSEN1", "PSEN2", "APOE", "MAPT", "TREM2", "BIN1",
    "CLU", "PICALM", "CD33", "CR1", "ABCA7", "SORL1", "CD2AP", "EPHA1",
    # AD药物靶标
    "ACHE", "BCHE", "BACE1", "GSK3B", "CDK5",
    # 神经炎症
    "TNF", "IL1B", "IL6", "PTGS2", "NOS2",
    # 氧化应激 (银杏叶EGb761核心靶标)
    "CAT", "SOD1", "SOD2", "GPX1", "GSR", "NFE2L2",
    # Aβ/Tau清除 / 神经营养 / 突触
    "IDE", "MME", "HSP90AA1", "BDNF", "NGF", "GDNF",
    "DLG4", "SYP", "GRIN2B",
}
# 疾病ID → {基因集: 权重} 映射 (扩展用)
DISEASE_GENE_WEIGHTS = {
    "AD": (KNOWN_AD_TARGET_GENES, KNOWN_TARGET_WEIGHT),
}

assert isinstance(TOP_BOTTOM_K, int) and 1 <= TOP_BOTTOM_K <= 20000

# ===========================================================================
# PPI 参数
# ===========================================================================
# Propagation Network Selection
#   "ppi"  -> traditional PPI protein interaction network
#   "kegg" -> KEGG pathway co-membership network (functional relationships)
NETWORK_TYPE = _env_override("SM_NETWORK_TYPE",
    _nested_get(_cfg, "network", "type", default="kegg"))
KEGG_MIN_SHARED = _nested_get(_cfg, "network", "min_shared", default=2)  # KEGG稀疏化: 剔除仅共享1条通路的弱连接
# ===========================================================================
PPI_N_PROTEINS = _nested_get(_cfg, "ppi", "n_proteins", default=13460)
PPI_N_INTERACTIONS = _nested_get(_cfg, "ppi", "n_interactions", default=141296)
PPI_MULTI_SOURCE = _env_override("SM_PPI_MULTI_SOURCE",
    _nested_get(_cfg, "ppi", "multi_source", default=True))
PPI_FUSION_STRATEGY = _nested_get(_cfg, "ppi", "fusion_strategy", default="evidence_count")
PPI_MIN_SOURCES = _nested_get(_cfg, "ppi", "min_sources", default=1)

# ===========================================================================
# 扩散参数
# ===========================================================================
DIFFUSION_METHOD = _nested_get(_cfg, "diffusion", "method", default="rwr")
THERMAL_ALPHA = _nested_get(_cfg, "diffusion", "thermal_alpha", default=1.0)
RWR_RESTART_PROB = _env_override("SM_RWR_RESTART_PROB",
    _nested_get(_cfg, "diffusion", "rwr_restart_prob", default=0.7))  # 调优: 0.85→0.7, 让信号传到 2-3 跳邻居, 扩大 TMS 范围
RWR_MAX_ITER = _nested_get(_cfg, "diffusion", "rwr_max_iter", default=100)
RWR_TOL = _nested_get(_cfg, "diffusion", "rwr_tol", default=1e-6)
assert THERMAL_ALPHA > 0
assert 0 < RWR_RESTART_PROB < 1
assert RWR_MAX_ITER >= 1
assert RWR_TOL > 0

# ===========================================================================
# DrugAI 参数
# ===========================================================================
DRUGAI_USE = _env_override("SM_DRUGAI_USE",
    _nested_get(_cfg, "drugai", "use", default=True))
DRUGAI_CONFIDENCE_THRESHOLD = _nested_get(_cfg, "drugai", "confidence_threshold", default=0.7)
DRUGAI_DEVICE = _nested_get(_cfg, "drugai", "device", default="cpu")
DRUGAI_USE_GPU = _nested_get(_cfg, "drugai", "use_gpu", default=False)
DRUGAI_DISEASE_VALIDATE = _nested_get(_cfg, "drugai", "disease_validate", default=True)  # AD表达谱方向验证

# 实验验证机制覆盖 (真值优先于 DrugAI 预测)
#   match_mode: "skeleton"(InChIKey骨架, 覆盖0.078%) | "exact"(canonical SMILES, 0.020%)
EXP_MECH_ENABLE = _nested_get(_cfg, "experimental_mechanisms", "enable", default=True)
EXP_MECH_MATCH_MODE = _nested_get(_cfg, "experimental_mechanisms", "match_mode",
                                  default="skeleton")

# ===========================================================================
# 边置信度加权 (解决"覆盖率提高反而降 AUROC"的核心机制)
#   签名权重 = action_type * confidence。下列参数让低质量边自动降权:
#   - OOD补全(PubChem)靶标: SMILES 为 OOD 天然产物, DrugAI 方向不可信 → 惩罚
#   - KEGG 网络: 通路共成员关系使生化激活/抑制方向噪声更大 → 方向可靠性折扣
#   - DrugAI 每对方向置信度(drugai_conf): 直接作为方向权重, 不可靠方向自动降权
# ===========================================================================
DIRECTION_CONFIDENCE_WEIGHTING = _env_override("SM_DIR_CONF_WEIGHT",
    _nested_get(_cfg, "direction_weighting", "enable", default=True))
OOD_COMPLETION_PENALTY = _nested_get(_cfg, "direction_weighting", "ood_completion_penalty", default=0.25)
KEGG_DIRECTION_RELIABILITY = _nested_get(_cfg, "direction_weighting", "kegg_direction_reliability", default=0.5)
PPI_DIRECTION_RELIABILITY = _nested_get(_cfg, "direction_weighting", "ppi_direction_reliability", default=0.9)
PRESENCE_FLOOR = _nested_get(_cfg, "direction_weighting", "presence_floor", default=0.2)

# ===========================================================================
# HDCTI 参数
# ===========================================================================
HDCTI_USE = _env_override("SM_HDCTI_USE",
    _nested_get(_cfg, "hdcti", "use", default=True))
HDCTI_CONFIDENCE_THRESHOLD = _nested_get(_cfg, "hdcti", "confidence_threshold", default=0.85)  # 调优: 0.7→0.85, 早期宽松参数下HDCTI噪声稀释TMS信号; 收紧后与DrugAI协同为正向贡献
HDCTI_TOP_K = _nested_get(_cfg, "hdcti", "top_k", default=8)  # 调优: 15→8, 减少每成分的噪声靶标, 保留最可靠的
HDCTI_USE_TCMSP_TARGETS = _nested_get(_cfg, "hdcti", "use_tcmsp_targets", default=True)
HDCTI_TCMSP_SVM_THRESHOLD = _nested_get(_cfg, "hdcti", "tcmsp_svm_threshold", default=0.7)
HDCTI_TCMSP_RF_THRESHOLD = _nested_get(_cfg, "hdcti", "tcmsp_rf_threshold", default=0.7)
HDCTI_USE_GCN_PREDICTION = _env_override("SM_HDCTI_GCN", 
    _nested_get(_cfg, "hdcti", "use_gcn_prediction", default=True))
HDCTI_ZSCORE_THRESHOLD = _nested_get(_cfg, "hdcti", "zscore_threshold", default=None)  # GCN预测z-score全局过滤, None=关闭
HDCTI_SIMILARITY_METHOD = _nested_get(_cfg, "hdcti", "similarity_method", default="tanimoto")  # "tanimoto" | "jaccard"
KEGG_FILTER_ENABLED = _nested_get(_cfg, "hdcti", "kegg_filter", default=True)  # KEGG通路2跳过滤

# ===========================================================================
# GA 遗传算法参数
# ===========================================================================
GA_POP_SIZE = _env_override("SM_GA_POP_SIZE",
    _nested_get(_cfg, "ga", "pop_size", default=2000))
GA_N_HERBS = _nested_get(_cfg, "ga", "n_herbs", default=502)
GA_MAX_GENERATIONS = _env_override("SM_GA_MAX_GENERATIONS",
    _nested_get(_cfg, "ga", "max_generations", default=500))
GA_N_TOP = _nested_get(_cfg, "ga", "n_top", default=50)
GA_MAX_COMBO_SIZE = _nested_get(_cfg, "ga", "max_combo_size", default=6)
GA_CROSSOVER_RATE = _nested_get(_cfg, "ga", "crossover_rate", default=0.8)
GA_MUTATION_RATE = _nested_get(_cfg, "ga", "mutation_rate", default=0.15)
GA_ELITE_SIZE = _nested_get(_cfg, "ga", "elite_size", default=12)
GA_SIZE_PENALTY_LAMBDA = _env_override("SM_GA_SIZE_PENALTY_LAMBDA",
    _nested_get(_cfg, "ga", "size_penalty_lambda", default=0.001))
GA_HOF_SIZE = _nested_get(_cfg, "ga", "hof_size", default=50)
# 双通道 GA: 按疾病机制先验 (antagonist/engagement) 选择复方 TMS 方向
GA_DUAL_MECHANISM = _nested_get(_cfg, "ga", "dual_mechanism", default=True)
# 是否对全部疾病运行 GA (逐病生效双通道机制); False 则只跑目标疾病(更快)
GA_RUN_ALL_DISEASES = _nested_get(_cfg, "ga", "run_all_diseases", default=False)
GA_RESTART_STAGNATION = _nested_get(_cfg, "ga", "restart_stagnation", default=15)
GA_RESTART_FRACTION = _nested_get(_cfg, "ga", "restart_fraction", default=0.4)
GA_MAX_RESTARTS = _nested_get(_cfg, "ga", "max_restarts", default=8)
GA_EARLY_STOP_GENERATIONS = _nested_get(_cfg, "ga", "early_stop_generations", default=50)
GA_DYNAMIC_EARLY_STOP = _nested_get(_cfg, "ga", "dynamic_early_stop_generations", default=20)
GA_MIN_IMPROVEMENT_RATE = _nested_get(_cfg, "ga", "min_improvement_rate", default=0.0005)
GA_EARLY_STOP_WINDOW = _nested_get(_cfg, "ga", "early_stop_window", default=20)
GA_DIVERSITY_THRESHOLD = _nested_get(_cfg, "ga", "diversity_threshold", default=0.85)
GA_FITNESS_EPSILON = _nested_get(_cfg, "ga", "fitness_epsilon", default=1e-10)
GA_NORM_EPSILON = _nested_get(_cfg, "ga", "norm_epsilon", default=1e-12)

# 验证 GA 参数
for _name, _val, _lo, _hi in [
    ("GA_POP_SIZE", GA_POP_SIZE, 10, 10000),
    ("GA_MAX_GENERATIONS", GA_MAX_GENERATIONS, 1, 5000),
    ("GA_CROSSOVER_RATE", GA_CROSSOVER_RATE, 0.0, 1.0),
    ("GA_MUTATION_RATE", GA_MUTATION_RATE, 0.0, 1.0),
    ("GA_ELITE_SIZE", GA_ELITE_SIZE, 1, 100),
    ("GA_HOF_SIZE", GA_HOF_SIZE, 1, 1000),
    ("GA_RESTART_STAGNATION", GA_RESTART_STAGNATION, 1, 500),
    ("GA_RESTART_FRACTION", GA_RESTART_FRACTION, 0.0, 1.0),
    ("GA_MAX_RESTARTS", GA_MAX_RESTARTS, 1, 50),
]:
    assert _lo <= _val <= _hi, f"{_name}={_val} out of range [{_lo}, {_hi}]"

# ===========================================================================
# 时间感知 / 迁移学习
# ===========================================================================

TEMPORAL_ENABLED = _nested_get(_cfg, "temporal", "enabled", default=False)
TEMPORAL_METHOD = _nested_get(_cfg, "temporal", "method", default="recent")
TEMPORAL_DECAY_FACTOR = _nested_get(_cfg, "temporal", "decay_factor", default=0.7)
TEMPORAL_TIME_COL = _nested_get(_cfg, "temporal", "time_col", default="time_point")
TEMPORAL_TOP_N = _nested_get(_cfg, "temporal", "top_n", default=50)

TRANSFER_ENABLED = _nested_get(_cfg, "transfer", "enabled", default=False)
TRANSFER_LAMBDA = _nested_get(_cfg, "transfer", "lambda", default=0.3)
TRANSFER_MIN_SIMILARITY = _nested_get(_cfg, "transfer", "min_similarity", default=0.05)
TRANSFER_USE_PPI_PROPAGATION = _nested_get(_cfg, "transfer", "use_ppi_propagation", default=True)

# ===========================================================================
# 置换检验 / 评估参数
# ===========================================================================
PERMUTATION_EARLY_STOP_EXTREME = _nested_get(_cfg, "permutation", "early_stop_extreme", default=20)

HPG_EDGE_THRESHOLD = _nested_get(_cfg, "evaluation", "hpg_edge_threshold", default=1e-6)
HPG_TOP_K = _nested_get(_cfg, "evaluation", "hpg_top_k", default=500)
ENRICHMENT_BACKGROUND_GENES = _nested_get(_cfg, "evaluation", "enrichment_background_genes", default=20000)
PUBMED_TOTAL_ARTICLES = _nested_get(_cfg, "evaluation", "pubmed_total_articles", default=22188039)
JACCARD_GENE_SET_SIZES = _nested_get(_cfg, "evaluation", "jaccard_gene_set_sizes", default=[200, 400, 600, 800, 1000])

# ===========================================================================
# 输出配置
# ===========================================================================
OUTPUT_INCLUDE_TIMESTAMP = _nested_get(_cfg, "output", "include_timestamp", default=True)
OUTPUT_TIMESTAMP_FORMAT = _nested_get(_cfg, "output", "timestamp_format", default="%Y%m%d_%H%M%S")
OUTPUT_LOG_FILE = _nested_get(_cfg, "output", "log_file", default="run.log")


def get_run_output_dir(disease_name: str = "", create: bool = True) -> str:
    """
    生成标准化的运行输出目录（含时间戳）
    
    Parameters
    ----------
    disease_name : str
        疾病名称，用于目录命名
    create : bool
        是否自动创建目录
    
    Returns
    -------
    str : 输出目录路径
        格式: output/results/{disease}_{timestamp}/
    """
    from datetime import datetime
    timestamp = datetime.now().strftime(OUTPUT_TIMESTAMP_FORMAT)
    if disease_name:
        run_dir = os.path.join(RESULTS_DIR, f"{disease_name}_{timestamp}")
    else:
        run_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    if create:
        os.makedirs(run_dir, exist_ok=True)
    return run_dir


def print_config_summary():
    """打印关键配置摘要（用于日志头部）"""
    print("\n" + "=" * 60)
    print("  配置摘要")
    print("=" * 60)
    print(f"  扩散方法:      {DIFFUSION_METHOD}")
    print(f"  传播网络:      {NETWORK_TYPE}")
    print(f"  RWR重启概率:   {RWR_RESTART_PROB}")
    print(f"  PPI多源融合:   {PPI_MULTI_SOURCE}")
    print(f"  DrugAI:        {'启用' if DRUGAI_USE else '关闭'}")
    print(f"  HDCTI:         {'启用' if HDCTI_USE else '关闭'}")
    print(f"  GA最大代数:    {GA_MAX_GENERATIONS}")
    print(f"  GA种群大小:    {GA_POP_SIZE}")
    print(f"  GA组合上限:    {GA_MAX_COMBO_SIZE}味")
    print(f"  Top/Bottom K:  {TOP_BOTTOM_K}")
    print(f"  TMS方法:       {TMS_METHOD}")
    print(f"  GA双通道机制:  {GA_DUAL_MECHANISM}")
    print(f"  置换次数:      {PERMUTATION_N}")
    print("=" * 60)
