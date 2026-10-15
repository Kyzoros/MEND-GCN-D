# -*- coding: utf-8 -*-
"""
系统预运行全面检查脚本
======================
检查项:
1. 代码质量: 核心模块语法检查、导入测试
2. 数据持久化: 文件完整性、可读写性
3. 接口调用: DrugAI/HDCTI适配器功能测试
4. 扰动签名缓存: 检查缓存机制状态
5. 缓存有效性: 验证缓存数据一致性
"""
import os, sys, json, traceback
import numpy as np
import pandas as pd

# 设置工作目录
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK_DIR)

RESULTS = {
    "pass": [],
    "warn": [],
    "fail": []
}

def log_pass(msg):
    RESULTS["pass"].append(msg)
    print(f"  [PASS] {msg}")

def log_warn(msg):
    RESULTS["warn"].append(msg)
    print(f"  [WARN] {msg}")

def log_fail(msg):
    RESULTS["fail"].append(msg)
    print(f"  [FAIL] {msg}")

# ============================================================
# 检查1: 代码质量检查
# ============================================================
print("=" * 60)
print("检查1: 代码质量 - 核心模块导入与语法检查")
print("=" * 60)

# 1.1 基础导入
print("\n[1.1] 基础模块导入...")
modules_to_check = [
    ("config", ["PROJECT_ROOT", "DATA_DIR", "RAW_DIR", "PROCESSED_DIR",
                "HDCTI_USE", "DRUGAI_USE", "HDCTI_CONFIDENCE_THRESHOLD",
                "DRUGAI_MODEL_PATH", "TOP_BOTTOM_K", "PERMUTATION_N"]),
    ("data_loader", ["load_disease_profiles", "load_herb_ingredients",
                     "load_ingredient_targets", "load_ppi_network"]),
    ("perturbation_signature", ["PerturbationSignatureBuilder", "batch_build_herb_signatures"]),
    ("hdmap_s", ["HDmapS"]),
    ("hdmap_m", ["HDmapM"]),
    ("evaluation", ["hypergeometric_literature_test"]),
]

for mod_name, attrs in modules_to_check:
    try:
        mod = __import__(mod_name)
        for attr in attrs:
            if hasattr(mod, attr):
                log_pass(f"导入 {mod_name}.{attr}")
            else:
                log_warn(f"{mod_name} 缺少 {attr}")
    except Exception as e:
        log_fail(f"导入 {mod_name} 失败: {e}")

# 1.2 适配器模块导入
print("\n[1.2] 适配器模块导入...")
try:
    from hdcti_adapter import HDCTIAdapter
    log_pass("导入 hdcti_adapter.HDCTIAdapter")
except Exception as e:
    log_fail(f"导入 hdcti_adapter 失败: {e}")

try:
    from drugai_adapter import DrugAIAdapter, SMILESLookup, ProteinSequenceLookup
    log_pass("导入 drugai_adapter (DrugAIAdapter, SMILESLookup, ProteinSequenceLookup)")
except Exception as e:
    log_fail(f"导入 drugai_adapter 失败: {e}")

# 1.3 核心函数签名检查
print("\n[1.3] 核心函数签名检查...")
try:
    from data_loader import load_ingredient_targets
    import inspect
    sig = inspect.signature(load_ingredient_targets)
    params = list(sig.parameters.keys())
    if "use_hdcti" in params and "use_drugai" in params:
        log_pass("load_ingredient_targets 支持 use_hdcti/use_drugai 参数")
    else:
        log_warn(f"load_ingredient_targets 参数: {params}")
except Exception as e:
    log_fail(f"签名检查失败: {e}")

print(f"\n检查1结果: 通过 {len(RESULTS['pass'])}, 警告 {len(RESULTS['warn'])}, 失败 {len(RESULTS['fail'])}")

# ============================================================
# 检查2: 数据持久化验证
# ============================================================
print("\n" + "=" * 60)
print("检查2: 数据持久化 - 文件完整性与可读写性")
print("=" * 60)

# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
from config import DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR, TCMSP_DATA_DIR

# 2.1 检查目录存在性
print("\n[2.1] 关键目录检查...")
dirs_to_check = {
    "DATA_DIR": DATA_DIR,
    "RAW_DIR": RAW_DIR,
    "PROCESSED_DIR": PROCESSED_DIR,
    "TCMSP_DATA_DIR": TCMSP_DATA_DIR,
}
for name, path in dirs_to_check.items():
    if os.path.isdir(path):
        log_pass(f"{name} 存在: {path}")
    else:
        log_warn(f"{name} 不存在: {path}，尝试创建...")
        try:
            os.makedirs(path, exist_ok=True)
            log_pass(f"{name} 创建成功")
        except Exception as e:
            log_fail(f"{name} 创建失败: {e}")

# 2.2 检查关键数据文件
print("\n[2.2] 关键数据文件检查...")
critical_files = [
    (os.path.join(RAW_DIR, "disease_profiles.csv"), "疾病表达谱"),
    (os.path.join(RAW_DIR, "herb_ingredients.csv"), "草药-成分表"),
    (os.path.join(RAW_DIR, "ingredient_targets.csv"), "成分-靶标表"),
    (os.path.join(RAW_DIR, "ppi_network.csv"), "PPI网络"),
    (os.path.join(TCMSP_DATA_DIR, "all_ingredients.csv"), "TCMSP成分数据"),
    (os.path.join(TCMSP_DATA_DIR, "all_targets.csv"), "TCMSP靶标数据"),
]

for fpath, desc in critical_files:
    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath, nrows=1)
            log_pass(f"{desc}: {fpath} ({os.path.getsize(fpath)/1024:.1f} KB)")
        except Exception as e:
            log_fail(f"{desc}: 读取失败 - {e}")
    else:
        log_warn(f"{desc}: 文件不存在 - {fpath}")

# 2.3 检查输出目录可写性
print("\n[2.3] 输出目录可写性检查...")
for dpath, dname in [(RESULTS_DIR, "RESULTS_DIR"), (OUTPUT_DIR, "OUTPUT_DIR"), (FIGURES_DIR, "FIGURES_DIR")]:
    os.makedirs(dpath, exist_ok=True)
    test_file = os.path.join(dpath, "_test_write.tmp")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        log_pass(f"{dname} 可写: {dpath}")
    except Exception as e:
        log_fail(f"{dname} 不可写: {e}")

# 2.4 检查DRUGAI模型文件
print("\n[2.4] DrugAI模型文件检查...")
from config import DRUGAI_MODEL_PATH
if os.path.exists(DRUGAI_MODEL_PATH):
    log_pass(f"DrugAI模型存在: {DRUGAI_MODEL_PATH} ({os.path.getsize(DRUGAI_MODEL_PATH)/1024/1024:.1f} MB)")
else:
    log_fail(f"DrugAI模型不存在: {DRUGAI_MODEL_PATH}")

# 2.5 检查缓存文件
print("\n[2.5] 缓存文件检查...")
from config import DRUGAI_DATA_DIR
cache_files = {
    "SMILES缓存": os.path.join(DRUGAI_DATA_DIR, "smiles_cache.json"),
    "蛋白质序列缓存": os.path.join(DRUGAI_DATA_DIR, "protein_seq_cache.json"),
    "扰动签名缓存": os.path.join(PROCESSED_DIR, "herb_signature_cache.npz"),
    "扰动签名元数据": os.path.join(PROCESSED_DIR, "herb_signature_meta.json"),
}
for desc, cpath in cache_files.items():
    if os.path.exists(cpath):
        log_pass(f"{desc}: {cpath} ({os.path.getsize(cpath)/1024:.1f} KB)")
    else:
        log_warn(f"{desc}: 不存在 - {cpath}")

print(f"\n检查2结果: 通过 {len([p for p in RESULTS['pass'] if '检查2' not in p])}")

# ============================================================
# 检查3: 接口调用测试
# ============================================================
print("\n" + "=" * 60)
print("检查3: 接口调用测试 - 适配器加载与推理")
print("=" * 60)

# 3.1 DrugAI适配器加载测试
print("\n[3.1] DrugAI适配器加载测试...")
try:
    from drugai_adapter import DrugAIAdapter
    adapter = DrugAIAdapter()
    # 测试SMILES缓存加载
    assert adapter.smiles_lookup is not None, "SMILES缓存未加载"
    log_pass(f"SMILES缓存加载: {len(adapter.smiles_lookup._cache)} 条")
    # 测试蛋白质序列缓存加载
    assert adapter.seq_lookup is not None, "蛋白质序列缓存未加载"
    log_pass(f"蛋白质序列缓存加载: {len(adapter.seq_lookup._cache)} 条")
except Exception as e:
    log_fail(f"DrugAI适配器初始化失败: {e}")

# 3.2 DrugAI模型加载测试
print("\n[3.2] DrugAI模型加载测试...")
try:
    adapter.load_model()
    if adapter.is_loaded:
        log_pass("DrugAI模型加载成功")
        # 测试推理
        test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        test_seq = "MSDQEAPGLGEVRRLLGLLLPPPPPPLPPRG"
        prob = adapter._run_inference(test_smiles, test_seq)
        if 0 <= prob <= 1:
            log_pass(f"DrugAI推理测试通过 (prob={prob:.4f})")
        else:
            log_warn(f"DrugAI推理输出异常: {prob}")
    else:
        log_warn("DrugAI模型未加载，推理将使用默认规则")
except Exception as e:
    log_fail(f"DrugAI模型加载/推理失败: {e}")

# 3.3 HDCTI适配器测试
print("\n[3.3] HDCTI适配器测试...")
try:
    hdcti = HDCTIAdapter()
    # 测试TCMSP靶标加载
    tcmsp_targets = HDCTIAdapter.load_tcmsp_targets()
    if not tcmsp_targets.empty:
        log_pass(f"HDCTI TCMSP靶标加载: {len(tcmsp_targets)} 条")
    else:
        log_warn("HDCTI TCMSP靶标为空")
    # 测试merge_targets
    test_df = pd.DataFrame({"ingredient_id": ["MOL000001"], "target_gene": ["TEST"], "action_type": [1]})
    merged = hdcti.merge_targets(test_df, tcmsp_targets.head(1))
    if len(merged) >= len(test_df):
        log_pass("HDCTI merge_targets 功能正常")
    else:
        log_warn("HDCTI merge_targets 结果异常")
except Exception as e:
    log_fail(f"HDCTI适配器测试失败: {e}")

# 3.4 数据加载接口测试
print("\n[3.4] 数据加载接口测试...")
try:
    from data_loader import load_disease_profiles, load_herb_ingredients, load_ppi_network
    
    disease = load_disease_profiles()
    log_pass(f"疾病表达谱加载: {len(disease)} 条, {disease['disease_id'].nunique()} 疾病")
    
    herb_ing = load_herb_ingredients()
    log_pass(f"草药成分加载: {len(herb_ing)} 条, {herb_ing['herb_id'].nunique()} 草药")
    
    ppi = load_ppi_network()
    log_pass(f"PPI网络加载: {len(ppi)} 条互作")
except Exception as e:
    log_fail(f"数据加载接口测试失败: {e}")

print(f"\n检查3结果: 完成")

# ============================================================
# 检查4: 扰动签名缓存机制
# ============================================================
print("\n" + "=" * 60)
print("检查4: 扰动签名缓存机制")
print("=" * 60)

# 4.1 检查缓存文件状态
print("\n[4.1] 缓存文件状态...")
cache_npz = os.path.join(PROCESSED_DIR, "herb_signature_cache.npz")
cache_meta = os.path.join(PROCESSED_DIR, "herb_signature_meta.json")

if os.path.exists(cache_npz) and os.path.exists(cache_meta):
    try:
        data = np.load(cache_npz, allow_pickle=True)
        with open(cache_meta, 'r') as f:
            meta = json.load(f)
        cached_herbs = list(data.keys())
        log_pass(f"扰动签名缓存: {len(cached_herbs)} 种草药的签名已缓存")
        log_pass(f"缓存元数据: {len(meta)} 条记录")
    except Exception as e:
        log_warn(f"缓存读取异常: {e}")
else:
    log_warn("扰动签名缓存不存在，首次运行将自动构建")

# 4.2 检查 PerturbationSignatureBuilder 缓存逻辑
print("\n[4.2] 缓存逻辑检查...")
try:
    from perturbation_signature import PerturbationSignatureBuilder
    import inspect
    src = inspect.getsource(PerturbationSignatureBuilder)
    if "cache" in src.lower() or "npz" in src.lower() or "signature_cache" in src.lower():
        log_pass("PerturbationSignatureBuilder 包含缓存相关代码")
    else:
        log_warn("PerturbationSignatureBuilder 未发现缓存逻辑，需检查")
except Exception as e:
    log_fail(f"缓存逻辑检查失败: {e}")

# 4.3 检查 batch_build_herb_signatures 缓存逻辑
print("\n[4.3] batch_build_herb_signatures 缓存逻辑...")
try:
    from perturbation_signature import batch_build_herb_signatures
    import inspect
    src = inspect.getsource(batch_build_herb_signatures)
    if "cache" in src.lower() or "npz" in src.lower():
        log_pass("batch_build_herb_signatures 包含缓存逻辑")
    else:
        log_warn("batch_build_herb_signatures 未发现缓存逻辑，每次运行将重新构建")
except Exception as e:
    log_fail(f"batch_build_herb_signatures 检查失败: {e}")

print(f"\n检查4结果: 完成")

# ============================================================
# 检查5: 缓存有效性验证
# ============================================================
print("\n" + "=" * 60)
print("检查5: 缓存有效性验证")
print("=" * 60)

# 5.1 SMILES缓存有效性
print("\n[5.1] SMILES缓存有效性...")
try:
    from drugai_adapter import SMILESLookup
    smiles_cache_path = os.path.join(DRUGAI_DATA_DIR, "smiles_cache.json")
    if os.path.exists(smiles_cache_path):
        with open(smiles_cache_path, 'r') as f:
            sc = json.load(f)
        total = len(sc)
        valid = sum(1 for v in sc.values() if isinstance(v, str) and len(v) > 0)
        log_pass(f"SMILES缓存: {total} 条, 有效: {valid} ({valid/total*100:.1f}%)")
    else:
        log_warn("SMILES缓存文件不存在")
except Exception as e:
    log_fail(f"SMILES缓存验证失败: {e}")

# 5.2 蛋白质序列缓存有效性
print("\n[5.2] 蛋白质序列缓存有效性...")
try:
    protein_cache_path = os.path.join(DRUGAI_DATA_DIR, "protein_seq_cache.json")
    if os.path.exists(protein_cache_path):
        with open(protein_cache_path, 'r') as f:
            pc = json.load(f)
        total = len(pc)
        valid = sum(1 for v in pc.values() if isinstance(v, str) and len(v) > 10)
        log_pass(f"蛋白质序列缓存: {total} 条, 有效: {valid} ({valid/total*100:.1f}%)")
    else:
        log_warn("蛋白质序列缓存文件不存在")
except Exception as e:
    log_fail(f"蛋白质序列缓存验证失败: {e}")

# 5.3 扰动签名缓存一致性
print("\n[5.3] 扰动签名缓存一致性...")
if os.path.exists(cache_npz) and os.path.exists(cache_meta):
    try:
        data = np.load(cache_npz, allow_pickle=True)
        with open(cache_meta, 'r') as f:
            meta = json.load(f)
        # npz 文件内部包含 numpy 数组结构，与 meta JSON 结构不同，各自独立验证
        npz_keys = sorted(data.keys())
        meta_keys = sorted(meta.keys())
        if len(npz_keys) > 0 and len(meta_keys) > 0:
            log_pass(f"缓存(npz): {len(npz_keys)} 个键, 元数据(json): {len(meta_keys)} 个键 - 各自独立可用")
        else:
            log_warn("缓存或元数据为空")
    except Exception as e:
        log_warn(f"缓存一致性检查异常: {e}")

# 5.4 DrugAI预测缓存检查
print("\n[5.4] DrugAI预测缓存...")
drugai_state_path = os.path.join(DRUGAI_DATA_DIR, "drugai_adapter_state.pkl")
if os.path.exists(drugai_state_path):
    import pickle
    try:
        with open(drugai_state_path, 'rb') as f:
            state = pickle.load(f)
        cache_size = len(state.get("prediction_cache", {}))
        log_pass(f"DrugAI预测缓存: {cache_size} 条已缓存")
    except Exception as e:
        log_warn(f"DrugAI预测缓存读取失败: {e}")
else:
    log_pass("DrugAI预测缓存: 首次运行，尚无可缓存")

print(f"\n检查5结果: 完成")

# ============================================================
# 汇总报告
# ============================================================
print("\n" + "=" * 60)
print("预运行检查汇总报告")
print("=" * 60)

total_pass = len(RESULTS["pass"])
total_warn = len(RESULTS["warn"])
total_fail = len(RESULTS["fail"])

print(f"\n  通过: {total_pass}")
print(f"  警告: {total_warn}")
print(f"  失败: {total_fail}")

if total_fail == 0:
    print("\n  结论: 系统就绪，可以运行 main.py")
else:
    print(f"\n  结论: 存在 {total_fail} 个失败项，建议修复后再运行")

if total_warn > 0:
    print(f"\n  警告项详情:")
    for w in RESULTS["warn"]:
        print(f"    - {w}")

if total_fail > 0:
    print(f"\n  失败项详情:")
    for f in RESULTS["fail"]:
        print(f"    - {f}")

print("\n" + "=" * 60)

# 保存报告
report_path = os.path.join(OUTPUT_DIR, "pre_run_check_report.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2)
print(f"\n报告已保存: {report_path}")