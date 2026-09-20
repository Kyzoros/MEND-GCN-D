# Systems-Mapping of Herbal Effects on Complex Diseases

## 项目概述

本代码是对论文 **"Systems-Mapping of Herbal Effects on Complex Diseases Using the Network-Perturbation Signatures"** (Chen et al., *Frontiers in Pharmacology*, 2018) 的 Python 复现实现。

### 论文方法核心框架

该研究提出了一种系统级计算框架，通过草药扰动签名与疾病病理状态特征的系统映射，建立草药与复杂疾病之间的联系：

1. **疾病病理状态谱 (Disease-Pathological State Profiles)**：从GEO数据库获取189种疾病的基因表达谱，差异分析后排序
2. **草药扰动签名 (Herb-Perturbation Signatures)**：基于TCMSP/ChEMBL/BindingDB的化合物-靶标数据，经两阶段热扩散模型生成
3. **HDmap-S (单草药-疾病映射)**：计算TMS (Total Matching Score) 评估草药对疾病的有效性
4. **HDmap-M (多草药组合映射)**：使用遗传算法搜索最优草药组合

## 项目结构

```
systems_mapping/
├── __init__.py                  # 包初始化
├── config.py                    # 配置参数
├── data_loader.py               # 数据加载模块
├── perturbation_signature.py    # 扰动签名构建 + 热扩散模型
├── hdmap_s.py                   # HDmap-S 单草药映射
├── hdmap_m.py                   # HDmap-M 遗传算法组合搜索
├── evaluation.py                # 评估验证模块
├── main.py                      # 主流程脚本
├── prepare_data.py              # 数据准备指导
├── requirements.txt             # 依赖
├── README.md                    # 本文件
├── data/
│   ├── raw/                     # 用户放置原始数据
│   └── processed/               # 预处理后数据
└── output/
    ├── results/                 # 分析结果
    └── figures/                 # 图表
```

## 安装

```bash
cd systems_mapping
pip install -r requirements.txt
```

## 需要准备的数据

**在运行代码前，您需要手动获取以下数据文件**（放入 `data/raw/` 目录）：

| 文件名 | 来源 | 描述 |
|--------|------|------|
| `disease_profiles.csv` | GEO数据库 | 疾病差异表达基因列表 |
| `herb_ingredients.csv` | TCMSP | 502种草药及其10,329个成分 |
| `ingredient_targets.csv` | ChEMBL v23 + BindingDB + WES | 成分-靶标关系 |
| `ppi_network.csv` | 整合PPI网络 | 13,460个蛋白, 141,296条互作 |
| `herb_metadata.csv` | TCMSP (可选) | 草药元信息 |

### 详细准备步骤

运行 `prepare_data.py` 查看详细的数据获取指南和模板:

```bash
python prepare_data.py
```

### 关于数据文件的详细说明

**1. disease_profiles.csv**
- 包含 `disease_id, gene_symbol, logFC, P_value` 四列
- 每个疾病有多个差异表达基因，按logFC降序排列
- 论文使用189个疾病、460个表达谱

**2. herb_ingredients.csv**
- 包含 `herb_id, herb_name, ingredient_id, ingredient_name, content_ratio` 五列
- 论文使用502个草药、10,329个成分
- 若成分含量未知，`content_ratio` 设为1.0

**3. ingredient_targets.csv**
- 包含 `ingredient_id, target_gene, action_type` 三列
- `action_type`: +1 (激活) 或 -1 (抑制)
- 非实验验证的靶标可以使用WES算法预测

**4. ppi_network.csv**
- 包含 `protein_A, protein_B` 两列
- 需要13,460个蛋白、141,296条互作
- 可以从STRING/BioGRID/InWeb等数据库组合获取

## 运行方法

### 步骤0: 获取疾病表达谱数据 (必须先完成!)

**推荐方式 — R脚本 (更稳定):**
```r
# 先安装依赖
install.packages(c("BiocManager", "data.table", "argparse"))
BiocManager::install(c("GEOquery", "limma"))

# 演示模式 (下载RA的2个数据集)
Rscript geo_download_R.R --demo

# 批量模式 (下载所有预定义疾病)
Rscript geo_download_R.R --batch

# 单个数据集
Rscript geo_download_R.R --gse GSE55235

# 自定义输出路径
Rscript geo_download_R.R --demo --output my_disease_data.csv
```

**Python方式 (备选):**
```bash
python geo_downloader.py --search "rheumatoid arthritis"
python geo_downloader.py --gse-list GSE55235,GSE77298
python geo_downloader.py --demo   # 演示模式
```

> **注意**: R版本的 `geo_download_R.R` 使用 limma 标准流程，结果与论文一致度更高，强烈推荐使用。

### 完整工作流

```bash
python main.py
```

### 分步运行

```python
# 在Python中导入
from data_loader import *
from perturbation_signature import *
from hdmap_s import HDmapS
from hdmap_m import HDmapM
from evaluation import *

# 1. 加载数据
disease_profiles = load_disease_profiles()
herb_ingredients = load_herb_ingredients()
ingredient_targets = load_ingredient_targets()
ppi_network = load_ppi_network()

# 2. 构建扰动签名
adj, gene2idx = build_ppi_adjacency(ppi_network)
builder = PerturbationSignatureBuilder(adj, gene2idx)
herb_signatures, builder = batch_build_herb_signatures(
    herb_ingredients, ingredient_targets, ppi_network, builder
)

# 3. HDmap-S 单草药映射
hdmap_s = HDmapS(herb_signatures, builder)
tms = hdmap_s.compute_tms("Herb_1", disease_genes_list)

# 4. HDmap-M 遗传算法搜索
hdmap_m = HDmapM(hdmap_s, list(herb_signatures.keys()), herb_signatures, builder)
result = hdmap_m.optimize(disease_genes_list)
```

## 关键模型参数

| 参数 | 论文值 | 说明 |
|------|--------|------|
| TOP_BOTTOM_K | 250 | 最优签名基因数 |
| GA_POP_SIZE | 500 | 遗传算法初始种群 |
| PERMUTATION_N | 10,000 | 置换检验次数 |
| GA_MAX_GENERATIONS | 200 | 最大遗传代数 |

## 输出结果

运行后将在 `output/` 目录生成:

- `results/herb_perturbation_signatures.csv` - 所有草药的扰动签名矩阵
- `results/hdmap_s_scores.csv` - HDmap-S得分矩阵 (草药×疾病)
- `results/hdmap_s_permutation.csv` - 置换检验P值
- `results/hdmap_m_results.csv` - 最优草药组合
- `results/hpg_network_RA.csv` - RA案例的H-PG网络 (可导入Cytoscape)

## 类风湿关节炎(RA)案例验证

论文以RA为例预测了一个新复方：
- Boschniakia Rossica (BR, 草苁蓉)
- Rhizoma Cimicifugae (RC, 升麻)
- Arisaematis Rhizoma (AR, 天南星)

代码将在 `step5` 中自动分析此复方的共同扰动基因和功能富集。

## 依赖

- numpy, pandas, scipy - 科学计算
- scikit-learn - 机器学习辅助
- matplotlib, seaborn - 可视化
- gseapy (可选) - 富集分析

## 引用

Chen X, Zheng C, Wang C, et al. Systems-Mapping of Herbal Effects on Complex Diseases Using the Network-Perturbation Signatures. *Frontiers in Pharmacology*, 2018, 9:1174.
