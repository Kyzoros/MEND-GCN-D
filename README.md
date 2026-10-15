# GCN-D: 基于图神经网络的草药-疾病关联预测模型

> 仓库名 `MEND-GCN-D`。本项目是一份**原创研究代码**：提出并训练GCN（图卷积网络）+ DistMult（可学习对角权重）解码器，在"草药—成分—靶基因—疾病"异构图上预测草药对疾病的治疗关联。**本文方法 ≠ 已有论文复现**；仓库中同时保留的无监督管线（扰动签名 / HDmap-S TMS / 遗传算法组合搜索）属于对比基线与数据基础。

---

## 项目概述

- **问题**：给定一种疾病和全库草药，预测哪些草药对该疾病具有治疗效果（二分类关联预测 / 排序问题）。
- **核心模型（本文方法，代码中命名为 GCN-D）**：
  1. 构建异构体 `草药(H)—成分(I)—靶基因(G)—疾病(D)`，可选加入 KEGG 基因-基因边；
  2. 两层 GCN 对对称归一化邻接矩阵做消息传递，得到各节点嵌入（含节点类型与基因差异表达等特征）；
  3. DistMult 解码器对每个 `(草药, 疾病)` 对打分：`s = Σ (z_h ⊙ w ⊙ z_d)`，`w` 为可学习对角权重；
  4. 纯二元交叉熵（BCE）损失 + 按疾病负采样训练。
- **代码位置**：模型定义在 [run_fair_comparison.py](mendgcn/run_fair_comparison.py) 的 `GCN` 类与 `train_gcn_model()` / `train_gcn()` 中，其余协议脚本均导入复用，全库统一。

---

## 方法与框架（与代码一一对应）

### 1. 异构图构建 — `build_graph()`（[run_fair_comparison.py](mendgcn/run_fair_comparison.py)）

| 项目 | 内容 |
|------|------|
| 节点 | 草药 H、成分 I、靶基因 G、疾病 D（4 类） |
| 边 | H–I（草药含成分）、I–G（成分作用靶基因）、G–D（疾病差异表达基因）；可选 KEGG G–G 边（`use_kegg=True`，来自 `kegg_network.build_kegg_adjacency()`）|
| 邻接矩阵 | 对称归一化 `A_sym = D^(-1/2) A D^(-1/2)`（加自环） |
| 节点特征 | 6 维：4 类节点类型的 one-hot + 度数开方的缩放特征 + 基因 |logFC| 表达特征（标准化） |

数据来源（`data/processed/`）：
- `herb_ingredients_processed.csv` — 草药-成分关系
- `ingredient_targets_enhanced_recovered.csv` — 成分-靶基因关系（可用环境变量 `SM_TGT_FILE` 覆盖，便于做清洗前后对照）
- `disease_profiles_processed.csv` — 疾病差异表达谱（`disease_id, gene_symbol, logFC, P_value`）

### 2. GCN 编码器 — `GCN` 类

```
h1 = ReLU( W1 · (A_sym @ X) )          # 第 1 层，输入 6 维 → 隐藏 64 维
z  = Wout · (A_sym @ h1)               # 第 2 层，64 维 → 输出 32 维
```

- `n_layers=1`：退化为无图传播的 MLP（结构消融对照）；`n_layers>=3`：更深版本。
- 默认超参：`hid=64, out=32, n_layers=2`。

### 3. 训练 — `train_gcn_model()` / `train_gcn()`

- 解码：`pos_s = Σ (z_h ⊙ z_d ⊙ w)`，DistMult 可学权重 `w`（32 维，初始化为 1）；`decoder="dot"` 时退化为纯点积（消融）。
- 负采样：每病每个正例按 `k_neg=5` 从该病候选草药中采样（通过 `build_exclude()` 排除该病全部已知正例，防止泄漏）。
- 损失：`L = −mean(log σ(pos_s)) − mean(log(1−σ(neg_s)))`（纯 BCE）。
- 优化：Adam `lr=1e-2`；epochs 冒烟 300 / 全量 3000；`torch.manual_seed` 固定随机性。
- 打分：`S = (z_h ⊙ w) @ z_dᵀ` 后按矩阵均值/标准差标准化。

### 4. 评估 — `evaluate()`

- `FULL`：正例 = 金标准中该病已知治疗草药，负例 = 全库其余草药；
- `labeled`：仅在金标准中有显式 `label∈{0,1}` 的草药上计算；
- 指标：AUROC（每病 + 8 病平均），另含留出泛化 AUROC。

---

## 评估协议（三种泛化设置）

| 协议 | 说明 | 脚本 | 输出表 |
|------|------|------|--------|
| **P1 · 5 折 transductive** | 306 个正例对划 5 折轮流留出，汇总后报 pair 级 AUROC / AUPRC / MRR / Hits@10 / Hits@20，bootstrap 95% CI | [run_cv_final.py](mendgcn/run_cv_final.py) | `Table_cv_clean.csv` |
| **P2 · LODO（留一病）** | 依次留出 1 个病：训练图彻底删除该病节点及其基因连边，冻结权重在全图推理（真 inductive） | [run_gnn_lodo_coldstart.py](experiments/run_gnn_lodo_coldstart.py) `--stage lodo` | `Table_lodo.csv` (+`_per_disease`) |
| **P3 · 冷启动草药** | 5 折各留出 1/5 草药节点及其成分连边，靠成分→基因→病传播给全新草药打分 | [run_gnn_lodo_coldstart.py](experiments/run_gnn_lodo_coldstart.py) `--stage coldstart` | `Table_coldstart.csv` (+`_per_fold`) |
| **公平对比主表** | 留出 1/5 正例对；Fisher / 网络邻近 / 靶点数量 / GCN-D 四方法同数据同评估 | [run_fair_comparison.py](mendgcn/run_fair_comparison.py) | `Table_fair_comparison.csv` |
| **定稿公平基线 + 干净留出 + 消融** | 6 方法（含 MLP 无图对照）干净对级评估、epoch 扫描、结构消融 | [run_fair_final.py](mendgcn/run_fair_final.py) | `Table_fair_final.csv`、`Table_holdout_clean.csv`、`Table_epoch_overfit.csv`、`Table_gnn_ablation2.csv` |
| **8 病扩展评估** | 新 4 病 curated AUC 与独立验证合并 | [run_8disease_curated_auc.py](experiments/run_8disease_curated_auc.py)、[run_8disease_master_table.py](experiments/run_8disease_master_table.py) | `Table_8disease_curated_auc.csv`、`Table_8disease_master.csv` |

## 金标准与疾病

金标准文件：`data/processed/gold_standard_final_8d_supplemented.csv`（8 病共 595 行，`label∈{0,1}`）：

- 老 4 病（人工金标准）：AD、PAH、RA、T2D
- 新 4 病（TCMBank indication 提取正例）：HCC、CRC、ASTHMA、OA

---

## 对比基线

全部在同一份数据、同一套评估代码下对比：

| 基线 | 思想 | 代码 |
|------|------|------|
| Fisher 富集 (D-TMS) | 草药靶点集与疾病基因集的富集显著性 | `score_fisher()`（[run_fair_comparison.py](mendgcn/run_fair_comparison.py)） |
| 网络邻近度 (Barabasi) | KEGG 网络上疾病基因到草药靶点的度校正 z-score | `score_proximity()`（同上） |
| 靶点数量 | 平凡基线：只数靶点数、无疾病特异性 | `score_targetcount()`（同上） |
| RWR 扩散 | 以疾病为种子在异构图上重启随机游走，稳态概率为分 | `score_rwr()`（[run_fair_final.py](mendgcn/run_fair_final.py)） |
| MLP 无图传播 | `n_layers=1` 的 GCN-D 对照，剥离图传播贡献 | [run_fair_final.py](mendgcn/run_fair_final.py) |

另保留完整的**无监督系统映射管线** [main.py](baselines/main.py)：RWR 草药扰动签名（`perturbation_signature.py`）→ HDmap-S TMS 得分与置换检验（`hdmap_s.py`）→ HDmap-M 遗传算法组合搜索（`hdmap_m.py`）→ H–PG 网络评估（`evaluation.py` / `evaluation_metrics.py`），作为传统方法的对照实现。

---

## 数据管线

- **靶标扩展与方向预测**（数据增强，非主模型）：
  - `hdcti_adapter.py` — 基于 TCMSP 支持向量机/随机森林与 GCN 图传播预测成分-蛋白新靶标；
  - `drugai_adapter.py` — DrugAI 深度学习模型（AttentiveFP + 1D-CNN）预测成分-靶标作用方向（激活/抑制）；
  - `gcn_target_expansion.py` — 草药-靶标二部图 GCN 靶标扩展对比实验。
- **KEGG 网络**：`kegg_network.py` 构建基因-基因 KEGG 邻接与归一化缓存。
- **注意**：原始数据与实验产出（`data/`、`output/`）不在本仓库内；数据获取与预处理脚本保留在本地未上传。运行主实验前需自行准备 `data/processed/` 下的数据文件（路径见 [config.py](config.py) / [config.yaml](config.yaml)）。

---

## 安装与快速开始

```bash
pip install -r requirements.txt
```

核心依赖：numpy / pandas / scipy / scikit-learn / matplotlib / pyyaml / requests；深度模型需 `torch`（可选：joblib、numba、gseapy、seaborn，见 [requirements.txt](requirements.txt)）。

在仓库根目录执行（脚本内部会自动把仓库根加入 `sys.path` 并据此解析数据/输出路径）：

```bash
# 公平对比主表（4 方法 × 8 病，冒烟参数）
python mendgcn/run_fair_comparison.py

# P1：5 折 transductive 对级评估（全量）
python mendgcn/run_cv_final.py --epochs 3000

# P2 / P3：留一病 与 冷启动草药（全量）
python experiments/run_gnn_lodo_coldstart.py --stage lodo --epochs 3000
python experiments/run_gnn_lodo_coldstart.py --stage coldstart --epochs 3000

# 定稿公平基线 + 干净留出 + 消融（4 张论文表）
python mendgcn/run_fair_final.py --epochs 3000 --seeds 0 1 2

# 无监督系统映射管线（对照方法）
python baselines/main.py --disease PAH
```

默认超参（所有 GCN-D 协议一致）：`lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2`。

---

## 仓库结构

```
systems_mapping/
├── README.md / ARCHITECTURE.md   # 项目说明 / 架构文档
├── requirements.txt              # 依赖
├── config.py / config.yaml       # 统一配置（数据路径、图/评估/增强参数）—— 留在根目录
├── .gitignore
│
├── mendgcn/                      # ★ 核心库：配置/数据/图构建/GCN-D 模型/评估/适配器
│   ├── run_fair_comparison.py    # ★ GCN 类 / build_graph / train_gcn / 公平对比主表
│   ├── run_cv_final.py           # ★ P1：5 折 transductive 对级评估
│   ├── run_fair_final.py         # ★ 定稿公平基线 + 干净留出 + 消融（4 表）
│   ├── run_gnn_final.py          #   定稿 epoch 扫描 / 多种子
│   ├── run_gnn_ablation.py       #   GNN 结构消融
│   ├── data_loader.py            #   数据加载 + HDCTI/DrugAI 增强集成
│   ├── kegg_network.py           #   KEGG 基因网络邻接与缓存
│   ├── kegg_pathway_filter.py    #   KEGG 通路过滤（data_loader 依赖）
│   ├── hdmap_s.py / hdmap_m.py   #   无监督基线：TMS 打分 / 遗传算法组合搜索
│   ├── perturbation_signature.py #   RWR / 热扩散扰动签名（基线组件）
│   ├── evaluation.py / evaluation_metrics.py  # 无监督评估（H-PG 网络等）
│   ├── tms_network_proximity.py  #   网络邻近度（基线）
│   ├── hdcti_adapter.py / drugai_adapter.py   # 靶标扩展 / 方向预测适配器
│   ├── gcn_target_expansion.py   #   二部图 GCN 靶标扩展对比实验
│   └── experimental_mechanisms.py / temporal_signature.py / transfer_learning.py
│
├── experiments/                  # 实验入口（论文表 / 扩展 / 验证）
│   ├── run_gnn_lodo_coldstart.py # ★ P2 LODO / P3 冷启动（inductive）
│   ├── run_8disease_*.py         #   8 病扩展评估（curated AUC / 主表 / top50）
│   ├── run_D5_* / run_D6_* / run_D7_* / run_D8_*.py   # 论文 D 组实验
│   ├── run_P0_* / run_P0a_* / run_P0b_* / run_positive_control.py  # P0 独立验证与阳性对照
│   ├── run_ablation.py / run_ablation_enrichment.py   # 组件与富集消融
│   ├── run_baseline_comparison.py / run_baseline_holdout.py  # 基线与留出对比
│   ├── run_gcn_case_predict.py / run_gcn_d_scores.py / interpret_gcn_d.py  # GCN-D 案例、打分与解释
│   ├── run_herb_external_validation.py / run_tms_independent_validation.py  # 外部验证
│   ├── compare_diffusion_rwr_vs_thermal.py / compare_ppi_vs_kegg.py / compare_target_expansion.py
│   └── run_coldstart.py / run_k_stratified_eval.py / run_conf_threshold_lodo.py /
│       run_final_tables.py / run_multidisease_date.py / run_pah_analysis.py /
│       run_tms_size_correction.py / run_B_probe.py / run_C_goldstandard_check.py /
│       method_comparison.py / permutation_check.py
│
├── baselines/                    # 无监督系统映射管线（对照方法）
│   ├── main.py                   # ★ 管线编排（扰动签名→HDmap-S→HDmap-M→评估）
│   ├── generate_random_baseline.py / generate_cosine_baseline.py /
│   │   generate_netprox_baseline.py / generate_dual_outputs.py  # 基线生成
│   └── evaluate_dual_channel.py / evaluate_gold_standard.py     # 基线评估
│
├── diagnostics/                  # 验证 / 诊断 / 统计（非论文主线）
│   ├── verify_gnn.py / verify_reproducibility.py / verify_reversal_hypothesis.py
│   ├── pre_run_check.py / check_drugai_gold_overlap.py / validate_drugai_direction.py
│   ├── direction_calibration.py / diagnose_tms_formula.py /
│   │   diagnose_target_count_confound.py / ablation_nongene_targets.py
│   └── mc2_stats.py / mc6_ablation_seeds.py / mc6_supervised.py /
│       run_confound_check.py / run_multigold_confound_check.py
│
├── preprocessing/                # 数据获取与预处理脚本（本地保留，不上传）
└── manuscript/                   # 论文排版辅助脚本（本地保留，不上传）
```

> 数据获取 / 预处理脚本（`preprocessing/`）与论文排版脚本（`manuscript/`）**本地保留但不上传**；仓库中只放模型与实验代码。所有入口脚本均把自身目录、仓库根、`mendgcn/` 加入 `sys.path`，因此可在任意工作目录直接运行。

## 输出产物

各协议脚本统一写到 `output/results/paper_deliverables/`（路径见 `config.py`），包括：

- `Table_fair_comparison.csv` / `Table_fair_final.csv` — 公平对比主表
- `Table_cv_clean.csv` — P1 交叉验证对级指标（含 bootstrap 95% CI）
- `Table_lodo.csv` / `Table_coldstart.csv` — P2 / P3 inductive 结果
- `Table_holdout_clean.csv` / `Table_epoch_overfit.csv` / `Table_gnn_ablation2.csv` — 干净留出、过拟合扫描、结构消融
- `Table_8disease_curated_auc.csv` / `Table_8disease_master.csv` — 8 病扩展评估