# Systems-Mapping 模型架构文档

## 项目概述

**Systems-Mapping of Herbal Effects on Complex Diseases** 是一个计算系统药理学管道，通过PPI网络传播分析草药成分对复杂疾病的扰动效应，并利用遗传算法搜索最优草药组合。

论文方法学基础：Chen et al., *Frontiers in Pharmacology*, 2018

---

## 一、总体数据流

```
┌─────────────────────────────────────────────────────────────┐
│                         输入数据                              │
│  disease_profiles.csv   herb_ingredients.csv                  │
│  ingredient_targets.csv  ppi_network.csv                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  [步骤1] data_loader.py                                      │
│  数据加载 + HDCTI靶标扩展 + DrugAI方向预测                     │
│  输出: ingredient_targets_enhanced.csv (262,269条靶标对)        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  [步骤2] perturbation_signature.py                           │
│  PPI邻接矩阵构建 → 成分级初始向量 → 草药级叠加 → 热扩散/RWR    │
│  输出: herb_perturbation_signatures.csv (499×16,201矩阵)      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  [步骤3] hdmap_s.py                                          │
│  TMS得分计算 (Top/Bottom K基因 + 疾病基因排位)                │
│  置换检验 (P值) + 可选跨疾病迁移学习                          │
│  输出: hdmap_s_scores.csv, {DISEASE}_top50_herbs.csv          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  [步骤4] hdmap_m.py                                          │
│  遗传算法搜索最优多草药组合 (Hall of Fame + 种群重启 + 早停)    │
│  输出: hdmap_m_results.csv, hdmap_m_topk_results.csv          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  [步骤5] evaluation_metrics.py + evaluation.py                │
│  H-PG网络、共同基因分析、Jaccard、PubMed验证                   │
│  输出: hpg_network.csv, common_genes.txt, run_metadata.json   │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、配置系统

### 2.1 config.py + config.yaml

**职责**：统一配置中心，三层优先级：`命令行 > 环境变量 > YAML > 代码默认值`

**关键参数一览**：

| 分类 | 参数 | 当前值 | 说明 |
|------|------|--------|------|
| HDmap-S | `TOP_BOTTOM_K` | 250 | 选取正/负签名最强的K个基因 |
| | `PERMUTATION_N` | 100 | 置换检验次数 |
| 扩散 | `DIFFUSION_METHOD` | `"rwr"` | 扩散算法 (`"rwr"` 或 `"thermal"`) |
| | `RWR_RESTART_PROB` | 0.5 | RWR 重启概率 |
| | `THERMAL_ALPHA` | 1.0 | 热扩散系数 |
| PPI | `PPI_MULTI_SOURCE` | `true` | 多源 PPI 融合 |
| | `PPI_N_PROTEINS` | 13,460 | PPI 网络蛋白数 |
| DrugAI | `DRUGAI_USE` | `true` | DrugAI 方向预测开关 |
| HDCTI | `HDCTI_USE` | `true` | HDCTI 靶标扩展开关 |
| GA | `GA_POP_SIZE` | 500 | 种群大小 |
| | `GA_MAX_GENERATIONS` | 200 | 最大迭代代数 |
| | `GA_MAX_COMBO_SIZE` | 4 | 组合最大草药数 |
| | `GA_SIZE_PENALTY_LAMBDA` | 0.01 | 组合大小正则化系数 |
| 迁移 | `TRANSFER_ENABLED` | `false` | 跨疾病迁移学习 |

---

## 三、核心模块详解

### 3.1 data_loader.py — 数据加载与增强

**职责**：加载原始数据、预处理、集成 HDCTI & DrugAI 进行靶标扩展和方向预测

**输入（raw/ 目录）**：
| 文件 | 格式 | 内容 |
|------|------|------|
| `disease_profiles.csv` | `gene_symbol, logFC, P_value` | AD差异表达基因 (17,654个) |
| `herb_ingredients.csv` | `herb_id, ingredient_id, content_ratio` | 499草药-13,728成分 |
| `ingredient_targets.csv` | `ingredient_id, target_gene, action_type` | 36,379条成分-靶标 |
| `ppi_network.csv` | `protein_A, protein_B` | 141,296条PPI互作 |

**增强流程**（仅当 `DRUGAI_USE`/`HDCTI_USE` 为 `true` 时执行）：

```
原始成分-靶标 (36,379条)
    │
    ├──[HDCTI]──→ TCMSP SVM/RF 靶标补充 (142,667条)
    │                 → GCN 图传播预测新靶标
    │                 → 合并去重 →
    │
    ├──[DrugAI]─→ SMILES查询 + 蛋白序列查询 
    │                → DrugAI模型推理(AttentiveFP + 1D-CNN)
    │                → 三分类阈值: prob≥0.7→+1, prob≤0.3→-1, else→保留原值
    │                → 覆写action_type列
    │
    └──→ ingredient_targets_enhanced.csv (262,269条)
           ├─ original (ChEMBL/BindingDB): 36,379条, 有方向
           └─ HDCTI_predicted (GCN): 225,890条, 方向=0(未知)
```

**关键函数**：
- `load_ingredient_targets(use_hdcti, use_drugai)` — 核心函数，支持缓存
- `build_ppi_adjacency()` — 构建稀疏邻接矩阵 + gene→idx映射
- `build_fused_ppi_adjacency()` — 多源PPI融合

**输出**：
- `ingredient_targets_enhanced.csv`（中间缓存）
- `scipy.sparse.csr_matrix`（邻接矩阵）
- `Dict[str, int]`（gene2idx 映射，~16,201个基因）

---

### 3.2 perturbation_signature.py — 扰动签名构建

**职责**：将靶标方向信息转化为PPI网络上的扰动签名向量

**两种扩散方法**：

| 方法 | 类 | 算法 |
|------|------|------|
| 热扩散 | `PerturbationSignatureBuilder` (基类) | 两步邻居加权平均：`α · A @ (c/D)` × 2 |
| RWR | `RWRPerturbationSignatureBuilder` (子类) | 迭代：`p = (1-r)·P·p + r·p₀` |

**签名构建流程**：

```
1. 成分级初始向量
   对每个成分，遍历其靶标:
   vec[gene_idx] = action_type (+1/-1/0)
   → 对每个成分得到一个 16,201维稀疏向量

2. 草药级叠加
   按含量比(content_ratio)加权叠加所有成分向量:
   herb_init = Σ(content_ratio_i × vec_i)
   → 对每种草药得到一个 16,201维叠加向量

3. PPI扩散
   diffuse(herb_init):
   a) 热扩散: vec/degree → adj@ → ×α → 再扩散一次
   b) RWR: 归一化 → 迭代(K·P + r·p₀) → 收敛
   
4. 后处理归一化
   result = result / max(|result|)  # → [-1, +1]
   → herb_perturbation_signatures.csv (499×16,201)
```

**关键方法**：
- `_vectorize_initial()` — 成分靶标→初始向量
- `diffuse()` — 执行PPI扩散
- `thermal_diffusion_step1/step2()` — 两阶段热扩散步骤
- `extract_top_bottom_genes()` — 提取正/负扰动最强的K个基因

**输出**：
- `herb_perturbation_signatures.csv` (499行 × 16,201列)
- 缓存: `herb_signature_cache.npz` + `herb_signature_meta.json`

---

### 3.3 hdcti_adapter.py — HDCTI 靶标扩展适配器

**职责**：通过图神经网络(GCN)预测成分-蛋白结合关系，扩展靶标覆盖范围

**两阶段策略**：

| 阶段 | 方法 | 说明 |
|------|------|------|
| 阶段1 | TCMSP SVM/RF | 从TCMSP数据库补充已知靶标（高置信度阈值0.7） |
| 阶段2 | GCN图传播 | 构建成分-蛋白异构图，GCN传播预测新靶标 |

**核心方法**：
- `load_tcmsp_targets()` — 从TCMSP加载已有靶标数据
- `predict_new_targets()` — GCN 预测新靶标 + 置信度评分
- `merge_targets(existing_df, predictions, action_type_filler=0)` — 合并现有靶标与新预测靶标

**关键设计决策**：HDCTI 预测的 225,890 条靶标方向设为 `0`（未知），仅在 PPI 扩散中提供结构信息，不预设方向偏倚。

---

### 3.4 drugai_adapter.py — DrugAI 方向预测适配器

**职责**：使用深度学习模型重预测成分-靶标的作用方向（激活/抑制）

**模型架构**：
```
输入:
  ├── 化合物: SMILES → Morgan指纹 (167维) + GNN图特征
  ├── 蛋白质: 氨基酸序列 → 1D-CNN编码 (800×20)
  └── Node2Vec: 基于Hash的确定性伪嵌入 (256维)

处理:
  └── 多视图融合 → 全连接 → Sigmoid → P(activation)

输出:
  prob ≥ 0.7 → +1 (激活)
  prob ≤ 0.3 → -1 (抑制)
  0.3 < prob < 0.7 → 保留原值 (不确定)
```

**可复现性保证**：
- `PYTHONHASHSEED=0` 通过 `main.py` 自动设置
- `hash((smiles, protein_seq))` 确定性地生成 Node2Vec 嵌入
- `torch.inference_mode()` 无梯度随机性

**核心方法**：
- `predict_action_type(ingredient_id, target_gene)` — 单对预测
- `update_action_types(df, ingredients_df)` — 批量更新，支持断点续传
- `_run_inference(smiles, protein_seq)` — 模型推理
- `save_state()` / `load_state()` — 预测缓存持久化

**缓存机制**：
- `drugai_adapter_state.pkl` — 预测结果缓存（~582KB）
- `_drugai_update_checkpoint.json` — 批量处理断点
- `smiles_cache.json` — PubChem SMILES查询缓存
- `protein_seq_cache.json` — UniProt蛋白序列缓存

---

### 3.5 hdmap_s.py — HDmap-S 单草药匹配

**职责**：计算草药扰动签名与疾病病理表达的TMS匹配得分

**TMS (Transcriptional Matching Score) 公式**：

```
1. 从草药签名中提取:
   Top_genes    = 正值最大的 K 个基因 (激活)
   Bottom_genes = 负值最大的 K 个基因 (抑制)

2. 在疾病基因排位中查找:
   rank_top    = [disease_rank.get(g, n_disease+1) for g in Top_genes]
   rank_bottom = [disease_rank.get(g, n_disease+1) for g in Bottom_genes]

3. 计算匹配分数:
   MS_top    = 1 - mean(rank_top / n_disease)
   MS_bottom = 1 - mean(rank_bottom / n_disease)
   TMS = (MS_top - MS_bottom) / 2

   TMS < 0 → 草药的bottom基因匹配疾病上调基因 → 逆转疾病 ✓
   TMS > 0 → 草药的top基因匹配疾病上调基因 → 同向 ✗
```

**置换检验**：

随机打乱疾病基因排位 N 次，计算每次随机 TMS，构造零分布：
- `P_value = (#随机TMS ≤ 真实TMS) / N`
- 通过早停优化：如果已经观察到 ≥ early_stop_extreme 个极端值，可提前终止

**可选跨疾病迁移学习**（`TRANSFER_ENABLED = false`，当前未启用）：
- 计算疾病间 TMS 相似度矩阵
- 对其他疾病应用迁移权重：`adjusted_TMS = TMS_AD + λ·Σ(sim × TMS_other)`

**核心类**：`HDmapS`

**输出**：
- `hdmap_s_scores.csv` — 499草药×疾病TMS矩阵
- `{DISEASE}_top50_herbs.csv` — Top50草药排名（含P值）

---

### 3.6 hdmap_m.py — HDmap-M 多草药组合优化

**职责**：通过遗传算法搜索最优多草药组合，最大化疾病逆转效果

**适应度函数**：

```python
# 组合签名 = Σ(herb_signatures) → 再扩散
combo_sig = builder.build_combination_signature(herb_list)
top_genes, bottom_genes = extract_top_bottom_genes(combo_sig, k=250)

# 原始TMS + 组合大小正则化
tms = compute_tms(top_genes, bottom_genes, disease_rank)
adjusted = tms + GA_SIZE_PENALTY_LAMBDA × len(herbs)
return adjusted  # 越小越好 (越负 = 治疗方向越强)
```

**遗传算法设计**：

| 组件 | 值 | 说明 |
|------|------|------|
| 编码 | 草药ID列表 | 可变长度，≤4味 |
| 种群大小 | 500 | 随机初始化 |
| 选择 | 锦标赛选择 | 基于适应度排序 |
| 交叉 | 两点交叉 | crossover_rate=0.8 |
| 变异 | 随机替换/增删 | mutation_rate=0.15 |
| 精英保留 | 12 | 最佳12个个体直接保留 |

**高级机制**：

| 机制 | 说明 |
|------|------|
| **Hall of Fame** | 外部存档，容量50，跨代永不丢失最优解 |
| **种群重启** | 停滞8代后：保留50%精英 + 注入HoF + 随机填充 |
| **自适应早停** | 连续20代无改善 OR 滑动窗口改善率<0.1% |
| **禁忌搜索** | 检测并避免已知不良配方组合 |
| **大小正则化** | 越大组合受越重惩罚（λ=0.01），鼓励简洁复方 |

**核心类**：`HDmapM`，关键方法 `optimize()`

**输出**：
- `hdmap_m_results_{DISEASE}.csv` — #1 最优组合
- `hdmap_m_topk_results_{DISEASE}.csv` — Top-50 最优组合

---

### 3.7 evaluation_metrics.py + evaluation.py — 评估体系

**职责**：多维度评估草药-疾病预测结果

**评估维度**：

| 评估项 | 说明 |
|------|------|
| **H-PG 网络** | Herb-Perturbed-Gene网络，草药与其扰动基因的连接 (edge_threshold=1e-6) |
| **共同基因分析** | GA最优组合与疾病差异基因的交集 |
| **Jaccard 相似度** | 预测配方与疾病已知靶标的基因重叠 |
| **GO/KEGG 富集** | 扰动基因的通路富集分析接口 |
| **PubMed 验证** | 基于文献的配方-疾病关联验证接口 |
| **Novelty 评分** | 预测结果的新颖性评估（未被文献报道的程度） |

**输出**：
- `hpg_network_{DISEASE}.csv` — 草药-基因边列表
- `{DISEASE}_common_genes.txt` — 共同扰动基因
- `run_metadata.json` — 完整运行参数记录

---

## 四、消融实验系统 (ablation.py)

**职责**：逐个剥离核心模块，量化每个组件对最终结果的贡献

**8种消融场景**：

| 场景ID | 说明 | 测试目标 |
|------|------|------|
| `full` | 全部开启 | 基线 |
| `baseline` | 关闭 DrugAI + HDCTI | 纯 TCMSP 数据 |
| `no_drugai` | 关闭 DrugAI | DrugAI 方向预测贡献 |
| `no_hdcti` | 关闭 HDCTI | HDCTI 靶标扩展贡献 |
| `no_ppi_multi` | 关闭多源PPI | PPI 多源融合贡献 |
| `no_rwr` | 切换为热扩散 | RWR 扩散方法贡献 |
| `no_transfer` | 关闭迁移学习 | 迁移学习贡献 |
| `no_size_penalty` | 组合惩罚=0 | 组合大小正则化贡献 |

**评估指标**：Top1 TMS、TMS均值/标准差、负TMS比例、GA最优得分、GA代数、运行时间

---

## 五、数据规模总结

| 数据项 | 数量 |
|------|------|
| 草药数 | 499 |
| 成分数 | 13,728 |
| 原始靶标对 (ChEMBL/BindingDB) | 36,379 |
| HDCTI 扩展靶标对 (GCN预测) | 225,890 |
| 总靶标对 | **262,269** |
| PPI 蛋白数 | ~13,460 |
| PPI 互作数 | 141,296 |
| 疾病基因数 (AD) | 17,654 |
| 扰动签名维度 | 16,201 |
| GA 种群大小 | 500 |
| GA 最大代数 | 200 |

---

## 六、模块依赖关系

```
main.py
 ├── config.py ←── config.yaml
 ├── data_loader.py
 │    ├── hdcti_adapter.py ──→ TCMSP 数据库
 │    └── drugai_adapter.py
 │         ├── Drug_AI_best_view/best_model_*.pkl (预训练模型)
 │         ├── data/drugai_data/smiles_cache.json (PubChem)
 │         └── data/drugai_data/protein_seq_cache.json (UniProt)
 ├── perturbation_signature.py
 │    └── PPI 网络 (raw/ppi_network.csv + ppi_sources/)
 ├── hdmap_s.py
 │    └── 疾病表达谱 (raw/disease_profiles.csv)
 ├── hdmap_m.py
 │    ├── hdmap_s.py (复用 TMS 计算)
 │    └── perturbation_signature.py (复用组合签名构建)
 └── evaluation.py
      └── evaluation_metrics.py
```

---

## 七、Python 模块清单

| 文件 | 行数(估) | 作用 |
|------|------|------|
| `main.py` | ~830 | 主入口，流程编排 |
| `config.py` | ~330 | 统一配置 + YAML加载 |
| `config.yaml` | ~140 | 配置文件 |
| `data_loader.py` | ~450 | 数据加载 + HDCTI/DrugAI 集成 |
| `perturbation_signature.py` | ~600 | 扰动签名构建 + 扩散方法 |
| `hdcti_adapter.py` | ~600 | HDCTI GCN 靶标扩展 |
| `drugai_adapter.py` | ~1800 | DrugAI 深度学习模型 + 推理 |
| `hdmap_s.py` | ~500 | TMS计算 + 置换检验 |
| `hdmap_m.py` | ~600 | 遗传算法组合优化 |
| `evaluation.py` | ~300 | 评估报告生成 |
| `evaluation_metrics.py` | ~200 | H-PG网络 + Jaccard + PubMed |
| `ablation.py` | ~500 | 消融实验系统 |
| `temporal_signature.py` | ~200 | 时间感知签名 (实验性) |
| `transfer_learning.py` | ~200 | 跨疾病迁移学习 (实验性) |
