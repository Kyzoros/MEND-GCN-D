# GCN-D 模型架构文档

> 本文档从代码出发描述本项目的真实框架。本文方法为 **GCN-D（GCN + DistMult）**；扰动签名 / HDmap-S / HDmap-M 等无监督管线是**对比基线与数据基础**，不是本文方法。

---

## 一、总体数据流

```
data/processed/  （本地数据，不进仓库）
    herb_ingredients_processed.csv              草药-成分
    ingredient_targets_enhanced_recovered.csv   成分-靶基因(增强)
    disease_profiles_processed.csv              疾病差异表达谱(logFC/P)
    kegg 缓存                                    基因-基因边
    gold_standard_final_8d_supplemented.csv     8病金标准(595行)
        ↓
mendgcn/run_fair_comparison.py — build_graph()
    异构图 H-I-G-D + 可选 KEGG G-G 边 → 对称归一化 A_sym
    节点特征 6 维（类型one-hot + 度数 + |logFC|）
        ↓
mendgcn/run_fair_comparison.py — GCN 类 / train_gcn_model()
    2层图卷积(ReLU) → z_h, z_d (32维)
    DistMult 可学对角权重 w；k_neg=5 负采样；纯 BCE；Adam 1e-2
    打分 S = (z_h ⊙ w) @ z_dᵀ （标准化）
        ↓
评估协议（对级干净评估 / FULL-labeled AUROC）
    P1  mendgcn/run_cv_final.py               5折 transductive → Table_cv_clean.csv
    P2  experiments/run_gnn_lodo_coldstart.py LODO 留一病      → Table_lodo.csv
    P3  experiments/run_gnn_lodo_coldstart.py 冷启动草药       → Table_coldstart.csv
    公平对比  mendgcn/run_fair_comparison.py / run_fair_final.py
```

---

## 二、核心模型：GCN + DistMult

### 2.1 图构建 `build_graph()`

| 组件 | 实现 |
|------|------|
| 节点 | 草药 H、成分 I、基因 G、疾病 D（4 类偏移拼接） |
| 边 | H–I、I–G、G–D；`use_kegg=True` 时加 G–G（`kegg_network.build_kegg_adjacency`，`min_shared=KEGG_MIN_SHARED`）|
| 邻接 | 双向对称 + 自环；`A_sym = D⁻¹ᐟ² A D⁻¹ᐟ²`（稀疏 COO） |
| 特征 X | 6 维：4 类节点 one-hot；`√(deg/degmax)` 度数特征；基因 |logFC| 均值（标准化）|
| 消融开关 | `use_logfc`（去掉表达特征）、`use_kegg`（去掉 G–G 边）、`excl_diseases/excl_herbs`（留一剔除） |

### 2.2 编码器 `GCN`

```
n_layers=2（默认）:  h = ReLU(W1·(A@X));   z = Wout·(A@h)
n_layers=1       :  z = Wout·ReLU(W1·X)    ← 无图传播 MLP 对照
n_layers>=3      :  中间层继续图卷积
```

尺寸：输入 6 → 隐藏 64 → 输出 32。所有权重矩阵无偏置（`bias=False`）。

### 2.3 训练 `train_gcn_model()`

- 正例：金标准 `label=1` 的 (草药, 疾病) 对（训练协议按折/留一划分）；
- 负采样：每病 `k_neg=5` × 正例数，候选 = 全库草药 \\ 该病**全部已知正例**（`build_exclude()` 过滤，防泄漏；修复了早期"取候选下标当草药编号"的 bug）；
- 损失：`L = −mean(log σ(pos_s)) − mean(log(1−σ(neg_s)))`（纯 BCE，无正则项）；
- 优化：Adam `lr=1e-2`；`manual_seed` 固定（默认 42）；epochs 300（冒烟）/ 3000（全量）；
- 解码消融：`decoder="dot"` → `w` 固定为 1，退化为纯点积。

### 2.4 评分与评估

- 打分矩阵：`S = (z_h ⊙ w) @ z_dᵀ`，按均值/标准差标准化；
- `evaluate()`：FULL（负例=全库其余草药）与 labeled（仅金标有显式 label 的草药）两口径 AUROC；
- 对级干净指标（[run_cv_final.py](mendgcn/run_cv_final.py) `pair_metrics`）：pair AUROC、AUPRC、MRR、Hits@10/20；bootstrap 95% CI（`summarise`）。

---

## 三、评估协议矩阵

| 协议 | 性质 | 关键实现 | 输出表 |
|------|------|----------|--------|
| P1 5 折 CV | transductive，未见对 | 306 正例对划 5 折（FOLD_SEED=12345），每对恰好测一次 | `Table_cv_clean.csv` |
| P2 LODO | inductive，未见病 | `build_graph_exclude` 剔除该病节点+基因边；`infer_score` 冻结权重全图推理 | `Table_lodo.csv` (+`_per_disease`) |
| P3 冷启动 | inductive，未见草药 | 5 折各剔除 1/5 草药节点+成分边；靠成分→基因→病传播打分 | `Table_coldstart.csv` (+`_per_fold`) |
| 公平对比 | 同数据同评估 | 留出 1/5 正例对；4 方法 FULL/labeled + GCN 留出泛化 | `Table_fair_comparison.csv` |
| 定稿基线+消融 | 6 方法 | 干净留出、epoch 扫描、结构消融（无KEGG/点积/MLP/3层） | `Table_fair_final.csv` 等 4 表 |
| 8 病扩展 | 新 4 病 | TCMBank 提取正例 + curated AUC + 主表整合 | `Table_8disease_curated_auc.csv`、`Table_8disease_master.csv` |

统一超参：`{lr:1e-2, hid:64, out:32, k_neg:5, n_layers:2}`（`BEST` 常量在各协议脚本中一致）。

---

## 四、对比基线

### 4.1 无监督基线（与 GCN-D 同图/同数据/同评估）

| 基线 | 函数 | 思想 |
|------|------|------|
| Fisher 富集 (D-TMS) | `score_fisher()`（mendgcn/run_fair_comparison.py） | 草药靶点集与疾病基因集富集显著性（`hdmap_s.pair_single_herb_all_diseases_enrichment`） |
| 网络邻近度 (Barabasi) | `score_proximity()` | KEGG 网络 BFS 距离 + 度校正随机期望 z-score |
| 靶点数量 | `score_targetcount()` | 平凡：count(靶点)，无疾病特异性 |
| RWR 扩散 | `score_rwr()`（mendgcn/run_fair_final.py） | 以病为种子行归一游走（c=0.7，100 迭代），草药稳态概率为分 |
| MLP 无图传播 | GCN `n_layers=1` | 剥离图传播的深度对照 |

### 4.2 无监督系统映射管线 [main.py](baselines/main.py)

```
加载数据 → RWR 扰动签名(mendgcn/perturbation_signature.py)
        → HDmap-S TMS + 置换检验(mendgcn/hdmap_s.py)
        → HDmap-M 遗传算法组合搜索(mendgcn/hdmap_m.py)
        → H-PG 网络 / 共同基因评估(mendgcn/evaluation.py + mendgcn/evaluation_metrics.py)
```

输出：`output/results/{DISEASE}_{timestamp}/hdmap_s_scores.csv`、`herb_perturbation_signatures.csv`、`hdmap_m_results_{DISEASE}.csv`、`hpg_network_{DISEASE}.csv` 等。

---

## 五、数据增强组件（数据层，非主模型）

| 模块 | 作用 | 谁调用 |
|------|------|--------|
| `mendgcn/data_loader.py` | 加载 raw CSV（疾病谱/草药成分/成分靶标/PPI），整合增强管线输出 | `baselines/main.py` 与各协议脚本 |
| `mendgcn/hdcti_adapter.py` | TCMSP SVM/RF 补充 + G–C 异构图 GCN 传播预测新靶标，合并去重 | 数据准备阶段（本地脚本） |
| `mendgcn/drugai_adapter.py` | DrugAI（AttentiveFP+1D-CNN）三分类阈值预测作用方向（+1/−1/保留） | 数据准备阶段（本地脚本） |
| `mendgcn/gcn_target_expansion.py` | 草药-靶标二部图 2 层 GCN、BCE、1:1 负采样——靶标扩展**对比实验** | 独立实验脚本 |
| `mendgcn/kegg_network.py` | KEGG 基因互作邻接构建与归一化缓存 | `build_graph()`（G–G 边） |

---

## 六、配置系统（config.py + config.yaml）

三层优先级：`命令行 > 环境变量 > YAML > 代码默认值`。与主实验相关的关键项：

| 分类 | 参数 | 说明 |
|------|------|------|
| 路径 | `RAW_DIR / PROCESSED_DIR / RESULTS_DIR` | 原始数据 / 处理后数据 / 结果输出 |
| 疾病谱 | `DISEASE_LOGFC_THRESHOLD、DISEASE_PVALUE_THRESHOLD、DISEASE_DEG_TOP_K` | 疾病差异基因筛选 |
| KEGG | `KEGG_MIN_SHARED` | 共通路基因最小共享阈值 |
| 靶标 | `SM_TGT_FILE`（环境变量） | 覆盖默认增强靶标文件，做清洗前后对照 |
| 扩散签名 | `DIFFUSION_METHOD（rwr/thermal）、RWR_RESTART_PROB` | 无监督签名（基线用） |
| 增强 | `HDCTI_USE、DRUGAI_USE` 及阈值 | 靶标扩展与方向预测开关 |
| TMS | `TOP_BOTTOM_K、PERMUTATION_N、TMS_METHOD` | HDmap-S 基线参数 |
| GA | `GA_POP_SIZE、GA_MAX_GENERATIONS、GA_MAX_COMBO_SIZE` 等 | HDmap-M 基线参数 |

---

## 七、模块依赖关系

```
各 run_*.py 协议脚本（mendgcn/ 与 experiments/ 下）
 ├── config.py ←── config.yaml
 ├── mendgcn/run_fair_comparison.py ★（GCN 类 / build_graph / train_gcn / 基线函数）
 │    ├── mendgcn/kegg_network.py（G–G 边）
 │    └── mendgcn/hdmap_s.py（Fisher 富集复用）
 ├── mendgcn/run_fair_final.py    （score_rwr）
 └── mendgcn/run_cv_final.py      （pair_metrics / summarise）

baselines/main.py（无监督基线管线）
 ├── mendgcn/data_loader.py ── mendgcn/hdcti_adapter.py / mendgcn/drugai_adapter.py（增强，本地数据准备）
 ├── mendgcn/perturbation_signature.py（RWR/热扩散签名）
 ├── mendgcn/hdmap_s.py → mendgcn/hdmap_m.py（TMS → 遗传算法）
 └── mendgcn/evaluation.py → mendgcn/evaluation_metrics.py（H–PG 网络等）
```

---

## 八、输出表清单（output/results/paper_deliverables/）

| 表 | 内容 |
|----|------|
| `Table_fair_comparison.csv` | 4 方法 × 8 病 FULL/labeled AUROC + GCN 留出泛化 |
| `Table_fair_final.csv` | 6 方法 × 8 病（定稿） |
| `Table_cv_clean.csv` | P1 对级 AUROC/AUPRC/MRR/Hits@10/20 + bootstrap 95% CI |
| `Table_lodo.csv` (+`_per_disease`) | P2 留一病结果 |
| `Table_coldstart.csv` (+`_per_fold`) | P3 冷启动结果 |
| `Table_holdout_clean.csv` | 干净留出对级评估（61 对） |
| `Table_epoch_overfit.csv` | epoch 扫描：FULL vs 干净留出 |
| `Table_gnn_ablation2.csv` | 结构消融：无 KEGG / 点积 / MLP / 3 层 |
| `Table_8disease_curated_auc.csv` / `Table_8disease_master.csv` | 8 病扩展评估 |