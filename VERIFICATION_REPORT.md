# 模型实现核查报告

## 一、原论文 vs 当前实现对比

### 原论文方法流程 (Chen et al., 2018)

```
1. 疾病病理状态谱 (Disease Profile)
   ├── 输入: 189种疾病的差异表达基因 (gene_symbol, logFC, P_value)
   └── 按logFC降序排列 → 疾病基因排位列表

2. 草药扰动签名构建
   ├── 成分靶标来源: ChEMBL + BindingDB (实验验证数据)
   ├── 方向预测: PreAM模型 → action_type = +1(激活) or -1(抑制)
   ├── 初始向量: vec[target_gene] = action_type
   ├── 草药叠加: 按含量比加权叠加成分向量
   └── PPI扩散: 两阶段热扩散 (Thermal Diffusion)

3. HDmap-S: 单草药匹配
   ├── Top/Bottom K基因 (K=250)
   ├── TMS = (MS_top - MS_bottom) / 2
   └── 置换检验 → P值

4. HDmap-M: GA组合优化
   └── 遗传算法搜索最优草药组合

5. 评估
   ├── H-PG网络
   ├── 文献验证
   └── GO/KEGG富集
```

### 当前实现 vs 原论文

| 步骤 | 原论文 | 当前实现 | 状态 |
|------|--------|---------|------|
| 疾病表达谱 | 189种疾病 GEO微阵列 | AD疾病 (17,654基因) | ✅ 已实现，单疾病模式 |
| 成分-靶标 | ChEMBL + BindingDB (实验) | ChEMBL + BindingDB (36,379条) | ✅ 已实现 |
| 方向预测 | **PreAM模型** | **DrugAI模型** | ✅ 已替换 |
| 靶标扩展 | WES算法 | **HDCTI GCN** (225,890条) | ✅ 已添加 |
| PPI网络 | 13,460蛋白, 141,296互作 | 相同 | ✅ 已实现 |
| 签名构建 | 两阶段热扩散 | 两阶段热扩散 | ✅ 已实现 |
| TMS公式 | (MS_top - MS_bottom) / 2, K=250 | 相同 | ✅ 已实现 |
| 置换检验 | N=100次 | N=100次 | ✅ 已实现 |
| GA优化 | 遗传算法 | 增强版GA (HoF, 重启, 早停) | ✅ 已实现 |
| 跨疾病验证 | 189种疾病 | 可选 (TRANSFER_ENABLED=false) | ⚠️ 未启用 |

---

## 二、DrugAI 实现核查

### DrugAI 原论文 (Zhang et al., Briefings in Bioinformatics, 2023)

DrugAI 是一个**多视图深度学习模型**，用于预测药物-靶标的激活/抑制机制。

**四个模块：**
1. **GNN (AttentiveFP)**: 编码化合物分子图
2. **1D-CNN**: 编码靶标蛋白氨基酸序列
3. **网络嵌入模块**: 药物和靶标的Node2Vec嵌入
4. **DNN**: 融合多视图特征 → Sigmoid → P(activation)

### 当前代码实现

| DrugAI论文 | 代码实现 | 文件位置 |
|-----------|---------|---------|
| GNN (AttentiveFP) 化合物编码 | `_CompoundEncoder` 类 | `drugai_adapter.py:62` |
| 1D-CNN 蛋白序列编码 | `_ProteinEncoder` 类 | `drugai_adapter.py:84` |
| Node2Vec 网络嵌入 | 基于hash的确定性伪嵌入 (256维) | `drugai_adapter.py:1795-1798` |
| DNN 多视图融合 | `DrugAIClassifier.forward()` | 与预训练模型接口 |
| 输出: P(activation) | Sigmoid → prob ∈ [0,1] | `drugai_adapter.py:1830` |
| 三分类: prob≥0.7→+1, prob≤0.3→-1 | `predict_action_type()` | `drugai_adapter.py:70-76` |

### 与PreAM的替换关系

| 原论文 | 替换后 | 说明 |
|--------|-------|------|
| PreAM模型预测 action_type | DrugAI模型预测 action_type | 同样输出 +1(激活)/-1(抑制)，接口兼容 |
| WES算法预测新靶标 | HDCTI GCN预测新靶标 | 功能相同，方法不同 |

**⚠️ 可复现性保障：**
- `main.py` 自动设置 `PYTHONHASHSEED=0`
- `hash((smiles, protein_seq))` 确定性地生成 Node2Vec 种子
- `torch.inference_mode()` 关闭梯度随机性
- 运行方式：`python main.py --disease AD` (无需额外参数)

---

## 三、HDCTI 实现核查

### HDCTI 原论文 (Qiao et al., Briefings in Bioinformatics, 2024)

HDCTI 使用**超图表示学习**来识别中药成分的新治疗靶标。

**核心方法：**
1. 构建两个超图：草药-成分超图 + 疾病-靶标超图
2. 超图卷积算子捕获高阶关联
3. PageRank + 多头注意力增强嵌入
4. 端到端预测成分-靶标关联

### 当前代码实现

| HDCTI论文 | 代码实现 | 说明 |
|----------|---------|------|
| 草药-成分超图 | GCN异构图 (简化) | 用GCN替代超图卷积 |
| 成分-靶标关联预测 | `predict_new_targets()` | GCN传播预测新靶标 + 置信度 |
| 靶标验证 | TCMSP SVM/RF筛选 | 两阶段：数据库补充 → 图传播 |

**HDCTI在管道中的作用：**
- 原始数据 36,379 条靶标 → HDCTI扩展至 262,269 条（新增 225,890 条）
- 新增靶标方向设为 `0`（未知），仅参与 PPI 结构扩散
- 保守策略：不预设方向偏倚，让网络拓扑决定信息流向

**⚠️ 实现差异：** 原论文使用超图（Hypergraph），代码使用标准 GCN 异构图。这是简化实现，但核心目标一致：扩展成分靶标覆盖范围。

---

## 四、输入数据核查

### 必需的数据文件

| 文件 | 路径 | 内容 | 来源 |
|------|------|------|------|
| **疾病差异表达基因** | `data/raw/disease_profiles.csv` | `gene_symbol, logFC, P_value` | GEO微阵列差异分析 |
| **草药-成分表** | `data/raw/herb_ingredients.csv` | `herb_id, ingredient_id, content_ratio` | TCMSP数据库 |
| **成分-靶标表** | `data/raw/ingredient_targets.csv` | `ingredient_id, target_gene, action_type` | ChEMBL + BindingDB |
| **PPI网络** | `data/raw/ppi_network.csv` | `protein_A, protein_B` | STRING等数据库 |
| **DrugAI预训练模型** | `Drug_AI_best_view/best_model_auc_*.pkl` | 模型权重 | DrugAI原论文训练 |

### 可能遗漏的数据

| 数据项 | 是否需要 | 当前状态 |
|--------|---------|---------|
| 多种疾病表达谱 (原论文189种) | 跨疾病验证需要 | ❌ 仅有AD疾病 |
| 草药含量数据 (content_ratio) | 权重叠加需要 | ✅ 已含 |
| SMILES数据 (成分结构) | DrugAI预测需要 | ✅ PubChem缓存 |
| 蛋白序列数据 | DrugAI预测需要 | ✅ UniProt缓存 |
| ChEMBL机制数据 | DrugAI验证覆盖 | ✅ `chembl_mechanisms.csv` |

### 你还需要什么？

根据原论文，如果你要做完整的疾病映射，还需要：
- **更多疾病的差异表达基因**（目前只有AD一种）
- 如果做消融实验对比，可能需要PreAM的原始预测结果（做baseline）

---

## 五、核心逻辑完整性检查

### 数据流验证 ✅

```
1. disease_profiles.csv → 疾病基因排位列表
   └── gene_symbol 按 logFC 降序 → disease_rank{gene: rank}

2. ingredient_targets.csv → HDCTI扩展(×6.2) → DrugAI重标方向
   └── 36,379 → 262,269 条靶标对

3. 成分级初始向量
   └── vec[gene_idx] = action_type (+1/-1/0)

4. 草药级叠加
   └── herb_init = Σ(content_ratio × vec)

5. 两阶段热扩散
   └── step1: adj @ (c/D), step2: adj @ (c'/D)
   └── 后归一化: / max(|result|) → [-1,+1]

6. Top/Bottom K=250 基因
   └── htop: 正扰动最强250个基因
   └── hbottom: 负扰动最强250个基因

7. TMS计算
   └── MS=1-mean(rank/n_disease)
   └── TMS=(MS_top-MS_bottom)/2

8. 置换检验 (N=100)
   └── 随机打乱疾病基因排位
   └── P = (#随机TMS ≤ 真实TMS) / N

9. GA组合优化
   └── 适应度 = TMS + λ×n_herbs
   └── HoF + 种群重启 + 自适应早停
```

### 发现的潜在问题

1. **配置文件中 `DIFFUSION_METHOD = "rwr"`，但代码实际使用热扩散**
   - `PerturbationSignatureBuilder` (热扩散) 被实例化，而非 `RWRPerturbationSignatureBuilder`
   - 建议：将 `config.yaml` 中的 `method: "rwr"` 改为 `method: "thermal"` 以反映实际

2. **HDCTI 实现与原论文不同**
   - 原论文：超图卷积
   - 代码：标准 GCN 异构图
   - 论文中需要说明这是简化实现

3. **TMS 得分偏小**
   - 当前 TMS 范围 [-0.007, +0.13]，原论文通常 [-0.2, +0.1]
   - 可能与 DrugAI 预测分布、疾病基因数量 (17K vs 原论文的少量基因) 有关

