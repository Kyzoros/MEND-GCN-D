# -*- coding: utf-8 -*-
"""
D7: enrichment 方法组件消融 + 置换检验 (论文核心证据)
=====================================================
审稿人必问: "你的提升是靶点-DEG富集组件带来的, 还是碰巧/任意打分带来的?"

设计 (全部在 FULL 协议, curated 金标准, 与 D5/P0a 同一评估域):
  A. 完整方法 F (Fisher 富集 -log10 p, 当前流水线)          -> 4 病 AUROC
  B. 靶点置换检验 (permutation of herb targets):            保持每草药靶点数不变,
     从背景基因集随机抽同等数量的基因当靶点, 重算 Fisher, 100 次
     -> 每病 AUROC 零分布; 真实 F 相对零分布的 z-score / 百分位
     (若 F 显著 > 随机, 证明"靶点-疾病匹配"本身携带信号, 非任意打分的假象)
  C. DEG 置换检验 (permutation of disease DEGs):            保持 DEG 集合大小,
     从背景基因集随机抽同等数量基因当"DEG", 100 次
     -> 证明疾病 DEG 特异性(而非任何基因集)驱动富集
  D. 方向消融: F high=good vs low=good                      证明方向一致性不是碰巧
  E. 负对照: 随机打分 (uniform random)                       理论上界 = 0.5

输出: output/results/paper_deliverables/D7_ablation_permutation.csv
"""
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
from config import GOLD_STANDARD_PATH
from evaluation_metrics import load_gold_standard

PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "output", "results", "paper_deliverables")
os.makedirs(OUT, exist_ok=True)

DISEASES = ["AD", "PAH", "RA", "T2D"]
LOGFC_THR = 0.05
PVAL_THR = 0.05
N_PERM = 100
rng = np.random.default_rng(42)

# ---------- 1. 载入数据 (与 P0a 完全一致) ----------
universe = list(pd.read_csv(os.path.join(BASE, "output", "results",
            "AD_kegg_results", "hdmap_s_scores.csv"), index_col=0).index)
pos_dict, neg_dict, _ = load_gold_standard(GOLD_STANDARD_PATH, None)

dp = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
deg_sets, deg_signed = {}, {}
for d in DISEASES:
    sub = dp[dp.disease_id == d].copy()
    sub = sub[sub.logFC.abs() > LOGFC_THR]
    if PVAL_THR > 0 and "P_value" in sub.columns:
        sub = sub[sub.P_value < PVAL_THR]
    deg_sets[d] = set(sub.gene_symbol.astype(str))
    deg_signed[d] = dict(zip(sub.gene_symbol.astype(str), sub.logFC.astype(float)))

it = pd.read_csv(os.path.join(PROC, "ingredient_targets_enhanced.csv"))
hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
ing_targets = defaultdict(dict)
for _, r in it.iterrows():
    g = str(r["target_gene"]).strip()
    if g and g != "nan":
        c = float(r.get("confidence", 0.5) or 0.5)
        cur = ing_targets[int(r["ingredient_id"])].get(g)
        if cur is None or c > cur:
            ing_targets[int(r["ingredient_id"])][g] = c
herb_w = defaultdict(dict)   # herb -> gene -> sum conf
for _, r in hi.iterrows():
    hid = str(r["herb_id"])
    t = ing_targets.get(int(r["ingredient_id"]), {})
    for g, c in t.items():
        herb_w[hid][g] = herb_w[hid].get(g, 0.0) + c

bg = set().union(*deg_sets.values()) | set().union(*[set(w) for w in herb_w.values()])
BG_LIST = sorted(bg)
N_BG = len(BG_LIST)
print(f"草药: {len(universe)} | 有靶点: {sum(1 for h in herb_w if herb_w[h])} | 背景基因: {N_BG}")

# 每草药实际靶点集合
real_targets = {h: set(herb_w[h]) for h in universe}

# ---------- 2. 打分 ----------
def fisher_score(tg, dg):
    a = len(tg & dg)
    if a == 0:
        return 0.0
    b = len(tg - dg); c = len(dg - tg)
    dd = max(0, N_BG - a - b - c)
    _, p = fisher_exact([[a, b], [c, dd]])
    return -np.log10(max(p, 1e-300))

def score_all(target_sets, d):
    dg = deg_sets[d]
    return {h: fisher_score(target_sets[h], dg) for h in universe}

def eval_auroc(scores, d):
    pos = set(pos_dict.get(d, []))
    ps = [scores[h] for h in pos if h in scores]
    s_all = [scores[h] for h in universe if h in scores]
    if len(ps) == 0 or len(s_all) == 0:
        return None
    y = [1 if h in pos else 0 for h in universe if h in scores]
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s_all))

# ---------- 3. 主实验 ----------
rows = []

# A. 完整方法
f_full = {}
for d in DISEASES:
    f_full[d] = eval_auroc(score_all(real_targets, d), d)
    rows.append({"ablation": "A_full_Fisher", "disease": d,
                 "AUROC": f_full[d], "n_perm": None})
print("A 完整 Fisher:", {k: round(v, 3) if v else None for k, v in f_full.items()})

# B. 靶点置换检验 (保持每草药靶点数量)
print(f"\nB. 靶点置换检验 x{N_PERM} ...")
perm_auroc = {d: [] for d in DISEASES}
for rep in range(N_PERM):
    perm_targets = {}
    for h in universe:
        k = len(real_targets[h])
        perm_targets[h] = set(rng.choice(BG_LIST, size=k, replace=False))
    for d in DISEASES:
        a = eval_auroc(score_all(perm_targets, d), d)
        if a is not None:
            perm_auroc[d].append(a)
for d in DISEASES:
    z = np.array(perm_auroc[d])
    real = f_full[d]
    if real is None or len(z) == 0:
        continue
    pct = (np.sum(z >= real) + 1) / (len(z) + 1)   # 单侧经验 p
    zscore = (real - z.mean()) / (z.std() + 1e-12)
    rows.append({"ablation": "B_target_perm", "disease": d,
                 "AUROC": real, "n_perm": N_PERM,
                 "perm_mean": round(float(z.mean()), 3),
                 "perm_p95": round(float(np.percentile(z, 95)), 3),
                 "perm_z": round(float(zscore), 2),
                 "perm_pctile": round(float(pct), 4)})
    print(f"  {d}: real={real:.3f} perm_mean={z.mean():.3f} p95={np.percentile(z,95):.3f} z={zscore:.2f} p={pct:.4f}")

# C. DEG 置换检验 (保持 DEG 集合大小)
print(f"\nC. DEG 置换检验 x{N_PERM} ...")
perm_deg_auroc = {d: [] for d in DISEASES}
real_dg_save = {d: deg_sets[d] for d in DISEASES}
for rep in range(N_PERM):
    for d in DISEASES:
        k = len(real_dg_save[d])
        deg_sets[d] = set(rng.choice(BG_LIST, size=k, replace=False))
    for d in DISEASES:
        a = eval_auroc(score_all(real_targets, d), d)
        if a is not None:
            perm_deg_auroc[d].append(a)
deg_sets = real_dg_save
for d in DISEASES:
    z = np.array(perm_deg_auroc[d])
    real = f_full[d]
    if real is None or len(z) == 0:
        continue
    pct = (np.sum(z >= real) + 1) / (len(z) + 1)
    zscore = (real - z.mean()) / (z.std() + 1e-12)
    rows.append({"ablation": "C_DEG_perm", "disease": d,
                 "AUROC": real, "n_perm": N_PERM,
                 "perm_mean": round(float(z.mean()), 3),
                 "perm_p95": round(float(np.percentile(z, 95)), 3),
                 "perm_z": round(float(zscore), 2),
                 "perm_pctile": round(float(pct), 4)})
    print(f"  {d}: real={real:.3f} perm_mean={z.mean():.3f} p95={np.percentile(z,95):.3f} z={zscore:.2f} p={pct:.4f}")

# D. 方向消融 (low=good)
print("\nD. 方向消融 (low=good) ...")
for d in DISEASES:
    s = score_all(real_targets, d)
    a_lo = eval_auroc({h: -v for h, v in s.items()}, d)
    rows.append({"ablation": "D_dir_low", "disease": d, "AUROC": a_lo})
    print(f"  {d}: high={f_full[d]:.3f} low={a_lo:.3f}")

# E. 负对照: 随机打分
print("\nE. 负对照随机打分 ...")
for d in DISEASES:
    a = eval_auroc({h: rng.random() for h in universe}, d)
    rows.append({"ablation": "E_random_score", "disease": d, "AUROC": a})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "D7_ablation_permutation.csv"), index=False)

# ---------- 4. 汇总 ----------
print("\n" + "=" * 88)
print("D7 消融+置换汇总 (FULL 协议, curated 金标准)")
print("=" * 88)
mean_rows = []
for abl in ["A_full_Fisher", "B_target_perm", "C_DEG_perm", "D_dir_low", "E_random_score"]:
    sub = df[df.ablation == abl]
    vals = sub[sub.AUROC.notna()].AUROC
    if len(vals):
        mean_rows.append(f"{abl:16s} mean={vals.mean():.3f}")
print(" | ".join(mean_rows))
print("\n解读: A 应显著高于 E(随机,≈0.5); B/C 的 real 应 > perm_p95 (即置换后掉回随机)")
print("[已保存] output/results/paper_deliverables/D7_ablation_permutation.csv")
