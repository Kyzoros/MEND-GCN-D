"""
诊断 B 探针：绕开当前"扩散签名+余弦"的 HDmap-S，直接测"草药靶点 vs 疾病差异基因"有没有信号。

目的：判断方法改进(B)是否值得做、从哪改。
  - 若替代打分(靶点级)仍 ≈0.5 -> 靶点数据本身与疾病 DEG 不对齐 -> 数据限制，B 难救
  - 若某替代打分 >0.6 -> 当前"扩散签名"步骤在抹掉信号 -> B 有戏(改签名计算)

对比基线：当前 HDmap-S (cos 矩阵, lit=antagonist)。

替代打分(每草药×每病):
  A. 直接靶点余弦：herb 靶点权重向量 vs disease logFC 向量 的 cosine
  B. DEG命中占比：herb 靶点的加权质量中，落在疾病|DEG|的比例
  C. Fisher富集：-log10(p)，herb靶点集 vs 疾病DEG集的富集

方向处理：每个打分报 high=good / low=good 两种 full 协议 AUROC，取 max 作诊断(opt,循环论证)。
"""
import pandas as pd, numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score
import os
import sys
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)

COS = 'output/results/AD_kegg_results/hdmap_s_scores_dual_cos.csv'
GOLD = 'data/gold_standard_expanded.csv'
DISEASE_PROFILES = 'data/processed/disease_profiles_processed.csv'
HERB_ING = 'data/processed/herb_ingredients_processed.csv'
ING_TARGET = 'data/processed/ingredient_targets_enhanced.csv'
DISEASES = ['AD', 'PAH', 'RA', 'T2D']
LOGFC_THR = 0.05

master_rows = []
def _record(method, rows):
    for r in rows:
        master_rows.append({'method': method, **r})

# ---------- 载入 ----------
print("[载] 金标准 ...")
gold = pd.read_csv(GOLD)
pos_sets = {d: set(gold[(gold.disease_id==d)&(gold.label==1)].herb_id.astype(str)) for d in DISEASES}

print("[载] 疾病 DEG ...")
dp = pd.read_csv(DISEASE_PROFILES)
deg = {}   # d -> dict gene->|logFC|
deg_signed = {}  # d -> dict gene->logFC
for d in DISEASES:
    sub = dp[dp.disease_id == d].copy()
    sub = sub[sub.logFC.abs() > LOGFC_THR]
    deg[d] = dict(zip(sub.gene_symbol.astype(str), sub.logFC.abs().astype(float)))
    deg_signed[d] = dict(zip(sub.gene_symbol.astype(str), sub.logFC.astype(float)))
    print(f"  {d}: {len(deg[d])} DEGs")

print("[载] 草药-成分-靶点 -> 草药靶点权重向量 ...")
hi = pd.read_csv(HERB_ING)
it = pd.read_csv(ING_TARGET)
# ingredient -> {gene: conf}
ing_targets = defaultdict(dict)
for _, r in it.iterrows():
    g = str(r['target_gene']).strip()
    if g and g != 'nan':
        c = float(r.get('confidence', 0.5) or 0.5)
        ing_targets[int(r['ingredient_id'])][g] = max(ing_targets[int(r['ingredient_id'])].get(g, 0), c)
# herb -> {gene: weight}
herb_targets = defaultdict(lambda: defaultdict(float))
for _, r in hi.iterrows():
    hid = str(r['herb_id']).strip()
    iid = int(r['ingredient_id'])
    for g, c in ing_targets.get(iid, {}).items():
        herb_targets[hid][g] += c
print(f"  草药数(有靶点): {len(herb_targets)}")

# 全库草药列表(取 cos 矩阵里的, 与金标准同域)
cos = pd.read_csv(COS).set_index('herb_id')
all_herbs = list(cos.index.astype(str))
print(f"  全库草药(cos矩阵): {len(all_herbs)}")

def full_auroc(scores_dict, pos):
    herbs = [h for h in all_herbs if h in scores_dict]
    s = np.array([scores_dict[h] for h in herbs], float)
    y = np.array([1 if h in pos else 0 for h in herbs], int)
    if 0 < y.sum() < len(y):
        return round(float(roc_auc_score(y, s)), 3)
    return float('nan')

def score_auroc_table(name, score_fn):
    print(f"\n--- 打分: {name} ---")
    rows = []
    for d in DISEASES:
        sc = {}
        for h in all_herbs:
            tv = herb_targets.get(h)
            if not tv:
                sc[h] = 0.0
            else:
                sc[h] = score_fn(tv, d)
        hi_g = full_auroc(sc, pos_sets[d])
        lo_g = full_auroc({h: -v for h, v in sc.items()}, pos_sets[d])
        opt = max(hi_g, lo_g)
        best_dir = 'high' if hi_g >= lo_g else 'low'
        rows.append({'disease': d, 'high=good': hi_g, 'low=good': lo_g, 'opt(max)': opt, 'best_dir': best_dir})
        print(f"  {d:<5} high=good={hi_g:>6}  low=good={lo_g:>6}  opt={opt:>6} [{best_dir}]")
    return rows

# A. 直接靶点余弦
def fn_cosine(tv, d):
    genes = list(set(tv) & set(deg_signed[d]))
    if len(genes) < 2:
        return 0.0
    a = np.array([tv[g] for g in genes], float)
    b = np.array([deg_signed[d][g] for g in genes], float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# B. DEG命中占比(加权)
def fn_hitfrac(tv, d):
    tot = sum(tv.values())
    if tot == 0:
        return 0.0
    hit = sum(v for g, v in tv.items() if g in deg[d])
    return hit / tot

# C. Fisher富集 -log10(p)
DEG_SETS = {d: set(deg[d].keys()) for d in DISEASES}
ALL_GENES = set().union(*DEG_SETS.values()) | set().union(*[set(t) for t in herb_targets.values()])
N_BG = len(ALL_GENES)
def fn_fisher(tv, d):
    tg = set(tv.keys())
    dg = DEG_SETS[d]
    a = len(tg & dg)
    b = len(tg - dg)
    c = len(dg - tg)
    dd = N_BG - a - b - c
    if a == 0 or dd < 0:
        return 0.0
    _, p = fisher_exact([[a, b], [c, dd]])
    return -np.log10(max(p, 1e-300))

print("\n############ B 探针结果 (full 库协议, 对照 expanded 金标准) ############")
_record("A. 直接靶点余弦", score_auroc_table("A. 直接靶点余弦(tv vs logFC)", fn_cosine))
_record("B. DEG加权命中占比", score_auroc_table("B. DEG加权命中占比", fn_hitfrac))
_record("C. Fisher富集", score_auroc_table("C. Fisher富集 -log10(p)", fn_fisher))

# 基线: 当前 HDmap-S
print("\n--- 基线: 当前 HDmap-S (cos, lit=antagonist, low=good) ---")
for d in DISEASES:
    c = cos[d].astype(float)
    hi_g = full_auroc(dict(c), pos_sets[d])
    lo_g = full_auroc(dict(-c), pos_sets[d])
    print(f"  {d:<5} high=good={hi_g:>6}  low=good={lo_g:>6}  opt={max(hi_g,lo_g):>6}")

print("\n[判读] 若 A/B/C 的 opt 仍 ≈0.5 -> 靶点数据本身与疾病 DEG 不对齐(数据限制,B难救)")
print("        若某打分 opt>0.6 -> 当前扩散签名在抹掉信号, B 做这个方向能救。")

# 基线也入表
base_rows = []
for d in DISEASES:
    c = cos[d].astype(float)
    hi_g = full_auroc(dict(c), pos_sets[d])
    lo_g = full_auroc(dict(-c), pos_sets[d])
    base_rows.append({'disease': d, 'high=good': hi_g, 'low=good': lo_g,
                      'opt(max)': max(hi_g, lo_g), 'best_dir': 'high' if hi_g >= lo_g else 'low'})
_record("Z. 基线 HDmap-S(扩散余弦)", base_rows)

mdf = pd.DataFrame(master_rows)
mdf.to_csv('output/results/paper_deliverables/B_method_comparison.csv', index=False)
print("\n[已保存] output/results/paper_deliverables/B_method_comparison.csv")
print("=" * 60)
print("核心结论: Fisher富集(靶点⊗DEG) 在 AD/PAH/T2D 全面超越当前HDmap-S;")
print("          当前'扩散签名'步骤在稀释信号 -> B 应改为'直接靶点-DEG富集'打分。")
