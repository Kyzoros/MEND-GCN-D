# -*- coding: utf-8 -*-
"""
P0-a: 收紧靶点 + 加权打分 对照实验
=================================
目标: 验证"每草药靶点太滥交(平均202个/全库1222)"是 AUROC 上不去的主因,
      并找出最优的靶点收紧策略与加权打分。

打分策略:
  F     : Fisher 富集 -log10(p) (纯集合, 当前流水线方法, 基线)
  WHF   : 加权命中占比 = Σ(在DEG内靶点的confidence) / Σ(全部靶点confidence)
  DC    : 方向余弦 = cos(草药有向靶点向量, 疾病logFC向量)  (P2预览, 方向感知)

靶点收紧 N: full(=当前) | 150 | 100 | 75 | 50 | 30 | 20
  按每草药靶点 confidence 权重降序取 Top-N

评估:
  协议A(主): pos vs neg (curated 金标准, 复现 D5 的 0.667/0.750/0.462/0.509)
  协议B:      pos vs 全库 (full 协议)

输出: output/results/paper_deliverables/P0a_target_tightening_results.csv
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
N_CANDIDATES = [None, 150, 100, 75, 50, 30, 20]  # None=全保留

# ---------- 1. 载入数据 ----------
# 草药宇宙: 用当前 enrichment 产物 index (491), 保证与 D5 同域
universe = list(pd.read_csv(os.path.join(BASE, "output", "results",
            "AD_kegg_results", "hdmap_s_scores.csv"), index_col=0).index)
print(f"草药宇宙: {len(universe)}")

# 金标准 (curated)
pos_dict, neg_dict, _ = load_gold_standard(GOLD_STANDARD_PATH, None)

# 疾病 DEG (与流水线 step3 完全一致: |logFC|>0.05 且 P<0.05)
dp = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
deg_sets, deg_signed = {}, {}
for d in DISEASES:
    sub = dp[dp.disease_id == d].copy()
    sub = sub[sub.logFC.abs() > LOGFC_THR]
    if PVAL_THR > 0 and "P_value" in sub.columns:
        sub = sub[sub.P_value < PVAL_THR]
    deg_sets[d] = set(sub.gene_symbol.astype(str))
    deg_signed[d] = dict(zip(sub.gene_symbol.astype(str), sub.logFC.astype(float)))
    print(f"  DEG[{d}] = {len(deg_sets[d])}")

# 草药靶点 (与 hdmap_s.py 完全一致: 成分级 gene->max(conf); 草药级 sum)
it = pd.read_csv(os.path.join(PROC, "ingredient_targets_enhanced.csv"))
hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
ing_targets = defaultdict(dict)          # ingredient -> gene -> (conf, action)
for _, r in it.iterrows():
    g = str(r["target_gene"]).strip()
    if g and g != "nan":
        c = float(r.get("confidence", 0.5) or 0.5)
        a = r.get("action_type", 0)
        a = int(a) if pd.notna(a) else 0
        cur = ing_targets[int(r["ingredient_id"])].get(g)
        if cur is None or c > cur[0]:
            ing_targets[int(r["ingredient_id"])][g] = (c, a)
herb_w = defaultdict(lambda: defaultdict(float))    # herb -> gene -> sum conf
herb_s = defaultdict(lambda: defaultdict(float))    # herb -> gene -> sum conf*action
for _, r in hi.iterrows():
    hid = str(r["herb_id"])
    t = ing_targets.get(int(r["ingredient_id"]), {})
    for g, (c, a) in t.items():
        herb_w[hid][g] += c
        herb_s[hid][g] += c * a
print(f"有靶点草药: {sum(1 for h in herb_w if herb_w[h])}")

# 背景基因集 (Fisher 用, 与流水线一致)
bg = set().union(*deg_sets.values()) | set().union(*[set(w) for w in herb_w.values()])
N_BG = len(bg)
print(f"背景基因集: {N_BG}")

# ---------- 2. 打分函数 ----------
def score_fisher(tv, d):
    tg = set(tv)
    dg = deg_sets[d]
    a = len(tg & dg)
    if a == 0:
        return 0.0
    b = len(tg - dg); c = len(dg - tg)
    dd = max(0, N_BG - a - b - c)
    _, p = fisher_exact([[a, b], [c, dd]])
    return -np.log10(max(p, 1e-300))

def score_whf(tv, d):
    tot = sum(tv.values())
    if tot == 0:
        return 0.0
    hit = sum(v for g, v in tv.items() if g in deg_sets[d])
    return hit / tot

def score_dc(sv, d):
    genes = [g for g in sv if g in deg_signed[d]]
    if len(genes) < 2:
        return 0.0
    a = np.array([sv[g] for g in genes], float)
    b = np.array([deg_signed[d][g] for g in genes], float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def top_n(d, n):
    if n is None:
        return dict(d)
    return dict(sorted(d.items(), key=lambda x: -x[1])[:n])

# ---------- 3. 评估 ----------
def eval_auroc(scores, d, mode):
    pos = set(pos_dict.get(d, []))
    neg = set(neg_dict.get(d, []))
    if mode == "pos_neg":
        ps = [scores[h] for h in pos if h in scores]
        ns = [scores[h] for h in neg if h in scores]
        if len(ps) == 0 or len(ns) == 0:
            return None
        y = [1] * len(ps) + [0] * len(ns)
        s = ps + ns
    else:  # full
        ps = [scores[h] for h in pos if h in scores]
        s_all = [scores[h] for h in universe if h in scores]
        if len(ps) == 0 or len(s_all) == 0:
            return None
        y = [1 if h in pos else 0 for h in universe if h in scores]
        s = s_all
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s))

rows = []
for scorer, tag in [(score_fisher, "F"), (score_whf, "WHF"), (score_dc, "DC")]:
    for n in N_CANDIDATES:
        for d in DISEASES:
            scores = {}
            for h in universe:
                if scorer is score_dc:
                    if not herb_s.get(h):
                        scores[h] = 0.0
                    else:
                        scores[h] = score_dc(top_n(herb_s[h], n), d)
                else:
                    if not herb_w.get(h):
                        scores[h] = 0.0
                    else:
                        scores[h] = scorer(top_n(herb_w[h], n), d)
            a_pn = eval_auroc(scores, d, "pos_neg")
            a_full = eval_auroc(scores, d, "full")
            # 诊断: 翻转方向
            a_pn_lo = eval_auroc({h: -v for h, v in scores.items()}, d, "pos_neg") if a_pn else None
            rows.append({
                "score": tag, "topN": (n if n else "full"),
                "disease": d,
                "AUROC_pos_neg(high)": round(a_pn, 3) if a_pn is not None else None,
                "AUROC_pos_neg(low)": round(a_pn_lo, 3) if a_pn_lo is not None else None,
                "AUROC_full(high)": round(a_full, 3) if a_full is not None else None,
            })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "P0a_target_tightening_results.csv"), index=False)

# ---------- 4. 打印对比表 (主: 子集协议 high=good) ----------
print("\n" + "=" * 100)
print("P0-a 对照: AUROC (协议A: pos vs neg, curated 金标准, high=good)")
print("=" * 100)
piv = df.pivot_table(index=["score", "topN"], columns="disease",
                     values="AUROC_pos_neg(high)", aggfunc="first")
print(piv.to_string())
print("\n--- 诊断: 翻转方向后(pos_neg low=good), 看哪列有救 ---")
piv_lo = df.pivot_table(index=["score", "topN"], columns="disease",
                        values="AUROC_pos_neg(low)", aggfunc="first")
print(piv_lo.to_string())
print("\n[已保存] output/results/paper_deliverables/P0a_target_tightening_results.csv")
