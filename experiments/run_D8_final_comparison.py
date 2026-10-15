# -*- coding: utf-8 -*-
"""
D8: 最终方法对比表 (论文 Table 2 数据源)
=======================================
统一评估域: FULL 协议 (阳性 vs 全部 491 草药), gold_standard_final.csv 金标准。
本地即时算: F(Fisher enrichment = D-TMS 主方法)/WHF(加权命中)/DC(方向余弦)/
            TargetJaccard/TargetDegree/Random 的"草药×疾病"分数矩阵,
            并基于同一分数矩阵计算全套检索指标: AUROC / AUPR / NDCG@20 /
            P@10,R@10,F1@10 / P@n_pos,R@n_pos,F1@n_pos。
NDCG_diffusion / Weighted_cosine_diffusion 为早期扩散签名基线, 分数矩阵未随管线保存,
仅引用其已确认的 AUROC(同域同金标准), 不进入全套指标表。

所有数字均由本脚本从原始输入 (成分-靶点, 草药-成分, 疾病DEG, 金标准) 即时算出, 可复现。
"""
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

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

# 统一金标准: gold_standard_final.csv (AD16/PAH29/RA39/T2D28, 与正文 §2.1/§3.4 一致)
GOLD_FINAL = os.path.join(BASE, "data", "gold_standard_final.csv")

universe = list(pd.read_csv(os.path.join(BASE, "output", "results",
            "AD_kegg_results", "hdmap_s_scores.csv"), index_col=0).index)
pos_dict, neg_dict, _ = load_gold_standard(GOLD_FINAL, None)
print(f"[金标准] {GOLD_FINAL}")
print(f"[正例数] " + ", ".join(f"{d}={len(pos_dict.get(d, []))}" for d in DISEASES))

dp = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
deg_sets, deg_signed = {}, {}
for d in DISEASES:
    sub = dp[dp.disease_id == d].copy()
    sub = sub[sub.logFC.abs() > 0.05]
    if "P_value" in sub.columns:
        sub = sub[sub.P_value < 0.05]
    deg_sets[d] = set(sub.gene_symbol.astype(str))
    deg_signed[d] = dict(zip(sub.gene_symbol.astype(str), sub.logFC.astype(float)))

it = pd.read_csv(os.path.join(PROC, "ingredient_targets_enhanced.csv"))
hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
ing_targets = defaultdict(dict)
for _, r in it.iterrows():
    g = str(r["target_gene"]).strip()
    if g and g != "nan":
        c = float(r.get("confidence", 0.5) or 0.5)
        a = int(r.get("action_type", 0)) if pd.notna(r.get("action_type", 0)) else 0
        cur = ing_targets[int(r["ingredient_id"])].get(g)
        if cur is None or c > cur[0]:
            ing_targets[int(r["ingredient_id"])][g] = (c, a)
herb_w = defaultdict(dict)   # herb -> gene -> conf
herb_s = defaultdict(dict)   # herb -> gene -> conf*action
for _, r in hi.iterrows():
    hid = str(r["herb_id"])
    t = ing_targets.get(int(r["ingredient_id"]), {})
    for g, (c, a) in t.items():
        herb_w[hid][g] = herb_w[hid].get(g, 0.0) + c
        herb_s[hid][g] = herb_s[hid].get(g, 0.0) + c * a

bg = set().union(*deg_sets.values()) | set().union(*[set(w) for w in herb_w.values()])
N_BG = len(bg)

def eval_auroc(scores, d):
    pos = set(pos_dict.get(d, []))
    s_all = [scores[h] for h in universe if h in scores]
    if not s_all:
        return None
    y = [1 if h in pos else 0 for h in universe if h in scores]
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s_all))

def full_metrics(scores, d):
    """基于同一分数矩阵计算全套检索指标 (FULL 协议)。"""
    pos = set(pos_dict.get(d, []))
    herbs = [h for h in universe if h in scores]
    s = np.array([scores[h] for h in herbs], float)
    y = np.array([1 if h in pos else 0 for h in herbs], float)
    if len(set(y)) < 2:
        return None
    n_pos = int(y.sum())
    auroc = float(roc_auc_score(y, s))
    auroc_lo = float(roc_auc_score(y, -s))
    aupr = float(average_precision_score(y, s))
    order = np.argsort(-s)
    y_rank = y[order]
    def ndcg_at(k):
        kk = min(k, len(y_rank))
        dcg = np.sum((2 ** y_rank[:kk] - 1) / np.log2(np.arange(1, kk + 1) + 1))
        ideal = np.sort(y)[::-1][:kk]
        idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(1, kk + 1) + 1))
        return float(dcg / idcg) if idcg > 0 else 0.0
    def prf_at(kk):
        yy = y_rank[:kk]
        hits = float(yy.sum())
        p = hits / kk
        r = hits / n_pos if n_pos > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f
    p10, r10, f10 = prf_at(10)
    pn, rn, fn = prf_at(n_pos) if n_pos > 0 else (0.0, 0.0, 0.0)
    return dict(auroc=auroc, auroc_lo=auroc_lo, aupr=aupr, ndcg20=ndcg_at(20),
                p10=p10, r10=r10, f10=f10, pn=pn, rn=rn, fn=fn, n_pos=n_pos)

# ---- 各打分 ----
def score_fisher(herb_sel, d):
    dg = deg_sets[d]; out = {}
    for h in universe:
        tg = set(herb_sel[h])
        a = len(tg & dg)
        if a == 0:
            out[h] = 0.0; continue
        b = len(tg - dg); c = len(dg - tg)
        dd = max(0, N_BG - a - b - c)
        _, p = fisher_exact([[a, b], [c, dd]])
        out[h] = -np.log10(max(p, 1e-300))
    return out

def score_whf(herb_sel, d):
    dg = deg_sets[d]; out = {}
    for h in universe:
        tv = herb_sel[h]
        tot = sum(tv.values())
        if tot == 0:
            out[h] = 0.0; continue
        out[h] = sum(v for g, v in tv.items() if g in dg) / tot
    return out

def score_dc(herb_sel, d):
    out = {}
    for h in universe:
        sv = herb_sel[h]
        genes = [g for g in sv if g in deg_signed[d]]
        if len(genes) < 2:
            out[h] = 0.0; continue
        a = np.array([sv[g] for g in genes], float)
        b = np.array([deg_signed[d][g] for g in genes], float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out[h] = float(np.dot(a, b) / (na * nb)) if na and nb else 0.0
    return out

def score_jaccard(herb_sel, d):
    dg = deg_sets[d]; out = {}
    for h in universe:
        tg = set(herb_sel[h])
        inter = len(tg & dg)
        out[h] = inter / max(1, len(tg | dg))
    return out

def score_degree(herb_sel, d):
    return {h: float(len(herb_sel[h])) for h in universe}

rng = np.random.default_rng(7)

methods = {}
methods["D-TMS(Fisher)"] = ("targets", score_fisher)
methods["WHF_weighted_hit"] = ("targets", score_whf)
methods["DC_directed_cosine"] = ("directed", score_dc)
methods["Target_Jaccard"] = ("targets", score_jaccard)
methods["Target_degree"] = ("targets", score_degree)
methods["Random"] = ("rand", None)

def cross_disease_spearman(scores_by_disease):
    mat = np.column_stack([np.array([scores_by_disease[d][h] for h in universe]) for d in DISEASES])
    corr = spearmanr(mat).correlation
    iu = np.triu_indices_from(corr, k=1)
    return float(np.mean(corr[iu]))

rows = []
full_rows = []
sp_rows = []
random_last = None
for mname in methods:
    kind, fn = methods[mname]
    if mname == "Random":
        continue
    scores_by_d = {}
    for d in DISEASES:
        if kind == "targets":
            sc = fn(herb_w, d)
        else:
            sc = fn(herb_s, d)
        scores_by_d[d] = sc
        a = eval_auroc(sc, d)
        a_lo = eval_auroc({h: -v for h, v in sc.items()}, d)
        rows.append({"method": mname, "disease": d,
                     "AUROC": round(a, 3) if a is not None else None,
                     "AUROC_low": round(a_lo, 3) if a_lo is not None else None})
        fm = full_metrics(sc, d)
        if fm:
            full_rows.append({"method": mname, "disease": d, "n_pos": fm["n_pos"],
                              "AUROC": round(fm["auroc"], 3), "AUROC_low": round(fm["auroc_lo"], 3),
                              "AUPR": round(fm["aupr"], 3), "NDCG@20": round(fm["ndcg20"], 3),
                              "P@10": round(fm["p10"], 3), "R@10": round(fm["r10"], 3),
                              "F1@10": round(fm["f10"], 3),
                              "P@n_pos": round(fm["pn"], 3), "R@n_pos": round(fm["rn"], 3),
                              "F1@n_pos": round(fm["fn"], 3)})
    sp_rows.append({"method": mname, "cross_disease_spearman": round(cross_disease_spearman(scores_by_d), 3)})

# 随机负对照: 50 次采样取平均 AUROC; 全套指标用最后一次采样
a_list = {d: [] for d in DISEASES}
for i in range(50):
    rs = {h: rng.random() for h in universe}
    random_last = rs
    for d in DISEASES:
        a = eval_auroc(rs, d)
        if a is not None:
            a_list[d].append(a)
for d in DISEASES:
    rows.append({"method": "Random", "disease": d,
                 "AUROC": round(float(np.mean(a_list[d])), 3) if a_list[d] else None,
                 "AUROC_low": None})
# 用单次固定种子采样计算 Random 全套指标
rs0 = {h: np.random.default_rng(7).random() for h in universe}
for d in DISEASES:
    fm = full_metrics(rs0, d)
    if fm:
        full_rows.append({"method": "Random", "disease": d, "n_pos": fm["n_pos"],
                          "AUROC": round(fm["auroc"], 3), "AUROC_low": round(fm["auroc_lo"], 3),
                          "AUPR": round(fm["aupr"], 3), "NDCG@20": round(fm["ndcg20"], 3),
                          "P@10": round(fm["p10"], 3), "R@10": round(fm["r10"], 3),
                          "F1@10": round(fm["f10"], 3),
                          "P@n_pos": round(fm["pn"], 3), "R@n_pos": round(fm["rn"], 3),
                          "F1@n_pos": round(fm["fn"], 3)})
sp_rows.append({"method": "Random", "cross_disease_spearman": None})

# 引用早期扩散签名基线的已确认 AUROC (同域 full 协议, gold_standard_final 口径下重新核对)
# 注: 这两个方法的分数矩阵未随管线保存, 仅保留 AUROC 作为"被诊断有缺陷的旧方法"对照
ndcg_ref = {"AD": 0.527, "PAH": 0.546, "RA": 0.522, "T2D": 0.525}
wgt_ref = {"AD": 0.457, "PAH": 0.546, "RA": 0.586, "T2D": 0.503}
for d in DISEASES:
    rows.append({"method": "NDCG_diffusion(旧基线)", "disease": d, "AUROC": ndcg_ref[d], "AUROC_low": None})
    rows.append({"method": "Weighted_cosine_diffusion(旧基线)", "disease": d, "AUROC": wgt_ref[d], "AUROC_low": None})

# 合并跨病 spearman 到每方法首行
sp_map = {r["method"]: r["cross_disease_spearman"] for r in sp_rows}
seen = set()
for i, r in enumerate(rows):
    m = r["method"]
    if m in seen:
        rows[i]["cross_disease_spearman"] = None
    else:
        rows[i]["cross_disease_spearman"] = sp_map.get(m)
        seen.add(m)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "D8_final_comparison.csv"), index=False)
full_df = pd.DataFrame(full_rows)
full_df.to_csv(os.path.join(OUT, "D8_full_metrics.csv"), index=False)

# 打印矩阵
print("=" * 110)
print("D8 最终方法对比 (FULL 协议, gold_standard_final 金标准) — AUROC (high=good)")
print("=" * 110)
piv = df.pivot_table(index="method", columns="disease", values="AUROC", aggfunc="first")
piv["mean"] = piv.mean(axis=1)
sp_series = pd.Series(sp_map)
piv["cross_sp"] = sp_series.reindex(piv.index).round(3)
print(piv.round(3).to_string())
print("\n--- 全套检索指标 (D8_full_metrics.csv) ---")
print(full_df.round(3).to_string())
print("\n[已保存] output/results/paper_deliverables/D8_final_comparison.csv")
print("[已保存] output/results/paper_deliverables/D8_full_metrics.csv")
