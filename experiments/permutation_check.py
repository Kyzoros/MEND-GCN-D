# -*- coding: utf-8 -*-
"""
置换检验重算 (表4): 用与 run_D8_final_comparison.py 完全相同的打分函数与金标准,
重算 D-TMS 观测 AUROC 及 靶点置换 / DEG 置换 零分布的 p 值, 保证表4 与表1 完全一致、可复现。
"""
import os, sys
import numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import run_D8_final_comparison as d8   # 复用打分函数与全局数据 (import 会刷新 D8 指标 csv)

DISEASES = ["AD", "PAH", "RA", "T2D"]
bg_list = list(d8.bg)
rng = np.random.default_rng(20260830)  # 独立种子, 不依赖 import run_D8 的副作用, 保证可复现
N_PERM = 1000

obs = {}
for d in DISEASES:
    sc = d8.score_fisher(d8.herb_w, d)
    obs[d] = d8.eval_auroc(sc, d)

def score_fisher_deg(herb_w, deg_set, bgN):
    out = {}
    for h in d8.universe:
        tg = set(herb_w[h])
        a = len(tg & deg_set)
        if a == 0:
            out[h] = 0.0; continue
        b = len(tg - deg_set); c = len(deg_set - tg); dd = max(0, bgN - a - b - c)
        _, p = fisher_exact([[a, b], [c, dd]])
        out[h] = -np.log10(max(p, 1e-300))
    return out

target_null = {d: [] for d in DISEASES}
for it in range(N_PERM):
    hw_perm = {}
    for h in d8.universe:
        genes = list(d8.herb_w[h].keys())
        pg = rng.choice(bg_list, size=len(genes), replace=False)
        hw_perm[h] = {g: v for g, v in zip(pg, d8.herb_w[h].values())}
    for d in DISEASES:
        sc = score_fisher_deg(hw_perm, d8.deg_sets[d], d8.N_BG)
        target_null[d].append(d8.eval_auroc(sc, d))

deg_null = {d: [] for d in DISEASES}
for it in range(N_PERM):
    for d in DISEASES:
        pg = rng.choice(bg_list, size=len(d8.deg_sets[d]), replace=False)
        sc = score_fisher_deg(d8.herb_w, set(pg), d8.N_BG)
        deg_null[d].append(d8.eval_auroc(sc, d))

rows = []
for d in DISEASES:
    tn = np.array(target_null[d]); dn = np.array(deg_null[d]); o = obs[d]
    tp = (int(sum(tn >= o)) + 1) / (N_PERM + 1)   # 保守单侧经验 p
    dp = (int(sum(dn >= o)) + 1) / (N_PERM + 1)
    rows.append({"disease": d, "obs": round(o, 3), "targ_mean": round(tn.mean(), 3),
                 "targ_p95": round(np.percentile(tn, 95), 3), "targ_p": tp, "deg_p": dp})
print("=== 表4 置换检验 (D-TMS, gold_standard_final) ===")
for r in rows:
    print(f"{r['disease']}: obs={r['obs']} targ_mean={r['targ_mean']} targ_p95={r['targ_p95']} "
          f"targ_p={r['targ_p']} deg_p={r['deg_p']}")
