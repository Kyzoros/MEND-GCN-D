# -*- coding: utf-8 -*-
"""
P0-b: 重建"严格关键词"TCMBank 金标准 + 量化金标准不稳定性
=========================================================
目标:
  1. 用 TCMBank(Nature 2023) 以"严格疾病特异词"(删掉补肾/滋阴/关节等宽词)
     重建阳性集, 输出 data/processed/gold_standard_strict.csv
  2. 对同一份打分(enrichment F/full, weighted), 用 3 份金标准(curated/expanded/strict)
     × 2 种协议(pos_neg/full) 评估 AUROC, 量化"换量尺结论反转"的程度。

输出: output/results/paper_deliverables/P0b_gold_instability.csv
"""
import os, sys, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)
EXT = os.path.join(BASE, "data", "external", "herb")
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "output", "results", "paper_deliverables")
os.makedirs(OUT, exist_ok=True)

DISEASES = ["AD", "PAH", "RA", "T2D"]

# ---------- 0. 载入打分 (enrichment 与 weighted 的全库产物) ----------
enrich = pd.read_csv(os.path.join(BASE, "output", "results",
                    "AD_kegg_results", "hdmap_s_scores.csv"), index_col=0)
weight = pd.read_csv(os.path.join(BASE, "output", "results",
                    "AD_results", "hdmap_s_scores.csv"), index_col=0)
universe = list(enrich.index)
print(f"打分矩阵: enrichment {enrich.shape}, weighted {weight.shape}")

# ---------- 1. 重建严格 TCMBank 金标准 ----------
# Latin 归一化 (与 build_expanded_gold_standard.py 一致)
PARTS = {"herba","radix","rhizoma","folium","fructus","cortex","flos","semen",
         "caulis","ramulus","pericarpium","seed","leaf","root","bark","flower",
         "fruit","stem","tuber","bulbus","resina","oleum","lignum","spica",
         "herb","radixet","rhizomaet","et"}
def genus_base(g):
    g = g.lower()
    if len(g) > 4 and g.endswith("ae"): return g[:-2]
    if len(g) > 3 and g.endswith("i"): return g[:-1]
    return g
def norm_lat(s):
    if not isinstance(s, str): return None
    s = s.split(",")[0].lower()
    toks = re.findall(r"[a-z]+", s)
    sci = [t for t in toks if t not in PARTS]
    if not sci: return None
    return (genus_base(sci[0]), sci[1] if len(sci) > 1 else None)

hips = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
proj = hips[["herb_id", "herb_name"]].drop_duplicates()
proj["nl"] = proj["herb_name"].map(norm_lat)
proj_ids = set(proj["herb_id"])
name_by_id = dict(zip(proj["herb_id"], proj["herb_name"]))

sym = pd.read_excel(os.path.join(EXT, "SMHB file.xlsx"))
symmap_lat2cn = {}
for _, r in sym.iterrows():
    nl = norm_lat(str(r["Latin_name"]))
    if nl: symmap_lat2cn[nl] = str(r["Chinese_name"])
proj2symcn = {r["herb_id"]: symmap_lat2cn[r["nl"]] for _, r in proj.iterrows() if r["nl"] in symmap_lat2cn}

tb = pd.read_excel(os.path.join(EXT, "tcmbank_herb_all.xlsx"))
tb["nl"] = tb["Herb_latin_name"].map(norm_lat)
tcm_cn2nl = {}
tb_idx = {}
for _, r in tb.iterrows():
    cn = r["TCM_name"]; nl = r["nl"]
    if pd.notna(cn) and nl: tcm_cn2nl[cn] = nl
    if nl: tb_idx.setdefault(nl, []).append(r)
proj2tcm = {}
for _, r in proj.iterrows():
    hid, nl = r["herb_id"], r["nl"]
    if nl and nl in tb_idx:
        proj2tcm[hid] = (tb_idx[nl][0]["TCM_name"], tb_idx[nl][0]["Herb_latin_name"], "latin")
    elif hid in proj2symcn:
        cn = proj2symcn[hid]
        if cn in tcm_cn2nl:
            sub = tb[tb["TCM_name"] == cn]
            if len(sub):
                t = sub.iloc[0]
                proj2tcm[hid] = (t["TCM_name"], t["Herb_latin_name"], "chinese-bridge")

# 严格关键词 (删除 补肾/滋阴/生津/关节/风湿/健脑/补脑 等宽词)
STRICT_KW = {
    "AD": {"zh": ["痴呆", "老年痴呆", "阿尔茨海默", "健忘", "益智", "记忆力", "醒脑", "开窍", "髓海", "益髓", "痰迷心窍"],
           "en": ["alzheimer", "dementia", "memory", "cognitive", "amnesia", "neuroprotect"]},
    "RA": {"zh": ["类风湿", "痹证", "历节", "除痹", "祛风除湿"],
           "en": ["rheumatoid", "arthritis", "arthralgia"]},
    "T2D": {"zh": ["消渴", "消渴症", "糖尿病", "降糖", "降血糖"],
            "en": ["diabet", "glucose", "hypoglyc"]},
    "PAH": {"zh": [],
            "en": ["pulmonary hypertens", "cor pulmonale", "pulmonary arterial",
                   "right ventricular hypertroph", "pulmonary vascular"]},
}
def herb_matches(row, kw):
    cols = ["Indication", "Function", "Clinical_manifestations", "Therapeutic_en_class", "TCM_name_en"]
    texts = " ".join(str(row[c]) for c in cols if c in row and pd.notna(row[c])).lower()
    return any(k in texts for k in kw["zh"]) or any(k in texts for k in kw["en"])

strict_pos = {}
for dis, kw in STRICT_KW.items():
    hits = tb[tb.apply(lambda r: herb_matches(r, kw), axis=1)]
    s = set()
    for _, t in hits.iterrows():
        cn = t["TCM_name"]; lat = t["Herb_latin_name"]; nl = norm_lat(lat)
        hid = None
        if nl and nl in tb_idx:
            for h, v in proj2tcm.items():
                if norm_lat(v[1]) == nl:
                    hid = h; break
        if hid is None and cn in tcm_cn2nl:
            for h, v in proj2tcm.items():
                if v[0] == cn:
                    hid = h; break
        if hid is not None and hid in proj_ids:
            s.add(hid)
    strict_pos[dis] = s
    print(f"[严格TCMBank] {dis}: 阳性 {len(s)} 味 -> {[name_by_id.get(h,'?') for h in sorted(s)]}")

# 写严格金标准 (阳性 + 等量随机阴性, 固定种子42)
import random
random.seed(42)
rows = []
for dis in DISEASES:
    pos = strict_pos[dis]
    cand = [h for h in proj_ids if h not in pos]
    random.shuffle(cand)
    for h in pos:
        rows.append({"herb_id": h, "disease_id": dis, "label": 1, "source": "TCMBank-strict"})
    for h in cand[:len(pos)]:
        rows.append({"herb_id": h, "disease_id": dis, "label": 0, "source": "random-unrelated"})
gs_strict = pd.DataFrame(rows)
gs_strict.to_csv(os.path.join(BASE, "data", "processed", "gold_standard_strict.csv"), index=False)
print(f"[写] data/processed/gold_standard_strict.csv 共 {len(gs_strict)} 行")

# ---------- 2. 载入其他两份金标准 ----------
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
from config import GOLD_STANDARD_PATH
from evaluation_metrics import load_gold_standard
def pos_from_gold(path):
    pos_dict, neg_dict, _ = load_gold_standard(path, None)
    return pos_dict, neg_dict
pos_cur, neg_cur = pos_from_gold(GOLD_STANDARD_PATH)  # curated
pos_exp, neg_exp = pos_from_gold(os.path.join(BASE, "data", "gold_standard_expanded.csv"))  # expanded
pos_strict = {d: set(gs_strict[(gs_strict.disease_id == d) & (gs_strict.label == 1)].herb_id.astype(str)) for d in DISEASES}
neg_strict = {d: set(gs_strict[(gs_strict.disease_id == d) & (gs_strict.label == 0)].herb_id.astype(str)) for d in DISEASES}

GOLDS = {"curated": (pos_cur, neg_cur),
         "expanded": (pos_exp, neg_exp),
         "strictTCM": (pos_strict, neg_strict)}

# ---------- 3. 评估: 同一打分 × 多金标准 × 多协议 ----------
def auroc(scores_dict, pos, neg, mode):
    if mode == "pos_neg":
        ps = [scores_dict[h] for h in pos if h in scores_dict]
        ns = [scores_dict[h] for h in neg if h in scores_dict]
        if not ps or not ns: return None
        y = [1]*len(ps) + [0]*len(ns); s = ps + ns
    else:
        ps = [scores_dict[h] for h in pos if h in scores_dict]
        s = [scores_dict[h] for h in universe if h in scores_dict]
        y = [1 if h in pos else 0 for h in universe if h in scores_dict]
        if not ps: return None
    if len(set(y)) < 2: return None
    return round(float(roc_auc_score(y, s)), 3)

METHODS = {"enrichment(high好)": (enrich, False),
           "weighted(low好)": (weight, True)}  # flip: weighted low好 -> 反号后 high好
summary = []
for gname, (pos_d, neg_d) in GOLDS.items():
    for d in DISEASES:
        for mname, (df, flip) in METHODS.items():
            sc = {h: (-float(df.loc[h, d]) if flip else float(df.loc[h, d])) for h in universe}
            a_pn = auroc(sc, pos_d.get(d, set()), neg_d.get(d, set()), "pos_neg")
            a_full = auroc(sc, pos_d.get(d, set()), neg_d.get(d, set()), "full")
            npos = len([h for h in pos_d.get(d, set()) if h in universe])
            summary.append({"gold": gname, "disease": d, "method": mname,
                            "n_pos": npos,
                            "AUROC_pos_neg": a_pn, "AUROC_full": a_full})
sdf = pd.DataFrame(summary)
sdf.to_csv(os.path.join(OUT, "P0b_gold_instability.csv"), index=False)

# ---------- 4. 打印 ----------
print("\n" + "=" * 100)
print("金标准不稳定性: 同一打分 × 3 金标准 × 2 协议 (AUROC)")
print("=" * 100)
for d in DISEASES:
    print(f"\n--- {d} ---")
    for m in ["enrichment(high好)", "weighted(low好)"]:
        sub = sdf[(sdf.disease == d) & (sdf.method == m)]
        line = "  ".join(f"{r.gold}: pos_neg={r.AUROC_pos_neg}, full={r.AUROC_full} (n_pos={r.n_pos})"
                         for r in sub.itertuples())
        print(f"  {m:20s}  {line}")
print("\n[已保存] output/results/paper_deliverables/P0b_gold_instability.csv")
