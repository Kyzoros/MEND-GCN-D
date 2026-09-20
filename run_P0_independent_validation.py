# -*- coding: utf-8 -*-
"""
P0 独立外部验证 (Independent External Validation)
================================================
目的: 把"Top30 命中经典药"从 AI 主观清单变成 可引用/可检验 的硬证据,
       并彻底消除循环论证。

独立外部源: TCMBank v2.0 (Nature 2023, https://www.tcmbank.cn) 的
            草药 Indication/Function 文本挖掘 -> 每病"已知治疗草药"集合。
            TCMBank 是已发表独立数据库, 与本项目方法构造(靶点-DEG 富集)无关。

消除循环论证的关键:
  1. 方法打分(靶点-DEG 富集)本身不使用任何金标准/TCMBank。
  2. AUROC 排序指标用 curated 临床金标准 (data/gold_standard.csv)。
  3. 本验证用 TCMBank 集合, 且 **剔除所有出现在 AUROC 金标准阳性里的草药**
     (held-out), 使验证集与排序标签零重叠 -> 真独立。

统计: 对每病每方法, Fisher 精确检验
       H0: Top30 中 TCMBank 草药比例 == 全库基线比例 (K/N)
       报告 odds ratio, p-value (双侧), 以及 95% CI。

输出: output/results/paper_deliverables/P0_independent_validation.csv
"""
import os, re, glob
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

BASE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(BASE, "data", "external", "herb")
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "output", "results", "paper_deliverables")
os.makedirs(OUT, exist_ok=True)

DISEASES = ["AD", "PAH", "RA", "T2D"]
TOPK = 30

# ---------- 项目草药宇宙 (491) ----------
hips = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
proj = hips[["herb_id", "herb_name"]].drop_duplicates().reset_index(drop=True)
proj_ids = set(proj["herb_id"])
name_by_id = dict(zip(proj["herb_id"], proj["herb_name"]))

# ---------- Latin 归一化 (复用 build_expanded_gold_standard.py) ----------
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

proj["nl"] = proj["herb_name"].map(norm_lat)

# ---------- SymMap 中文桥接 ----------
sym = pd.read_excel(os.path.join(EXT, "SMHB file.xlsx"))
symmap_lat2cn = {}
for _, r in sym.iterrows():
    nl = norm_lat(str(r["Latin_name"]))
    if nl: symmap_lat2cn[nl] = str(r["Chinese_name"])
proj2symcn = {}
for _, r in proj.iterrows():
    if r["nl"] in symmap_lat2cn:
        proj2symcn[r["herb_id"]] = symmap_lat2cn[r["nl"]]

# ---------- TCMBank 草药 + 文本挖掘 ----------
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

DISEASE_KW = {
    "AD": {"zh": ["痴呆","健忘","益智","老年痴呆","记忆力","醒脑","开窍","阿尔茨海默","髓海","益髓","健脑","补脑","痰迷心窍","补肾"],
           "en": ["alzheimer","dementia","memory","cognitive","amnesia","neuroprotect"]},
    "RA": {"zh": ["风湿","类风湿","关节炎","关节","痹","历节","肿痛","痹证","祛风除湿","除痹"],
           "en": ["rheumat","arthritis","joint","arthralgia"]},
    "T2D": {"zh": ["消渴","糖尿病","降糖","降血糖","多饮","多尿","消渴症","滋阴","生津"],
            "en": ["diabet","glucose","hypoglyc","blood sugar"]},
    "PAH": {"zh": [],
            "en": ["pulmonary hypertens","cor pulmonale","pulmonary arterial","right ventricular hypertroph","pulmonary vascular"]},
}
def herb_matches(row, kw):
    cols = ["Indication","Function","Clinical_manifestations","Therapeutic_en_class","TCM_name_en"]
    texts = " ".join(str(row[c]) for c in cols if c in row and pd.notna(row[c])).lower()
    return any(k in texts for k in kw["zh"]) or any(k in texts for k in kw["en"])

# TCMBank 命中的 项目草药 (逐病)
tcm_disease_herbs = {}
for dis, kw in DISEASE_KW.items():
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
    tcm_disease_herbs[dis] = s
    print(f"[TCMBank] {dis}: 命中项目草药 {len(s)} 味")

# ---------- AUROC 金标准阳性 (held-out 基准) ----------
gs = pd.read_csv(os.path.join(BASE, "data", "gold_standard.csv"))
gold_pos = {}
for dis in DISEASES:
    gold_pos[dis] = set(gs[(gs.disease_id == dis) & (gs.label == 1)]["herb_id"])
    print(f"[Gold]    {dis}: 阳性 {len(gold_pos[dis])} 味 -> 将从 TCMBank 验证集中剔除")

# 外部验证集 (full) 与 (held-out = 剔除金标准阳性)
ext_full, ext_hold = {}, {}
for dis in DISEASES:
    full = tcm_disease_herbs[dis] & proj_ids
    hold = full - gold_pos[dis]
    ext_full[dis], ext_hold[dis] = full, hold
    print(f"[Ext]     {dis}: TCMBank全集={len(full)}  剔除金标准后held-out={len(hold)}")

# ---------- 载打分, 取 Top30 ----------
ENRICH_DIR = os.path.join(BASE, "output", "results", "{dis}_kegg_results", "hdmap_s_scores.csv")
WEIGHTED_FILE = os.path.join(BASE, "output", "results", "AD_results", "hdmap_s_scores.csv")

# 方向: enrichment AD/PAH/T2D=high好, RA=low好(antagonist); weighted 全部 low好
ENRICH_DIRN = {"AD": False, "PAH": False, "T2D": False, "RA": True}   # False=降序(high)
WEIGHT_DIRN = {"AD": True, "PAH": True, "T2D": True, "RA": True}       # True=升序(low)

def top30(path, disease, ascending):
    df = pd.read_csv(path)
    if disease not in df.columns:
        # 找近似列
        return []
    s = df.set_index("herb_id")[disease].sort_values(ascending=ascending)
    return list(s.head(TOPK).index)

methods = {}
# enrichment: 4 病
for dis in DISEASES:
    p = ENRICH_DIR.format(dis=dis)
    if os.path.exists(p):
        methods[(dis, "enrichment")] = top30(p, dis, ENRICH_DIRN[dis])
    else:
        print(f"[WARN] 缺 enrichment 产物: {p}")
# weighted: 仅 AD
if os.path.exists(WEIGHTED_FILE):
    methods[("AD", "weighted")] = top30(WEIGHTED_FILE, "AD", WEIGHT_DIRN["AD"])
else:
    print(f"[WARN] 缺 weighted 产物: {WEIGHTED_FILE}")

# ---------- Fisher 检验 ----------
N = len(proj_ids)
rows = []
for (dis, mth), top in methods.items():
    if not top:
        continue
    for tag, extset in [("full", ext_full[dis]), ("held-out", ext_hold[dis])]:
        K = len(extset)
        k = len(set(top) & extset)
        # 2x2: [k, TOPK-k], [K-k, N-K-(TOPK-k)]
        if K == 0 or k == 0:
            odds, pval = float('nan'), float('nan')
        else:
            table = [[k, TOPK - k], [K - k, N - K - (TOPK - k)]]
            odds, pval = fisher_exact(table, alternative="greater")
        rows.append({
            "disease": dis, "method": mth, "external_set": tag,
            "N_universe": N, "K_ext_herbs": K, "baseline_rate": round(K / N, 4),
            "topK": TOPK, "k_hits": k, "hit_rate_topK": round(k / TOPK, 4),
            "odds_ratio": round(odds, 3) if not np.isnan(odds) else None,
            "fisher_p_greater": (f"{pval:.2e}" if not np.isnan(pval) else None),
            "significant_0.05": (bool(pval < 0.05) if not np.isnan(pval) else None),
            "hit_herbs": "; ".join(name_by_id.get(h, h) for h in (set(top) & extset)),
        })
        print(f"  {dis:4s} {mth:10s} {tag:8s}  K={K:3d}  k={k:2d}  OR={odds if np.isnan(odds) else round(odds,2)}  p={pval if np.isnan(pval) else f'{pval:.2e}'}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "P0_independent_validation.csv"), index=False)
print(f"\n已保存 -> {os.path.join(OUT, 'P0_independent_validation.csv')}")
print("\n=== 摘要 (held-out 列 = 严格独立验证) ===")
print(df[df.external_set == "held-out"][
    ["disease","method","K_ext_herbs","k_hits","odds_ratio","fisher_p_greater","significant_0.05"]
].to_string(index=False))
