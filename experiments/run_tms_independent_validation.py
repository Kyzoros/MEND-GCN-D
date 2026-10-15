"""
run_tms_independent_validation.py
=================================
用两套"不偏的尺子"评估 TMS，验证它是否真能识别治疗草药(而非靠靶点数量水分)：
  尺子A: TCMBank 主治文本金标准 (独立来源，不靠靶点数量)
  尺子B: CMap 外部基因扰动验证 (完全独立于靶点数量)
同时测"原始 enrichment TMS" 和 "size-corrected TMS"(扣靶点数量水分)，
对比两者在独立尺子上的表现，回答方案A去偏倚到底有没有用。
"""
import os
import sys
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
PROC = os.path.join(HERE, "data", "processed")
RES = os.path.join(HERE, "output", "results")
RAW_TMS = os.path.join(PROC, "hdmap_s_scores_enrichment_8d.csv")
TCM_GOLD = os.path.join(PROC, "gold_tcmbank_indication.csv")
TARGETS = os.path.join(PROC, "ingredient_targets_processed.csv")
HERB_ING = os.path.join(PROC, "herb_ingredients_processed.csv")
CMAP_DIR = os.path.join(PROC, "external_raw")
OUT = os.path.join(PROC, "tms_independent_validation.csv")

DISEASES = ["AD", "PAH", "RA", "T2D", "HCC", "CRC", "ASTHMA", "OA"]
CMAP_FILE = {"AD": "Alzheimer_cmap2_data.xlsx", "PAH": "PAH_cmap2_data.xlsx",
             "RA": "RA_cmap2_data.xlsx", "T2D": "T2D_cmap2_data.xlsx"}
STOP = {'acid', 'the', 'and', 'of', 'extract', 'oil', 'root', 'chinese', 'powder',
        'membranous', 'leaf', 'fruit', 'seed', 'flower', 'stem', 'rootof'}

# ---- 载入原始 enrichment TMS ----
raw = pd.read_csv(RAW_TMS).set_index("herb_id")
all_herbs = raw.index.tolist()

# ---- 靶点数量 + size-corrected (线性残差扣水分) ----
ing = pd.read_csv(HERB_ING)
tg = pd.read_csv(TARGETS)
hi = "herb_id" if "herb_id" in ing.columns else ing.columns[0]
ii = "ingredient_id" if "ingredient_id" in ing.columns else ing.columns[1]
ti = "ingredient_id" if "ingredient_id" in tg.columns else tg.columns[0]
tgc = "target_gene" if "target_gene" in tg.columns else tg.columns[1]
mm = ing[[hi, ii]].merge(tg[[ti, tgc]], left_on=ii, right_on=ti)
tcount = mm.groupby(hi)[tgc].nunique().reindex(raw.index)

corr = raw.copy()
for d in raw.columns:
    s = raw[d].values.astype(float)
    x = tcount.values.astype(float)
    mask = ~np.isnan(s) & ~np.isnan(x)
    coef = np.polyfit(x[mask], s[mask], 1)
    corr[d] = s - np.polyval(coef, x)

# ---- 尺子A: TCMBank 主治文本金标准 ----
tcm = pd.read_csv(TCM_GOLD)
tcm_pos = {d: set(tcm[(tcm.disease_id == d) & (tcm.label == 1)]["herb_id"])
           for d in DISEASES}


def auc(y, s):
    s = np.asarray(s, float)
    y = np.asarray(y)
    m = ~np.isnan(s)
    s = s[m]
    y = y[m]
    if len(set(y)) < 2:
        return float('nan')
    return roc_auc_score(y, s)


# ---- 尺子B: CMap 外部基因扰动 (成分->草药 hybrid 匹配 + max 聚合) ----
hip = pd.read_csv(HERB_ING)
ing2herbs = {}
for _, r in hip.iterrows():
    n = str(r['ingredient_name']).lower().strip()
    n = re.sub(r'[^a-z0-9 ]', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()
    if len(n) >= 4:
        ing2herbs.setdefault(n, set()).add(str(r['herb_id']))


def core_terms(name):
    toks = re.split(r'[^a-z0-9]+', name.lower())
    return [t for t in toks if len(t) >= 4 and t not in STOP]


def norm_ing(s):
    if not isinstance(s, str):
        return ''
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def match_hybrid(name):
    h = set(ing2herbs.get(norm_ing(name), set()))
    if h:
        return h
    terms = set(core_terms(name))
    if not terms:
        return set()
    hit = set()
    for k, hset in ing2herbs.items():
        if terms.issubset(set(k.split())):
            hit |= hset
    return hit


rows = []
for d in DISEASES:
    # 尺子A
    y = np.array([1 if h in tcm_pos[d] else 0 for h in all_herbs])
    a_auc_raw = auc(y, raw[d].values)
    a_auc_corr = auc(y, corr[d].values)

    # 尺子B (仅 4 病有 CMap 外部数据; 新 4 病 HCC/CRC/ASTHMA/OA 暂缺)
    b_rho_raw = b_p_raw = b_auc_raw = b_rho_corr = b_p_corr = b_auc_corr = float('nan')
    if d in CMAP_FILE:
        df = pd.read_excel(os.path.join(CMAP_DIR, CMAP_FILE[d]))
        ing_df = df[df['Type'] == 'Ingredient']
        tmp = {}
        for _, r in ing_df.iterrows():
            hs = match_hybrid(str(r['Drug name']))
            sc = float(r['Summary Score'])
            for h in hs:
                tmp.setdefault(str(h), []).append(sc)
        ext = pd.DataFrame([(h, float(np.max(sl))) for h, sl in tmp.items()],
                           columns=['herb_id', 'ext_score'])
        ext['herb_id'] = ext['herb_id'].astype(str)
        for tag, tms in [("raw", raw), ("corrected", corr)]:
            m2 = ext.merge(tms[[d]].reset_index(), on='herb_id', how='inner').dropna()
            if len(m2) < 10:
                print(f"  {d}/{tag}: CMap 重叠草药过少 ({len(m2)}) 跳过")
                continue
            our = m2[d].values.astype(float)
            exs = m2['ext_score'].values.astype(float)
            rho, p = spearmanr(our, exs)
            thr = np.quantile(exs, 0.70)
            label = (exs >= thr).astype(int)
            b_auc = auc(label, our)
            if tag == "raw":
                b_rho_raw, b_p_raw, b_auc_raw = rho, p, b_auc
            else:
                b_rho_corr, b_p_corr, b_auc_corr = rho, p, b_auc
    else:
        print(f"  {d}: 无 CMap 外部数据, 尺子B 跳过")

    rows.append({
        "disease": d,
        "A_auc_raw": round(a_auc_raw, 3), "A_auc_corrected": round(a_auc_corr, 3),
        "B_rho_raw": round(b_rho_raw, 3), "B_rho_corrected": round(b_rho_corr, 3),
        "B_auc_raw": round(b_auc_raw, 3), "B_auc_corrected": round(b_auc_corr, 3),
    })
    print(f"{d}: 尺子A(TCMBank) AUROC {a_auc_raw:.3f}->{a_auc_corr:.3f} | "
          f"尺子B(CMap) rho {b_rho_raw:+.3f}->{b_rho_corr:+.3f}, "
          f"AUROC {b_auc_raw:.3f}->{b_auc_corr:.3f}")

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f"\n写入: {OUT}")
print("解读: corrected 在独立尺子上 >= raw -> 方案A去偏倚有效且非靠curated水分")
