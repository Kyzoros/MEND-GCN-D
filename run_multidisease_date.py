#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多病 DATE 打分 (杠杆①): 用 TCM-suite 疾病基因集 + 我们 491 草药靶点,
在 TCM-suite herb-disease 金标准 (映射回项目 Herb_N) 上算多病 AUROC。

- degs: 从 tcmsuite_disease_target.csv 取选定疾病的 gene 集合
- 金标准: tcmsuite_herb_disease.csv 已知 (herb,disease) -> fuzzy 映射回 Herb_N
- 富集: 复用 pair_single_herb_all_diseases_enrichment (直接靶点-DEG Fisher)
- 评估: FULL 协议 (正例=已知草药, 负例=全部项目草药)
"""
import os
import re
import sys
from collections import defaultdict
from difflib import get_close_matches

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = "c:/Users/alikex/Desktop/Work/systems_mapping/"
RAW = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
PAPER = os.path.join(BASE, "output", "results", "paper_deliverables")
sys.path.insert(0, BASE)
from hdmap_s import pair_single_herb_all_diseases_enrichment  # noqa: E402

# 选定多病 (TCM-suite disease_id), 基于候选筛选 + 医学/中药证据
DISEASE_IDS = [2117, 264, 328, 335, 1920, 3000, 2661, 3505,
               2845, 3443, 1770, 2102, 2380, 881, 1170, 1244]


def norm(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z ]", " ", s)
    for w in ["herba", "radix", "rhizoma", "caulis", "fructus", "cortex",
              "flos", "folium", "semen", "root", "leaf", "herb", "chinese"]:
        s = s.replace(w, "")
    return re.sub(r"\s+", " ", s).strip()


def main():
    # 项目草药名 -> Herb_N
    proj = pd.read_csv(os.path.join(RAW, "herb_metadata.csv"))
    proj_norm = {norm(n): n for n in proj["herb_name"].dropna()}

    # TCM herb -> 项目 Herb_N (fuzzy)
    tcm = pd.read_csv(os.path.join(RAW, "tcmsuite_herb.csv"), encoding="latin-1")
    tcm2proj = {}
    for _, r in tcm.iterrows():
        nl = norm(r.get("latinName", ""))
        if nl and nl in proj_norm:
            tcm2proj[r["id1"]] = proj_norm[nl]
        else:
            c = get_close_matches(nl, list(proj_norm.keys()), n=1, cutoff=0.80)
            if c:
                tcm2proj[r["id1"]] = proj_norm[c[0]]
    print(f"[映射] TCM herb -> 项目: {len(tcm2proj)}/{len(tcm)}")

    # 桥接: TCM-suite disease id1 <-> DisGeNET xbDiseaseId
    dc = pd.read_csv(os.path.join(RAW, "tcmsuite_disease.csv"), encoding="latin-1")
    tcm2xb = {}
    for _, r in dc.iterrows():
        try:
            tcm2xb[int(r["id1"])] = int(r["xbDiseaseId"])
        except (ValueError, TypeError):
            pass

    # 疾病基因集 (DisGeNET xbDiseaseId)
    dt = pd.read_csv(os.path.join(RAW, "tcmsuite_disease_target.csv"))
    degs = {}
    for did in DISEASE_IDS:
        xb = tcm2xb.get(did)
        if xb is None:
            print(f"  [跳过] TCM病 {did} 无 DisGeNET 映射")
            continue
        genes = dt[dt["diseaseId"] == xb]["gene"].dropna().astype(str).tolist()
        if len(genes) >= 20:
            degs[xb] = set(genes)
    print(f"[degs] 构建 {len(degs)} 病: " +
          ", ".join(f"{k}:{len(v)}" for k, v in degs.items()))

    # 金标准: herb-disease (col0=diseaseId, col1=herbId)
    hd = pd.read_csv(os.path.join(RAW, "tcmsuite_herb_disease.csv"), header=None)
    gold_pos = defaultdict(set)
    for _, r in hd.iterrows():
        xb = tcm2xb.get(r[0])
        if xb is not None and xb in degs and r[1] in tcm2proj:
            gold_pos[xb].add(tcm2proj[r[1]])
    print(f"[金标准] 有正例的病: {sum(1 for d in degs if gold_pos.get(d))}/{len(degs)}")

    # herb ids
    herb_ids = (pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
                ["ingredient_id"].dropna().unique().tolist())

    # 富集 (DATE = 直接靶点-DEG Fisher)
    mat = pair_single_herb_all_diseases_enrichment(
        degs, herb_ids,
        os.path.join(PROC, "ingredient_targets_enhanced.csv"),
        os.path.join(PROC, "herb_ingredients_processed.csv"))
    mat.index = mat.index.astype(str).str.replace(r"\.0$", "", regex=True)
    hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
    id2herb = dict(zip(hi["ingredient_id"].astype(int).astype(str),
                       hi["herb_id"].astype(str)))
    mat.index = [id2herb.get(str(h), str(h)) for h in mat.index]

    # AUROC (FULL)
    rows = []
    for d in mat.columns:
        pos = gold_pos.get(d, set()) & set(mat.index)
        if len(pos) < 3:
            continue
        y = np.array([1 if h in pos else 0 for h in mat.index], dtype=float)
        s = mat[d].values.astype(float)
        if len(set(y)) < 2:
            continue
        rows.append((d, round(float(roc_auc_score(y, s)), 4), len(pos)))
    df = pd.DataFrame(rows, columns=["disease_id", "AUROC", "n_pos"])
    df.loc["MEAN"] = [np.nan, round(df["AUROC"].mean(), 4), df["n_pos"].sum()]
    out = os.path.join(PAPER, "multidisease_date.csv")
    df.to_csv(out, index=False)
    print("\n" + "=" * 60)
    print("多病 DATE AUROC (TCM-suite 金标准)")
    print("=" * 60)
    print(df.to_string())
    print(f"[写入] {out}")


if __name__ == "__main__":
    main()
