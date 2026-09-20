#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多病候选筛选 (杠杆①): 从 TCM-suite herb-disease 实测每个疾病能映射到
我们项目 Herb_N 的草药数量, 选 top 高置信病作为多病 AUROC 评估集。

映射: 项目 herb_metadata.herb_name(拉丁) <-> TCM-suite herb.latinName(拉丁)
      先精确标准化匹配, 再用 difflib fuzzy(cutoff=0.80) 补。
"""
import os
import re
from collections import defaultdict
from difflib import get_close_matches
import pandas as pd

BASE = "c:/Users/alikex/Desktop/Work/systems_mapping/"
RAW = os.path.join(BASE, "data", "raw")


def norm(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z ]", " ", s)
    for w in ["herba", "radix", "rhizoma", "caulis", "fructus", "cortex",
              "flos", "folium", "semen", "root", "leaf", "herb", "chinese"]:
        s = s.replace(w, "")
    return re.sub(r"\s+", " ", s).strip()


def main():
    # 1) 项目草药名
    proj = pd.read_csv(os.path.join(RAW, "herb_metadata.csv"))
    proj_names = [str(x) for x in proj["herb_name"].dropna()]
    proj_norm = {norm(n): n for n in proj_names}

    # 2) TCM 草药
    tcm = pd.read_csv(os.path.join(RAW, "tcmsuite_herb.csv"), encoding="latin-1")
    tcm_map = {}
    for _, r in tcm.iterrows():
        tid = r["id1"]
        tcm_map[tid] = {
            "latin": norm(r.get("latinName", "")),
            "pin": norm(r.get("pinyinName", "")),
            "eng": norm(r.get("englishName", "")),
        }

    # 3) fuzzy 映射 TCM id -> 项目原始名
    tcm2proj = {}
    for tid, info in tcm_map.items():
        nl = info["latin"]
        if nl and nl in proj_norm:
            tcm2proj[tid] = proj_norm[nl]
        else:
            cand = get_close_matches(nl, proj_norm.keys(), n=1, cutoff=0.80)
            if cand:
                tcm2proj[tid] = proj_norm[cand[0]]
    print(f"[映射] TCM 草药 -> 项目: {len(tcm2proj)}/{len(tcm_map)}")

    # 4) herb-disease
    hd = pd.read_csv(os.path.join(RAW, "tcmsuite_herb_disease.csv"))
    print("[herb-disease] 列:", hd.columns.tolist(), "| 行:", len(hd))
    herb_col = "herbId" if "herbId" in hd.columns else hd.columns[1]
    dis_col = "diseaseId" if "diseaseId" in hd.columns else hd.columns[0]

    dis2proj = defaultdict(set)
    for _, r in hd.iterrows():
        tid = r[herb_col]
        if tid in tcm2proj:
            dis2proj[r[dis_col]].add(tcm2proj[tid])

    # 5) 疾病名
    dn = {}
    dname_path = os.path.join(RAW, "tcmsuite_disease.csv")
    if os.path.exists(dname_path):
        d = pd.read_csv(dname_path, encoding="latin-1")
        k0 = d.columns[0]
        for _, r in d.iterrows():
            dn[r[k0]] = r.iloc[1] if len(d.columns) > 1 else r[k0]

    rows = [(did, len(s), str(dn.get(did, did))) for did, s in dis2proj.items()
            if len(s) >= 8]
    rows.sort(key=lambda x: -x[1])
    print(f"\n[候选疾病] 可映射>=8 草药: {len(rows)} 个")
    print("=" * 60)
    for did, n, nm in rows[:40]:
        print(f"{str(did):>10} | {n:>4} herbs | {nm}")
    print("=" * 60)
    # 导出候选
    out = os.path.join(BASE, "output", "results", "paper_deliverables",
                       "candidate_diseases.csv")
    pd.DataFrame(rows, columns=["disease_id", "n_mapped_herbs", "disease_name"]
                 ).to_csv(out, index=False)
    print(f"[写入] {out}")


if __name__ == "__main__":
    main()
