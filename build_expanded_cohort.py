#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建扩病种候选集（方案 C：把病种从 8 扩到 20-30）
==================================================
背景
--------------------------------------------------------------------------
论文目前只有 8 个病种，统计效力与结论泛化性受同行质疑（冲二区的短板之一）。
本脚本评估：**在本地已有数据上，能稳定新增多少个病种**。

数据源（全部本地，且与我们现有自建金标准相互独立）
--------------------------------------------------------------------------
1. data/raw/tcmsuite_herb_disease.csv    草药-疾病关联（121.5 万条）
2. data/raw/tcmsuite_disease_target.csv  疾病-靶点基因（13.9 万条，14110 病）
3. data/raw/tcmsuite_herb.csv            草药字典（id1 -> 拼音/英文/拉丁名）
4. data/raw/tcmsuite_disease.csv         疾病字典（id1 -> 英文病名 / ICD10）
5. data/processed/herb_name_map.csv      我们的 Herb_XXX -> TCMSuite 桥接表

新增一个病种需要同时满足两件事
--------------------------------------------------------------------------
(a) **疾病基因集**：该病要能作为节点接入 H-I-G-D 图（连到基因节点）
    -> 来自 tcmsuite_disease_target.csv 的 gene 列
(b) **草药正例**：该病要有足够多的已知治疗草药（金标准正例）
    -> 来自 tcmsuite_herb_disease.csv，且草药必须落在我们的 499 味之内

只有同时满足 (a)+(b) 的疾病才是可评估的候选。

为什么这不算循环论证
--------------------------------------------------------------------------
候选疾病与其草药正例均来自 TCMSuite 数据库，**未使用任何模型输出参与筛选**；
TCMSuite 与现有金标准（文献 curated + TCMBank）来源相互独立，
可用于构建扩展队列；主结论仍以原有 GEO-DEG 8 病为准，扩展队列作稳健性验证。

用法
--------------------------------------------------------------------------
    python build_expanded_cohort.py                       # 默认门槛
    python build_expanded_cohort.py --min-herbs 15 --top 40
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg                                        # noqa: E402

RAW = os.path.join(BASE, "data", "raw")
PROC = cfg.PROCESSED_DIR
OUT = os.path.join(BASE, "data", "expanded_disease_candidates.csv")

# 现有 8 个病种（需排除，避免与既有队列重复）
EXISTING_KEYWORDS = {
    "AD": ["alzheimer"],
    "ASTHMA": ["asthma"],
    "CRC": ["colorectal", "colon", "rectal"],
    "HCC": ["hepatocellular", "liver"],
    "OA": ["osteoarthritis"],
    "PAH": ["pulmonary arterial", "pulmonary hypertension"],
    "RA": ["rheumatoid"],
    "T2D": ["diabetes mellitus, type 2", "diabetes, type 2", "type 2 diabetes"],
}


def _norm_pinyin(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.upper().strip())


def _norm_latin(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z ]", " ", s.lower()).strip()


def load_herb_bridge():
    """我们的 Herb_XXX -> TCMSuite 草药 id。

    优先 tcms_id 直接对应；缺失时用拼音兜底；再用拉丁名兜底。
    返回 (映射 dict, 诊断信息 dict)
    """
    hm = pd.read_csv(os.path.join(PROC, "herb_name_map.csv"))
    th = pd.read_csv(os.path.join(RAW, "tcmsuite_herb.csv"),
                     encoding="gb18030", encoding_errors="replace")
    th["id1"] = pd.to_numeric(th["id1"], errors="coerce")

    by_id = {int(v): int(v) for v in th["id1"].dropna().astype(int)}
    by_pin, by_lat = {}, {}
    for _, r in th.iterrows():
        p = _norm_pinyin(r.get("pinyinName"))
        if p and p not in by_pin:
            by_pin[p] = int(r["id1"])
        l = _norm_latin(r.get("latinName"))
        if l and l not in by_lat:
            by_lat[l] = int(r["id1"])

    mapping, how = {}, {"tcms_id": 0, "pinyin": 0, "latin": 0, "unmapped": 0}
    for _, r in hm.iterrows():
        hid = str(r["herb_id"])
        tid = r.get("tcms_id")
        if pd.notna(tid):
            try:
                tid = int(float(tid))
                if tid in by_id:
                    mapping[hid] = tid
                    how["tcms_id"] += 1
                    continue
            except (TypeError, ValueError):
                pass
        p = _norm_pinyin(r.get("pinyin"))
        if p and p in by_pin:
            mapping[hid] = by_pin[p]
            how["pinyin"] += 1
            continue
        l = _norm_latin(r.get("herb_latin"))
        if l and l in by_lat:
            mapping[hid] = by_lat[l]
            how["latin"] += 1
            continue
        how["unmapped"] += 1
    return mapping, how


def main():
    ap = argparse.ArgumentParser(description="构建扩病种候选集")
    ap.add_argument("--min-herbs", type=int, default=10,
                    help="候选病至少需要的我方草药正例数")
    ap.add_argument("--min-genes", type=int, default=20,
                    help="候选病至少需要的疾病基因数")
    ap.add_argument("--top", type=int, default=40, help="输出前 N 个候选")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # --- 1. 草药桥接 ---
    hmap, how = load_herb_bridge()
    print("=" * 78)
    print("我方草药 -> TCMSuite 桥接")
    print("=" * 78)
    print(f"  tcms_id 直接命中 : {how['tcms_id']}")
    print(f"  拼音兜底命中     : {how['pinyin']}")
    print(f"  拉丁名兜底命中   : {how['latin']}")
    print(f"  未能映射         : {how['unmapped']}")
    print(f"  合计可映射       : {len(hmap)} / {sum(how.values())}")
    tcms_of_ours = set(hmap.values())

    # --- 2. 草药-疾病关联 ---
    hd = pd.read_csv(os.path.join(RAW, "tcmsuite_herb_disease.csv"))
    hd.columns = ["herb_id", "disease_id"]
    hd["herb_id"] = pd.to_numeric(hd["herb_id"], errors="coerce")
    hd["disease_id"] = pd.to_numeric(hd["disease_id"], errors="coerce")
    hd = hd.dropna()
    hd = hd[hd["herb_id"].astype(int).isin(tcms_of_ours)]
    print(f"\n[关联] 落在我方草药范围内的关联数 = {len(hd)}")

    cnt = (hd.groupby("disease_id")["herb_id"].nunique()
             .reset_index(name="n_our_herbs"))

    # --- 3. 疾病基因集 ---
    dt = pd.read_csv(os.path.join(RAW, "tcmsuite_disease_target.csv"))
    dt["diseaseId"] = pd.to_numeric(dt["diseaseId"], errors="coerce")
    dt = dt.dropna(subset=["diseaseId"])
    dt = dt[dt["gene"].notna()]
    genes = (dt.groupby("diseaseId")["gene"].nunique()
               .reset_index(name="n_genes"))

    # --- 4. 合并 + 过滤 ---
    cand = cnt.merge(genes, left_on="disease_id", right_on="diseaseId",
                     how="inner")
    cand = cand[cand["n_our_herbs"] >= args.min_herbs]
    cand = cand[cand["n_genes"] >= args.min_genes]

    dd = pd.read_csv(os.path.join(RAW, "tcmsuite_disease.csv"),
                     encoding="gb18030", encoding_errors="replace")
    dd["id1"] = pd.to_numeric(dd["id1"], errors="coerce")
    name_map = dict(zip(dd["id1"].dropna().astype(int), dd["name"]))
    icd_map = dict(zip(dd["id1"].dropna().astype(int), dd["xbIcd10"]))
    cand["disease_name"] = cand["disease_id"].astype(int).map(name_map)
    cand["icd10"] = cand["disease_id"].astype(int).map(icd_map)

    # 排除与现有 8 病重复的
    def is_existing(name):
        n = str(name).lower()
        return any(k in n for ks in EXISTING_KEYWORDS.values() for k in ks)

    cand["is_existing"] = cand["disease_name"].map(is_existing)
    cand = cand[~cand["is_existing"]]

    cand = cand.sort_values(["n_our_herbs", "n_genes"], ascending=False)
    cand = cand.head(args.top)
    cand["disease_id"] = cand["disease_id"].astype(int)
    cand["n_our_herbs"] = cand["n_our_herbs"].astype(int)
    cand["n_genes"] = cand["n_genes"].astype(int)

    print("\n" + "=" * 78)
    print(f"候选病种（我方草药正例 >= {args.min_herbs}，"
          f"疾病基因 >= {args.min_genes}，已排除现有 8 病）")
    print("=" * 78)
    show = cand[["disease_id", "disease_name", "n_our_herbs", "n_genes",
                 "icd10"]].reset_index(drop=True)
    print(show.to_string(index=False))
    print(f"\n满足门槛的候选病种总数（截断前）见下方统计；已输出前 "
          f"{len(show)} 个")

    cand.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {args.out}")


if __name__ == "__main__":
    main()
