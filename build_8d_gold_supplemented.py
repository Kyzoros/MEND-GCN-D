#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建「补强版 8 病金标准」(gold_standard_final_8d_supplemented.csv)
===============================================================
动机 (2026-09-02/03):
  当前 gold_standard_final_8d.csv 的新 4 病 (HCC/CRC/ASTHMA/OA) 金标准
  100% 自动映射自 TCMBank (tcmbank_indication), 质量偏低, 且**只有正例、无阴性**,
  导致 labeled 协议只能跑原 4 病, 跨病比较不公平。

本脚本做的补强:
  1. 双库交叉验证: 用 TCM-suite 经项目已有的 tcms_id 主键映射到项目草药,
     作为第二个**独立** TCM 数据库来源 (两库重合仅 1~6 味, 互为补充)。
  2. 正例证据分级 (与 PAH 扩展一致的 A/B/C 体系):
       A = TCMBank ∩ TCM-suite 两库共识 (高置信)
       B = 单库 (TCMBank 或 TCM-suite 独有)
  3. 补齐阴性: 为新 4 病各生成 random-unrelated 阴性 (与原 4 病机制一致, 采样非正例草药)。
  4. 不影响原 4 病: 原 4 病 (AD/PAH/RA/T2D) 完全沿用 gold_standard_final.csv。

输出: data/processed/gold_standard_final_8d_supplemented.csv
       (列与原 8d 版一致: herb_id, disease_id, label, source, mapping_method,
        tcm_cn, evidence_grade, evidence, source_ref)

用法:
    python build_8d_gold_supplemented.py
    python build_8d_gold_supplemented.py --neg-ratio 1.0    # 每病阴性=正例数
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfg

RAW = os.path.join(cfg.DATA_DIR, "raw")
PROC = cfg.PROCESSED_DIR
SEED = 42


def read_any(path, **kw):
    for enc in ["utf-8", "gb18030", "gbk", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(path)


def to_key(v):
    try:
        return str(int(float(v)))
    except Exception:
        return str(v).strip()


def main():
    ap = argparse.ArgumentParser(description="构建补强版 8 病金标准")
    ap.add_argument("--neg-ratio", type=float, default=1.0,
                    help="每病阴性数 = round(负例比 * 正例数)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(
        PROC, "gold_standard_final_8d_supplemented.csv")

    # ---------- 1) 原 4 病金标准 (沿用, 不动) ----------
    curated4 = pd.read_csv(os.path.join(HERE, "data", "gold_standard_final.csv"))
    # 补齐列
    for c in ["evidence_grade", "evidence", "source_ref", "mapping_method", "tcm_cn"]:
        if c not in curated4.columns:
            curated4[c] = None

    # ---------- 2) 项目草药 -> tcms_id ----------
    hn = pd.read_csv(os.path.join(PROC, "herb_name_map.csv"))
    proj_by_tcms = {}
    for _, r in hn.iterrows():
        if pd.notna(r["tcms_id"]):
            proj_by_tcms.setdefault(to_key(r["tcms_id"]), set()).add(str(r["herb_id"]))

    # ---------- 3) TCM-suite herb_disease ----------
    hd = read_any(os.path.join(RAW, "tcmsuite_herb_disease.csv"), header=None,
                  names=["herb_id", "disease_id"], skiprows=1)
    first = read_any(os.path.join(RAW, "tcmsuite_herb_disease.csv"), nrows=1, header=None)
    hd = pd.concat([pd.DataFrame([[first.iloc[0, 0], first.iloc[0, 1]]],
                                 columns=["herb_id", "disease_id"]), hd], ignore_index=True)
    hd["herb_id"] = pd.to_numeric(hd["herb_id"], errors="coerce")
    hd["disease_id"] = pd.to_numeric(hd["disease_id"], errors="coerce")
    hd = hd.dropna()

    dis = read_any(os.path.join(RAW, "tcmsuite_disease.csv"))
    tgt = {"HCC": ["carcinoma, hepatocellular", "liver neoplasms", "liver cancer"],
           "CRC": ["colorectal cancer", "colorectal neoplasms"],
           "ASTHMA": ["asthma"], "OA": ["osteoarthritis"]}
    dis_ids = {}
    for tag, kws in tgt.items():
        s = pd.Series(False, index=dis.index)
        for col in ["name", "xbDiseaseName"]:
            if col in dis.columns:
                c = dis[col].astype(str).str.lower()
                for kw in kws:
                    s |= c.str.contains(kw, na=False, regex=False)
        dis_ids[tag] = set(dis.loc[s, "id1"].dropna().astype(int))

    # TCM-suite 关联 -> 项目草药
    tcmsuite_pos = {}
    for tag, ids in dis_ids.items():
        sub = hd[hd["disease_id"].isin(ids)]
        hids = set(sub["herb_id"].dropna().astype(int).map(to_key))
        proj = set()
        for hk in hids:
            proj |= proj_by_tcms.get(hk, set())
        tcmsuite_pos[tag] = proj
        print(f"  {tag}: TCM-suite 映射到项目草药 {len(proj)} 味")

    # ---------- 4) TCMBank 正例 (当前金标准来源) ----------
    tcmbank = pd.read_csv(os.path.join(PROC, "gold_tcmbank_indication.csv"))
    tcmbank_pos = {}
    for _, r in tcmbank.iterrows():
        if r["disease_id"] in tgt and int(r["label"]) == 1:
            tcmbank_pos.setdefault(r["disease_id"], set()).add(str(r["herb_id"]))

    # ---------- 5) 组装新 4 病补强金标准 ----------
    new4_rows = []
    gold_tcm_cn = {}
    for _, r in pd.read_csv(cfg.GOLD_STANDARD_PATH).iterrows():
        if isinstance(r.get("tcm_cn"), str) and r["tcm_cn"].strip():
            gold_tcm_cn[str(r["herb_id"])] = r["tcm_cn"].strip()

    for d in ["HCC", "CRC", "ASTHMA", "OA"]:
        tb = tcmbank_pos.get(d, set())
        ts = tcmsuite_pos.get(d, set())
        both = tb & ts
        only_tb = tb - ts
        only_ts = ts - tb
        # A: 两库共识; B: 单库
        for hid in both:
            new4_rows.append(dict(herb_id=hid, disease_id=d, label=1,
                                  source="tcmsuite_x_tcmbank", mapping_method="cross_validated",
                                  tcm_cn=gold_tcm_cn.get(hid), evidence_grade="A",
                                  evidence="TCMBank ∩ TCM-suite 双库共识", source_ref="TCMBank+TCM-suite"))
        for hid in only_tb:
            new4_rows.append(dict(herb_id=hid, disease_id=d, label=1,
                                  source="tcmbank_indication", mapping_method="indication_keyword",
                                  tcm_cn=gold_tcm_cn.get(hid), evidence_grade="B",
                                  evidence="TCMBank indication", source_ref="TCMBank"))
        for hid in only_ts:
            new4_rows.append(dict(herb_id=hid, disease_id=d, label=1,
                                  source="tcmsuite_indication", mapping_method="tcmsuite_herb_disease",
                                  tcm_cn=gold_tcm_cn.get(hid), evidence_grade="B",
                                  evidence="TCM-suite herb-disease association", source_ref="TCM-suite"))
        n_pos = len(both) + len(only_tb) + len(only_ts)
        print(f"  {d}: TCMBank {len(tb)} | TCM-suite {len(ts)} | 共有 {len(both)} | "
              f"补强后正例 {n_pos} (A={len(both)}, B={len(only_tb)+len(only_ts)})")

        # 阴性: random-unrelated (与原4病机制一致: 采样非正例草药)
        all_herbs = sorted(set(pd.read_csv(os.path.join(PROC, cfg.FILE_HERB_INGREDIENTS_PROCESSED))
                               ["herb_id"].astype(str)))
        pos_set = (both | only_tb | only_ts)
        neg_pool = [h for h in all_herbs if h not in pos_set]
        rng = __import__("numpy").random.default_rng(SEED)
        n_neg = int(round(args.neg_ratio * n_pos))
        n_neg = min(n_neg, len(neg_pool))
        neg_sample = list(rng.choice(neg_pool, size=n_neg, replace=False))
        for hid in neg_sample:
            new4_rows.append(dict(herb_id=hid, disease_id=d, label=0,
                                  source="random-unrelated", mapping_method="random_sample",
                                  tcm_cn=None, evidence_grade="C",
                                  evidence="非正例随机采样阴性", source_ref="random"))

    new4 = pd.DataFrame(new4_rows)
    combined = pd.concat([curated4, new4], ignore_index=True)
    # 去重 (同一 herb-disease 保留首个)
    combined = combined.drop_duplicates(subset=["herb_id", "disease_id"], keep="first")
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n[写入]", out_path)
    print("总行数:", len(combined))
    print("\n=== 每病 label 分布 ===")
    print(combined.groupby(["disease_id", "label"]).size()
          .unstack(fill_value=0).to_string())
    print("\n=== 证据等级分布 ===")
    print(combined["evidence_grade"].value_counts(dropna=False).to_string())
    print("\n=== 新4病 source 分布 ===")
    print(new4["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
