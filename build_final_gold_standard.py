#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建最终金标准 = 扩充版 + PAH 文献候选
========================================
合并:
  1. data/gold_standard_expanded.csv
     (192 条 = curated 105 + TCMBank 自动挖 39 正 + 随机无关负 48)
  2. data/pah_expansion_candidates.csv
     (PAH 文献候选 16 味, 证据分级 A/B/C, 每条带 evidence 与出处)

输出: data/gold_standard_final.csv
新增字段:
  evidence_grade —— 证据等级, 便于论文做敏感性分析(例如只用 A+B 级重跑对比)
  evidence / source_ref —— 新增条目的依据与文献出处

理论依据 (为什么可以并入而不构成循环论证):
  - 候选依据来自独立的临床诊疗指南(T/CACM 1170-2019)与药理学综述,
    与 HDmap-S 的打分方法完全无关, 未使用任何模型输出参与选择。
  - 金标准的定义即"已知治疗该病的草药", 用文献确定是标准做法,
    且质量高于自动挖掘(TCMBank)条目。
  - 主评估协议为 FULL (阳性 vs 全部 491 味草药),
    仅增加正例不会造成类别不平衡。
  - 注意: labeled 子集协议下 PAH 将变为 29 正 13 负(不平衡),
    故 labeled 结果仅作诊断参考, 主结论以 FULL 为准。

用法:
    python build_final_gold_standard.py                 # 并入 A+B+C
    python build_final_gold_standard.py --grades A,B    # 只并入 A/B 级(保守)
"""
import argparse
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

SRC_MAP = {
    "curated": "curated",
    "TCMBank": "mined",
    "random-unrelated": "negative",
}


def main():
    ap = argparse.ArgumentParser(description="构建最终金标准")
    ap.add_argument("--grades", default="A,B,C",
                    help="并入哪些证据等级, 逗号分隔 (默认 A,B,C)")
    ap.add_argument("--out", default=None, help="输出路径")
    args = ap.parse_args()

    exp_path = os.path.join(BASE, "data", "gold_standard_expanded.csv")
    cand_path = os.path.join(BASE, "data", "pah_expansion_candidates.csv")
    exp = pd.read_csv(exp_path)
    cand = pd.read_csv(cand_path)

    # 现有条目补 evidence_grade
    exp["evidence_grade"] = exp["source"].map(SRC_MAP).fillna("unknown")
    for c in ["evidence", "source_ref"]:
        if c not in exp.columns:
            exp[c] = None

    grades = [g.strip().upper() for g in args.grades.split(",")]
    sub = cand[(cand["status"] == "待核实") &
               (cand["evidence_grade"].isin(grades))]

    pah_pos = set(exp[(exp["disease_id"] == "PAH") &
                      (exp["label"] == 1)]["herb_id"].astype(str))
    print(f"PAH 现有正例: {len(pah_pos)}")
    print(f"待并入候选 ({','.join(grades)} 级): {len(sub)}")

    rows = []
    for _, r in sub.iterrows():
        hid = str(r["herb_id"])
        if hid in pah_pos:
            print(f"  [跳过] {hid} 已在 PAH 正例中")
            continue
        rows.append({
            "herb_id": hid,
            "disease_id": "PAH",
            "label": 1,
            "source": "literature",
            "mapping_method": "guideline_review",
            "tcm_cn": None,
            "evidence_grade": r["evidence_grade"],
            "evidence": r["evidence"],
            "source_ref": r["source"],
        })

    new = pd.DataFrame(rows)
    if new.empty:
        print("无新增条目, 未写入。")
        return

    out = pd.concat([exp, new], ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(subset=["herb_id", "disease_id"], keep="first")
    if len(out) < before:
        print(f"[去重] 移除 {before - len(out)} 条重复")

    path = args.out or os.path.join(BASE, "data", "gold_standard_final.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"\n[写入] {path}")
    print(f"总行数: {len(exp)} → {len(out)}  (+{len(out) - len(exp)})")

    print("\n每病 label 分布:")
    print(out.groupby(["disease_id", "label"]).size()
          .unstack(fill_value=0).to_string())

    print("\n证据等级分布:")
    print(out["evidence_grade"].value_counts().to_string())

    print("\nPAH 新增条目:")
    for _, r in new.iterrows():
        print(f"  [{r['evidence_grade']}] {r['herb_id']:<10} "
              f"{str(r['evidence'])[:56]}")


if __name__ == "__main__":
    main()
