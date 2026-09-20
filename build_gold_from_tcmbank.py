"""
build_gold_from_tcmbank.py
==========================
从 TCMBank Indication(主治)/Function(功能) 文本, 按 8 病英文关键词抽取治疗草药,
生成"基于文献主治"的多套金标准之一 (source = tcmbank_indication)。

输入: data/processed/herb_name_map.csv  (build_herb_name_mapping.py 输出, 含 indication 列)
输出: data/processed/gold_tcmbank_indication.csv
  列: herb_id, disease_id, label, source, mapping_method, tcm_cn, matched_keyword, evidence

说明:
  - 关键词覆盖中西医术语 (如 T2D 含 diabetes/thirst/xiaoke; RA 含 rheumatoid/wind-damp)
  - 正例来源独立于人工 curated 金标准, 二者可交叉验证 (论文多金标准核心)
  - 仅作为金标准"正例候选"; 负例由 run_multigold_confound_check.py 统一处理
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROC = os.path.join(HERE, "data", "processed")

NAME_MAP = os.path.join(PROC, "herb_name_map.csv")
OUT = os.path.join(PROC, "gold_tcmbank_indication.csv")

# 每病英文关键词 (regex, 对 indication+function 小写文本检索)
DISEASE_KEYWORDS = {
    "AD":     [r"alzheimer", r"dementia", r"\bmemory\b", r"cognitive", r"amnesia",
               r"neuro", r"\bbrain\b"],
    "PAH":    [r"pulmonary arterial hypertension", r"pulmonary hypertension",
               r"\bpulmonary\b", r"hypoxia", r"\bhypoxic\b"],
    "RA":     [r"rheumatoid arthritis", r"\brheumatic\b", r"wind-damp",
               r"arthralgia", r"antirheumatic"],
    "T2D":    [r"\bdiabet", r"hyperglycemia", r"blood glucose", r"\bthirst\b",
               r"xiaoke", r"glycosuria"],
    "HCC":    [r"hepatocellular carcinoma", r"liver cancer", r"hepatic carcinoma",
               r"hepatitis", r"cirrhosis", r"hepatocarcin"],
    "CRC":    [r"colorectal", r"colorectal cancer", r"colorectal carcin",
               r"intestinal cancer", r"\bbowel\b", r"dysentery", r"bloody stool",
               r"\bcolon\b", r"colonic", r"tenesmus", r"abdominal mass",
               r"intestinal polyp"],
    "ASTHMA": [r"\basthma", r"wheezing", r"bronchial asthma", r"\bbronchial\b"],
    "OA":     [r"osteoarthritis", r"knee joint", r"cartilage", r"\bbone\b",
               r"osteoarthr", r"\bjoint\b", r"arthralgia"],
}


def main():
    df = pd.read_csv(NAME_MAP)
    df = df[df["match_method"] != "unmatched"]
    df = df.dropna(subset=["indication"])
    df["_text"] = (df["indication"].fillna("") + " " + df["function"].fillna("")).str.lower()

    compiled = {d: [re.compile(k) for k in kws] for d, kws in DISEASE_KEYWORDS.items()}

    rows = []
    for _, r in df.iterrows():
        text = r["_text"]
        for did, pats in compiled.items():
            hits = [k for p, k in zip(pats, DISEASE_KEYWORDS[did]) if p.search(text)]
            if hits:
                rows.append({
                    "herb_id": r["herb_id"],
                    "disease_id": did,
                    "label": 1,
                    "source": "tcmbank_indication",
                    "mapping_method": "indication_keyword",
                    "tcm_cn": r["tcm_name"],
                    "matched_keyword": "; ".join(hits),
                    "evidence": r["indication"],
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = (out.sort_values(["disease_id", "herb_id"])
                 .drop_duplicates(subset=["herb_id", "disease_id"]))
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"从 {len(df)} 味有 Indication 的匹配草药中, 抽到正例:")
    print(f"总条数: {len(out)}")
    if not out.empty:
        print(out.groupby("disease_id").size().to_string())
    print(f"\n写入: {OUT}")


if __name__ == "__main__":
    main()
