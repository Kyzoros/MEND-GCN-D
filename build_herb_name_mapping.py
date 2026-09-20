"""
build_herb_name_mapping.py
==========================
将项目 500 味草药 (data/raw/herb_metadata.csv) 映射到 TCMBank 全表 (data/raw/herb_all.xlsx),
补上中文名 / 拼音名 / Indication(主治) / Function(功能) / SymMap_id / TCMSP_id。

匹配键: 拉丁名 (herb_metadata.herb_latin / herb_name  vs  herb_all.Herb_latin_name)
匹配方法:
  1. exact   : 小写去非字母后完全相等
  2. token_set: 标准化(小写, 去标点, 去药用部位停用词 herba/radix/...)后 token 集合相等
                能处理词序差异 (Ardisiae Japonicae Herba  <->  Herba Ardisiae Japonicae)

输出: data/processed/herb_name_map.csv
  列: herb_id, herb_latin, tcm_name, pinyin, tcmbank_herb_id, symmap_id, tcms_id,
      function, indication, match_method
供 build_gold_from_tcmbank.py 从 Indication 文本抽取 8 病治疗草药 (多套金标准)。
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RAW = os.path.join(HERE, "data", "raw")

META = os.path.join(RAW, "herb_metadata.csv")
TCMBANK = os.path.join(RAW, "herb_all.xlsx")

# 药用拉丁部位/连接词停用词 (标准化时剔除, 仅保留"属+种"核心 token)
STOP = {
    "herba", "radix", "rhizoma", "fructus", "cortex", "flos", "folium", "semen",
    "caulis", "planta", "seu", "et", "cum", "sui", "radicis", "folia", "lignum",
    "resina", "oleum", "pericarpium", "herb", "herber", "root", "leaf", "seed",
    "stem", "fruit", "flower", "bark", "whole",
}


def norm_exact(s):
    if pd.isna(s):
        return ""
    return re.sub(r"[^a-z]", "", str(s).lower())


def norm_tokens(s):
    if pd.isna(s):
        return frozenset()
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return frozenset(t for t in s.split() if t and t not in STOP)


def main():
    meta = pd.read_csv(META)
    bank = pd.read_excel(TCMBANK)

    bank["_exact"] = bank["Herb_latin_name"].map(norm_exact)
    bank["_tok"] = bank["Herb_latin_name"].map(norm_tokens)
    exact_idx, tok_idx = {}, {}
    for i, r in bank.iterrows():
        if r["_exact"]:
            exact_idx.setdefault(r["_exact"], i)
        if r["_tok"]:
            tok_idx.setdefault(r["_tok"], i)

    rows = []
    c_exact = c_tok = 0
    for _, m in meta.iterrows():
        lat = m["herb_latin"] if pd.notna(m.get("herb_latin")) else m.get("herb_name")
        # 先试 herb_latin, 再试 herb_name
        bi = None
        method = None
        for cand in [m.get("herb_latin"), m.get("herb_name")]:
            ex = norm_exact(cand)
            tok = norm_tokens(cand)
            if ex and ex in exact_idx:
                bi = exact_idx[ex]; method = "exact"; break
            if tok and tok in tok_idx:
                bi = tok_idx[tok]; method = "token_set"; break
        if method == "exact":
            c_exact += 1
        elif method == "token_set":
            c_tok += 1

        if bi is not None:
            b = bank.iloc[bi]
            rows.append({
                "herb_id": m["herb_id"],
                "herb_latin": lat,
                "tcm_name": b["TCM_name"],
                "pinyin": b["Herb_pinyin_name"],
                "tcmbank_herb_id": b["Herb_ID"],
                "symmap_id": b["SymMap_id"],
                "tcms_id": b["TCMSP_id"],
                "function": b["Function"],
                "indication": b["Indication"],
                "match_method": method,
            })
        else:
            rows.append({
                "herb_id": m["herb_id"],
                "herb_latin": lat,
                "tcm_name": None, "pinyin": None, "tcmbank_herb_id": None,
                "symmap_id": None, "tcms_id": None, "function": None,
                "indication": None, "match_method": "unmatched",
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "data", "processed", "herb_name_map.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    matched = (out["match_method"] != "unmatched").sum()
    print(f"草药总数: {len(out)}")
    print(f"匹配: {matched} ({matched / len(out) * 100:.1f}%)  "
          f"exact={c_exact}  token_set={c_tok}")
    print(f"未匹配: {len(out) - matched}")
    print(f"有 Indication 的匹配草药: {out[out['indication'].notna() & (out['match_method'] != 'unmatched')].shape[0]}")
    print(f"写入: {out_path}")

    # 打印若干未匹配样例, 便于判断是否要增强匹配
    unmatched = out[out["match_method"] == "unmatched"]["herb_latin"].head(10).tolist()
    if unmatched:
        print("\n未匹配样例 (前10):")
        for u in unmatched:
            print(f"  {u}")


if __name__ == "__main__":
    main()
