"""汇总 8 病 Top-50 推荐为两张扫览表:
   1) 8disease_top50_combined.csv  : 长表 (disease, rank, herb_id, herb_name, TMS_score) 共 400 行
   2) 8disease_top50_cross.csv     : 交叉频次表 (草药 x 8病, 值为排名; n_diseases=进Top50的病数)
"""
import os
import sys
import glob

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "results", "8disease_enrichment")
files = sorted(glob.glob(os.path.join(OUT, "*_top50_herbs.csv")))

frames = []
for f in files:
    d = os.path.basename(f).replace("_top50_herbs.csv", "")
    df = pd.read_csv(f)
    df.insert(0, "disease", d)
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
combined.to_csv(os.path.join(OUT, "8disease_top50_combined.csv"),
                index=False, encoding="utf-8-sig")

pivot = combined.pivot_table(index=["herb_id", "herb_name"], columns="disease",
                             values="rank", aggfunc="first")
pivot["n_diseases"] = pivot.notna().sum(axis=1)
pivot = pivot.sort_values("n_diseases", ascending=False)
pivot.to_csv(os.path.join(OUT, "8disease_top50_cross.csv"),
             encoding="utf-8-sig")

print(f"合并长表: {combined.shape}  -> 8disease_top50_combined.csv")
print(f"交叉表  : {pivot.shape}  -> 8disease_top50_cross.csv")
print("\n进 Top-50 病种数 >=2 的广谱草药 (前15):")
print(pivot[pivot['n_diseases'] >= 2].head(15).to_string())
