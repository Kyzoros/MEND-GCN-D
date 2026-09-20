"""
run_8disease_top50.py
=====================
从已有的 8 病 enrichment TMS (hdmap_s_scores_enrichment_8d.csv) 直接产出每病 Top-50 推荐草药表。
轻量、不跑 GA、不依赖金标准。仅做"按 TMS 降序取前 50 + 中文药名"。
输出: output/results/8disease_enrichment/<DISEASE>_top50_herbs.csv
(注: 完整主流程的 permutation p 值版在 AD_kegg_results/ 仅 4 病; 此处先给排名, 如需 p 值可后续补)
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfg

PROC = cfg.PROCESSED_DIR
OUT = os.path.join(HERE, "output", "results", "8disease_enrichment")
os.makedirs(OUT, exist_ok=True)

tms = pd.read_csv(os.path.join(PROC, "hdmap_s_scores_enrichment_8d.csv"), index_col=0)
tms.index = tms.index.astype(str)

hi = pd.read_csv(os.path.join(PROC, cfg.FILE_HERB_INGREDIENTS_PROCESSED))
name_col = 'herb_name' if 'herb_name' in hi.columns else hi.columns[-1]
name_map = dict(zip(hi['herb_id'].astype(str), hi[name_col].astype(str)))

n_written = 0
for d in tms.columns:
    top = tms[d].sort_values(ascending=False).head(50)
    rows = [{"rank": i + 1, "herb_id": h, "herb_name": name_map.get(h, h),
             "TMS_score": float(s)} for i, (h, s) in enumerate(top.items())]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{d}_top50_herbs.csv"),
                             index=False, encoding="utf-8-sig")
    n_written += 1
    print(f"  {d}: Top-50 写出 ({len(top)} 草药)")

print(f"\n完成 {n_written} 病 Top-50, 输出目录: {OUT}")
