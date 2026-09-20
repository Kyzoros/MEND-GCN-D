"""run_8disease_curated_auc.py
(2) 补新 4 病 curated 金标准 + 算 8 病 curated AUROC
- 老 4 病 (AD/PAH/RA/T2D): 用原人工 gold_standard_final.csv
- 新 4 病 (HCC/CRC/ASTHMA/OA): 从 TCMBank indication 提取正例 (label=1)
- 合并 -> data/processed/gold_standard_final_8d.csv
- 用 8 病 enrichment TMS 算每病 AUROC (全库草药为负例)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfg

PROC = cfg.PROCESSED_DIR
RES = os.path.join(HERE, "output", "results", "8disease_enrichment")
os.makedirs(RES, exist_ok=True)

# ---------- 1) 合并 8 病 curated 金标准 ----------
curated = pd.read_csv(os.path.join(HERE, "data", "gold_standard_final.csv"))
tcm = pd.read_csv(os.path.join(PROC, "gold_tcmbank_indication.csv"))
new4 = ['HCC', 'CRC', 'ASTHMA', 'OA']
add = tcm[tcm['disease_id'].isin(new4)].copy()
for c in ['evidence_grade', 'source_ref']:
    if c not in add.columns:
        add[c] = 'NA'
add['source_ref'] = 'tcmbank_indication'
add = add[curated.columns]
combined = pd.concat([curated, add], ignore_index=True)
out8 = os.path.join(PROC, "gold_standard_final_8d.csv")
combined.to_csv(out8, index=False, encoding="utf-8-sig")
print("8病金标准疾病分布:", combined['disease_id'].value_counts().to_dict())
print("  -> 写出", out8)

# ---------- 2) 8 病 curated AUROC (enrichment TMS) ----------
tms = pd.read_csv(os.path.join(PROC, "hdmap_s_scores_enrichment_8d.csv"), index_col=0)
tms.index = tms.index.astype(str)
hi = pd.read_csv(os.path.join(PROC, cfg.FILE_HERB_INGREDIENTS_PROCESSED))
all_herbs = [h for h in hi['herb_id'].astype(str).tolist() if h in tms.index]

rows = []
for d in tms.columns:
    pos = set(combined[(combined['disease_id'] == d) & (combined['label'] == 1)]['herb_id'].astype(str))
    y_true = np.array([1 if h in pos else 0 for h in all_herbs])
    y_score = tms[d].loc[all_herbs].values.astype(float)
    auc = roc_auc_score(y_true, y_score)
    rows.append({'disease': d, 'n_pos': len(pos), 'AUROC': round(float(auc), 3)})

auc_df = pd.DataFrame(rows)
print("\n8 病 curated AUROC (enrichment TMS):")
print(auc_df.to_string(index=False))
auc_df.to_csv(os.path.join(RES, "Table_8disease_curated_auc.csv"),
              index=False, encoding="utf-8-sig")
print("\n  -> 写出", os.path.join(RES, "Table_8disease_curated_auc.csv"))
