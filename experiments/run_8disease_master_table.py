"""run_8disease_master_table.py
整合 8 病主评估总表:
  disease | TCMBank_AUROC(尺子A, raw) | CMap_rho(尺子B, raw) | CMap_AUROC | Curated_AUROC | Curated_n_pos
来源:
  - tms_independent_validation.csv (尺子A/B)
  - Table_8disease_curated_auc.csv (curated)
"""
import os
import sys

import pandas as pd
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data", "processed")
RES = os.path.join(HERE, "output", "results", "8disease_enrichment")

ind = pd.read_csv(os.path.join(PROC, "tms_independent_validation.csv"))
cur = pd.read_csv(os.path.join(RES, "Table_8disease_curated_auc.csv"))

m = ind[["disease", "A_auc_raw", "B_rho_raw", "B_auc_raw"]].merge(
    cur, on="disease", how="outer")
m = m.rename(columns={
    "A_auc_raw": "TCMBank_AUROC", "B_rho_raw": "CMap_rho",
    "B_auc_raw": "CMap_AUROC", "AUROC": "Curated_AUROC", "n_pos": "Curated_n_pos"})
m = m[["disease", "TCMBank_AUROC", "CMap_rho", "CMap_AUROC",
       "Curated_AUROC", "Curated_n_pos"]]
m.to_csv(os.path.join(RES, "Table_8disease_master.csv"),
         index=False, encoding="utf-8-sig")
print(m.to_string(index=False))
print("\n-> 写出", os.path.join(RES, "Table_8disease_master.csv"))
