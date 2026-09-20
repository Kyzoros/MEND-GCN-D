"""
build_paper_validation_table.py
================================
把 run_tms_independent_validation.py 的 8 病结果, 整理成两张论文用表:
  Table 1 (Table_tms_independent_validation.csv):
    核心结论 - TMS 在独立尺子上的原始表现 (raw), 每病标注独立信号强度。
  Table 2 (Table_sizecorrection_ablation.csv):
    消融 - 原始 vs size-corrected, 证明"扣靶点数量水分"(方案A)是有害误诊。
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROC = os.path.join(HERE, "data", "processed")
OUT_DIR = os.path.join(HERE, "output", "results", "paper_deliverables")
os.makedirs(OUT_DIR, exist_ok=True)

src = pd.read_csv(os.path.join(PROC, "tms_independent_validation.csv"))
tcm = pd.read_csv(os.path.join(PROC, "gold_tcmbank_indication.csv"))

# 每病 TCMBank 正例数 (独立金标准规模)
pos_n = tcm[tcm.label == 1].groupby("disease_id").size().to_dict()


def verdict(a):
    if pd.isna(a):
        return "n/a"
    if a >= 0.65:
        return "Strong"
    if a >= 0.55:
        return "Moderate"
    return "Weak"


# ---- Table 1: 独立尺子验证 (raw, 核心结论) ----
rows = []
for _, r in src.iterrows():
    d = r["disease"]
    rows.append({
        "Disease": d,
        "TCMBank_positive_n": int(pos_n.get(d, 0)),
        "TCMBank_AUROC_raw": r["A_auc_raw"],
        "CMap_rho_raw": r["B_rho_raw"],
        "CMap_AUROC_raw": r["B_auc_raw"],
        "Independent_signal": verdict(r["A_auc_raw"]),
    })
main_tbl = pd.DataFrame(rows)
main_tbl.to_csv(os.path.join(OUT_DIR, "Table_tms_independent_validation.csv"),
                index=False)

# ---- Table 2: size-correction 消融 (证明方案A有害) ----
ab_rows = []
for _, r in src.iterrows():
    d = r["disease"]
    ab_rows.append({
        "Disease": d,
        "TCMBank_AUROC_raw": r["A_auc_raw"],
        "TCMBank_AUROC_sizecorrected": r["A_auc_corrected"],
        "CMap_rho_raw": r["B_rho_raw"],
        "CMap_rho_sizecorrected": r["B_rho_corrected"],
        "CMap_AUROC_raw": r["B_auc_raw"],
        "CMap_AUROC_sizecorrected": r["B_auc_corrected"],
    })
ab_tbl = pd.DataFrame(ab_rows)
ab_tbl.to_csv(os.path.join(OUT_DIR, "Table_sizecorrection_ablation.csv"),
              index=False)

print("=== Table 1: Independent-scale validation of TMS (raw) ===")
print(main_tbl.to_string(index=False))
print("\n=== Table 2: Size-correction ablation (shows method A is harmful) ===")
print(ab_tbl.to_string(index=False))
print(f"\n写入:\n  {os.path.join(OUT_DIR, 'Table_tms_independent_validation.csv')}")
print(f"  {os.path.join(OUT_DIR, 'Table_sizecorrection_ablation.csv')}")
