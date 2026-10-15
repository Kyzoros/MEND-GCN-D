#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正对照实验 (路线 C): 区分"模型/签名不行" 还是 "金标准标签不行"
================================================================================
思路:
  用【独立实测数据】(HERB CMap 成分级连接性, 是真实实验测的, 非本项目模型产出)
  作为预测器, 检验它能否识别金标准中的"治疗草药"。

  CMap 约定: Connectivity Score 越负 = 越逆转疾病表达谱 = 越治疗
  故: 预测分 = -CMap_score (越高越治疗)

判读:
  AUROC 明显 > 0.5 且显著
    => 实测数据能识别金标准草药
    => 标签有效, 且"逆转"原理在【实测】数据上成立
    => 问题出在我们【模拟】的签名 -> 选路线 B (换实测数据重建签名)
  AUROC ≈ 0.5
    => 实测数据也识别不出
    => 标签本身不可靠, 或逆转原理在中药上不适用
    => 选路线 A (诊断性论文)

注: 这里评估在"有 CMap 分数的草药"子集上(约 360-405 味), 与 FULL 协议(491味)不同,
    因为只有匹配到 HERB 成分的草药才有实测分数。这是正对照, 不影响判读。
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg  # noqa: E402
# 复用外部验证脚本中已验证的成分->草药匹配逻辑
from run_herb_external_validation import DISEASES, RAW, match_hybrid  # noqa: E402

PAPER = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")
OUT = os.path.join(PAPER, "positive_control_check.csv")


def build_herb_cmap(fname, agg="max"):
    """草药级 CMap 分数: 成分 Summary Score 按味聚合"""
    df = pd.read_excel(os.path.join(RAW, f"{fname}_cmap2_data.xlsx"))
    ing = df[df["Type"] == "Ingredient"].copy()
    tmp = {}
    n_matched = 0
    for _, r in ing.iterrows():
        hs = match_hybrid(str(r["Drug name"]))
        if hs:
            n_matched += 1
        sc = float(r["Summary Score"])
        for h in hs:
            tmp.setdefault(h, []).append(sc)
    rows = []
    for h, sl in tmp.items():
        v = max(sl) if agg == "max" else float(np.mean(sl))
        rows.append((str(h), v, len(sl)))
    ext = pd.DataFrame(rows, columns=["herb_id", "cmap_score", "n_ing"])
    return ext, len(ing), n_matched


def boot_ci(y, s, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)
    rows = []

    for fname, dk in DISEASES.items():
        ext, n_ing, n_mat = build_herb_cmap(fname)
        if ext.empty:
            continue
        pos_all = set(gold[(gold["disease_id"] == dk) & (gold["label"] == 1)]["herb_id"].astype(str))
        herbs = list(ext["herb_id"])
        y = np.array([1 if h in pos_all else 0 for h in herbs], dtype=float)
        n_pos = int(y.sum())
        if n_pos < 3:
            print(f"{fname}: 子集内正例不足({n_pos}), 跳过")
            continue

        cmap = ext["cmap_score"].values.astype(float)
        score_rev = -cmap          # CMap 约定: 越负越治疗 -> 取负后越高越治疗
        score_raw = cmap           # 方向对照

        a_rev = float(roc_auc_score(y, score_rev))
        a_raw = float(roc_auc_score(y, score_raw))
        lo, hi = boot_ci(y, score_rev)
        try:
            p = float(mannwhitneyu(score_rev[y == 1], score_rev[y == 0],
                                   alternative="greater").pvalue)
        except Exception:
            p = float("nan")

        rows.append({
            "disease": fname,
            "n_herbs_in_subset": len(herbs),
            "n_pos_in_subset": n_pos,
            "AUROC_CMap实测(逆转方向)": round(a_rev, 4),
            "AUROC_CMap(对照方向)": round(a_raw, 4),
            "CI_lo": round(lo, 4),
            "CI_hi": round(hi, 4),
            "p_value": f"{p:.3g}",
            "判定": "显著" if (lo > 0.5) else ("反向显著" if hi < 0.5 else "不显著"),
        })
        print(f"{fname}: 子集{len(herbs)}味(正例{n_pos}) | "
              f"AUROC={a_rev:.4f} CI[{lo:.3f},{hi:.3f}] p={p:.3g}")

    df = pd.DataFrame(rows)
    mean_row = {"disease": "MEAN", "n_herbs_in_subset": "-", "n_pos_in_subset": "-"}
    for c in ["AUROC_CMap实测(逆转方向)", "AUROC_CMap(对照方向)"]:
        mean_row[c] = round(df[c].astype(float).mean(), 4)
    mean_row["CI_lo"] = "-"
    mean_row["CI_hi"] = "-"
    mean_row["p_value"] = "-"
    mean_row["判定"] = "-"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    os.makedirs(PAPER, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 96)
    print("正对照: HERB CMap【实测】数据能否识别金标准治疗草药?  (CMap约定: 越负=越逆转=越治疗)")
    print("=" * 96)
    print(df.to_string(index=False))
    print("\n判读: AUROC 明显>0.5 且显著 => 标签有效+逆转子实测成立 => 问题在模拟签名 => 选路线B")
    print("      AUROC ≈0.5            => 标签不可靠或逆转不适用 => 选路线A")
    print(f"\n[写入] {OUT}")


if __name__ == "__main__":
    main()
