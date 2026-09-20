#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向校准 (数据驱动, 替代 a-priori 药理先验)
=========================================
问题:
  config.hdmap_s.disease_tms_mechanism 用药理先验决定"高分=治疗 vs 低分=治疗",
  但对 3/4 病 (AD/PAH/T2D) 先验是反的, 导致下游所有 TMS/AUROC/案例研究方向反置。

解决:
  不用先验, 用金标准"已知治疗草药"在 dual 通道 (rev=-cos, eng=+cos) 上的得分,
  自动学出每个病的正确治疗方向:
    - 正例在 rev(-cos) 上更高 -> 治疗是"低分好" -> antagonist
    - 正例在 eng(+cos) 上更高 -> 治疗是"高分好" -> engagement
  并用 5-fold 留出法避免"同一批金标准既学方向又评估"的循环论证。

产物:
  - direction_calibration.csv               每病: 学到方向 / rev_mean / eng_mean / 留出AUROC / 最终AUROC
  - hdmap_s_scores_dual_calibrated.csv      用学到方向合并的 TMS 矩阵 (越高=越治疗)

用法 (PyCharm 直接 Run):
  python direction_calibration.py
  python direction_calibration.py --gold data/gold_standard.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from config import RESULTS_DIR, GOLD_STANDARD_PATH
from data_loader import load_herb_ingredients
from evaluation_metrics import load_gold_standard_mechanism


def load_dual(out_dir):
    """载入 dual 通道矩阵, 索引统一转 str 以便与金标准 herb_id 对齐"""
    rev = pd.read_csv(os.path.join(out_dir, "hdmap_s_scores_dual_reverse.csv")).set_index("herb_id")
    eng = pd.read_csv(os.path.join(out_dir, "hdmap_s_scores_dual_engage.csv")).set_index("herb_id")
    rev.index = rev.index.astype(str)
    eng.index = eng.index.astype(str)
    return rev, eng


def load_positive(gold_path):
    """返回 ({disease: set(正例 herb_id_str)}, {disease: set(负例 herb_id_str)})

    修复历史 bug (2026-08-30): 旧版写成 `gold, _, _, _ = ...`,
    把 load_gold_standard_mechanism 返回的 labeled negatives (label=0) 丢掉了,
    导致后续只能在"正例 vs 全体背景"的 PU 偏倚口径下选方向 (见 STATUS §6)。"""
    herb_ing = load_herb_ingredients()
    gold, _, gold_neg, _ = load_gold_standard_mechanism(gold_path, herb_ing)
    # gold / gold_neg: {disease: [herb_id, ...]}
    return ({d: set(map(str, hs)) for d, hs in gold.items()},
            {d: set(map(str, hs)) for d, hs in gold_neg.items()})


def learn_direction(rev, eng, pos_herbs, neg_herbs, disease):
    """用 *正例 vs 负例* 的判别力学方向, 返回 (direction, (rev_mean, eng_mean))

    修复历史 bug (2026-08-30):
      旧判据 `direction = antagonist if rev_mean > eng_mean else engagement` 是错的。
      因 rev=-cos / eng=+cos, rev_mean 恒等于 -eng_mean, 该比较等价于
      "只看正例平均 cos 的正负号", 完全没有与负例比较 -> 方向学反,
      且与本文件算出的 final_AUROC 自相矛盾 (标签 engagement 却报出反向的 AUROC)。

    新判据 (正确):
      在 正例 vs 负例 上算 eng(+cos) 的 AUROC。因 rev=-eng, AUROC(rev)=1-AUROC(eng), 故
          AUROC(eng) >= 0.5 -> engagement (高分=治疗)
          AUROC(eng) <  0.5 -> antagonist  (低分=治疗)
      缺 labeled 负例时回退到旧的符号启发式并打印警告 (该结果不可靠, 属 PU 偏倚口径)。

    另注 (历史 bug 修复): 必须按 disease 取列, 否则 rev.loc[p].values 是
    (n_pos, 4) 的 2D 矩阵, np.nanmean 会跨 4 病混合求均值 (错)。"""
    p = [h for h in pos_herbs if h in eng.index]
    n = [h for h in (neg_herbs or []) if h in eng.index]
    if len(p) < 2:
        return None, None
    rev_mean = float(np.nanmean(rev.loc[p, disease].values))
    eng_mean = float(np.nanmean(eng.loc[p, disease].values))

    if len(n) < 2:
        print(f"    [警告] {disease}: labeled 负例不足 (n={len(n)}), "
              f"回退到符号启发式(只看正例cos正负, 不可靠)")
        direction = "antagonist" if rev_mean > eng_mean else "engagement"
        return direction, (rev_mean, eng_mean)

    sp = eng.loc[p, disease].values.astype(float)
    sn = eng.loc[n, disease].values.astype(float)
    y = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
    s = np.concatenate([sp, sn])
    if len(set(y)) < 2:
        direction = "antagonist" if rev_mean > eng_mean else "engagement"
    else:
        a_eng = float(roc_auc_score(y, s))
        direction = "engagement" if a_eng >= 0.5 else "antagonist"
    return direction, (rev_mean, eng_mean)


def cv_auroc(rev, eng, pos_herbs, neg_herbs, disease, n_fold=5):
    """5-fold 留出: 每折用 train 正例学方向, 在 val 正例上算 *该病* AUROC, 返回均值。
    避免循环论证: 评估用的正例不参与学方向。

    评估对照 (2026-08-30 修正):
      优先用 labeled 负例 (label=0) 作对照 —— 这是 STATUS §6 推荐的诚实口径。
      缺负例时才回退到"val 正例 vs 全体背景", 该口径含 PU 偏倚, 仅供粗略参考。

    修复历史 bug: 之前 tms.loc[all_h].values 是 (n_herbs, 4) 的 2D score,
    新版 sklearn (>=1.5) 把它当 4 分类, 强制要求 multi_class 参数。
    正确做法是按 disease 切单列, 拿到 1D score。"""
    p = np.array([h for h in pos_herbs if h in rev.index])
    if len(p) < n_fold * 2:
        return None
    neg_h = [h for h in (neg_herbs or []) if h in rev.index]
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    all_h = list(rev.index)
    aucs = []
    for tr, va in kf.split(p):
        direction, _ = learn_direction(rev, eng, p[tr], neg_herbs, disease)
        tms = rev if direction == "antagonist" else eng
        val_pos = list(p[va])
        if len(neg_h) >= 2:
            # 诚实口径: val 正例 vs labeled 负例
            eval_h = val_pos + neg_h
            y = np.array([1.0] * len(val_pos) + [0.0] * len(neg_h), dtype=float)
        else:
            # 回退: val 正例 vs 全体背景 (PU 偏倚)
            eval_h = all_h
            vp = set(val_pos)
            y = np.array([1.0 if h in vp else 0.0 for h in all_h], dtype=float)
        s = tms.loc[eval_h, disease].values.astype(float)
        if len(set(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, s))
    return float(np.mean(aucs)) if aucs else None


def auroc_full(tms_col, pos_set):
    """全库协议 AUROC: 正例=pos_set, 负例=其余全部草药。
    tms_col 已是 Series (单列), 所以 1D score, 不会触发 multi_class 错。"""
    all_h = list(tms_col.index)
    y = np.array([1 if h in pos_set else 0 for h in all_h], dtype=float)
    s = tms_col.loc[all_h].values.astype(float)
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser(description="数据驱动方向校准 (替代 a-priori)")
    ap.add_argument("--gold", default=GOLD_STANDARD_PATH, help="金标准 CSV 路径")
    ap.add_argument("--n-fold", type=int, default=5)
    args = ap.parse_args()

    # 定位 dual 矩阵
    out_dir = None
    for d in [os.path.join(RESULTS_DIR, "weighted_kegg"), RESULTS_DIR]:
        if os.path.exists(os.path.join(d, "hdmap_s_scores_dual_reverse.csv")):
            out_dir = d
            break
    if out_dir is None:
        print("[错误] 找不到 dual 矩阵, 请先运行 generate_dual_outputs.py")
        sys.exit(1)

    rev, eng = load_dual(out_dir)
    diseases = list(rev.columns)
    pos, neg = load_positive(args.gold)
    print(f"金标准正例数: {{ {', '.join(f'{d}:{len(pos.get(d, []))}' for d in diseases)} }}")
    print(f"金标准负例数: {{ {', '.join(f'{d}:{len(neg.get(d, []))}' for d in diseases)} }}")
    if not any(len(neg.get(d, [])) for d in diseases):
        print("[警告] 金标准无 labeled 负例 (label=0), 方向将回退到不可靠的符号启发式")

    rows = []
    calib = pd.DataFrame(index=rev.index)
    better = 0
    for d in diseases:
        p = pos.get(d, set())
        n = neg.get(d, set())
        direction, means = learn_direction(rev, eng, p, n, d)
        if direction is None:
            print(f"  {d}: 正例不足, 保持 engagement(默认)")
            calib[d] = eng[d]
            rows.append({"disease": d, "direction": "engagement(默认)",
                         "rev_mean": None, "eng_mean": None,
                         "cv_AUROC": None, "final_AUROC": None, "n_pos": len(p)})
            continue

        cv = cv_auroc(rev, eng, p, n, d, n_fold=args.n_fold)
        calib[d] = rev[d] if direction == "antagonist" else eng[d]
        final_auc = auroc_full(calib[d], p)
        rows.append({"disease": d, "direction": direction,
                     "rev_mean": round(means[0], 4), "eng_mean": round(means[1], 4),
                     "cv_AUROC": round(cv, 4) if cv else None,
                     "final_AUROC": round(final_auc, 4) if final_auc else None,
                     "n_pos": len(p)})
        if final_auc is not None and final_auc > 0.5:
            better += 1

    cdf = pd.DataFrame(rows)
    out_csv = os.path.abspath(os.path.join(out_dir, "..", "direction_calibration.csv"))
    cdf.to_csv(out_csv, index=False)
    calib.index.name = "herb_id"
    calib.to_csv(os.path.join(out_dir, "hdmap_s_scores_dual_calibrated.csv"))

    print("\n" + "=" * 82)
    print("方向校准结果 (数据驱动, 替代 a-priori 先验)")
    print("=" * 82)
    print(cdf.to_string(index=False))
    print(f"\n校准后 {better}/{len(diseases)} 病 final_AUROC > 0.5 (随机基线)")
    print(f"[写入] {os.path.join(out_dir, 'hdmap_s_scores_dual_calibrated.csv')}")
    print(f"[写入] {out_csv}")

    print("\n>>> 把下列机制填入 config.yaml 的 hdmap_s.disease_tms_mechanism:")
    print("  disease_tms_mechanism:")
    for r in rows:
        if str(r["direction"]).endswith("(默认)"):
            print(f"    {r['disease']}: engagement")
        else:
            print(f"    {r['disease']}: {r['direction']}")


if __name__ == "__main__":
    main()
