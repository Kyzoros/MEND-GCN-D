"""
诊断 C：用扩充金标准复测，排除"金标准太弱导致测不出信号"的可能。

同时对比两个金标准文件:
  - data/gold_standard.csv            (原始, 每病阳性少)
  - data/gold_standard_expanded.csv   (扩充, 每病阳性更多)

对每个文件、每种 AUROC 协议(full 全库 / subset 子集) 与 每个方向先验
(naive=engagement高=好, lit=antagonist低=好) 都算一遍，看:
  1) 换大金标准后，full 协议 AUROC 是否明显离开 ~0.5?
  2) 若仍未离开 -> 方法本身问题(走 B 改进方法)
     若离开   -> 之前是金标准太小测不出(可继续走 A/诚实诊断但更有底气)
"""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
import os
import sys
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)

COS = 'output/results/AD_kegg_results/hdmap_s_scores_dual_cos.csv'
cos = pd.read_csv(COS).set_index('herb_id')

GOLDS = {
    'orig': 'data/gold_standard.csv',
    'expanded': 'data/gold_standard_expanded.csv',
}

def auroc_full(s, pos):
    s = s.dropna()
    y = np.array([1 if h in pos else 0 for h in s.index.astype(str)], int)
    if 0 < y.sum() < len(y):
        return round(float(roc_auc_score(y, s.values.astype(float))), 3)
    return float('nan')

def auroc_subset(s, pos, neg):
    sub = [h for h in s.index.astype(str) if h in (pos | neg)]
    if not sub:
        return float('nan')
    ss = s.loc[sub]
    y = np.array([1 if h in pos else 0 for h in ss.index.astype(str)], int)
    if 0 < y.sum() < len(y):
        return round(float(roc_auc_score(y, ss.values.astype(float))), 3)
    return float('nan')

MECH = {'AD': 'antagonist', 'PAH': 'antagonist', 'RA': 'antagonist', 'T2D': 'antagonist'}

print("=" * 100)
print("诊断 C: 扩充金标准复测 (full=全库协议, subset=子集协议; naive=高=好, lit=低=好)")
print("=" * 100)

summary = []
for gname, gpath in GOLDS.items():
    gold = pd.read_csv(gpath)
    print(f"\n########## 金标准文件: {gname}  ({gpath}, n={len(gold)}) ##########")
    print(f"{'disease':<6} {'n_pos':>5} {'n_neg':>5} | {'naive_full':>10} {'naive_sub':>9} | {'lit_full':>9} {'lit_sub':>8} | {'opt_full*':>9}")
    print("-" * 78)
    for dis in ['AD', 'PAH', 'RA', 'T2D']:
        sub = gold[gold['disease_id'] == dis]
        pos = set(sub[sub['label'] == 1]['herb_id'].astype(str))
        neg = set(sub[sub['label'] == 0]['herb_id'].astype(str))
        c = cos[dis].astype(float)
        naive_full = auroc_full(c, pos)
        naive_sub = auroc_subset(c, pos, neg)
        lit_full = auroc_full(-c, pos)
        lit_sub = auroc_subset(-c, pos, neg)
        # optimal 方向(循环论证诊断用)
        if abs(lit_full - 0.5) >= abs(naive_full - 0.5):
            opt_full = lit_full
        else:
            opt_full = naive_full
        print(f"{dis:<6} {len(pos):>5} {len(neg):>5} | {str(naive_full):>10} {str(naive_sub):>9} | {str(lit_full):>9} {str(lit_sub):>8} | {str(opt_full):>9}")
        summary.append({'gold': gname, 'disease': dis, 'n_pos': len(pos), 'n_neg': len(neg),
                        'naive_full': naive_full, 'naive_sub': naive_sub,
                        'lit_full': lit_full, 'lit_sub': lit_sub, 'opt_full': opt_full})

print("\n" + "=" * 100)
print("判读:")
print("  - 若 lit_full(文献方向, 全库) 在任一大金标准下仍 ≈0.5 -> 方法本身信号弱(走 B 改进方法)")
print("  - 若 expanded 的 lit_full 明显 >0.6 -> 之前是金标准太小测不出(可走 A/诚实诊断)")
print("  - *opt_full 为循环论证列(在测试集上挑最优方向)，仅供判断方向先验是否合理，论文不可用。")
print("=" * 100)

sdf = pd.DataFrame(summary)
sdf.to_csv('output/results/paper_deliverables/C_goldstandard_check.csv', index=False)
print("已保存: output/results/paper_deliverables/C_goldstandard_check.csv")
