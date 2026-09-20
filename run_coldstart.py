# -*- coding: utf-8 -*-
"""冷启动 / 鲁棒性评估 (repeated random hold-out of positives).
验证 D-TMS 是否依赖特定正例:
  (1) 随机移除 frac 比例正例后, AUROC 是否仍稳定 (应接近 FULL AUROC -> 鲁棒);
  (2) held-out 正例在全部 491 草药排序中的绝对位置 (冷启动召回 / 平均排名百分位).
D-TMS 无监督、不打分依赖金标准, 故 hold-out 只影响"评估所用的正例集", 不改变分数;
这正好隔离出"方法对未见/被剔除正例的排序能力".
"""
import csv, os, random, numpy as np

BASE = r"c:\Users\alikex\Desktop\Work\systems_mapping"
PD = os.path.join(BASE, "output", "results", "paper_deliverables")
PROC = os.path.join(BASE, "data", "processed")
DATA = os.path.join(BASE, "data")

def load_scores(path):
    rows = {}; dis = None
    with open(path, newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        dis = [c for c in r.fieldnames if c != 'herb_id']
        for row in r:
            rows[row['herb_id']] = {d: float(row[d]) for d in dis}
    return rows, dis

def load_gold(path):
    pos = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['label'].strip() == '1':
                pos.setdefault(row['disease_id'], set()).add(row['herb_id'])
    return pos

def roc_auc(scores, y):
    s = np.asarray(scores, float); lab = np.asarray(y, float)
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return (ranks[lab==1].sum() - n1*(n1+1)/2)/(n1*n0)

s, dis = load_scores(os.path.join(PD, "hdmap_s_scores_enrichment_method.csv"))
pos = load_gold(os.path.join(DATA, "gold_standard_final.csv"))
DISEASES = ["AD", "PAH", "RA", "T2D"]
M = 200
fracs = [0.1, 0.2, 0.3, 0.5]
out = []
print("=== Cold-start / robustness (repeated random hold-out of positives, M=%d) ===" % M)
for d in DISEASES:
    herbs = list(s.keys())
    sc = np.array([s[h][d] for h in herbs])
    pset = pos.get(d, set()); n = len(pset)
    y_full = [1 if h in pset else 0 for h in herbs]
    auc_full = roc_auc(sc, y_full)
    order = np.argsort(-sc)
    rank = {herbs[i]: i for i in order}
    for frac in fracs:
        k = max(1, int(round(frac*n)))
        aucs = []; top10 = []; rp = []
        for t in range(M):
            rng = random.Random(1000+t)
            held = set(rng.sample(list(pset), k))
            train = pset - held
            y_tr = [1 if h in train else 0 for h in herbs]
            aucs.append(roc_auc(sc, y_tr))
            rks = [rank[h] for h in held]
            top10.append(np.mean([1 if r < int(0.1*len(herbs)) else 0 for r in rks]))
            rp.append(np.mean([r/len(herbs) for r in rks]))
        out.append([d, frac, f"{auc_full:.3f}", f"{np.mean(aucs):.3f}", f"{np.std(aucs):.3f}",
                    f"{np.mean(top10):.3f}", f"{np.mean(rp):.3f}"])
        print(f"{d} drop={frac:.0%}: AUROC_full={auc_full:.3f}  AUROC_holdout={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
              f"heldout_top10%recall={np.mean(top10):.3f}  heldout_mean_rank%={np.mean(rp):.3f}")

with open(os.path.join(PD, "coldstart_table.csv"), "w", newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(["disease", "drop_frac", "AUROC_full", "AUROC_holdout_mean", "AUROC_holdout_std",
                "heldout_top10%_recall", "heldout_mean_rank_pct"])
    for row in out:
        w.writerow(row)
print("saved coldstart_table.csv")
