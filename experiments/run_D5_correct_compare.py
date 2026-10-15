"""
导师对比 D5 (修正版): 用正确的文件路径
  Proposed (enrichment): output/results/AD_kegg_results/hdmap_s_scores.csv  (08-22 05:55)
  Baseline (weighted):   output/results/AD_results/hdmap_s_scores.csv        (08-22 06:19)
先校验形状/草药集合一致, 再算:
  1. 固定方向全库 AUROC (enrichment=high好, weighted=low好)  <- 论文该报的, 不是取max
  2. Top30 经典药命中率
"""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
from config import GOLD_STANDARD_PATH
from evaluation_metrics import load_gold_standard

ENRICH = r"output\results\AD_kegg_results\hdmap_s_scores.csv"
WEIGHT = r"output\results\AD_results\hdmap_s_scores.csv"
enrich = pd.read_csv(ENRICH, index_col=0)
weight = pd.read_csv(WEIGHT, index_col=0)

print("############ 0. 形状与草药集合校验 ############")
print(f"  enrichment: shape={enrich.shape}  cols={list(enrich.columns)}")
print(f"  weighted:   shape={weight.shape}  cols={list(weight.columns)}")
print(f"  草药集合完全一致? {set(enrich.index)==set(weight.index)}  (重叠 {len(set(enrich.index)&set(weight.index))}/{len(enrich.index)})")

pos_dict, neg_dict, _ = load_gold_standard(GOLD_STANDARD_PATH, None)

def fixed_auroc(scores_dict, disease, higher_is_better):
    pos = set(pos_dict.get(disease, [])); neg = set(neg_dict.get(disease, []))
    ps = [scores_dict[h] for h in pos if h in scores_dict]
    ns = [scores_dict[h] for h in neg if h in scores_dict]
    if len(ps)==0 or len(ns)==0: return None, None
    y = [1]*len(ps) + [0]*len(ns); s = ps + ns
    if not higher_is_better: s = [-x for x in s]
    return roc_auc_score(y, s), average_precision_score(y, s)

CLASSIC = {
    'AD':  ['Ginkgo','Huperzia','Ginseng','Polygala','Salvia','Polygonum Multiflorum','Yuanzhi','Baical','Coptis','Cinnamom','Zingiber','Paeonia','Bupleur','Saposhnikov','Artemisi'],
    'PAH': ['Pueraria','Salvia','Ligustic','Angelica','Astragalus','Ginkgo','Carthamus','Ephedra','Linder','Zingiber','Artemisi','Curcum'],
    'RA':  ['Paeonia','Tripteryg','Sinomen','Smilac','Curcum','Celastr','Euonym','Caulis'],
    'T2D': ['Dioscore','Salvia','Rehmann','Anemarrhen','Momordic','Ginseng','Polygonat','Asparag','Cassia','Cuscut','Figwort','Pseudostell'],
}
names = pd.read_csv(r"data\processed\herb_ingredients_processed.csv")[['herb_id','herb_name']].drop_duplicates()
nmap = dict(zip(names['herb_id'], names['herb_name']))

def top30_classic(df, disease, ascending):
    s = df[disease].sort_values(ascending=ascending).head(30).index
    kw = CLASSIC[disease]
    hit = [h for h in s if any(k.lower() in str(nmap.get(h,'')).lower() for k in kw)]
    return len(hit), [nmap.get(h,'') for h in hit]

print("\n############ Table 2 (修正): Proposed(enrichment) vs Baseline(weighted) ############")
rows = []
for d in ['AD','PAH','RA','T2D']:
    ea, ep = fixed_auroc(enrich[d].astype(float).to_dict(), d, True)   # enrichment: high好
    wa, wp = fixed_auroc(weight[d].astype(float).to_dict(), d, False)  # weighted: low好
    ec, el = top30_classic(enrich, d, False)   # enrichment high好 -> 降序
    wc, wl = top30_classic(weight, d, True)    # weighted low好 -> 升序
    rows.append({
        'Disease': d,
        'Proposed_AUROC(high好)': round(ea,3), 'Proposed_AUPRC': round(ep,3), f'Proposed_Top30_classic': f"{ec}: {', '.join(el)}",
        'Baseline_AUROC(low好)': round(wa,3), 'Baseline_AUPRC': round(wp,3), f'Baseline_Top30_classic': f"{wc}: {', '.join(wl)}",
        'AUROC_gain': round(ea-wa,3),
    })
    print(f"\n  [{d}] Proposed AUROC={ea:.3f} (Top30经典药 {ec}: {el})")
    print(f"  [{d}] Baseline  AUROC={wa:.3f} (Top30经典药 {wc}: {wl})")
    print(f"  [{d}] => AUROC 提升 {ea-wa:+.3f}")

df = pd.DataFrame(rows)
out = r"output\results\paper_deliverables\D5_baseline_comparison_correct.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"\n[已保存] {out}")
