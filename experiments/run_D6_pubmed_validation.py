# -*- coding: utf-8 -*-
"""
D6: PubMed 文献共现 独立外部验证
================================
目的: 用 PubMed 文献共现(草药科学名 × 疾病)作为**独立于金标准**的外部证据,
      检验 enrichment 方法的 Top-K 推荐是否显著富集"文献支持的草药"。

设计(规避循环论证):
  - 金标准(curated/expanded)完全不用。
  - "文献支持" = 该草药科学名与该疾病在 PubMed 有 ≥1 篇共现文献。
  - Fisher 精确检验: Top-K 中"文献支持"比例 vs 全库基线(K/N)。

名字映射: 由草药拉丁名 norm_lat 提取 (属, 种); 有种则查 "属 种", 无种则查 "属"。
          PubMed 对科学名召回良好("Panax ginseng"+AD=8 篇, "Ginkgo biloba"+AD=41 篇)。

可断点续跑: 结果缓存到 data/processed/pubmed_cooccurrence_cache.csv, 重跑只补缺。

输出: output/results/paper_deliverables/D6_pubmed_validation.csv
"""
import os, sys, re, time
import urllib.request, urllib.parse
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from collections import defaultdict
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "output", "results", "paper_deliverables")
CACHE = os.path.join(PROC, "pubmed_cooccurrence_cache.csv")
os.makedirs(OUT, exist_ok=True)

DISEASES = ["AD", "PAH", "RA", "T2D"]
DISEASE_QUERY = {
    "AD": '"Alzheimer disease"[Title/Abstract]',
    "PAH": '"pulmonary arterial hypertension"[Title/Abstract]',
    "RA": '"rheumatoid arthritis"[Title/Abstract]',
    "T2D": '"type 2 diabetes"[Title/Abstract]',
}
TOPK = 30
DELAY = 0.35
TIMEOUT = 15

# ---------- 草药科学名 ----------
PARTS = {"herba","radix","rhizoma","folium","fructus","cortex","flos","semen",
         "caulis","ramulus","pericarpium","seed","leaf","root","bark","flower",
         "fruit","stem","tuber","bulbus","resina","oleum","lignum","spica",
         "herb","radixet","rhizomaet","et"}
def genus_base(g):
    g = g.lower()
    if len(g) > 4 and g.endswith("ae"): return g[:-2]
    if len(g) > 3 and g.endswith("i"): return g[:-1]
    return g
def norm_lat(s):
    if not isinstance(s, str): return None
    s = s.split(",")[0].lower()
    toks = re.findall(r"[a-z]+", s)
    sci = [t for t in toks if t not in PARTS]
    if not sci: return None
    return (genus_base(sci[0]), sci[1] if len(sci) > 1 else None)
def sci_name(nl):
    if not nl: return None
    g, sp = nl
    if sp:
        return f"{g.capitalize()} {sp}"
    return g.capitalize()

hips = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
proj = hips[["herb_id", "herb_name"]].drop_duplicates()
universe = list(pd.read_csv(os.path.join(BASE, "output", "results",
             "AD_kegg_results", "hdmap_s_scores.csv"), index_col=0).index)
name_by_id = dict(zip(proj["herb_id"], proj["herb_name"]))
nl_by_id = dict(zip(proj["herb_id"], proj["herb_name"].map(norm_lat)))
sci_by_id = {h: sci_name(nl_by_id.get(h)) for h in universe}
no_name = [h for h in universe if not sci_by_id[h]]
print(f"草药宇宙: {len(universe)}, 有科学名: {len(universe)-len(no_name)}, 无: {len(no_name)}")

# ---------- PubMed 查询 ----------
def esearch_count(term):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "term": term, "rettype": "count", "retmode": "xml"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                xml = r.read().decode("utf-8")
            m = re.search(r"<Count>(\d+)</Count>", xml)
            return int(m.group(1)) if m else 0
        except Exception:
            time.sleep(2 * (attempt + 1))
    return -1  # 失败标记

# 自检: 打印 5 个已知草药的计数
print("\n[自检] 科学名查询样例:")
for probe in ["Ginkgo biloba", "Panax ginseng", "Tripterygium wilfordii",
              "Rehmannia glutinosa", "Momordica charantia"]:
    c = esearch_count(f'"{probe}"[Title/Abstract] AND {DISEASE_QUERY["AD" if probe!="Tripterygium wilfordii" else "RA"]}')
    time.sleep(DELAY)
    print(f"  {probe}: {c}")

# ---------- 缓存加载 ----------
cache = {}
if os.path.exists(CACHE):
    cdf = pd.read_csv(CACHE)
    for _, r in cdf.iterrows():
        cache[(str(r["herb_id"]), str(r["disease"]))] = int(r["count"])
    print(f"[缓存] 已有 {len(cache)} 条")
else:
    print("[缓存] 无, 从头查询")

# ---------- 查询缺失对 ----------
pairs = [(h, d) for h in universe for d in DISEASES if (h, d) not in cache]
print(f"[查询] 待查 {len(pairs)} 条")
for i, (h, d) in enumerate(pairs):
    nm = sci_by_id.get(h)
    if not nm:
        cache[(h, d)] = 0
        continue
    cnt = esearch_count(f'"{nm}"[Title/Abstract] AND {DISEASE_QUERY[d]}')
    cache[(h, d)] = cnt if cnt >= 0 else 0
    if (i + 1) % 50 == 0:
        pd.DataFrame([{"herb_id": k[0], "disease": k[1], "count": v}
                      for k, v in cache.items()]).to_csv(CACHE, index=False)
        print(f"  ... {i+1}/{len(pairs)} 完成 (已缓存)")
    time.sleep(DELAY)

pd.DataFrame([{"herb_id": k[0], "disease": k[1], "count": v}
              for k, v in cache.items()]).to_csv(CACHE, index=False)
print(f"[完成] 缓存 {len(cache)} 条 -> {CACHE}")

# ---------- 评估: 全库基线 vs Top-K 富集 ----------
def support(h, d):
    return cache.get((h, d), 0) > 0

def load_top30(path, disease, ascending):
    df = pd.read_csv(path)
    s = df.set_index("herb_id")[disease].sort_values(ascending=ascending)
    return list(s.head(TOPK).index)

enrich_top = {d: load_top30(os.path.join(BASE, "output", "results",
            "AD_kegg_results", "hdmap_s_scores.csv"), d, False) for d in DISEASES}
weight_top = {d: load_top30(os.path.join(BASE, "output", "results",
            "AD_results", "hdmap_s_scores.csv"), d, True) for d in DISEASES}

rows = []
print("\n" + "=" * 100)
print("D6 PubMed 独立验证: Top30 文献支持富集 (Fisher)")
print("=" * 100)
for d in DISEASES:
    N = len(universe)
    K = sum(1 for h in universe if support(h, d))
    print(f"\n--- {d} (疾病查询: {DISEASE_QUERY[d]}) ---")
    print(f"  全库基线: {K}/{N} 味草药有文献支持 ({100*K/N:.1f}%)")
    for tag, top in [("enrichment", enrich_top[d]), ("weighted", weight_top[d])]:
        k = sum(1 for h in top if support(h, d))
        hit = [name_by_id.get(h, h) for h in top if support(h, d)]
        table = [[k, TOPK - k], [K - k, N - K - (TOPK - k)]]
        if k > 0 and K > 0:
            odds, p = fisher_exact(table, alternative="greater")
            odds = round(odds, 2); p = f"{p:.2e}"
        else:
            odds, p = "NA", "NA"
        print(f"  {tag:10s} Top{TOPK} 命中 {k:2d} ({100*k/TOPK:.0f}%)  OR={odds}  p={p}  {'***' if p!='NA' and float(p)<0.05 else ''}")
        rows.append({"disease": d, "method": tag, "N": N, "K_baseline": K,
                     "baseline_rate": round(K/N, 4), "topK": TOPK, "k_hits": k,
                     "hit_rate": round(k/TOPK, 4), "odds_ratio": odds,
                     "fisher_p": p, "hit_herbs": "; ".join(hit)})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "D6_pubmed_validation.csv"), index=False)
print(f"\n[已保存] {os.path.join(OUT, 'D6_pubmed_validation.csv')}")
