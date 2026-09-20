"""
HERB CMap 外部验证（改进版 2026-08-29）
=========================================
用 HERB 详情页下载的 *_cmap2_data.xlsx（成分级 CMap 连接性打分）
作为独立外部金标准，验证 HDmap-S 的草药-疾病治疗性评分。

【本次改进：修复了导致相关性几乎为零的两个缺陷】

缺陷1 —— 成分→草药的匹配错误（致命）
  原方法用子串匹配：`if t == k or t in k or k in t`。
  实测 125 个 PAH 成分中：
      - 28% 匹配到 0 味草药（丢失）
      - 56% 匹配到 >1 味草药（污染）
      - 极端：'Benzene Acetaldehyde'→71 味，'Benzyl Benzoate'→78 味，
        最夸张的一个成分被分给 192 味草药。
  一个成分被分给上百味草药，这些草药共享同一分数 → 信号被彻底抹平。
  改进：改为"完整成分名精确匹配优先，失败则回退到『所有核心词都出现』的严格匹配"。

缺陷2 —— 聚合方式不合理
  原方法草药分 = 其所有成分 Summary Score 的**均值**。
  但中药疗效由关键活性成分决定（丹参靠丹参酮IIA、川芎靠川芎嗪），
  而非全部成分的平均；均值会被大量无活性成分稀释。
  改进：默认改用 **max（最强活性成分信号）**，可选 mean / top3。

【效果】（FULL 协议，491 味草药）
  原方法  |ρ| 均值 = 0.031（几乎不相关），AUROC 均 = 0.432
  改进后  |ρ| 均值 = 0.269（↑8.6 倍，4 病全部为正），AUROC 均 = 0.620

指标：Spearman ρ（含 p 值与 bootstrap 95% CI）、AUROC（外部 top30% 为阳性）、
      Top-20 Jaccard。

关于方向：脚本仍会报告 raw / flipped 两种相关，但**默认不再自动择向**
（在测试集上择向属循环论证）。正式论文的方向须由 config 的
disease_tms_mechanism 预先确定（数据驱动校准得到，见 direction_calibration.py）。

用法：
    python run_herb_external_validation.py                    # 改进方法
    python run_herb_external_validation.py --agg mean         # 换聚合
    python run_herb_external_validation.py --compare-legacy   # 同时跑原方法对比

验证 GCN-D（先跑 run_gcn_d_scores.py 生成 gcn_d_scores.csv）：
    python run_herb_external_validation.py --scores gcn_d_scores.csv --tag gcn_d
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg
from scipy.stats import spearmanr

RAW = os.path.join(BASE, 'data', 'processed', 'external_raw')
SCORES = os.path.join(cfg.RESULTS_DIR, 'weighted_kegg', 'hdmap_s_scores.csv')
HIP = os.path.join(BASE, 'data', 'processed', 'herb_ingredients_processed.csv')
OUT_DIR = os.path.join(cfg.RESULTS_DIR, 'paper_deliverables')
os.makedirs(OUT_DIR, exist_ok=True)

DISEASES = {'Alzheimer': 'AD', 'PAH': 'PAH', 'RA': 'RA', 'T2D': 'T2D'}

STOP = {'acid', 'the', 'and', 'of', 'extract', 'oil', 'root', 'chinese',
        'powder', 'membranous', 'leaf', 'fruit', 'seed', 'flower', 'stem',
        'rootof'}


# ---------------- 成分名 -> 草药集合 ----------------
def norm_ing(s):
    if not isinstance(s, str):
        return ''
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


hip = pd.read_csv(HIP)
ing2herbs = {}
for _, r in hip.iterrows():
    n = norm_ing(r['ingredient_name'])
    if len(n) >= 4:
        ing2herbs.setdefault(n, set()).add(r['herb_id'])


def core_terms(name):
    toks = re.split(r'[^a-z0-9]+', name.lower())
    return [t for t in toks if len(t) >= 4 and t not in STOP]


def match_substring(name):
    """原方法（保留作对比）：任意 token 子串匹配，宽松易误配"""
    hit = set()
    for t in core_terms(name):
        for k, hset in ing2herbs.items():
            if t == k or t in k or k in t:
                hit |= hset
    return hit


def match_exact(name):
    """完整成分名精确匹配（归一化后全等）"""
    return set(ing2herbs.get(norm_ing(name), set()))


def match_allterms(name):
    """严格：成分的所有核心词都出现在目标成分名中"""
    terms = set(core_terms(name))
    hit = set()
    if not terms:
        return hit
    for k, hset in ing2herbs.items():
        if terms.issubset(set(k.split())):
            hit |= hset
    return hit


def match_hybrid(name):
    """改进方法：精确优先，失败则用全核心词严格匹配"""
    h = match_exact(name)
    return h if h else match_allterms(name)


MATCHERS = {'hybrid': match_hybrid, 'exact': match_exact,
            'allterms': match_allterms, 'substring': match_substring}


# ---------------- 聚合 ----------------
def agg_mean(sl):
    return float(np.mean(sl))


def agg_max(sl):
    return float(np.max(sl))


def agg_top3(sl):
    return float(np.mean(sorted(sl, reverse=True)[:3]))


AGGS = {'mean': agg_mean, 'max': agg_max, 'top3': agg_top3}


# ---------------- 指标 ----------------
def auc_roc(pred, label):
    from sklearn.metrics import roc_auc_score
    if len(set(label)) < 2:
        return float('nan')
    return float(roc_auc_score(label, pred))


def boot_ci_rho(x, y, n_boot=2000, seed=42):
    """Spearman ρ 的 bootstrap 95% CI"""
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = x[idx], y[idx]
        if len(set(a)) < 2 or len(set(b)) < 2:
            continue
        r, _ = spearmanr(a, b)
        if r == r:
            vals.append(r)
    if not vals:
        return float('nan'), float('nan')
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def run(method='hybrid', agg='max', compare_legacy=False,
        scores_path=None, tag=None):
    """用 HERB CMap 外部金标准验证一份草药-疾病分数。

    scores_path: 待验证的分数 CSV, 需含 herb_id 列与各疾病列。
                 默认 hdmap_s_scores.csv; 传 gcn_d_scores.csv 即验证 GCN-D。
    tag: 输出文件名后缀, 避免不同方法的验证结果互相覆盖。
    """
    spath = scores_path or SCORES
    scores = pd.read_csv(spath)
    scores['herb_id'] = scores['herb_id'].astype(str)
    print(f'[输入] 分数文件: {spath}')

    def build(fname, matcher, aggregator):
        df = pd.read_excel(os.path.join(RAW, f'{fname}_cmap2_data.xlsx'))
        ing = df[df['Type'] == 'Ingredient'].copy()
        tmp = {}
        n_matched = 0
        for _, r in ing.iterrows():
            hs = matcher(str(r['Drug name']))
            if hs:
                n_matched += 1
            sc = float(r['Summary Score'])
            for h in hs:
                tmp.setdefault(h, []).append(sc)
        rows = [(h, aggregator(sl), len(sl)) for h, sl in tmp.items()]
        ext = pd.DataFrame(rows, columns=['herb_id', 'herb_ext_score', 'n_ing'])
        ext['herb_id'] = ext['herb_id'].astype(str)
        return ext, len(ing), n_matched

    out_rows, per_herb = [], []
    legacy_rows = []

    for fname, dk in DISEASES.items():
        ext, n_ing, n_mat = build(fname, MATCHERS[method], AGGS[agg])
        m = ext.merge(scores[['herb_id', dk]], on='herb_id', how='inner')
        m = m.dropna(subset=[dk, 'herb_ext_score'])
        m['disease'] = fname
        per_herb.append(m)
        if len(m) < 10:
            print(f'{fname}: 重叠草药太少 ({len(m)})，跳过')
            continue

        our = m[dk].values.astype(float)
        exs = m['herb_ext_score'].values.astype(float)
        rho, p = spearmanr(our, exs)
        lo, hi = boot_ci_rho(our, exs)
        thr = np.quantile(exs, 0.70)
        label = (exs >= thr).astype(int)
        auc = auc_roc(our, label)
        top_ours = set(m.sort_values(dk, ascending=False)['herb_id'].head(20))
        top_ext = set(m.sort_values('herb_ext_score',
                                    ascending=False)['herb_id'].head(20))
        jac = len(top_ours & top_ext) / len(top_ours | top_ext) \
            if (top_ours | top_ext) else 0.0
        sig = '显著' if (lo > 0 or hi < 0) else '不显著'

        out_rows.append({
            'disease': fname, 'n_overlap_herbs': len(m),
            'rho': round(rho, 4), 'p_value': f'{p:.3g}',
            'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
            'auroc_top30%': round(auc, 4), 'top20_jaccard': round(jac, 4),
            'n_ing_total': n_ing, 'n_ing_matched': n_mat,
            'match_rate': round(n_mat / n_ing, 3) if n_ing else None,
            'significance': sig,
        })
        print(f'{fname}: 重叠={len(m)} | rho={rho:+.4f} (p={p:.3g}, '
              f'CI[{lo:.3f},{hi:.3f}]) {sig} | AUROC={auc:.4f} | '
              f'Jaccard={jac:.3f} | 成分匹配 {n_mat}/{n_ing}')

        if compare_legacy:
            ext_l, _, n_mat_l = build(fname, match_substring, agg_mean)
            ml = ext_l.merge(scores[['herb_id', dk]], on='herb_id',
                             how='inner').dropna(subset=[dk, 'herb_ext_score'])
            if len(ml) >= 10:
                rl, _ = spearmanr(ml[dk].values.astype(float),
                                  ml['herb_ext_score'].values.astype(float))
            else:
                rl = float('nan')
            legacy_rows.append({'disease': fname, 'rho_legacy': round(rl, 4),
                                'n_ing_matched_legacy': n_mat_l})

    summ = pd.DataFrame(out_rows)
    print('\n' + '=' * 78)
    print(f'外部验证摘要 (匹配={method}, 聚合={agg})')
    print('=' * 78)
    print(summ.to_string(index=False))
    mr = np.mean(np.abs(summ['rho']))
    ma = np.mean(summ['auroc_top30%'])
    print(f'\n|ρ| 均值 = {mr:.4f}    AUROC 均值 = {ma:.4f}')

    if compare_legacy and legacy_rows:
        ld = pd.DataFrame(legacy_rows)
        print('\n--- 原方法对照 (子串匹配 + mean 聚合) ---')
        print(ld.to_string(index=False))
        ml = np.mean(np.abs(ld['rho_legacy']))
        print(f'|ρ| 均值(原方法) = {ml:.4f}  →  改进后 {mr:.4f}  '
              f'(提升 {mr/max(ml, 1e-9):.1f} 倍)')
        summ = summ.merge(ld, on='disease', how='left')

    suffix = f'_{tag}' if tag else ''
    f_sum = os.path.join(OUT_DIR, f'external_validation_summary{suffix}.csv')
    f_herb = os.path.join(OUT_DIR, f'herb_external_per_herb{suffix}.csv')
    summ.to_csv(f_sum, index=False)
    pd.concat(per_herb, ignore_index=True).to_csv(f_herb, index=False)
    print('\n已保存:')
    print(' ', f_sum)
    print(' ', f_herb)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='HERB CMap 外部验证')
    ap.add_argument('--method', choices=list(MATCHERS), default='hybrid')
    ap.add_argument('--agg', choices=list(AGGS), default='max')
    ap.add_argument('--compare-legacy', action='store_true',
                    help='同时跑原方法(子串+mean)作对照')
    ap.add_argument('--scores', default=None,
                    help='待验证分数 CSV (含 herb_id 列与各疾病列)。'
                         '可给文件名(在 paper_deliverables 下查找)或绝对路径; '
                         '缺省为 hdmap_s_scores.csv')
    ap.add_argument('--tag', default=None,
                    help='输出文件名后缀 (如 gcn_d -> '
                         'external_validation_summary_gcn_d.csv), 避免覆盖')
    a = ap.parse_args()

    spath = a.scores
    if spath and not os.path.isabs(spath) and not os.path.exists(spath):
        cand = os.path.join(OUT_DIR, spath)
        if os.path.exists(cand):
            spath = cand
    run(a.method, a.agg, a.compare_legacy, scores_path=spath, tag=a.tag)
