#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公平对比脚本 (2026-09-04)
==========================
在**同一份数据、同一套评估代码**下对比 4 类方法, 产出论文主表。

  1. Fisher 富集 (D-TMS)      —— 无监督, 单跳集合重叠
  2. 网络邻近度 (Barabasi)    —— 无监督, KEGG 网络拓扑 z-score
  3. 靶点数量 (平凡基线)      —— 无监督, 仅 count(herb targets), 无疾病特异性
  4. GCN + DistMult (本文方法) —— 有监督, 多跳传播

协议 (两个都报):
  - FULL    : 正例 = 已知治疗草药; 负例 = 全部其余草药 (严格全库, 不挑负例)
  - labeled : 仅在金标准中有显式 label∈{0,1} 的草药上计算

金标准: data/processed/gold_standard_final_8d_supplemented.csv (8 病, 595 行)

输出: output/results/paper_deliverables/Table_fair_comparison.csv
用法:
    python run_fair_comparison.py
    python run_fair_comparison.py --epochs 300 --seed 42
"""
import argparse
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg
from kegg_network import build_kegg_adjacency
from hdmap_s import build_herb_target_sets, pair_single_herb_all_diseases_enrichment

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
# 靶点文件可用环境变量 SM_TGT_FILE 覆盖, 便于做"清洗前/清洗后"对照实验,
# 而不必改动主流程默认数据。
TGT_FILE = os.environ.get("SM_TGT_FILE") or os.path.join(
    PROC, "ingredient_targets_enhanced_recovered.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")


# --------------------------------------------------------------------------
# 图构建 (与 run_gnn_pilot_v3.py 完全一致)
# --------------------------------------------------------------------------
def build_graph(use_logfc=True, use_kegg=True, excl_diseases=None, excl_herbs=None):
    """构建异质图。

    use_kegg: 是否加入 KEGG 基因-基因边 (消融用; 关掉后图只剩 H-I-G-D 骨架)。
    返回 10 元组, 末位 g2i 为 基因符号 -> 节点下标 映射 (外部基线/诊断用)。
    """
    hi = pd.read_csv(os.path.join(PROC, "herb_ingredients_processed.csv"))
    it = pd.read_csv(TGT_FILE)
    dp = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))

    ing_with_target = set(it["ingredient_id"].astype(str))
    hi = hi[hi["ingredient_id"].astype(str).isin(ing_with_target)]
    excl_diseases = set(excl_diseases or [])
    excl_herbs = set(excl_herbs or [])
    # 健壮构造: dropna + 显式 str/strip + 排除空串与 'nan' 字面值,
    # 杜绝混合类型列(float 混入 object)时 sorted 报 "'<' not supported between float and str"。
    def _strs(col):
        s = col.dropna().astype(object).map(lambda x: str(x).strip())
        return {v for v in s.tolist() if v and v.lower() != "nan"}
    herbs    = sorted(_strs(hi["herb_id"])        - excl_herbs)
    ings     = sorted(_strs(hi["ingredient_id"]))
    genes    = sorted(_strs(pd.concat([it["target_gene"], dp["gene_symbol"]])))
    diseases = sorted(_strs(dp["disease_id"])     - excl_diseases)

    h2i = {n: i for i, n in enumerate(herbs)}
    i2i = {n: i for i, n in enumerate(ings)}
    g2i = {n: i for i, n in enumerate(genes)}
    d2i = {n: i for i, n in enumerate(diseases)}
    off_h, off_i, off_g, off_d = (0, len(herbs), len(herbs) + len(ings),
                                  len(herbs) + len(ings) + len(genes))
    N = off_d + len(diseases)

    def node(t, k):
        if t == "h":
            return off_h + h2i[k]
        if t == "i":
            return off_i + i2i[k]
        if t == "g":
            return off_g + g2i[k]
        return off_d + d2i[k]

    edges = []
    for _, r in hi.iterrows():
        if str(r["herb_id"]) in h2i:
            edges.append((node("h", str(r["herb_id"])), node("i", str(r["ingredient_id"]))))
    for _, r in it.iterrows():
        g = str(r["target_gene"])
        if g in g2i:
            edges.append((node("i", str(r["ingredient_id"])), node("g", g)))
    for _, r in dp.iterrows():
        if str(r["disease_id"]) in d2i:
            g = str(r["gene_symbol"])
            if g in g2i:
                edges.append((node("g", g), node("d", str(r["disease_id"]))))

    if use_kegg:
        adj_norm, gene2idx = build_kegg_adjacency(cache_dir=PROC, min_shared=cfg.KEGG_MIN_SHARED)
        gi = {g: gene2idx[g] for g in genes if g in gene2idx}
        inv = {v: s for s, v in gi.items()}
        sp = adj_norm.tocsr()
        for g in gi:
            for j in sp[gi[g]].indices:
                if j in inv:
                    edges.append((node("g", g), node("g", inv[j])))
    else:
        print("  [图] use_kegg=False —— 不加入 KEGG 基因-基因边 (消融设置)")

    eset = set()
    for u, v in edges:
        eset.add((u, v))
        eset.add((v, u))
    for n in range(N):
        eset.add((n, n))
    idx = torch.tensor(list(eset), dtype=torch.long).T
    A = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1]), (N, N)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
    d_inv_sqrt = torch.pow(deg, -0.5)
    row, col = A.indices()
    norm = d_inv_sqrt[row] * d_inv_sqrt[col]
    A_sym = torch.sparse_coo_tensor(A.indices(), norm, (N, N)).coalesce()

    X = torch.zeros(N, 6)
    for n in range(off_h, off_i):
        X[n, 0] = 1
    for n in range(off_i, off_g):
        X[n, 1] = 1
    for n in range(off_g, off_d):
        X[n, 2] = 1
    for n in range(off_d, N):
        X[n, 3] = 1
    X[:, 4] = (deg / deg.max()).sqrt()
    if use_logfc:
        expr = dp.groupby("gene_symbol")["logFC"].apply(lambda s: s.abs().mean())
        for g in genes:
            if g in g2i and g in expr.index:
                X[off_g + g2i[g], 5] = float(expr[g])
    X[:, 5] = (X[:, 5] - X[:, 5].mean()) / (X[:, 5].std() + 1e-8)
    return (A_sym, X, N, off_h, off_d, len(herbs), len(diseases), herbs,
            list(d2i.keys()), g2i)


def build_graph_exclude(use_logfc=True, use_kegg=True, excl_diseases=None,
                         excl_herbs=None):
    """build_graph 的排除版 (供 inductive 留一评估剔除被测节点)。"""
    return build_graph(use_logfc=use_logfc, use_kegg=use_kegg,
                       excl_diseases=excl_diseases, excl_herbs=excl_herbs)


# --------------------------------------------------------------------------
# 基线 1: Fisher 富集 (D-TMS)
# --------------------------------------------------------------------------
def score_fisher(herbs, disease_ids):
    dp = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    disease_genes = {}
    for did in disease_ids:
        sub = dp[dp["disease_id"] == did]
        if cfg.DISEASE_LOGFC_THRESHOLD > 0:
            sub = sub[sub["logFC"].abs() > cfg.DISEASE_LOGFC_THRESHOLD]
        if cfg.DISEASE_DEG_TOP_K and cfg.DISEASE_DEG_TOP_K > 0:
            sub = sub.sort_values("P_value", ascending=True).head(cfg.DISEASE_DEG_TOP_K)
        elif cfg.DISEASE_PVALUE_THRESHOLD > 0:
            sub = sub[sub["P_value"] < cfg.DISEASE_PVALUE_THRESHOLD]
        disease_genes[did] = sub.sort_values("logFC", ascending=False)["gene_symbol"].tolist()
    tms = pair_single_herb_all_diseases_enrichment(disease_genes, herbs)
    tms.index = tms.index.astype(str)
    return tms.reindex(index=herbs, columns=disease_ids)


# --------------------------------------------------------------------------
# 基线 2: 网络邻近度 (Barabasi)
# --------------------------------------------------------------------------
def score_proximity(herbs, disease_ids, min_targets=5):
    adj_norm, gene2idx = build_kegg_adjacency(cache_dir=PROC, min_shared=cfg.KEGG_MIN_SHARED)
    n = adj_norm.shape[0]
    adj = adj_norm.copy()
    adj.data = np.ones_like(adj.data)
    adj = ((adj + adj.T) > 0).astype(np.int8).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    deg = np.asarray(adj.sum(axis=1)).flatten().astype(np.int64)
    indptr, indices = adj.indptr, adj.indices

    dp = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    disease_genes = {}
    for did in disease_ids:
        sub = dp[dp["disease_id"] == did]
        if cfg.DISEASE_LOGFC_THRESHOLD > 0:
            sub = sub[sub["logFC"].abs() > cfg.DISEASE_LOGFC_THRESHOLD]
        if cfg.DISEASE_DEG_TOP_K and cfg.DISEASE_DEG_TOP_K > 0:
            sub = sub.sort_values("P_value", ascending=True).head(cfg.DISEASE_DEG_TOP_K)
        disease_genes[did] = sub["gene_symbol"].tolist()

    herb_targets = build_herb_target_sets()

    out = {}
    for dis in disease_ids:
        src = [gene2idx[g] for g in disease_genes[dis] if g in gene2idx]
        if not src:
            out[dis] = pd.Series(0.0, index=herbs)
            continue
        dist = np.full(n, -1, dtype=np.int32)
        q = deque()
        for s in src:
            if 0 <= s < n and dist[s] < 0:
                dist[s] = 0
                q.append(s)
        while q:
            u = q.popleft()
            du = dist[u] + 1
            for v in indices[indptr[u]:indptr[u + 1]]:
                if dist[v] < 0:
                    dist[v] = du
                    q.append(v)
        reach = dist >= 0
        if reach.sum() == 0:
            out[dis] = pd.Series(0.0, index=herbs)
            continue
        g_mu = float(dist[reach].mean())
        g_var = float(dist[reach].astype(float).var())
        mu_by_deg, var_by_deg = {}, {}
        reach_deg = deg[reach]
        reach_dist = dist[reach].astype(float)
        for dv in np.unique(reach_deg):
            m = reach_deg == dv
            if m.sum() < 5:
                continue
            mu_by_deg[int(dv)] = float(reach_dist[m].mean())
            var_by_deg[int(dv)] = float(reach_dist[m].var())

        scores = {}
        for hid in herbs:
            tgts = herb_targets.get(hid, set())
            tidx = [gene2idx[t] for t in tgts if t in gene2idx]
            tidx = [i for i in tidx if dist[i] >= 0]
            if len(tidx) < min_targets:
                scores[hid] = 0.0
                continue
            d_obs = float(dist[tidx].mean())
            dgs = [int(deg[i]) for i in tidx]
            mus = np.array([mu_by_deg.get(d, g_mu) for d in dgs])
            vs = np.array([var_by_deg.get(d, g_var) for d in dgs])
            mu_rand = float(mus.mean())
            var_rand = float(vs.sum()) / (len(tidx) ** 2)
            scores[hid] = 0.0 if var_rand <= 1e-12 else -float(
                (d_obs - mu_rand) / np.sqrt(var_rand))
        out[dis] = pd.Series(scores)
    return pd.DataFrame(out).reindex(index=herbs, columns=disease_ids)


# --------------------------------------------------------------------------
# 基线 3: 靶点数量 (平凡基线, 无疾病特异性)
# --------------------------------------------------------------------------
def score_targetcount(herbs, disease_ids):
    herb_targets = build_herb_target_sets()
    cnt = pd.Series({h: float(len(herb_targets.get(h, set()))) for h in herbs})
    return pd.DataFrame({d: cnt for d in disease_ids}).reindex(index=herbs, columns=disease_ids)


# --------------------------------------------------------------------------
# 本文方法: GCN + DistMult
# --------------------------------------------------------------------------
class GCN(nn.Module):
    def __init__(self, fin, fhid, fout, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.w1 = nn.Linear(fin, fhid, bias=False)
        self.w2 = nn.Linear(fhid, fhid, bias=False) if n_layers >= 3 else None
        self.wout = nn.Linear(fhid if n_layers >= 3 else fhid, fout, bias=False)
        if n_layers < 3:
            self.wout = nn.Linear(fhid, fout, bias=False)

    def forward(self, A, X):
        if self.n_layers <= 1:          # 对照: 纯 MLP, 不做任何图传播
            return self.wout(F.relu(self.w1(X)))
        h = F.relu(self.w1(A @ X))
        if self.w2 is not None:
            h = F.relu(self.w2(A @ h))
        return self.wout(A @ h)


def train_gcn_model(A, X, off_h, off_d, nH, nD, train_pairs, n_herb_total,
                    epochs=300, lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2,
                    seed=42, exclude_pos=None, decoder="distmult"):
    """训练 GCN+DistMult, 返回 (model, w, last_loss)。

    与 train_gcn 的区别: 只返回模型与可学 DistMult 权重 w, 不在训练图上算 S。
    需要做 inductive 推理 (在含未见节点的图上打分) 时, 用返回的 model 在推理图上
    再跑一次前向即可 (见 run_gnn_lodo_coldstart.infer_score)。

    exclude_pos: {disease_idx: set(herb_idx)} —— 负采样时的排除集。
      None         -> 只排除训练正例 (旧行为, 有泄漏: 留出正例会被当负例抽到)
      传入全集      -> 排除该病**全部**已知正例 (正确做法, filtered setting)
    decoder: "distmult" (可学权重 w) | "dot" (纯点积, 消融用)
    n_layers: 1 = 纯 MLP 无图传播 (消融对照); 2 = 默认; >=3 = 更深
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = GCN(X.shape[1], hid, out, n_layers=n_layers)
    if decoder == "dot":
        w = torch.ones(out)                       # 不可学, 纯点积解码 (消融)
    else:
        w = nn.Parameter(torch.ones(out))         # DistMult 可学权重
    opt = torch.optim.Adam(
        list(model.parameters()) + ([w] if isinstance(w, nn.Parameter) else []), lr=lr)

    pos_per_d = {d: set() for d in range(nD)}
    for h, d in train_pairs:
        pos_per_d[d].add(h)
    # 负采样排除集: 默认(旧)只排除训练正例; 给了 exclude_pos 就排除全部已知正例
    excl_per_d = exclude_pos if exclude_pos is not None else pos_per_d

    # 预计算每病负采样候选集 (训练期间不变)。
    # 【2026-09-04 修复】原实现有严重 bug:
    #     nh.append(int(rng.integers(0, len(cand))))     # 取的是 cand 的"下标"
    #     neg_s = zh[torch.tensor(nh)]                   # 却直接当"草药编号"用
    # 后果: 负例永远只从编号 0..len(cand)-1 的前段草药里抽, 末尾约 50 味药从未当过
    #       负例; 且 excl 过滤器仅通过缩小 len(cand) 间接起作用, 基本失效。
    # 现改为 cand[pick] 正确索引, 并预计算候选 (顺带把每 epoch O(nH×|pairs|) 降到 O(nH))。
    cand_per_d, n_pos_per_d = {}, {}
    for d in range(nD):
        cand_per_d[d] = np.array([hh for hh in range(nH)
                                  if hh not in excl_per_d.get(d, set())], dtype=np.int64)
        n_pos_per_d[d] = 0
    for _h, d in train_pairs:
        n_pos_per_d[d] += 1

    for ep in range(epochs):
        model.train()
        Z = model(A, X)
        zh = Z[off_h:off_h + nH]
        zd = Z[off_d:off_d + nD]
        ph = torch.tensor([p[0] for p in train_pairs])
        pdd = torch.tensor([p[1] for p in train_pairs])
        pos_s = (zh[ph] * zd[pdd] * w).sum(1)

        nh_list, nd_list = [], []
        for d in range(nD):
            m = n_pos_per_d[d] * k_neg
            if m == 0 or len(cand_per_d[d]) == 0:
                continue
            picks = cand_per_d[d][rng.integers(0, len(cand_per_d[d]), size=m)]
            nh_list.append(picks)
            nd_list.append(np.full(m, d, dtype=np.int64))
        nh = np.concatenate(nh_list)
        nd = np.concatenate(nd_list)

        neg_s = (zh[torch.from_numpy(nh)] * zd[torch.from_numpy(nd)] * w).sum(1)
        loss = -torch.log(torch.sigmoid(pos_s)).mean() - torch.log(1 - torch.sigmoid(neg_s)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep == epochs - 1:
            last_loss = loss.item()

    model.eval()
    with torch.no_grad():
        Z = model(A, X)
        zh = Z[off_h:off_h + nH]
        zd = Z[off_d:off_d + nD]
        S = (zh * w) @ zd.T
        S = (S - S.mean()) / (S.std() + 1e-8)
    return model, w, last_loss


def train_gcn(A, X, off_h, off_d, nH, nD, train_pairs, n_herb_total,
              epochs=300, lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2, seed=42,
              exclude_pos=None, decoder="distmult"):
    """训练 + 在训练图上推理, 返回 (S, last_loss), 与原 train_gcn 行为完全一致。

    若需 inductive 推理 (在含未见节点的图上打分), 请直接调用 train_gcn_model 取
    (model, w), 再用 model(推理图) 自行计算 S。"""
    model, w, last_loss = train_gcn_model(
        A, X, off_h, off_d, nH, nD, train_pairs, n_herb_total,
        epochs=epochs, lr=lr, hid=hid, out=out, k_neg=k_neg, n_layers=n_layers,
        seed=seed, exclude_pos=exclude_pos, decoder=decoder)
    with torch.no_grad():
        Z = model(A, X)
        zh = Z[off_h:off_h + nH]
        zd = Z[off_d:off_d + nD]
        S = (zh * w) @ zd.T
        S = (S - S.mean()) / (S.std() + 1e-8)
    return S.numpy(), last_loss


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
def evaluate(S, herbs, disease_ids, gold):
    """返回 {method: {disease: {'full':..., 'labeled':...}}}"""
    hidx = {h: i for i, h in enumerate(herbs)}
    res = {}
    for di, dis in enumerate(disease_ids):
        sub = gold[gold["disease_id"] == dis]
        pos = {str(r.herb_id) for r in sub[sub["label"] == 1].itertuples() if str(r.herb_id) in hidx}
        neg = {str(r.herb_id) for r in sub[sub["label"] == 0].itertuples() if str(r.herb_id) in hidx}
        if len(pos) < 3:
            continue
        # FULL
        y_full = np.array([1 if h in pos else 0 for h in herbs], dtype=float)
        s = S[:, di].astype(float)
        if np.isnan(s).any():
            s = np.nan_to_num(s, nan=0.0)
        a_full = roc_auc_score(y_full, s) if len(set(y_full)) == 2 else float("nan")
        # labeled
        li = [hidx[h] for h in (pos | neg)]
        y_lab = np.array([1 if herbs[i] in pos else 0 for i in li], dtype=float)
        a_lab = roc_auc_score(y_lab, s[li]) if len(set(y_lab)) == 2 else float("nan")
        res[dis] = {"n_pos": len(pos), "n_neg": len(neg),
                    "full": float(a_full), "labeled": float(a_lab)}
    return res


def build_exclude(gold, herbs, disease_ids):
    """全部已知正例按病归集 -> 负采样排除集 (filtered setting)。

    不用它的话, 留出的正例对会在训练时被当负例抽到, 模型被训练着把它们排低,
    协议是错的 (虽然效果是让留出数偏悲观而非虚高)。
    """
    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}
    excl = {d: set() for d in range(len(disease_ids))}
    for r in gold[gold["label"] == 1].itertuples():
        h, d = str(r.herb_id), str(r.disease_id)
        if h in hidx and d in didx:
            excl[didx[d]].add(hidx[h])
    return excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--out-dim", type=int, default=32)
    ap.add_argument("--k-neg", type=int, default=5)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    print("=" * 78)
    print("公平对比: Fisher富集 / 网络邻近 / 靶点数量 / GCN+DistMult")
    print(f"金标准: {os.path.basename(GOLD_PATH)}  ({len(gold)} 行)")
    print("=" * 78)

    # --- 图 ---
    # build_graph 返回 10 个值 (末位 g2i = 基因符号->节点下标 映射)
    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, g2i = build_graph()
    print(f"[图] 节点={N}  草药={nH}  疾病={nD}  特征维={X.shape[1]}")
    print(f"[病] {disease_ids}\n")

    # --- 基线 ---
    print("[1/4] Fisher 富集 ...")
    Sf = score_fisher(herbs, disease_ids).values
    print("[2/4] 网络邻近度 ...")
    Sp = score_proximity(herbs, disease_ids).values
    print("[3/4] 靶点数量 (平凡基线) ...")
    Sc = score_targetcount(herbs, disease_ids).values

    # --- GCN (留出 1/5) ---
    print("[4/4] GCN + DistMult ...")
    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}
    pos_pairs = [(hidx[str(r.herb_id)], didx[str(r.disease_id)])
                 for r in gold[gold["label"] == 1].itertuples()
                 if str(r.herb_id) in hidx and str(r.disease_id) in didx]
    pos_pairs = sorted(set(pos_pairs))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pos_pairs))
    n_hold = len(pos_pairs) // 5
    hold = set(perm[:n_hold].tolist())
    train_pairs = [p for i, p in enumerate(pos_pairs) if i not in hold]
    test_pairs = [p for i, p in enumerate(pos_pairs) if i in hold]
    print(f"  正例对={len(pos_pairs)}  训练={len(train_pairs)}  留出={len(test_pairs)}")

    EXCL = build_exclude(gold, herbs, disease_ids)
    Sg, last_loss = train_gcn(A, X, off_h, off_d, nH, nD, train_pairs, nH,
                              epochs=args.epochs, lr=args.lr, hid=args.hid,
                              out=args.out_dim, k_neg=args.k_neg,
                              n_layers=args.layers, seed=args.seed,
                              exclude_pos=EXCL)
    print(f"  末轮 loss={last_loss:.4f}")

    # 留出泛化: 被留出的正例对, 在 FULL 协议下的 AUROC
    hold_aucs = []
    for h, d in test_pairs:
        sub = gold[gold["disease_id"] == disease_ids[d]]
        pos = {str(r.herb_id) for r in sub[sub["label"] == 1].itertuples()
               if str(r.herb_id) in hidx}
        y = np.array([1 if hh in pos else 0 for hh in herbs], dtype=float)
        if len(set(y)) == 2:
            hold_aucs.append(roc_auc_score(y, Sg[:, d]))
    hold_auc = float(np.mean(hold_aucs)) if hold_aucs else float("nan")

    # --- 评估 ---
    methods = {
        "Fisher_enrichment": Sf,
        "Network_proximity": Sp,
        "Target_count": Sc,
        "GCN_DistMult": Sg,
    }
    rows = []
    for mname, S in methods.items():
        r = evaluate(S, herbs, disease_ids, gold)
        for dis, v in r.items():
            rows.append({"method": mname, "disease": dis, "n_pos": v["n_pos"],
                         "n_neg": v["n_neg"], "AUROC_full": v["full"],
                         "AUROC_labeled": v["labeled"]})
    df = pd.DataFrame(rows)

    # 打印
    print("\n" + "=" * 78)
    print("每病 AUROC")
    print("=" * 78)
    for mname in methods:
        sub = df[df["method"] == mname]
        print(f"\n--- {mname} ---")
        print(f"{'disease':<9}{'n_pos':>6}{'n_neg':>6}{'FULL':>10}{'labeled':>10}")
        for _, r in sub.iterrows():
            print(f"{r['disease']:<9}{int(r['n_pos']):>6}{int(r['n_neg']):>6}"
                  f"{r['AUROC_full']:>10.4f}{r['AUROC_labeled']:>10.4f}")
        print(f"{'平均':<9}{'':>6}{'':>6}"
              f"{sub['AUROC_full'].mean():>10.4f}{sub['AUROC_labeled'].mean():>10.4f}")

    print("\n" + "=" * 78)
    print("汇总 (均值)")
    print("=" * 78)
    print(f"{'method':<20}{'FULL':>10}{'labeled':>10}")
    print("-" * 78)
    for mname in methods:
        sub = df[df["method"] == mname]
        print(f"{mname:<20}{sub['AUROC_full'].mean():>10.4f}{sub['AUROC_labeled'].mean():>10.4f}")
    print("-" * 78)
    print(f"{'GCN 留出泛化 FULL':<20}{hold_auc:>10.4f}   (n_test={len(test_pairs)})")
    print("=" * 78)

    out = os.path.join(OUT_DIR, "Table_fair_comparison.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    # 留出泛化单独存
    with open(out, "a", encoding="utf-8-sig") as f:
        f.write(f"\n# GCN holdout_generalization_FULL,{hold_auc:.4f},n_test,{len(test_pairs)}\n")
        f.write(f"# GCN final_loss,{last_loss:.4f}\n")
        f.write(f"# config,epochs,{args.epochs},hid,{args.hid},out,{args.out_dim},"
                f"k_neg,{args.k_neg},layers,{args.layers},lr,{args.lr},seed,{args.seed}\n")
    print(f"\n[写入] {out}")


if __name__ == "__main__":
    main()
