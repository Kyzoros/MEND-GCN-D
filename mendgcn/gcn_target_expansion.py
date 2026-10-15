#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCN 靶标扩展 (对比实验)
========================
目的:
  对比"可学习的图卷积"与"无参数的相似度传播"在靶标扩展任务上的效果,
  回答: HDCTI 模块该维持相似度传播, 还是真的加上图卷积?

设计:
  - 图: 草药-靶点二部图 (节点 = 草药 + 靶标; 边 = 已知互作, 来自
        ingredient_targets_enhanced 经草药-成分聚合)
  - 模型: 2 层 GCN, 链路预测 (BCE loss), 负采样 1:1
  - 输出: 每个草药的"扩展靶标" (按预测概率取 top-K)
  - 评估: 把扩展靶标喂给 enrichment 打分, 比较下游金标准 AUROC

关键: 这是**对比实验**, 不是替换主方法。
  无论 GCN 更好还是更差, 结果都有价值:
    - GCN 更好 -> 可升级主方法
    - GCN 更差 -> "小样本中药数据下简单图传播优于深度学习"本身是可发表发现

用法:
    python gcn_target_expansion.py                 # 训练 + 扩展 + 评估
    python gcn_target_expansion.py --epochs 200 --dim 64 --top-k 300
    python gcn_target_expansion.py --compare-only  # 只画对比, 不训练
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("[错误] 需要 PyTorch。请安装: pip install torch")
    sys.exit(1)


# ------------------------- 数据 -------------------------
def load_herb_target_map():
    """herb_id -> set(target_gene)  (经草药-成分聚合)"""
    proc = cfg.PROCESSED_DIR
    it = pd.read_csv(os.path.join(proc, "ingredient_targets_enhanced.csv"))
    hi = pd.read_csv(os.path.join(proc, "herb_ingredients_processed.csv"))
    ing2tgt = it.groupby("ingredient_id")["target_gene"].apply(set).to_dict()

    out = {}
    for hid, grp in hi.groupby("herb_id"):
        s = set()
        for ing in grp["ingredient_id"]:
            s |= ing2tgt.get(ing, set())
        out[str(hid)] = s
    return out


# ------------------------- 模型 -------------------------
class GCN(nn.Module):
    """2 层 GCN, 用于二部图链路预测"""

    def __init__(self, n_nodes, dim, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(n_nodes, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, adj_norm):
        x = self.emb.weight
        x = torch.sparse.mm(adj_norm, self.w1(x))
        x = torch.relu(x)
        x = self.dropout(x)
        x = torch.sparse.mm(adj_norm, self.w2(x))
        return x                       # (n_nodes, dim)


def build_adj(herb_ids, targets, herb_tgt, self_loop=True):
    """构建归一化邻接 (含自环) 的稀疏 tensor"""
    tlist = sorted(targets)
    t2i = {t: i for i, t in enumerate(tlist)}
    h2i = {h: i + len(tlist) for i, h in enumerate(herb_ids)}
    n = len(tlist) + len(herb_ids)

    rows, cols = [], []
    for h, ts in herb_tgt.items():
        if h not in h2i:
            continue
        hi_ = h2i[h]
        for t in ts:
            if t in t2i:
                ti = t2i[t]
                rows += [hi_, ti]
                cols += [ti, hi_]
    if self_loop:
        rows += list(range(n))
        cols += list(range(n))

    v = np.ones(len(rows), dtype=np.float32)
    idx = torch.LongTensor([rows, cols])
    adj = torch.sparse_coo_tensor(idx, torch.FloatTensor(v), (n, n))

    # 对称归一化 D^{-1/2} A D^{-1/2}
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    vals = adj._values()
    r, c = adj._indices()
    newv = vals * deg_inv_sqrt[r] * deg_inv_sqrt[c]
    adj_norm = torch.sparse_coo_tensor(adj._indices(), newv, (n, n)).coalesce()
    return adj_norm, h2i, t2i, n


def train_model(adj_norm, n, edges, epochs, dim, lr=0.01, seed=42):
    """链路预测训练。edges: List[(node_i, node_j, label)]"""
    torch.manual_seed(seed)
    model = GCN(n, dim, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    ei = torch.LongTensor([e[0] for e in edges])
    ej = torch.LongTensor([e[1] for e in edges])
    lab = torch.FloatTensor([e[2] for e in edges])

    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        z = model(adj_norm)
        logits = (z[ei] * z[ej]).sum(dim=1)
        loss = lossf(logits, lab)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            with torch.no_grad():
                p = torch.sigmoid(logits)
                acc = ((p > 0.5).float() == lab).float().mean().item()
            print(f"    epoch {ep+1:>4}/{epochs}  loss={loss.item():.4f}  "
                  f"acc={acc:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser(description="GCN 靶标扩展对比实验")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=300,
                    help="每草药保留的扩展靶标数")
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        cfg.RESULTS_DIR, "weighted_kegg", "gcn_expanded_targets.csv")

    print("[1/5] 加载草药-靶点 ...")
    herb_tgt = load_herb_target_map()
    herb_ids = sorted(herb_tgt.keys())
    all_t = set()
    for s in herb_tgt.values():
        all_t |= s
    print(f"  草药 {len(herb_ids)}  靶标 {len(all_t)}  "
          f"已知边 {sum(len(s) for s in herb_tgt.values())}")

    print("[2/5] 构图 ...")
    adj_norm, h2i, t2i, n = build_adj(herb_ids, all_t, herb_tgt)
    print(f"  节点数 {n}")

    print("[3/5] 构造训练样本 (正负 1:{args.neg_ratio}) ...")
    rng = np.random.default_rng(42)
    pos_edges, neg_edges = [], []
    t_idx = np.array(list(t2i.values()))
    for h, ts in herb_tgt.items():
        if h not in h2i:
            continue
        hi_ = h2i[h]
        for t in ts:
            if t in t2i:
                pos_edges.append((hi_, t2i[t], 1.0))
        n_neg = int(len(ts) * args.neg_ratio)
        known = {t2i[t] for t in ts if t in t2i}
        cand = t_idx[~np.isin(t_idx, list(known))] if len(known) else t_idx
        if len(cand) == 0:
            continue
        pick = rng.choice(cand, size=min(n_neg, len(cand)), replace=False)
        for ti in pick:
            neg_edges.append((hi_, int(ti), 0.0))
    edges = pos_edges + neg_edges
    rng.shuffle(edges)
    print(f"  正样本 {len(pos_edges)}  负样本 {len(neg_edges)}")

    print(f"[4/5] 训练 GCN ({args.epochs} epochs, dim={args.dim}) ...")
    model = train_model(adj_norm, n, edges, args.epochs, args.dim,
                        lr=args.lr)

    print("[5/5] 预测扩展靶标 ...")
    model.eval()
    with torch.no_grad():
        z = model(adj_norm)
    zh = z[[h2i[h] for h in herb_ids]]                 # (n_herb, dim)
    zt = z[[t2i[t] for t in sorted(all_t)]]            # (n_target, dim)
    scores = zh @ zt.T                                  # (n_herb, n_target)
    score_df = pd.DataFrame(scores.numpy(),
                            index=herb_ids,
                            columns=sorted(all_t))

    # 每草药取 top-K (排除已知靶标 -> 这才是"扩展")
    expanded = {}
    for h in herb_ids:
        known = herb_tgt[h]
        s = score_df.loc[h]
        s = s.drop(index=[t for t in known if t in s.index], errors="ignore")
        top = s.sort_values(ascending=False).head(args.top_k)
        expanded[h] = set(known) | set(top.index)
    print(f"  扩展后每草药靶标数(中位数): "
          f"{np.median([len(v) for v in expanded.values()]):.0f}")

    # 保存: herb_id, target_gene (长表)
    rows = [(h, t, 1.0) for h, ts in expanded.items() for t in ts]
    df = pd.DataFrame(rows, columns=["herb_id", "target_gene", "confidence"])
    df.to_csv(out_path, index=False)
    print(f"\n[写入] {out_path}   行数={len(df)}")
    print("\n提示: 用扩展后的靶标重跑 enrichment 打分即可对比下游 AUROC。")


if __name__ == "__main__":
    main()
