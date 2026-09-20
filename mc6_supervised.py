#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MC6 Part A: 有监督 GNN 基线 (GAT / GraphSAGE / R-GCN) 2026-09-06
=================================================================
审稿意见 MC6: 基线全是无监督 + 1 个 MLP 消融, 是"稻草人"; 至少补 GAT、
GraphSAGE、R-GCN 等有监督 GNN 基线, 在**完全相同的**图、特征、负采样、
折划分与训练轮数下, 仅替换编码器。

本脚本在纯 PyTorch 中手写三种编码器 (不依赖 torch_geometric):
  - GCN-D (本文, 对照): GCN(nn.Module) 即 run_fair_comparison.GCN
  - GraphSAGE: 均值聚合, 邻接 = 行归一化 (含自环) 的原始图
  - R-GCN:    按边类型 (self/H-I/I-G/G-D/G-G) 分解, 每关系独立权重
  - GAT:      2 层稀疏图注意力 (首层 8 头, 拼接; 次层 1 头)
复用 run_cv_final 的 5 折划分、pair_metrics/summarise, 指标口径与表 2 完全一致。

用法:
  python mc6_supervised.py --epochs 3000 --encoders sage rgcn gat   # 正式
  python mc6_supervised.py --epochs 15 --encoders sage rgcn gat gcn --fold 0  # 冒烟
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg
from run_fair_comparison import build_graph, build_exclude, GCN
from run_cv_final import make_folds, pair_metrics, summarise

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")
BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)
N_FOLDS = 5
FOLD_SEED = 12345


# ---------------------------------------------------------------------------
# 邻接预处理
# ---------------------------------------------------------------------------
def prep_adjacencies(A_sym, X, nH, off_d, N, device="cuda"):
    """从 build_graph 的对称归一化 A_sym 反解原始图, 派生各编码器所需邻接.

    节点布局: [herbs | ingredients | genes | diseases], 类型可经 X one-hot 识别.
    返回 dict:
      A_sym : GCN 用 (保持原样)
      A_row : GraphSAGE 均值聚合用 (行归一化, 含自环)
      rels  : list of (name, A_r_row)  R-GCN 用 (每关系行归一化, 不含自环)
      I     : 恒等 (R-GCN self 关系)
      edge_index / edge_attr: GAT 用
    """
    A_sym = A_sym.coalesce()
    idx = A_sym.indices()                     # (2, E)
    vals = A_sym.values().double()            # (E,)
    # --- 自环值 = 1/d_i  -> 反解原始度 d ---
    self_mask = idx[0] == idx[1]
    diag_nodes = idx[0][self_mask]
    diag_vals = vals[self_mask]
    d = torch.ones(N, dtype=torch.double, device=A_sym.device)
    d[diag_nodes] = 1.0 / diag_vals.clamp(min=1e-9)
    # --- 原始二值邻接 (含自环, 对称) ---
    raw_vals = (vals * torch.sqrt(d[idx[0]] * d[idx[1]])).round().clamp(0, 1)
    raw_idx = idx
    # 行归一化 (GraphSAGE mean): 1/d[row]
    sage_vals = (1.0 / d[idx[0]]).float()
    A_row = torch.sparse_coo_tensor(raw_idx, sage_vals, (N, N)).coalesce()
    # 类型块
    type_of = X.float().argmax(dim=1)         # 0=herb 1=ing 2=gene 3=disease
    off_i = int((type_of == 1).nonzero()[0].min())
    off_g = int((type_of == 2).nonzero()[0].min())
    # 关系拆分
    u, v = raw_idx[0], raw_idx[1]
    rel_of = torch.full((u.shape[0],), -1, dtype=torch.long, device=A_sym.device)
    tu, tv = type_of[u], type_of[v]
    hi = ((tu == 0) & (tv == 1)) | ((tu == 1) & (tv == 0))
    ig = ((tu == 1) & (tv == 2)) | ((tu == 2) & (tv == 1))
    gd = ((tu == 2) & (tv == 3)) | ((tu == 3) & (tv == 2))
    gg = (tu == 2) & (tv == 2) & (u != v)
    rel_of[hi] = 1
    rel_of[ig] = 2
    rel_of[gd] = 3
    rel_of[gg] = 4
    names = {1: "hi", 2: "ig", 3: "gd", 4: "gg"}
    rels = []
    for r in [1, 2, 3, 4]:
        m = rel_of == r
        ru, rv, rval = u[m], v[m], torch.ones(m.sum(), dtype=torch.double, device=A_sym.device)
        rdeg = torch.zeros(N, dtype=torch.double, device=A_sym.device)
        rdeg.index_add_(0, ru, rval)
        rdeg = rdeg.clamp(min=1.0)
        rrow = torch.sparse_coo_tensor(
            torch.stack([ru, rv]), (1.0 / rdeg[ru]).float(), (N, N)).coalesce()
        rels.append((names[r], rrow))
    I = torch.eye(N, dtype=torch.float32, device=A_sym.device) if device == "cpu" \
        else torch.eye(N, dtype=torch.float32, device=A_sym.device)
    # GAT edge index (src->dst, 含自环), attn over in-edges of dst
    edge_index = raw_idx
    return {"A_sym": A_sym.float(), "A_row": A_row, "rels": rels, "I": I,
            "edge_index": edge_index}


# ---------------------------------------------------------------------------
# 编码器
# ---------------------------------------------------------------------------
class SAGE(nn.Module):
    """GraphSAGE-mean, 2 层 (均值聚合, 含自环)."""
    def __init__(self, fin, fhid, fout, n_layers=2):
        super().__init__()
        self.w1 = nn.Linear(fin, fhid, bias=False)
        self.w2 = nn.Linear(fhid, fhid, bias=False) if n_layers >= 3 else None
        self.wout = nn.Linear(fhid if self.w2 is not None else fhid, fout, bias=False)

    def forward(self, A, X):
        if self.w2 is None:
            h = F.relu(self.w1(A @ X))
            return self.wout(A @ h)
        h = F.relu(self.w1(A @ X))
        h = F.relu(self.w2(A @ h))
        return self.wout(A @ h)


class RGCN(nn.Module):
    """R-GCN 风格: 关系分解消息传递 (self/H-I/I-G/G-D/G-G 五关系).
    含 bias + ELU, 避免无 bias 时全负预激活导致死 ReLU / 折叠到常数输出."""
    def __init__(self, fin, fhid, fout, R=5):
        super().__init__()
        self.act = F.elu
        self.l1 = nn.ModuleList([nn.Linear(fin, fhid, bias=True) for _ in range(R)])
        self.l2 = nn.ModuleList([nn.Linear(fhid, fhid, bias=True) for _ in range(R)])
        self.wout = nn.Linear(fhid, fout, bias=True)

    def forward(self, A_tuple, X):
        rels, I = A_tuple
        h = sum(A @ self.l1[r](X) for r, (_, A) in enumerate(rels))
        h = h + self.l1[-1](X)               # self 关系 = I @ (X W_self)
        h = self.act(h)
        h2 = sum(A @ self.l2[r](h) for r, (_, A) in enumerate(rels))
        h2 = h2 + self.l2[-1](h)
        return self.wout(self.act(h2))


class GATLayer(nn.Module):
    def __init__(self, fin, fout, heads=1, concat=True, negative_slope=0.2):
        super().__init__()
        self.heads, self.fout, self.concat = heads, fout, concat
        self.W = nn.Linear(fin, heads * fout, bias=False)
        self.a = nn.Parameter(torch.Tensor(1, heads, 2 * fout))
        nn.init.xavier_uniform_(self.a)
        self.leaky = nn.LeakyReLU(negative_slope)

    def forward(self, x, edge_index):
        N = x.shape[0]
        h = self.W(x)                          # (N, heads*fout)
        h = h.view(N, self.heads, self.fout)   # (N,H,F)
        src, dst = edge_index[0], edge_index[1]
        hs, hd = h[src], h[dst]                # (E,H,F)
        scores = (hs * self.a[0, :, :self.fout]).sum(-1) \
               + (hd * self.a[0, :, self.fout:]).sum(-1)   # (E,H)
        scores = self.leaky(scores)
        e = (scores - scores.max(dim=0, keepdim=True).values).exp()  # (E,H)
        denom = torch.zeros(N, self.heads, device=x.device)
        denom.index_add_(0, dst, e)
        alpha = e / denom[dst].clamp(min=1e-9)  # (E,H)
        msg = alpha.unsqueeze(-1) * hs           # (E,H,F)
        out = torch.zeros(N, self.heads, self.fout, device=x.device)
        out.index_add_(0, dst, msg)              # 沿 dim0 按 dst 累加 (E,)
        if self.concat:
            return out.view(N, self.heads * self.fout)
        return out.mean(dim=1)


class GAT(nn.Module):
    """2 层稀疏 GAT: L1 8 头拼接, L2 单头."""
    def __init__(self, fin, fhid=64, fout=32, heads=8, edge_index=None):
        super().__init__()
        self.heads = heads
        head_dim = fhid // heads
        self.l1 = GATLayer(fin, head_dim, heads=heads, concat=True)
        self.l2 = GATLayer(fhid, fout, heads=1, concat=False)

    def forward(self, A_tuple, X):
        edge_index = A_tuple[2]
        h = F.elu(self.l1(X, edge_index))
        return self.l2(h, edge_index)


# ---------------------------------------------------------------------------
# 通用训练器 (梯度裁剪 + NaN 重试), 与 train_gcn_model 训练循环/负采样一致
# ---------------------------------------------------------------------------
def train_enc(encoder, A_tuple, X, off_h, off_d, nH, nD, train_pairs,
              epochs=3000, lr=1e-2, hid=64, out=32, k_neg=5, seed=42,
              exclude_pos=None, device="cuda", tag=""):
    A_sym, A_row, rels, I, edge_index = A_tuple
    fin = X.shape[1]
    for attempt, use_lr in enumerate([lr, lr * 0.5, lr * 0.25]):
        torch.manual_seed(seed + attempt * 1000)
        np.random.seed(seed + attempt * 1000)
        rng = np.random.default_rng(seed + attempt * 1000)
        if encoder == "gcn":
            model = GCN(fin, hid, out, n_layers=2)
            mfwd = lambda: model(A_sym, X)
        elif encoder == "mlp":
            model = GCN(fin, hid, out, n_layers=1)
            mfwd = lambda: model(A_sym, X)
        elif encoder == "sage":
            model = SAGE(fin, hid, out)
            mfwd = lambda: model(A_row, X)
        elif encoder == "rgcn":
            model = RGCN(fin, hid, out, R=5)
            mfwd = lambda: model((rels, I), X)
        elif encoder == "gat":
            model = GAT(fin, hid, out, heads=8)
            mfwd = lambda: model((rels, I, edge_index), X)
        else:
            raise ValueError(encoder)
        w = nn.Parameter(torch.ones(out, device=device))
        opt = torch.optim.Adam(list(model.parameters()) + [w], lr=use_lr)
        pos_per_d = {d: set() for d in range(nD)}
        for h, d in train_pairs:
            pos_per_d[d].add(h)
        excl_per_d = exclude_pos if exclude_pos is not None else pos_per_d
        cand_per_d, n_pos_per_d = {}, {}
        for d in range(nD):
            cand_per_d[d] = np.array([hh for hh in range(nH)
                                      if hh not in excl_per_d.get(d, set())], dtype=np.int64)
            n_pos_per_d[d] = 0
        for _h, d in train_pairs:
            n_pos_per_d[d] += 1
        ph = torch.tensor([p[0] for p in train_pairs], device=device)
        pdd = torch.tensor([p[1] for p in train_pairs], device=device)
        last_loss = float("nan")
        for ep in range(epochs):
            model.train()
            Z = mfwd()
            zh = Z[off_h:off_h + nH]
            zd = Z[off_d:off_d + nD]
            pos_s = (zh[ph] * zd[pdd] * w).sum(1)
            nh_list, nd_list = [], []
            for d in range(nD):
                m = n_pos_per_d[d] * k_neg
                if m == 0 or len(cand_per_d[d]) == 0:
                    continue
                picks = cand_per_d[d][rng.integers(0, len(cand_per_d[d]), size=m)]
                nh_list.append(torch.as_tensor(picks, device=device))
                nd_list.append(torch.full((m,), d, dtype=torch.int64, device=device))
            nh = torch.cat(nh_list)
            nd = torch.cat(nd_list)
            neg_s = (zh[nh] * zd[nd] * w).sum(1)
            loss = -torch.log(torch.sigmoid(pos_s)).mean() \
                   - torch.log(1 - torch.sigmoid(neg_s)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + [w], 2.0)
            opt.step()
            last_loss = loss.item()
        model.eval()
        with torch.no_grad():
            Z = mfwd().cpu()
            zh = Z[off_h:off_h + nH]
            zd = Z[off_d:off_d + nD]
            S = (zh * w.cpu()) @ zd.T
            S = (S - S.mean()) / (S.std() + 1e-8)
        S = S.numpy()
        if not np.isnan(S).any():
            return S, last_loss
        print(f"    [{tag}] attempt {attempt}: NaN, retry lr={use_lr:.1e}")
    return S, last_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--encoders", nargs="+", default=["sage", "rgcn", "gat"])
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    A_cpu, X_cpu, N, off_h, off_d, nH, nD, herbs, disease_ids, _g = build_graph(use_logfc=True)
    print(f"[图] 节点={N} 草药={nH} 疾病={nD}", flush=True)
    EXCL = build_exclude(gold, herbs, disease_ids)
    all_pairs, folds = make_folds(herbs, disease_ids, gold)
    print(f"[划分] 正例对={len(all_pairs)} 折={N_FOLDS}", flush=True)

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.from_numpy = lambda a: torch.as_tensor(np.asarray(a), device="cuda")
        A_sym = A_cpu.cuda()
        X = X_cpu.cuda()
        dev = "cuda"
        print("[设备] CUDA", flush=True)
    else:
        A_sym, X, dev = A_cpu, X_cpu, "cpu"
    prep = prep_adjacencies(A_sym, X, nH, off_d, N)
    A_tuple = (prep["A_sym"], prep["A_row"], prep["rels"], prep["I"], prep["edge_index"])
    print("[邻接] 派生完成 (sage/rgcn/gat 用)", flush=True)

    METHODS = {
        "sage": "GraphSAGE_mean", "rgcn": "R-GCN",
        "gat": "GAT_8head", "gcn": "GCN_DistMult", "mlp": "MLP_no_graph",
    }
    rows, per_pair = [], []

    for enc in args.encoders:
        mname = METHODS[enc]
        dfs = []
        for k in range(N_FOLDS):
            test = folds[k]
            train = [p for j, f in enumerate(folds) if j != k for p in f]
            S, loss = train_enc(enc, A_tuple, X, off_h, off_d, nH, nD, train,
                                epochs=args.epochs, seed=42, exclude_pos=EXCL,
                                device=dev, tag=f"{mname}")
            dfk = pair_metrics(S, herbs, disease_ids, gold, test)
            dfk["fold"] = k
            dfs.append(dfk)
            print(f"  {mname:<20} fold {k+1} n_test={len(dfk)} "
                  f"fold_AUROC={dfk['auc'].mean():.4f} loss={loss:.4f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
        df = pd.concat(dfs, ignore_index=True)
        r = summarise(df, mname)
        r["n_folds"] = N_FOLDS
        rows.append(r)
        df["method"] = mname
        per_pair.append(df)
        print(f"  --> {mname}: pair AUROC={r['pair_AUROC']:.4f} "
              f"[{r['ci95_lo']:.4f},{r['ci95_hi']:.4f}] MRR={r['MRR']:.4f} "
              f"Hits@10={r['Hits@10']:.3f} Hits@20={r['Hits@20']:.3f}\n", flush=True)

    out = os.path.join(OUT_DIR, "_mc6_supervised_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    p = os.path.join(OUT_DIR, "_mc6_supervised_per_pair.csv")
    pd.concat(per_pair, ignore_index=True).to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {out}\n[写入] {p}\n总耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
