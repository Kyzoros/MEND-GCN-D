#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MC6 Part B: 消融变体 5 种子误差棒 (2026-09-06)
================================================
审稿意见 MC6 / minor m2: 表 3 消融各变体为单点估计, 无误差棒, 与正文
"±0.03 属噪声" 的说法自相矛盾。此处对 6 个消融变体各跑 5 个随机种子
(改变训练与留出划分), 报告 clean-holdout AUROC 的 mean ± std。

6 变体 (与 build_manuscript 表 3 / Table_gnn_ablation2 一致):
  默认(2层/DistMult/KEGG/logFC) | 去 KEGG 边 | 点积解码 | 无图传播(MLP) |
  无 logFC 特征 | 3 层 GCN
输出: output/results/paper_deliverables/_mc6_ablation_seeds.csv
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)
import config as cfg
from run_fair_comparison import build_graph, build_exclude, GCN
from run_fair_final import clean_holdout
from run_gnn_ablation import build_split

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")
SEEDS = [42, 7, 123, 2024, 999]
EPOCHS = 1200


def train_variant(graph, X, off_h, off_d, nH, nD, train_pairs, n_layers,
                  decoder, exclude_pos, seed, tag=""):
    for attempt, use_lr in enumerate([1e-2, 5e-3, 2.5e-3]):
        torch.manual_seed(seed + attempt * 1000)
        np.random.seed(seed + attempt * 1000)
        rng = np.random.default_rng(seed + attempt * 1000)
        model = GCN(X.shape[1], 64, 32, n_layers=n_layers)
        if decoder == "dot":
            w = torch.ones(32, device="cuda")          # 固定点积 (不可学, 消融)
            opt = torch.optim.Adam(list(model.parameters()), lr=use_lr)
        else:
            w = nn.Parameter(torch.ones(32, device="cuda"))  # DistMult 可学
            opt = torch.optim.Adam(list(model.parameters()) + [w], lr=use_lr)
        plist = list(model.parameters()) + ([w] if decoder != "dot" else [])
        cand_per_d, n_pos_per_d = {}, {}
        for d in range(nD):
            cand_per_d[d] = np.array([hh for hh in range(nH)
                                      if hh not in exclude_pos.get(d, set())],
                                     dtype=np.int64)
            n_pos_per_d[d] = 0
        for _h, d in train_pairs:
            n_pos_per_d[d] += 1
        ph = torch.tensor([p[0] for p in train_pairs], device="cuda")
        pdd = torch.tensor([p[1] for p in train_pairs], device="cuda")
        last = float("nan")
        for ep in range(EPOCHS):
            model.train()
            Z = model(graph, X)
            zh = Z[off_h:off_h + nH]
            zd = Z[off_d:off_d + nD]
            pos_s = (zh[ph] * zd[pdd] * w).sum(1)
            nh_list, nd_list = [], []
            for d in range(nD):
                m = n_pos_per_d[d] * 5
                if m == 0 or len(cand_per_d[d]) == 0:
                    continue
                picks = cand_per_d[d][rng.integers(0, len(cand_per_d[d]), size=m)]
                nh_list.append(torch.as_tensor(picks, device="cuda"))
                nd_list.append(torch.full((m,), d, dtype=torch.int64, device="cuda"))
            nh = torch.cat(nh_list)
            nd = torch.cat(nd_list)
            neg_s = (zh[nh] * zd[nd] * w).sum(1)
            loss = -torch.log(torch.sigmoid(pos_s)).mean() \
                   - torch.log(1 - torch.sigmoid(neg_s)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(plist, 2.0)
            opt.step()
            last = loss.item()
        model.eval()
        with torch.no_grad():
            Z = model(graph, X).cpu()
            zh = Z[off_h:off_h + nH]
            zd = Z[off_d:off_d + nD]
            S = (zh * w.cpu()) @ zd.T
            S = (S - S.mean()) / (S.std() + 1e-8)
        S = S.numpy()
        if not np.isnan(S).any():
            return S, last
    return S, last


def main():
    t0 = time.time()
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.from_numpy = lambda a: torch.as_tensor(np.asarray(a), device="cuda")
        dev = "cuda"
    else:
        dev = "cpu"
    print(f"[设备] {dev}")

    variants = [
        ("默认 (2层/DistMult/KEGG/logFC)", dict(use_kegg=True, use_logfc=True, n_layers=2, decoder="distmult")),
        ("去掉 KEGG 基因-基因边",            dict(use_kegg=False, use_logfc=True, n_layers=2, decoder="distmult")),
        ("点积解码 (无DistMult权重)",         dict(use_kegg=True, use_logfc=True, n_layers=2, decoder="dot")),
        ("无图传播 (MLP, 1层)",               dict(use_kegg=True, use_logfc=True, n_layers=1, decoder="distmult")),
        ("无 |logFC| 基因特征",               dict(use_kegg=True, use_logfc=False, n_layers=2, decoder="distmult")),
        ("3 层 GCN",                         dict(use_kegg=True, use_logfc=True, n_layers=3, decoder="distmult")),
    ]
    cache = {}
    rows = []
    for vname, v in variants:
        vals = []
        for seed in SEEDS:
            key = (v["use_kegg"], v["use_logfc"])
            if key not in cache:
                A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, _ = \
                    build_graph(use_kegg=v["use_kegg"], use_logfc=v["use_logfc"])
                cache[key] = (A.cuda(), X.cuda(), off_h, off_d, nH, nD, herbs, disease_ids)
            A, X, off_h, off_d, nH, nD, herbs, disease_ids = cache[key]
            EXCL = build_exclude(gold, herbs, disease_ids)
            pos_pairs, train_pairs, test_pairs = build_split(herbs, disease_ids, gold, seed=seed)
            S, loss = train_variant(A, X, off_h, off_d, nH, nD, train_pairs,
                                    n_layers=v["n_layers"], decoder=v["decoder"],
                                    exclude_pos=EXCL, seed=seed, tag=vname)
            hm = clean_holdout(S, herbs, disease_ids, gold, test_pairs)
            vals.append(hm["holdout_AUROC"])
            print(f"  {vname:<30} seed={seed:<5} clean_holdout={hm['holdout_AUROC']:.4f} "
                  f"loss={loss:.4f} t={time.time()-t0:.0f}s", flush=True)
        rows.append({"variant": vname,
                     "mean": round(float(np.mean(vals)), 4),
                     "std": round(float(np.std(vals)), 4),
                     "min": round(float(np.min(vals)), 4),
                     "max": round(float(np.max(vals)), 4),
                     "seeds": ",".join(map(str, SEEDS)),
                     "per_seed": [round(x, 4) for x in vals]})
    out = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(f"{'variant':<30}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    for _, r in out.iterrows():
        print(f"{r['variant']:<30}{r['mean']:>8.4f}{r['std']:>8.4f}{r['min']:>8.4f}{r['max']:>8.4f}")
    op = os.path.join(OUT_DIR, "_mc6_ablation_seeds.csv")
    out.to_csv(op, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {op}  总耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
