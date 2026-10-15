#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MC2 统计检验: GCN-D vs 每个基线的 DeLong 检验 (2026-09-06)
============================================================
审稿意见 MC2: 主表 6 个方法缺统计显著性检验; 且 306 对共享草药/疾病节点
(非 i.i.d.), 普通 bootstrap 低估方差, 正确做法是 DeLong 法或按疾病分层的
block bootstrap.

本脚本:
  1. 复跑 5 折 CV 拿到 GCN-D 与 MLP 的原始逐对分数 (3000 epoch, 与主表一致);
  2. 在同一套 306 测试对上计算 4 个无监督基线的原始逐对分数 (无需训练);
  3. 按疾病对每方法分数做中心化 (保持病内序, 使跨病池化合法), 跑 DeLong
     比较 GCN-D vs 每个基线 (池化), 给出精确 p 值;
  4. 对 5 个比较做 BH-FDR 多重校正;
  5. 同时给按疾病聚类的 block bootstrap 作为稳健性检查。

使用 CUDA (RTX 4060) 加速训练; 稳定训练器带梯度裁剪 + NaN 重试.
输出: output/results/paper_deliverables/_mc2_delong.csv (仅摘要, 不覆盖任何交付物)
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import torch
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
from run_fair_comparison import (build_graph, build_exclude, GCN,
                                 score_fisher, score_proximity, score_targetcount)
from run_fair_final import score_rwr

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")
BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)
N_FOLDS = 5
FOLD_SEED = 12345


def make_folds(herbs, disease_ids, gold, n_folds=N_FOLDS, seed=FOLD_SEED):
    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}
    pairs = sorted({
        (hidx[str(r.herb_id)], didx[str(r.disease_id)])
        for r in gold[gold["label"] == 1].itertuples()
        if str(r.herb_id) in hidx and str(r.disease_id) in didx})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    folds = [[pairs[i] for i in perm[k::n_folds]] for k in range(n_folds)]
    return pairs, folds


def psi_gt(x):
    return np.where(x > 0, 1.0, np.where(x == 0, 0.5, 0.0))


def delong_auc(pos, neg):
    S_pos = pos[:, None] - neg[None, :]
    V10 = psi_gt(S_pos).mean(axis=1)
    S_neg = pos[None, :] - neg[:, None]
    V01 = psi_gt(S_neg).mean(axis=1)
    return V10.mean(), V10, V01


def delong_test(pos1, neg1, pos2, neg2):
    nx, ny = len(pos1), len(neg1)
    A1, V10_1, V01_1 = delong_auc(pos1, neg1)
    A2, V10_2, V01_2 = delong_auc(pos2, neg2)
    S10 = ((V10_1 - A1) * (V10_2 - A2)).sum() / (nx - 1)
    S01 = ((V01_1 - A1) * (V01_2 - A2)).sum() / (ny - 1)
    S11 = (((V10_1 - A1)[:, None] * (V01_2 - A2)[None, :]).sum()
           + ((V10_2 - A2)[:, None] * (V01_1 - A1)[None, :]).sum()) / (nx * ny)
    var = S10 / nx + S01 / ny - S11
    if var <= 0:
        var = 1e-12
    z = (A1 - A2) / np.sqrt(var)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return A1, A2, z, p


def clustered_bootstrap(per_gcn, per_b, n_boot=2000, seed=11):
    """按疾病聚类 block bootstrap: 检验 H0: AUC(GCN) <= AUC(baseline).
    内部用 roc_auc_score (rank-based, O(n log n)) 计算每次重采样的 AUROC 差,
    避免两两 DeLong 的 O(n_pos*n_neg) 开销."""
    diseases = list(per_gcn.keys())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sel = rng.choice(diseases, size=len(diseases), replace=True)
        rp1 = np.concatenate([per_gcn[d][0] for d in sel])
        rn1 = np.concatenate([per_gcn[d][1] for d in sel])
        rp2 = np.concatenate([per_b[d][0] for d in sel])
        rn2 = np.concatenate([per_b[d][1] for d in sel])
        y1 = np.concatenate([np.ones(len(rp1)), np.zeros(len(rn1))])
        y2 = np.concatenate([np.ones(len(rp2)), np.zeros(len(rn2))])
        a1 = roc_auc_score(y1, np.concatenate([rp1, rn1]))
        a2 = roc_auc_score(y2, np.concatenate([rp2, rn2]))
        diffs.append(a1 - a2)
    diffs = np.array(diffs)
    p_one = (diffs <= 0).mean()
    p_two = 2 * min(p_one, 1 - p_one)
    return float(p_one), float(p_two), float(diffs.mean()), \
           float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def train_stable(A, X, off_h, off_d, nH, nD, train_pairs, n_herb_total,
                 epochs=3000, lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2,
                 seed=42, exclude_pos=None, decoder="distmult", device="cuda"):
    """稳定训练: 梯度裁剪 + NaN 重试 (降 lr). 返回 S(numpy, 已中心化)."""
    for attempt, use_lr in enumerate([lr, lr * 0.5, lr * 0.25]):
        torch.manual_seed(seed + attempt * 1000)
        np.random.seed(seed + attempt * 1000)
        rng = np.random.default_rng(seed + attempt * 1000)
        model = GCN(X.shape[1], hid, out, n_layers=n_layers)
        if decoder == "dot":
            w = torch.ones(out, device=device)
        else:
            w = torch.nn.Parameter(torch.ones(out, device=device))
        opt = torch.optim.Adam(
            list(model.parameters()) + ([w] if isinstance(w, torch.nn.Parameter) else []),
            lr=use_lr)
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
        pos_t = torch.tensor([p[0] for p in train_pairs], device=device)
        pdd_t = torch.tensor([p[1] for p in train_pairs], device=device)
        last_loss = float("nan")
        for ep in range(epochs):
            model.train()
            Z = model(A, X)
            zh = Z[off_h:off_h + nH]
            zd = Z[off_d:off_d + nD]
            pos_s = (zh[pos_t] * zd[pdd_t] * w).sum(1)
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
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) +
                                          ([w] if isinstance(w, torch.nn.Parameter) else []), 2.0)
            opt.step()
            if ep == epochs - 1:
                last_loss = loss.item()
        with torch.no_grad():
            Z = model(A, X).cpu()
            zh = Z[off_h:off_h + nH]
            zd = Z[off_d:off_d + nD]
            S = (zh * w.cpu()) @ zd.T
            S = (S - S.mean()) / (S.std() + 1e-8)
        S = S.numpy()
        if not np.isnan(S).any():
            return S, last_loss, use_lr
        else:
            print(f"    [重试] attempt {attempt} lr={use_lr:.2e} 出现 NaN, 降 lr 重试")
    print(f"    [警告] 训练仍 NaN (lr={use_lr:.2e}), 返回最后结果 (含 NaN)")
    return S, last_loss, use_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    A_cpu, X_cpu, N, off_h, off_d, nH, nD, herbs, disease_ids, _g = build_graph(use_logfc=True)
    print(f"[图] 节点={N} 草药={nH} 疾病={nD}")
    EXCL = build_exclude(gold, herbs, disease_ids)
    all_pairs, folds = make_folds(herbs, disease_ids, gold)
    print(f"[划分] 正例对={len(all_pairs)} 折={N_FOLDS}")

    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}
    pos_by_d = {d: set() for d in range(nD)}
    for r in gold[gold["label"] == 1].itertuples():
        h, d = str(r.herb_id), str(r.disease_id)
        if h in hidx and d in didx:
            pos_by_d[didx[d]].add(hidx[h])

    S_fisher = score_fisher(herbs, disease_ids).values
    S_rwr = score_rwr(A_cpu, N, off_h, nH, off_d, nD)
    S_prox = score_proximity(herbs, disease_ids).values
    S_tc = score_targetcount(herbs, disease_ids).values
    print("[基线] 分数矩阵已计算 (CPU)")

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.from_numpy = lambda a: torch.as_tensor(np.asarray(a), device="cuda")
        A = A_cpu.cuda()
        X = X_cpu.cuda()
        dev = "cuda"
        print("[设备] CUDA 已启用")
    else:
        A, X, dev = A_cpu, X_cpu, "cpu"
        print("[设备] CPU")

    METHODS = ["GCN_DistMult", "MLP_no_graph", "D-TMS_Fisher",
               "RWR_diffusion", "Network_proximity", "Target_count"]
    coll = {dname: {m: {"pos": [], "neg": []} for m in METHODS} for dname in disease_ids}

    for k in range(N_FOLDS):
        test = folds[k]
        train = [p for j, f in enumerate(folds) if j != k for p in f]
        S_gcn, loss_g, lr_g = train_stable(
            A, X, off_h, off_d, nH, nD, train, nH, epochs=args.epochs,
            seed=42, exclude_pos=EXCL, device=dev, **BEST)
        c_mlp = dict(BEST); c_mlp["n_layers"] = 1
        S_mlp, loss_m, lr_m = train_stable(
            A, X, off_h, off_d, nH, nD, train, nH, epochs=args.epochs,
            seed=42, exclude_pos=EXCL, device=dev, **c_mlp)
        Smap = {"GCN_DistMult": S_gcn, "MLP_no_graph": S_mlp,
                "D-TMS_Fisher": S_fisher, "RWR_diffusion": S_rwr,
                "Network_proximity": S_prox, "Target_count": S_tc}
        print(f"  fold {k+1}/{N_FOLDS} GCN loss={loss_g:.4f}(lr={lr_g:.2e}) "
              f"MLP loss={loss_m:.4f} n_test={len(test)} t={time.time()-t0:.0f}s")
        for (h, d) in test:
            negs = np.array([i for i in range(nH) if i not in pos_by_d[d]], dtype=np.int64)
            dname = disease_ids[d]
            for m in METHODS:
                sp = float(Smap[m][h, d])
                sn = Smap[m][negs, d].astype(float)
                coll[dname][m]["pos"].append(sp)
                coll[dname][m]["neg"].append(sn)

    per_m = {m: {} for m in METHODS}
    nan_flag = False
    for dname in coll:
        for m in METHODS:
            pos = np.array(coll[dname][m]["pos"])
            neg = np.concatenate(coll[dname][m]["neg"])
            if np.isnan(pos).any() or np.isnan(neg).any():
                nan_flag = True
            mu = np.concatenate([pos, neg]).mean()
            per_m[m][dname] = (pos - mu, neg - mu)
    if nan_flag:
        print("[警告] 仍存在 NaN 分数!")

    def concat(method):
        pos = np.concatenate([per_m[method][d][0] for d in per_m[method]])
        neg = np.concatenate([per_m[method][d][1] for d in per_m[method]])
        return pos, neg

    gcn_pos, gcn_neg = concat("GCN_DistMult")
    gcn_per = per_m["GCN_DistMult"]
    compare = ["MLP_no_graph", "D-TMS_Fisher", "RWR_diffusion",
               "Network_proximity", "Target_count"]
    rows = []
    for m in compare:
        b_pos, b_neg = concat(m)
        A1, A2, z, p = delong_test(gcn_pos, gcn_neg, b_pos, b_neg)
        p1, p2, dmean, dlo, dhi = clustered_bootstrap(gcn_per, per_m[m])
        rows.append({"GCN-D vs": m, "GCN_AUROC": round(A1, 4),
                     f"{m}_AUROC": round(A2, 4), "DeLong_z": round(z, 3),
                     "DeLong_p": f"{p:.2e}", "block_boot_p2": round(p2, 4),
                     "boot_d_mean": round(dmean, 4),
                     "boot_CI_lo": round(dlo, 4), "boot_CI_hi": round(dhi, 4)})
        print(f"  GCN-D vs {m:<18} AUROC {A1:.4f} vs {A2:.4f} "
              f"DeLong z={z:.2f} p={p:.2e}  block-boot p2={p2:.3f}")

    ps = np.array([float(r["DeLong_p"]) for r in rows])
    order = np.argsort(ps)
    ps_sorted = ps[order]
    n = len(ps_sorted)
    bh = np.minimum(1.0, ps_sorted * n / (np.arange(1, n + 1)))
    for i in range(n - 2, -1, -1):
        bh[i] = min(bh[i], bh[i + 1])
    adj = np.empty(n)
    adj[order] = bh
    for i, r in enumerate(rows):
        r["DeLong_p_BH"] = f"{adj[i]:.2e}"

    out = pd.DataFrame(rows)
    print("\n=== DeLong 检验 (GCN-D vs 各基线, 池化, BH 校正) ===")
    print(out.to_string(index=False))
    op = os.path.join(OUT_DIR, "_mc2_delong.csv")
    out.to_csv(op, index=False, encoding="utf-8-sig")
    print(f"\n[写入] {op}  总耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
