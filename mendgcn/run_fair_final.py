#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公平基线 + 干净留出评估 + 过拟合/结构消融  (2026-09-04, v2)
============================================================

产出的 4 张论文表:
  Table_fair_final.csv     6 方法 × 8 病: FULL / labeled AUROC
  Table_holdout_clean.csv  干净留出对级评估 (61 对训练未见): AUROC / MRR / Hits@10 / Hits@20
  Table_epoch_overfit.csv  epoch 扫描: FULL vs 干净留出 (判断是否记忆/过拟合)
  Table_gnn_ablation2.csv  结构消融: 无KEGG边 / 点积解码 / 无图传播(MLP) / 3层

★ 本脚本修正的关键度量错误 ★
------------------------------------------------------------------
旧代码 (run_gnn_final.metrics / run_gnn_ablation.metrics) 里的"留出泛化"是错的:
它对每个留出对 (h,d) 计算的是 **该病全部已知正例 vs 全库 499 味** 的 AUROC,
然后把 61 个留出对的结果取均值。训练时用过的那 80% 正例**仍然留在评估正例集里**,
所以这个数字并不是"留出泛化", 只是按留出对数量重新加权的每病 FULL AUROC ——
它高并不代表模型没有记忆, 因此旧稿"留出 ≈ FULL 所以排除记忆"的推论不成立。

本脚本改用 **对级干净留出 (pair-level clean holdout)**:
对每个留出正例 (h,d), 把 h 与该病**全部已知正例之外**的所有草药排序比较
(即候选负例 = {全部草药} \\ {该病全部正例}, 训练正例也被排除),
得到每个留出对一个 AUROC, 再对 61 对取均值, 并同时报 MRR / Hits@K。
这样评估正例集里不含任何训练正例, "没有记忆"的推论才站得住。

方法清单 (全部在同一份数据、同一套评估代码下比较):
  无监督: Fisher_enrichment / RWR_diffusion / Network_proximity / Target_count
  有监督: MLP_no_graph (n_layers=1, 只做特征变换不做传播) / GCN_DistMult (本文方法)

用法:
    python run_fair_final.py                    # 全量 (约 90 分钟)
    python run_fair_final.py --stage baselines  # 只跑无监督基线 (快)
    python run_fair_final.py --epochs 3000 --seeds 0 1 2
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
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
from run_fair_comparison import (build_graph, build_exclude, train_gcn,
                                 score_fisher, score_proximity,
                                 score_targetcount)
from run_gnn_final import build_split

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")

BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)


# --------------------------------------------------------------------------
# 度量
# --------------------------------------------------------------------------
def per_disease_metrics(S, herbs, disease_ids, gold):
    """每病 FULL (正例=已知治疗草药, 负例=全库其余) 与 labeled AUROC。"""
    hidx = {h: i for i, h in enumerate(herbs)}
    rows = []
    for di, dis in enumerate(disease_ids):
        sub = gold[gold["disease_id"] == dis]
        pos = {str(r.herb_id) for r in sub[sub["label"] == 1].itertuples()
               if str(r.herb_id) in hidx}
        neg = {str(r.herb_id) for r in sub[sub["label"] == 0].itertuples()
               if str(r.herb_id) in hidx}
        if len(pos) < 3:
            continue
        s = np.nan_to_num(S[:, di].astype(float), nan=0.0)
        y = np.array([1.0 if h in pos else 0.0 for h in herbs])
        a_full = roc_auc_score(y, s) if len(set(y)) == 2 else np.nan
        li = [hidx[h] for h in (pos | neg)]
        yl = np.array([1.0 if herbs[i] in pos else 0.0 for i in li])
        a_lab = roc_auc_score(yl, s[li]) if len(set(yl)) == 2 else np.nan
        rows.append({"disease": dis, "n_pos": len(pos), "n_neg": len(neg),
                     "AUROC_full": a_full, "AUROC_labeled": a_lab})
    return rows


def clean_holdout(S, herbs, disease_ids, gold, test_pairs, topk=(10, 20)):
    """对级干净留出评估 (见文件头说明)。

    对每个留出正例 (h,d): 候选集 = {全部草药} \\ {d 的全部已知正例}
      AUROC_pair = P(score(neg) < score(h)) + 0.5 * P(相等)
      rank       = #{neg : score(neg) > score(h)} + 1
    """
    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}
    pos_by_d = {d: set() for d in range(len(disease_ids))}
    for r in gold[gold["label"] == 1].itertuples():
        h, d = str(r.herb_id), str(r.disease_id)
        if h in hidx and d in didx:
            pos_by_d[didx[d]].add(hidx[h])

    aucs, mrrs = [], []
    hits = {k: [] for k in topk}
    for h, d in test_pairs:
        negs = np.array([i for i in range(len(herbs)) if i not in pos_by_d[d]],
                        dtype=np.int64)
        if len(negs) == 0:
            continue
        s_neg = np.nan_to_num(S[negs, d].astype(float), nan=-1e9)
        s_pos = float(np.nan_to_num(S[h, d].astype(float), nan=-1e9))
        aucs.append(float((s_neg < s_pos).mean() + 0.5 * (s_neg == s_pos).mean()))
        rank = int((s_neg > s_pos).sum()) + 1
        mrrs.append(1.0 / rank)
        for k in topk:
            hits[k].append(1.0 if rank <= k else 0.0)
    out = {"holdout_AUROC": float(np.mean(aucs)) if aucs else float("nan"),
           "MRR": float(np.mean(mrrs)) if mrrs else float("nan"),
           "n_pairs": len(aucs)}
    for k in topk:
        out[f"Hits@{k}"] = float(np.mean(hits[k])) if hits[k] else float("nan")
    return out


# --------------------------------------------------------------------------
# 新增无监督基线: RWR 扩散 (疾病节点作种子, 在异质图上游走)
# --------------------------------------------------------------------------
def score_rwr(A_sym, N, off_h, nH, off_d, nD, c=0.7, n_iter=100):
    """重启随机游走。以疾病节点为种子在整张异质图上游走, 草药节点稳态概率即得分。

    P[u,v] = A[u,v]/deg(u) (行归一); r_{t+1} = (1-c) P^T r_t + c*r0。
    """
    from scipy import sparse as sp
    idx = A_sym.indices().numpy()
    P = sp.csr_matrix((np.ones(idx.shape[1], dtype=np.float64),
                       (idx[0], idx[1])), shape=(N, N))
    rs = np.asarray(P.sum(1)).flatten()
    rs[rs == 0] = 1.0
    P = (sp.diags(1.0 / rs) @ P).T.tocsr()
    S = np.zeros((nH, nD))
    for d in range(nD):
        r = np.zeros(N)
        r[off_d + d] = 1.0
        for _ in range(n_iter):
            r = (1.0 - c) * (P @ r)
            r[off_d + d] += c
        S[:, d] = r[off_h:off_h + nH]
    return S


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def save(df, name, header=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    if header:
        with open(out, "a", encoding="utf-8-sig") as f:
            for line in header:
                f.write("\n# " + line)
    print(f"[写入] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "baselines", "gcn", "overfit", "ablation"])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--overfit-epochs", type=int, default=6000)
    ap.add_argument("--abl-epochs", type=int, default=1200)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)

    print("=" * 78)
    print("公平基线 + 干净留出评估  (v2)")
    print("=" * 78)
    print(f"金标准: {os.path.basename(GOLD_PATH)} ({len(gold)} 行, "
          f"正例 {(gold['label'] == 1).sum()})")

    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, _g2i = build_graph(use_logfc=True)
    print(f"[图] 节点={N}  草药={nH}  疾病={nD}  特征维={X.shape[1]}")
    EXCL = build_exclude(gold, herbs, disease_ids)

    fair_rows, hold_rows = [], []

    # 全部正例对 (草药下标, 疾病下标) —— 无监督方法的"干净对级"评估用
    _hidx = {h: i for i, h in enumerate(herbs)}
    _didx = {d: i for i, d in enumerate(disease_ids)}
    ALL_PAIRS = sorted({
        (_hidx[str(r.herb_id)], _didx[str(r.disease_id)])
        for r in gold[gold["label"] == 1].itertuples()
        if str(r.herb_id) in _hidx and str(r.disease_id) in _didx})

    # ---------------------------------------------------------------- 基线
    if args.stage in ("all", "baselines"):
        print("\n" + "=" * 78)
        print("[1/5] 无监督基线")
        print("=" * 78)
        print("  Fisher 富集 (D-TMS) ...")
        S_fisher = score_fisher(herbs, disease_ids).values
        print("  RWR 扩散 (疾病节点种子, c=0.7) ...")
        S_rwr = score_rwr(A, N, off_h, nH, off_d, nD)
        print("  网络邻近度 (degree-matched z-score) ...")
        S_prox = score_proximity(herbs, disease_ids).values
        print("  靶点数量 (平凡基线) ...")
        S_tc = score_targetcount(herbs, disease_ids).values

        for name, S in [("Fisher_enrichment", S_fisher), ("RWR_diffusion", S_rwr),
                        ("Network_proximity", S_prox), ("Target_count", S_tc)]:
            for r in per_disease_metrics(S, herbs, disease_ids, gold):
                fair_rows.append({"method": name, **r})
            # 无监督方法的"干净对级"评估: 在全部 306 个正例对上算 (它们没见过任何训练数据)
            hm = clean_holdout(S, herbs, disease_ids, gold, ALL_PAIRS)
            hm.update({"method": name, "note": "unsupervised: 全部正例对, 无训练"})
            hold_rows.append(hm)
            sub = [r["AUROC_full"] for r in per_disease_metrics(S, herbs, disease_ids, gold)]
            print(f"    {name:<20} FULL={np.mean(sub):.4f}  干净对级AUROC={hm['holdout_AUROC']:.4f}")

        save(pd.DataFrame(fair_rows), "Table_fair_final.csv")
        save(pd.DataFrame(hold_rows), "Table_holdout_clean.csv")

    # ------------------------------------------------- MLP 无图传播 (对照)
    if args.stage in ("all", "baselines"):
        print("\n[2/5] MLP 无图传播对照 (n_layers=1, 相同监督/相同负采样) ...")
        tr, te = build_split(herbs, disease_ids, gold, seed=42)
        S_mlp, loss_mlp = train_gcn(A, X, off_h, off_d, nH, nD, tr, nH,
                                    epochs=args.abl_epochs, seed=42,
                                    exclude_pos=EXCL, n_layers=1, **{
                                        k: v for k, v in BEST.items() if k != "n_layers"})
        mrows = per_disease_metrics(S_mlp, herbs, disease_ids, gold)
        for r in mrows:
            fair_rows.append({"method": "MLP_no_graph", **r})
        hm = clean_holdout(S_mlp, herbs, disease_ids, gold, te)
        hm.update({"method": "MLP_no_graph",
                   "note": f"supervised, {len(te)} held-out pairs, {args.abl_epochs}ep"})
        hold_rows.append(hm)
        print(f"    MLP_no_graph        FULL={np.mean([r['AUROC_full'] for r in mrows]):.4f}  "
              f"干净留出={hm['holdout_AUROC']:.4f}  loss={loss_mlp:.4f}")

    # ------------------------------------------------------------ GCN-D 主结果
    if args.stage in ("all", "gcn"):
        print("\n" + "=" * 78)
        print(f"[3/5] GCN-D 主结果 (epochs={args.epochs}, seeds={args.seeds})")
        print("=" * 78)
        per_seed, gcn_hold, gcn_full = {}, [], []
        for sd in args.seeds:
            tr, te = build_split(herbs, disease_ids, gold, seed=sd)
            S, loss = train_gcn(A, X, off_h, off_d, nH, nD, tr, nH,
                                epochs=args.epochs, seed=sd,
                                exclude_pos=EXCL, **BEST)
            mrows = per_disease_metrics(S, herbs, disease_ids, gold)
            hm = clean_holdout(S, herbs, disease_ids, gold, te)
            f = np.nanmean([r["AUROC_full"] for r in mrows])
            l = np.nanmean([r["AUROC_labeled"] for r in mrows])
            per_seed[sd] = {r["disease"]: r["AUROC_full"] for r in mrows}
            gcn_full.append(f)
            gcn_hold.append(hm["holdout_AUROC"])
            print(f"  seed={sd}  FULL={f:.4f}  labeled={l:.4f}  "
                  f"干净留出={hm['holdout_AUROC']:.4f}  "
                  f"MRR={hm['MRR']:.4f}  Hits@10={hm['Hits@10']:.3f}  loss={loss:.4f}")
            if sd == args.seeds[0]:
                for r in mrows:
                    fair_rows.append({"method": "GCN_DistMult", **r})
                hm2 = dict(hm)
                hm2.update({"method": "GCN_DistMult",
                            "note": f"supervised, {len(te)} held-out pairs, "
                                    f"{args.epochs}ep, mean over seeds"})
                hold_rows.append(hm2)

        # 多种子汇总 (均值写回 CSV, 供论文引用)
        all_d = sorted(set().union(*[set(p.keys()) for p in per_seed.values()]))
        mean_row = []
        for d in all_d:
            vals = [per_seed[sd].get(d, np.nan) for sd in args.seeds]
            vals = [v for v in vals if not np.isnan(v)]
            mean_row.append(np.mean(vals))
        print("-" * 78)
        print(f"  FULL(3seed)     = {np.mean(gcn_full):.4f} ± {np.std(gcn_full):.4f}")
        print(f"  干净留出(3seed) = {np.mean(gcn_hold):.4f} ± {np.std(gcn_hold):.4f}")
        print(f"  每病均值         = {np.mean(mean_row):.4f}")
        print("=" * 78)
        with open(os.path.join(OUT_DIR, "Table_gnn_final_clean.txt"), "w",
                  encoding="utf-8") as f:
            f.write(f"# GCN-D epochs={args.epochs} seeds={args.seeds}\n")
            f.write(f"FULL_mean,{np.mean(gcn_full):.4f}\n")
            f.write(f"FULL_std,{np.std(gcn_full):.4f}\n")
            f.write(f"CLEAN_HOLDOUT_mean,{np.mean(gcn_hold):.4f}\n")
            f.write(f"CLEAN_HOLDOUT_std,{np.std(gcn_hold):.4f}\n")
            for d in all_d:
                vals = [per_seed[sd].get(d, np.nan) for sd in args.seeds]
                vals = [v for v in vals if not np.isnan(v)]
                f.write(f"per_disease,{d},{np.mean(vals):.4f},{np.std(vals):.4f}\n")

        save(pd.DataFrame(fair_rows), "Table_fair_final.csv")
        save(pd.DataFrame(hold_rows), "Table_holdout_clean.csv")

    # ------------------------------------------------------------ 过拟合检查
    if args.stage in ("all", "overfit"):
        print("\n" + "=" * 78)
        print("[4/5] 过拟合/记忆检查: FULL vs 干净留出 随 epoch 变化")
        print("=" * 78)
        rows = []
        tr, te = build_split(herbs, disease_ids, gold, seed=42)
        for ep in sorted({3000, args.overfit_epochs}):
            S, loss = train_gcn(A, X, off_h, off_d, nH, nD, tr, nH,
                                epochs=ep, seed=42, exclude_pos=EXCL, **BEST)
            mrows = per_disease_metrics(S, herbs, disease_ids, gold)
            f = np.nanmean([r["AUROC_full"] for r in mrows])
            l = np.nanmean([r["AUROC_labeled"] for r in mrows])
            hm = clean_holdout(S, herbs, disease_ids, gold, te)
            rows.append({"epochs": ep, "AUROC_full": round(f, 4),
                         "AUROC_labeled": round(l, 4),
                         "clean_holdout": round(hm["holdout_AUROC"], 4),
                         "MRR": round(hm["MRR"], 4),
                         "Hits@10": round(hm["Hits@10"], 4),
                         "loss": round(loss, 4)})
            print(f"  epochs={ep:<6} FULL={f:.4f}  labeled={l:.4f}  "
                  f"干净留出={hm['holdout_AUROC']:.4f}  loss={loss:.4f}")
        save(pd.DataFrame(rows), "Table_epoch_overfit.csv",
             header=["判读: 若 FULL 继续涨而干净留出持平/下降 => 超出该 epoch 后是记忆而非泛化"])

    # ------------------------------------------------------------ 结构消融
    if args.stage in ("all", "ablation"):
        print("\n" + "=" * 78)
        print(f"[5/5] 结构消融 (epochs={args.abl_epochs}, seed=42)")
        print("=" * 78)
        tr, te = build_split(herbs, disease_ids, gold, seed=42)
        rows = []

        def run(tag, note, A_, X_, **kw):
            c = dict(BEST)
            c.update(kw)
            S, loss = train_gcn(A_, X_, off_h, off_d, nH, nD, tr, nH,
                                epochs=args.abl_epochs, seed=42,
                                exclude_pos=EXCL, **c)
            mrows = per_disease_metrics(S, herbs, disease_ids, gold)
            hm = clean_holdout(S, herbs, disease_ids, gold, te)
            f = np.nanmean([r["AUROC_full"] for r in mrows])
            l = np.nanmean([r["AUROC_labeled"] for r in mrows])
            rows.append({"variant": tag, "note": note,
                         "AUROC_full": round(f, 4), "AUROC_labeled": round(l, 4),
                         "clean_holdout": round(hm["holdout_AUROC"], 4),
                         "MRR": round(hm["MRR"], 4), "loss": round(loss, 4)
                         if loss == loss else np.nan})
            print(f"  {tag:<26} FULL={f:.4f}  labeled={l:.4f}  "
                  f"干净留出={hm['holdout_AUROC']:.4f}  loss={loss:.4f}")

        run("默认 (2层/DistMult/KEGG/logFC)", "full model", A, X)
        # 无 KEGG 边
        A_nok, X_nok, *_rest = build_graph(use_logfc=True, use_kegg=False)
        run("去掉 KEGG 基因-基因边", "ablate KEGG edges", A_nok, X_nok)
        # 点积解码
        run("点积解码 (无DistMult权重)", "ablate decoder", A, X, decoder="dot")
        # 无图传播
        run("无图传播 (MLP, 1层)", "ablate propagation", A, X, n_layers=1)
        # 无 logFC 特征
        _, X_nolog, *_r2 = build_graph(use_logfc=False)
        run("无 |logFC| 基因特征", "ablate expression feature", A, X_nolog)
        # 3 层
        run("3 层 GCN", "deeper", A, X, n_layers=3)

        save(pd.DataFrame(rows), "Table_gnn_ablation2.csv")

    print(f"\n[完成] 总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
