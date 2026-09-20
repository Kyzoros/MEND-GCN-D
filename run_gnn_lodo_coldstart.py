#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCN-D 的 inductive 泛化评估 (2026-09-04, v5 补充)
====================================================
主表 (run_cv_final) 只测了"未见对" (unseen pair), 仍属 transductive:
训练图里 8 个病、全部草药节点都在。审稿人必问的硬伤是:
  "模型能否泛化到**训练时从未见过的治疗方法信号**的病 / 药?"
本脚本补这块, 做两种留一 (leave-X-out) 协议, 全部对级干净评估
(候选负例 = {全部草药} \\ {该病全部已知正例}), 与无监督基线口径一致:

(A) 留一病 (Leave-One-Disease-Out, LODO):
    依次留出 1 个病, 训练图**彻底剔除该病的节点及其全部基因连边**,
    模型从未见过这个病。推理时把冻结权重的模型跑在含该病的全图上,
    靠其基因邻居 (GCN 传播) 算出该病的嵌入, 再对其正例对打分。
    —— 测: 给定一病的分子谱, 能否从没见过它任何中药治疗的前提下排好它的药?

(B) 冷启动草药 (Cold-start herb):
    5 折: 每折留出 1/5 草药, 训练图剔除这些草药节点及其成分连边,
    模型从未见过这些药。推理时冻结权重跑在全图上, 靠其成分->基因->病
    传播算出这些药的嵌入。
    —— 测: 一味全新草药 (无训练信号) 能否被正确排到它该治的病里?

关键实现 (inductive 不是作弊):
    训练用 build_graph_exclude 把被测节点从图里删掉; 推理用 infer_score
    把冻结 model 跑在含被测节点的全图上。model 是标准 nn.Module, 前向只做
    A@X, 不硬编码节点下标, 故可在任意同构图上复用。被测节点在训练时完全
    不可见, 其嵌入纯由冻结权重的图传播得出 => 真 inductive。

产出: output/results/paper_deliverables/Table_lodo.csv (+_per_disease)
      output/results/paper_deliverables/Table_coldstart.csv (+_per_fold)
用法:
    python run_gnn_lodo_coldstart.py --stage lodo --epochs 300     # 冒烟
    python run_gnn_lodo_coldstart.py --stage all  --epochs 3000    # 全量
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg
from run_fair_comparison import (build_graph, build_graph_exclude, build_exclude,
                                 train_gcn_model, score_fisher, score_proximity,
                                 score_targetcount)
from run_fair_final import score_rwr
from run_cv_final import pair_metrics, summarise

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROC = cfg.PROCESSED_DIR
GOLD_PATH = os.path.join(PROC, "gold_standard_final_8d_supplemented.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")
BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)


def infer_score(model, w, A, X, off_h, off_d, nH, nD):
    """用冻结的 model 在推理图上算 herb×disease 分数 (inductive)。"""
    with torch.no_grad():
        Z = model(A, X)
        zh = Z[off_h:off_h + nH]
        zd = Z[off_d:off_d + nD]
        S = (zh * w) @ zd.T
        S = (S - S.mean()) / (S.std() + 1e-8)
    return S.numpy()


def build_train_pairs(gold, hidx_train, didx_train, skip_herbs=set(),
                      skip_diseases=set()):
    pairs = []
    for r in gold[gold["label"] == 1].itertuples():
        h, d = str(r.herb_id), str(r.disease_id)
        if h in skip_herbs or d in skip_diseases:
            continue
        if h in hidx_train and d in didx_train:
            pairs.append((hidx_train[h], didx_train[d]))
    return pairs


def build_test_pairs(gold, hidx_full, didx_full, sel_herbs=set(),
                     sel_diseases=set()):
    pairs = []
    for r in gold[gold["label"] == 1].itertuples():
        h, d = str(r.herb_id), str(r.disease_id)
        if sel_herbs and h not in sel_herbs:
            continue
        if sel_diseases and d not in sel_diseases:
            continue
        if h in hidx_full and d in didx_full:
            pairs.append((hidx_full[h], didx_full[d]))
    return pairs


def run_lodo(epochs):
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)
    A_full, X_full, N_full, off_h, off_d, nH, nD, herbs, disease_ids, _ = \
        build_graph(use_logfc=True)

    baselines = {
        "D-TMS_Fisher": score_fisher(herbs, disease_ids).values,
        "RWR_diffusion": score_rwr(A_full, N_full, off_h, nH, off_d, nD),
        "Network_proximity": score_proximity(herbs, disease_ids).values,
        "Target_count": score_targetcount(herbs, disease_ids).values,
    }
    hidx_full = {h: i for i, h in enumerate(herbs)}
    didx_full = {d: i for i, d in enumerate(disease_ids)}

    gcn_all, base_all = [], {k: [] for k in baselines}
    per_disease = []  # rows

    print("=" * 78)
    print(f"留一病 (LODO) · GCN-D inductive vs 4 无监督基线  (epochs={epochs})")
    print("=" * 78)
    for di, d_test in enumerate(disease_ids):
        train_diseases = [d for d in disease_ids if d != d_test]
        didx_train = {d: i for i, d in enumerate(train_diseases)}
        hidx_train = {h: i for i, h in enumerate(herbs)}  # 草药不剔除
        excl_train = build_exclude(gold, herbs, train_diseases)
        train_pairs = build_train_pairs(gold, hidx_train, didx_train,
                                         skip_diseases={d_test})

        A_tr, X_tr, N_tr, off_h_tr, off_d_tr, nH_tr, nD_tr, herbs_tr, \
            train_diseases2, _ = build_graph_exclude(excl_diseases={d_test},
                                                     use_logfc=True)

        model, w, loss = train_gcn_model(
            A_tr, X_tr, off_h, off_d, nH, nD_tr, train_pairs, nH,
            epochs=epochs, seed=42, exclude_pos=excl_train, **BEST)
        S = infer_score(model, w, A_full, X_full, off_h, off_d, nH, nD)

        test_pairs = build_test_pairs(gold, hidx_full, didx_full,
                                       sel_diseases={d_test})
        gdf = pair_metrics(S, herbs, disease_ids, gold, test_pairs)
        gdf["disease"] = d_test
        gcn_all.append(gdf)
        for name, Sb in baselines.items():
            bdf = pair_metrics(Sb, herbs, disease_ids, gold, test_pairs)
            bdf["disease"] = d_test
            base_all[name].append(bdf)

        gr = summarise(gdf, "GCN_D_LODO")
        print(f"  留出 {d_test:<6} n_test={len(gdf):>3}  GCN-D={gr['pair_AUROC']:.4f}  "
              f"loss={loss:.4f}  | "
              f"Fisher={summarise(base_all['D-TMS_Fisher'][-1],'x')['pair_AUROC']:.4f} "
              f"RWR={summarise(base_all['RWR_diffusion'][-1],'x')['pair_AUROC']:.4f}")

    gcn_sum = summarise(pd.concat(gcn_all, ignore_index=True), "GCN_D_LODO")
    rows = [gcn_sum]
    for name, dl in base_all.items():
        rows.append(summarise(pd.concat(dl, ignore_index=True), name))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "Table_lodo.csv"),
                              index=False, encoding="utf-8-sig")

    # 每病明细
    method_names = ["GCN_D_LODO"] + list(baselines.keys())
    all_method_dfs = [gcn_all] + [base_all[k] for k in baselines]
    det_rows = []
    for mname, mdfs in zip(method_names, all_method_dfs):
        for d_test in disease_ids:
            dd = [d for d in mdfs if d["disease"].iloc[0] == d_test]
            if not dd:
                continue
            r = summarise(pd.concat(dd, ignore_index=True), mname)
            r["disease"] = d_test
            det_rows.append(r)
    pd.DataFrame(det_rows).to_csv(os.path.join(OUT_DIR, "Table_lodo_per_disease.csv"),
                                  index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("LODO 汇总 (306 对中的该病子集, 与无监督基线同口径)")
    print("=" * 78)
    print(f"{'method':<20}{'n_pairs':>8}{'AUROC':>10}{'95%CI_lo':>10}{'95%CI_hi':>10}"
          f"{'MRR':>8}{'Hits@10':>9}")
    for r in rows:
        print(f"{r['method']:<20}{r['n_pairs']:>8}{r['pair_AUROC']:>10.4f}"
              f"{r['ci95_lo']:>10.4f}{r['ci95_hi']:>10.4f}{r['MRR']:>8.4f}"
              f"{r['Hits@10']:>9.3f}")
    print(f"[写入] Table_lodo.csv / Table_lodo_per_disease.csv")


def run_coldstart(epochs):
    gold = pd.read_csv(GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)
    A_full, X_full, N_full, off_h, off_d, nH, nD, herbs, disease_ids, _ = \
        build_graph(use_logfc=True)

    baselines = {
        "D-TMS_Fisher": score_fisher(herbs, disease_ids).values,
        "RWR_diffusion": score_rwr(A_full, N_full, off_h, nH, off_d, nD),
        "Network_proximity": score_proximity(herbs, disease_ids).values,
        "Target_count": score_targetcount(herbs, disease_ids).values,
    }
    hidx_full = {h: i for i, h in enumerate(herbs)}
    didx_full = {d: i for i, d in enumerate(disease_ids)}

    # 有正例对的草药才做冷启动测试
    pos_herbs = sorted({str(r.herb_id) for r in gold[gold["label"] == 1].itertuples()})
    rng = np.random.default_rng(2024)
    perm = rng.permutation(len(pos_herbs))
    folds = [set(pos_herbs[i] for i in perm[k::5]) for k in range(5)]

    gcn_all, base_all = [], {k: [] for k in baselines}
    per_fold = []

    print("=" * 78)
    print(f"冷启动草药 (5 折) · GCN-D inductive vs 4 无监督基线  (epochs={epochs})")
    print("=" * 78)
    for k, held in enumerate(folds):
        herbs_train = [h for h in herbs if h not in held]
        hidx_train = {h: i for i, h in enumerate(herbs_train)}
        excl_train = build_exclude(gold, herbs_train, disease_ids)
        train_pairs = build_train_pairs(gold, hidx_train, didx_full,
                                         skip_herbs=held)

        A_tr, X_tr, N_tr, off_h_tr, off_d_tr, nH_tr, nD_tr, herbs_tr, \
            disease_ids_tr, _ = build_graph_exclude(excl_herbs=held,
                                                     use_logfc=True)

        model, w, loss = train_gcn_model(
            A_tr, X_tr, off_h, off_d_tr, nH_tr, nD, train_pairs, nH_tr,
            epochs=epochs, seed=42, exclude_pos=excl_train, **BEST)
        S = infer_score(model, w, A_full, X_full, off_h, off_d, nH, nD)

        test_pairs = build_test_pairs(gold, hidx_full, didx_full,
                                      sel_herbs=held)
        gdf = pair_metrics(S, herbs, disease_ids, gold, test_pairs)
        gdf["fold"] = k
        gcn_all.append(gdf)
        for name, Sb in baselines.items():
            bdf = pair_metrics(Sb, herbs, disease_ids, gold, test_pairs)
            bdf["fold"] = k
            base_all[name].append(bdf)

        gr = summarise(gdf, "GCN_D_coldstart")
        print(f"  折 {k + 1}/5  留出药={len(held):>3}  n_test={len(gdf):>3}  "
              f"GCN-D={gr['pair_AUROC']:.4f}  loss={loss:.4f}  | "
              f"Fisher={summarise(base_all['D-TMS_Fisher'][-1],'x')['pair_AUROC']:.4f} "
              f"RWR={summarise(base_all['RWR_diffusion'][-1],'x')['pair_AUROC']:.4f}")

    gcn_sum = summarise(pd.concat(gcn_all, ignore_index=True), "GCN_D_coldstart")
    rows = [gcn_sum]
    for name, dl in base_all.items():
        rows.append(summarise(pd.concat(dl, ignore_index=True), name))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "Table_coldstart.csv"),
                              index=False, encoding="utf-8-sig")

    # 每折明细
    method_names = ["GCN_D_coldstart"] + list(baselines.keys())
    all_method_dfs = [gcn_all] + [base_all[k] for k in baselines]
    det_rows = []
    for mname, mdfs in zip(method_names, all_method_dfs):
        for k in range(5):
            dd = [d for d in mdfs if int(d["fold"].iloc[0]) == k]
            if not dd:
                continue
            r = summarise(pd.concat(dd, ignore_index=True), mname)
            r["fold"] = k
            det_rows.append(r)
    pd.DataFrame(det_rows).to_csv(
        os.path.join(OUT_DIR, "Table_coldstart_per_fold.csv"),
        index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("冷启动草药 汇总 (对级干净评估)")
    print("=" * 78)
    print(f"{'method':<20}{'n_pairs':>8}{'AUROC':>10}{'95%CI_lo':>10}{'95%CI_hi':>10}"
          f"{'MRR':>8}{'Hits@10':>9}")
    for r in rows:
        print(f"{r['method']:<20}{r['n_pairs']:>8}{r['pair_AUROC']:>10.4f}"
              f"{r['ci95_lo']:>10.4f}{r['ci95_hi']:>10.4f}{r['MRR']:>8.4f}"
              f"{r['Hits@10']:>9.3f}")
    print(f"[写入] Table_coldstart.csv / Table_coldstart_per_fold.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["lodo", "coldstart", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=3000)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.stage in ("lodo", "all"):
        run_lodo(args.epochs)
    if args.stage in ("coldstart", "all"):
        run_coldstart(args.epochs)
    print(f"\n[完成] 总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
