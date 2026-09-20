#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验：移除"非基因符号"靶点对 GCN-D 的影响（多种子）
==========================================================
背景
--------------------------------------------------------------------------
`ingredient_targets_enhanced_recovered.csv` 的 target_gene 列有 43% 是
UniProt 的**蛋白质/酶名称**（含细菌/病毒蛋白），而非人类基因符号。

本脚本回答一个决策问题：**该不该把这些条目删掉？**

两种对立考量
--------------------------------------------------------------------------
支持删除（数据卫生）：
  它们永远匹配不到疾病 DEG 基因符号（交集恒为 0），
  在图中是挂在成分节点上的叶子节点，语义上是"冒充基因符号"的噪声。

支持保留（信息价值）：
  这些靶点仍然刻画了"某草药成分能作用于哪些蛋白"的**生物活性指纹**，
  是一种真实的化学-生物活性信号；删掉它们会同时删掉 822 个成分的全部靶点
  （成分数 6564 -> 5742），可能损失草药表征信息。

为什么必须多种子
--------------------------------------------------------------------------
单次运行存在明显的运行间波动：同一配置、同一随机种子下，
原始数据两次独立运行得到 GCN-D FULL 0.7787 与 0.7957（波动约 0.017），
已接近甚至超过清洗带来的差异。**单次对比不足以定论**，故本脚本跑多个种子。

实现说明
--------------------------------------------------------------------------
`run_fair_comparison.build_graph()` 在**调用时**读取模块级全局 TGT_FILE，
因此可在同一进程内通过修改 `rf.TGT_FILE` 切换靶点文件，无需重启或设环境变量。

输出
--------------------------------------------------------------------------
- 控制台：每个种子 × 每种靶点版本的 FULL / labeled AUROC
- paper_deliverables/ablation_nongene_targets.csv

用法
--------------------------------------------------------------------------
    python ablation_nongene_targets.py --seeds 0 1 2 --epochs 3000
    python ablation_nongene_targets.py --seeds 0 --epochs 300   # 冒烟
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import config as cfg                                        # noqa: E402
import run_fair_comparison as rf                            # noqa: E402

PROC = cfg.PROCESSED_DIR
ORIG = os.path.join(PROC, "ingredient_targets_enhanced_recovered.csv")
CLEANED = os.path.join(PROC, "ingredient_targets_enhanced_cleaned.csv")
OUT_DIR = os.path.join(cfg.RESULTS_DIR, "paper_deliverables")

BEST = dict(lr=1e-2, hid=64, out=32, k_neg=5, n_layers=2)


def run_once(tgt_path, seed, epochs):
    """用指定靶点文件跑一次 GCN-D, 返回 (FULL均值, labeled均值, 节点数)"""
    rf.TGT_FILE = tgt_path
    gold = pd.read_csv(rf.GOLD_PATH)
    gold["herb_id"] = gold["herb_id"].astype(str)
    gold["disease_id"] = gold["disease_id"].astype(str)

    A, X, N, off_h, off_d, nH, nD, herbs, disease_ids, _g = rf.build_graph()
    hidx = {h: i for i, h in enumerate(herbs)}
    didx = {d: i for i, d in enumerate(disease_ids)}
    pos = sorted({(hidx[str(r.herb_id)], didx[str(r.disease_id)])
                  for r in gold[gold["label"] == 1].itertuples()
                  if str(r.herb_id) in hidx and str(r.disease_id) in didx})
    EXCL = rf.build_exclude(gold, herbs, disease_ids)
    S, loss = rf.train_gcn(A, X, off_h, off_d, nH, nD, pos, nH,
                           epochs=epochs, seed=seed, exclude_pos=EXCL, **BEST)
    res = rf.evaluate(S, herbs, disease_ids, gold)
    full = float(np.mean([v["full"] for v in res.values()]))
    lab = float(np.mean([v["labeled"] for v in res.values()]))
    return full, lab, N, loss


def main():
    ap = argparse.ArgumentParser(description="非基因靶点消融实验")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--alt", default=CLEANED,
                    help="对照组的靶点文件路径")
    ap.add_argument("--alt-name", default="清洗(仅基因符号)",
                    help="对照组名称")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CLEANED):
        print(f"[错误] 找不到清洗版文件 {CLEANED}")
        print("请先运行: python clean_target_genes.py")
        return

    rows = []
    for name, path in [("原始(含非基因靶点)", ORIG),
                       (args.alt_name, args.alt)]:
        fs, ls, ns = [], [], []
        for s in args.seeds:
            f, l, N, loss = run_once(path, s, args.epochs)
            fs.append(f)
            ls.append(l)
            ns.append(N)
            print(f"  {name:<20} seed={s:<3} FULL={f:.4f}  "
                  f"labeled={l:.4f}  节点={N}  loss={loss:.4f}")
            rows.append({"targets": name, "seed": s, "AUROC_full": round(f, 4),
                         "AUROC_labeled": round(l, 4), "n_nodes": N,
                         "epochs": args.epochs})
        print(f"  --> {name} 均值: FULL={np.mean(fs):.4f} "
              f"(±{np.std(fs):.4f})  labeled={np.mean(ls):.4f} "
              f"(±{np.std(ls):.4f})\n")

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "ablation_nongene_targets.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("=" * 78)
    print("汇总（多种子均值 ± 标准差）")
    print("=" * 78)
    for name in df["targets"].unique():
        sub = df[df["targets"] == name]
        print(f"{name:<20} FULL={sub.AUROC_full.mean():.4f}"
              f"(±{sub.AUROC_full.std():.4f})   "
              f"labeled={sub.AUROC_labeled.mean():.4f}"
              f"(±{sub.AUROC_labeled.std():.4f})")
    a = df[df.targets.str.startswith("原始")].AUROC_full.mean()
    b = df[df.targets.str.startswith("清洗")].AUROC_full.mean()
    print(f"\n清洗 - 原始 (FULL)  = {b - a:+.4f}")
    print(f"[写入] {out}")


if __name__ == "__main__":
    main()
