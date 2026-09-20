#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重建增强靶点文件: 找回加工过程中丢失的靶基因
============================================
问题 (2026-09-02/03 诊断):
  ingredient_targets.csv (原始) 有 1177 个靶点(蛋白质全名),
  但加工成 ingredient_targets_processed.csv 后只剩 368 个 —— 丢了 809 个(69%)。
  之后 HDCTI 预测补了 955 个, 增强文件合计归一化后仅 1152 基因,
  DEG∩U 仅 549。根因是早期映射表太小。

修复: 用项目现有的 70,488 条映射表 (gene_alias 76,984 + mygene 14,287 + tcmsp 1,677)
       重新把原始 1177 个蛋白质全名映射到基因符号, 还原成 1,157 符号 / 44,812 条,
       替换增强文件里"被阉割过的 368-基因 original 部分", HDCTI_predicted 部分原样保留。

产出: data/processed/ingredient_targets_enhanced_recovered.csv
验证: 唯一基因应 ~1485, DEG∩U 应 ~705 (原 549)。

用法:
    python rebuild_enhanced_targets_recovered.py            # 重建并写盘
    python rebuild_enhanced_targets_recovered.py --dry     # 只验证不写盘
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfg
from hdmap_s import load_gene_symbol_map

RAW = os.path.join(cfg.DATA_DIR, "raw")
PROC = cfg.PROCESSED_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只验证不写盘")
    ap.add_argument("--out", default=os.path.join(
        PROC, "ingredient_targets_enhanced_recovered.csv"))
    args = ap.parse_args()

    gmap = load_gene_symbol_map()
    known = set(gmap.values())
    print(f"映射表: {len(gmap):,} 条, 目标基因符号 {len(known):,}")

    # ---------- 1) 原始基线: 蛋白质全名 -> 基因符号 ----------
    raw = pd.read_csv(os.path.join(RAW, "ingredient_targets.csv"))
    print(f"原始基线: {len(raw):,} 行, {raw['target_gene'].nunique()} 个靶点名")

    def norm(s):
        s = str(s).strip()
        if not s or s.lower() == "nan":
            return None
        if s in known:
            return s
        h = gmap.get(s.lower())
        if h:
            return h
        if " " not in s and 2 <= len(s) <= 15:
            return s
        return None

    raw = raw.copy()
    raw["sym"] = raw["target_gene"].map(norm)
    mapped = raw.dropna(subset=["sym"])
    print(f"  映射成功: {len(mapped):,}/{len(raw):,} "
          f"({len(mapped)/len(raw):.1%}) -> {mapped['sym'].nunique()} 唯一符号")

    rec = pd.DataFrame({
        "ingredient_id": mapped["ingredient_id"].astype(int).values,
        "target_gene": mapped["sym"].values,
        "action_type": mapped["action_type"].astype(int).values,
        "source": "original",
        "action_source": "raw_baseline",
        "drugai_conf": np.nan,
        "exp_verified": 0,
        "base_confidence": 0.5,
        "confidence": 0.5,
    })
    # 去重 (同一成分-靶点)
    rec = rec.drop_duplicates(subset=["ingredient_id", "target_gene"], keep="first")
    print(f"  还原后 original 部分: {len(rec):,} 行, {rec['target_gene'].nunique()} 基因")

    # ---------- 2) 保留 HDCTI_predicted 部分 ----------
    enh = pd.read_csv(os.path.join(PROC, cfg.FILE_INGREDIENT_TARGETS_ENHANCED))
    hdcti = enh[enh["source"] == "HDCTI_predicted"].copy()
    print(f"  HDCTI_predicted 部分保留: {len(hdcti):,} 行, {hdcti['target_gene'].nunique()} 基因")

    # ---------- 3) 合并 ----------
    out = pd.concat([rec, hdcti], ignore_index=True)
    out = out.drop_duplicates(subset=["ingredient_id", "target_gene"], keep="first")
    print(f"\n合并后: {len(out):,} 行, {out['target_gene'].nunique()} 唯一靶点")

    # ---------- 4) 验证 DEG∩U ----------
    dfp = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    deg_all = set()
    for did in dfp["disease_id"].unique():
        sub = dfp[dfp["disease_id"] == did]
        if cfg.DISEASE_LOGFC_THRESHOLD > 0:
            sub = sub[sub["logFC"].abs() > cfg.DISEASE_LOGFC_THRESHOLD]
        if cfg.DISEASE_DEG_TOP_K and cfg.DISEASE_DEG_TOP_K > 0:
            sub = sub.sort_values("P_value", ascending=True).head(cfg.DISEASE_DEG_TOP_K)
        deg_all |= set(g for g in sub["gene_symbol"].astype(str) if g and g != "nan")
    # 目标基因需经 hdmap_s 归一化(这里直接按已知符号集合近似)
    target_syms = set(out["target_gene"]) & known
    inter = len(target_syms & deg_all)
    print(f"  DEG 空间: {len(deg_all):,} 基因")
    print(f"  靶点宇宙(已归一化估计): {len(target_syms):,} 基因")
    print(f"  DEG ∩ 靶点宇宙: {inter}  (原 549, 预期 ~705)")

    if args.dry:
        print("\n[DRY] 未写盘。")
        return

    # ---------- 5) 备份 + 写盘 ----------
    backup = os.path.join(PROC, cfg.FILE_INGREDIENT_TARGETS_ENHANCED + ".bak")
    if not os.path.exists(backup):
        enh.to_csv(backup, index=False)
        print(f"\n[备份] 原文件 -> {backup}")
    out.to_csv(args.out, index=False)
    print(f"[写入] {args.out}")


if __name__ == "__main__":
    main()
