#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把外部 TCM 靶点库 (HERB 等) 并入现有 成分-靶点 表, 扩大靶点宇宙, 重跑 GNN。
=====================================================================================
输入:
  --external   外部 成分-靶点 表 (如 HERB chemical_target.csv)
  --chem-map   外部 成分-元数据 表 (如 HERB herb_ingredient_chemical.csv, 提供 SMILES/InChIKey)
  --source     source 标记 (默认 HERB)

匹配逻辑:
  用 InChIKey 首段(14字符, 忽略立体/质子化) 把外部成分 映射到我们的 ingredient_id
  (我们用 ingredient_chembl_mapping.csv 已有 inchikey; 外部成分从 chem-map 的 SMILES 算 inchikey)

输出:
  data/processed/ingredient_targets_expanded.csv  (现有 + 外部 并集)
  -> 改 config 指向它后, 重跑 run_gnn_pilot_v3.py 即可看 AUROC 是否提升。
"""
import os, sys, argparse
import pandas as pd
from rdkit import Chem

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg

def first_block(ik):
    """InChIKey 首段(连接性), 忽略立体/质子化, TCM 常缺立体故用此匹配"""
    if not isinstance(ik, str) or "-" not in ik:
        return ik
    return ik.split("-")[0].upper()

def inchikey_from_smiles(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.MolToInchiKey(m) if m else None
    except Exception:
        return None

def detect_col(df, *names):
    for n in names:
        for c in df.columns:
            if str(c).strip().lower() == n.lower():
                return c
    # 子串兜底
    for n in names:
        for c in df.columns:
            if n.lower() in str(c).lower():
                return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True, help="外部 成分-靶点 csv")
    ap.add_argument("--chem-map", default=None, help="外部 成分-元数据 csv (给 SMILES/InChIKey)")
    ap.add_argument("--source", default="HERB")
    ap.add_argument("--out", default=os.path.join(cfg.PROCESSED_DIR, "ingredient_targets_expanded.csv"))
    args = ap.parse_args()

    # 我们的成分 InChIKey -> ingredient_id
    mp = pd.read_csv(os.path.join(cfg.RAW_DIR if hasattr(cfg, "RAW_DIR") else os.path.join(BASE, "data", "raw"),
                                   "ingredient_chembl_mapping.csv"))
    our_ik = {}
    for _, r in mp.iterrows():
        ik = r.get("inchikey")
        if not isinstance(ik, str) or len(ik) < 14:
            ik = inchikey_from_smiles(r.get("smiles"))
        if ik:
            our_ik.setdefault(first_block(ik), []).append(str(r["ingredient_id"]))
    print(f"[我们的成分] {len(mp)} 个, 可匹配 InChIKey 首段 {len(our_ik)} 个")

    # 外部成分 -> InChIKey 首段
    ext_ik = {}
    if args.chem_map:
        cm = pd.read_csv(args.chem_map)
        ck = detect_col(cm, "inchikey", "inchi_key")
        sk = detect_col(cm, "smiles", "smile")
        cid = detect_col(cm, "chemical", "ingredient", "id")
        print(f"[外部成分表] 列={list(cm.columns)[:8]}")
        for _, r in cm.iterrows():
            key = str(r[cid]) if cid else None
            ik = r.get(ck) if ck else None
            if not isinstance(ik, str) or len(ik) < 14:
                ik = inchikey_from_smiles(r.get(sk)) if sk else None
            if ik and key:
                ext_ik[str(key)] = first_block(ik)

    # 外部 成分-靶点
    et = pd.read_csv(args.external)
    tcol = detect_col(et, "target", "target_gene", "gene", "gene_symbol", "protein")
    ccol = detect_col(et, "chemical", "ingredient", "compound", "id")
    print(f"[外部靶点表] 列={list(et.columns)[:10]}  target_col={tcol} chemical_col={ccol}")

    # 现有靶点
    cur = pd.read_csv(os.path.join(cfg.PROCESSED_DIR, "ingredient_targets_enhanced_recovered.csv"))
    rows = [(str(r["ingredient_id"]), str(r["target_gene"]),
             str(r.get("action_type", "legacy")), "legacy") for _, r in cur.iterrows()]
    matched_ings = set()
    new_edges = 0
    for _, r in et.iterrows():
        chem = str(r[ccol]) if ccol else None
        gene = r.get(tcol)
        if not chem or not isinstance(gene, str) or not gene:
            continue
        fb = ext_ik.get(chem)
        if not fb:  # 外部表自带 inchikey 的情况
            ik = r.get(detect_col(et, "inchikey")) if detect_col(et, "inchikey") else None
            fb = first_block(ik) if isinstance(ik, str) else None
        if not fb or fb not in our_ik:
            continue
        for ing in our_ik[fb]:
            rows.append((ing, gene, "interacts", args.source))
            matched_ings.add(ing)
            new_edges += 1

    out = pd.DataFrame(rows, columns=["ingredient_id", "target_gene", "action_type", "source"])
    out = out.drop_duplicates()
    out.to_csv(args.out, index=False)
    print(f"\n[输出] {args.out}")
    print(f"  总边数={len(out)}  成分数={out['ingredient_id'].nunique()}  基因数={out['target_gene'].nunique()}")
    print(f"  外部匹配到成分={len(matched_ings)}  新增边={new_edges}")
    print(f"  现有基线: 边数={len(cur)} 成分={cur['ingredient_id'].nunique()} 基因={cur['target_gene'].nunique()}")

if __name__ == "__main__":
    main()
