#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断 TMS 公式: 余弦归一化是否稀释了"逆转"信号
======================================================================
当前公式:  TMS = cos(sig, v_d) = Σ_g sig_g·logFC_g / (‖sig‖ · ‖v_d‖)
                                   ^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^
                                   真实逆转信号(仅DEG处)    分母含8077维扩散能量

嫌疑:
  疾病向量 v_d 仅在 ~1500 个 DEG 基因上非零, 分子只在 DEG 处有贡献;
  但 sig 经网络扩散后几乎 8077 维全非零, ‖sig‖ 包含了大量与疾病无关的能量。
  不同草药的 ‖sig‖ 差异(由扩散程度/靶点数决定)会主导 cos 值 -> 稀释真实逆转信号。

对照方案 (统一按"逆转=治疗"评估, score = -value):
  A. cos 全基因空间 (当前公式)
  B. cos 限制在 DEG 基因子集 (‖sig‖ 只含 DEG 处能量)
  C. dot 点积 (分子本身, 不除以 ‖sig‖) —— 保留纯逆转信号

若 C(或B) 明显高于 A => 归一化稀释被证实, 公式层面的缺陷确认。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = "c:/Users/alikex/Desktop/Work/systems_mapping/"
PROC = os.path.join(BASE, "data", "processed")
sys.path.insert(0, BASE)
import config as cfg  # noqa: E402

OUT = os.path.join(cfg.RESULTS_DIR, "paper_deliverables", "tms_formula_diagnosis.csv")


def build_disease_vec(prof, did, genes_index, top_k=1500, logfc_thr=0.05):
    """与 regenerate_enrichment_scores.filter_genes 一致"""
    sub = prof[prof["disease_id"] == did]
    if logfc_thr > 0:
        sub = sub[sub["logFC"].abs() > logfc_thr]
    sub = sub.sort_values("P_value", ascending=True).head(top_k)
    v = np.zeros(len(genes_index), dtype=np.float64)
    n_ov = 0
    for g, lfc in zip(sub["gene_symbol"].astype(str), sub["logFC"].astype(float)):
        if g in genes_index:
            v[genes_index[g]] = lfc
            n_ov += 1
    return v, n_ov, len(sub)


def auroc_reverse(score_vec, herbs, gold, did):
    """逆转逻辑: score = -value, 越高越治疗"""
    pos = set(gold[(gold["disease_id"] == did) & (gold["label"] == 1)]["herb_id"].astype(str))
    pos = {h for h in pos if h in set(herbs)}
    if len(pos) < 3:
        return None
    y = np.array([1 if h in pos else 0 for h in herbs], dtype=float)
    s = np.nan_to_num(np.asarray(score_vec, dtype=float))
    if len(set(y)) < 2:
        return None
    return round(float(roc_auc_score(y, -s)), 4)


def main():
    d = np.load(os.path.join(PROC, "herb_signature_cache_kegg.npz"))
    S = d["signatures"]
    gene_names = [str(g) for g in d["gene_names"]]
    meta = json.load(open(os.path.join(PROC, "herb_signature_meta_kegg.json"), encoding="utf-8"))
    herb_ids = [str(h) for h in meta["herb_ids"]]
    genes_index = {g: i for i, g in enumerate(gene_names)}

    prof = pd.read_csv(os.path.join(PROC, "disease_profiles_processed.csv"))
    gold = pd.read_csv(cfg.GOLD_STANDARD_PATH)

    nz = (np.abs(S) > 1e-12).sum(axis=1)
    print(f"签名矩阵 {S.shape} | 平均非零基因数/草药 = {nz.mean():.1f} / {S.shape[1]}")
    sig_norm = np.linalg.norm(S, axis=1)

    rows = []
    for did in ["AD", "PAH", "RA", "T2D"]:
        v, n_ov, n_deg = build_disease_vec(prof, did, genes_index)
        v_norm = np.linalg.norm(v)
        if v_norm <= 0:
            continue

        dot = S @ v                                    # 分子 (= 纯逆转信号)
        cos_full = dot / (sig_norm * v_norm)           # A. 当前公式

        mask = (v != 0)                                # B. 限制在 DEG 基因子集
        S_sub = S[:, mask]
        v_sub = v[mask]
        sn_sub = np.linalg.norm(S_sub, axis=1)
        vn_sub = np.linalg.norm(v_sub)
        cos_sub = (S_sub @ v_sub) / np.where(sn_sub > 0, sn_sub, 1) / vn_sub

        rows.append({
            "disease": did,
            "DEG数": n_deg,
            "落在签名空间的DEG数": n_ov,
            "A_cos全空间(当前公式)": auroc_reverse(cos_full, herb_ids, gold, did),
            "B_cos_DEG子集": auroc_reverse(cos_sub, herb_ids, gold, did),
            "C_dot点积(不除以‖sig‖)": auroc_reverse(dot, herb_ids, gold, did),
        })

    df = pd.DataFrame(rows)
    mean_row = {"disease": "MEAN", "DEG数": "-", "落在签名空间的DEG数": "-"}
    for c in df.columns[3:]:
        mean_row[c] = round(df[c].astype(float).mean(), 4)
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\n" + "=" * 84)
    print("TMS 公式诊断 (评分=逆转逻辑 -value, 越高越治疗)")
    print("=" * 84)
    print(df.to_string(index=False))
    print("\n判读: 若 C 或 B 明显高于 A => 余弦归一化稀释了信号, 公式存在缺陷")
    print(f"\n[写入] {OUT}")


if __name__ == "__main__":
    main()
