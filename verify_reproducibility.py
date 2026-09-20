#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键可复现性校验脚本
====================
自动执行：备份关键文件 → 重算 enrichment (regenerate_enrichment_scores.py)
→ 重跑 HERB 外部验证 (run_herb_external_validation.py --compare-legacy)
→ 对比论文已知值并打出 PASS / FAIL。

用法：
    python verify_reproducibility.py            # 含备份
    python verify_reproducibility.py --no-backup

要点：
    - 仅“重算”，不重跑整条 main.py 管线；不依赖 herb_signature_cache。
    - 自动注入 KMP_DUPLICATE_LIB_OK=TRUE，避免 OMP Error#15。
    - 论文已知值取自 manuscript_zh_final.md（扩充金标准下：enrichment 均值 0.601；
      外部验证 ρ∈[0.186,0.351]、AUROC∈[0.536,0.664]，4/4 显著）。
    - 容差：enrichment AUROC ±0.05；外部 ρ ±0.12；外部 AUROC ±0.10。
      超出容差即判 FAIL，提示“可能发现隐藏错误/漂移”。
"""

import argparse
import os
import sys
import shutil
import subprocess
import re
import datetime

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import config as cfg  # noqa: E402

# ---------- 论文已知值（来自 manuscript_zh_final.md） ----------
EXPECTED_ENRICH_MEAN = 0.601
EXPECTED_ENRICH_PER = {"AD": 0.625, "PAH": 0.664, "RA": 0.503, "T2D": 0.612}
EXPECTED_EXT = {
    "Alzheimer": {"rho": 0.186, "auc": 0.536},
    "PAH":       {"rho": 0.302, "auc": 0.664},
    "RA":        {"rho": 0.351, "auc": 0.662},
    "T2D":       {"rho": 0.235, "auc": 0.618},
}

TOL_E = 0.05    # enrichment AUROC 容差
TOL_RHO = 0.12  # 外部 rho 容差
TOL_AUC = 0.10  # 外部 AUROC 容差


def backup():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(cfg.RESULTS_DIR, f"_verify_backup_{ts}")
    os.makedirs(dst, exist_ok=True)
    files = [
        os.path.join(cfg.RESULTS_DIR, "weighted_kegg", "hdmap_s_scores.csv"),
        os.path.join(cfg.RESULTS_DIR, "paper_deliverables", "external_validation_summary.csv"),
        cfg.GOLD_STANDARD_PATH,
        os.path.join(cfg.PROCESSED_DIR, "ingredient_targets_enhanced.csv"),
    ]
    print(f"[备份] -> {dst}")
    n = 0
    for f in files:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
            n += 1
            print(f"  ✓ {os.path.basename(f)}")
        else:
            print(f"  - 跳过(不存在): {f}")
    print(f"  共备份 {n} 个文件\n")
    return dst


def run(cmd, env):
    print("=" * 78)
    print("$ " + " ".join(cmd))
    print("=" * 78)
    # 显式 UTF-8：Windows 默认用 GBK 解码子进程 stdout，会被 regenerate / run_herb
    # 打印的中文（UTF-8 字节）炸出 UnicodeDecodeError;0x86 / 0xa6 即典型 GBK 错位字节
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env=env, cwd=BASE)
    print(r.stdout)
    if r.returncode != 0:
        print("!!! 进程返回非零，stderr 末尾如下：")
        print(r.stderr[-3000:])
        return None
    return r.stdout


def parse_mean(s):
    m = re.search(r"平均 AUROC = ([\d.]+)", s)
    return float(m.group(1)) if m else None


def parse_per(s):
    out = {}
    # 行格式: AD         16   0.6250  0.5690  0.6810   显著
    for m in re.finditer(r"^(\w+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)",
                         s, re.MULTILINE):
        out[m.group(1)] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backup", action="store_true", help="跳过备份步骤")
    args = ap.parse_args()

    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if not args.no_backup:
        backup()

    out1 = run([sys.executable, "regenerate_enrichment_scores.py"], env)
    out2 = run([sys.executable, "run_herb_external_validation.py", "--compare-legacy"], env)

    mean = parse_mean(out1) if out1 else None
    per = parse_per(out1) if out1 else {}
    ext = None
    p = os.path.join(cfg.RESULTS_DIR, "paper_deliverables", "external_validation_summary.csv")
    if os.path.exists(p):
        ext = pd.read_csv(p)

    rows = []
    fails = 0

    # --- enrichment 均值 ---
    if mean is None:
        rows.append(("enrichment 均值 AUROC", "0.601", "<未解析/运行失败>", "-", "FAIL"))
        fails += 1
    else:
        d = abs(mean - EXPECTED_ENRICH_MEAN)
        ok = d <= TOL_E
        fails += 0 if ok else 1
        rows.append(("enrichment 均值 AUROC", "0.601", f"{mean:.4f}", f"{d:+.4f}",
                     "PASS" if ok else "FAIL"))

    # --- enrichment 每病 ---
    for k, v in EXPECTED_ENRICH_PER.items():
        got = per.get(k)
        if got is None:
            rows.append((f"  {k} AUROC", f"{v:.3f}", "<缺失>", "-", "FAIL"))
            fails += 1
        else:
            d = abs(got - v)
            ok = d <= TOL_E
            fails += 0 if ok else 1
            rows.append((f"  {k} AUROC", f"{v:.3f}", f"{got:.4f}", f"{d:+.4f}",
                         "PASS" if ok else "FAIL"))

    # --- 外部验证 ---
    if ext is None:
        rows.append(("外部验证 CSV", "-", "<未生成>", "-", "FAIL"))
        fails += 1
    else:
        extd = {r["disease"]: r for _, r in ext.iterrows()}
        for k, exp in EXPECTED_EXT.items():
            if k not in extd:
                rows.append((f"  外部 {k} rho/auc", "-", "<缺失>", "-", "FAIL"))
                fails += 1
                continue
            r = extd[k]
            dr = abs(float(r["rho"]) - exp["rho"])
            da = abs(float(r["auroc_top30%"]) - exp["auc"])
            okr = dr <= TOL_RHO
            oka = da <= TOL_AUC
            oksig = str(r["significance"]).strip() == "显著"
            ok = okr and oka and oksig
            fails += 0 if ok else 1
            rows.append((f"  外部 {k} rho", f"{exp['rho']:.3f}",
                         f"{float(r['rho']):.4f}", f"{dr:+.4f}", "PASS" if okr else "FAIL"))
            rows.append((f"  外部 {k} auc", f"{exp['auc']:.3f}",
                         f"{float(r['auroc_top30%']):.4f}", f"{da:+.4f}", "PASS" if oka else "FAIL"))

    # --- 报告 ---
    print("\n" + "=" * 78)
    print("可复现性校验报告")
    print("=" * 78)
    print(f"{'指标':<26}{'论文已知':>12}{'重算值':>12}{'偏差':>10}{'判定':>8}")
    print("-" * 78)
    for name, exps, got, diff, verdict in rows:
        print(f"{name:<26}{exps:>12}{got:>12}{diff:>10}{verdict:>8}")
    print("-" * 78)
    if fails == 0:
        print("✅ 全部 PASS：当前代码+数据稳定复现论文数字，缓存有效，结果可复现。")
    else:
        print(f"⚠️  发现 {fails} 项 FAIL：存在漂移/错误，请检查 "
              f"ingredient_targets_enhanced.csv、config.yaml 或运行日志。")
    print("=" * 78)


if __name__ == "__main__":
    main()
