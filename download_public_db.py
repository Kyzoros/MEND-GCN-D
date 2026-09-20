#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载公开 TCM 数据集 (Zenodo TCM-suite, CC BY 4.0) 用于扩病种 / 扩金标准。
====================================================================
来源: https://zenodo.org/records/10432947
      DOI: 10.5281/zenodo.10432947
      (zhang, TCM-SUITE DATA, 2023, CC BY 4.0)

本脚本下载两份关键关联表:
  - herb-disease.csv           草药-疾病关联  -> 扩金标准 / 评估病种
  - target_protein-disease.csv 疾病-靶点关联  -> 构建多病疾病基因集 (扩病种)

用法:
    python download_public_db.py
"""
import os
import sys
import urllib.request

BASE = "https://zenodo.org/api/records/10432947/files/"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(OUTDIR, exist_ok=True)

FILES = {
    "herb-disease.csv": "tcmsuite_herb_disease.csv",
    "target_protein-disease.csv": "tcmsuite_disease_target.csv",
}


def main():
    for fname, saveas in FILES.items():
        url = BASE + fname + "/content?download=1"
        dst = os.path.join(OUTDIR, saveas)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[跳过] 已存在: {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
            continue
        print(f"[下载] {fname} -> {dst}")
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"  OK ({os.path.getsize(dst)/1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL: {e}")
            sys.exit(1)
    print("下载完成。文件位于:", OUTDIR)


if __name__ == "__main__":
    main()
