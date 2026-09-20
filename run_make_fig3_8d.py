"""run_make_fig3_8d.py — 生成 8 病 D-TMS 跨病热图 (figure3_tms_heatmap_8d.png)

读取 data/processed/hdmap_s_scores_enrichment_8d.csv (500 草药 x 8 病 D-TMS 富集分数),
取每病 top-20 草药, 绘制 diverging heatmap。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "data", "processed")
OUT = os.path.join(HERE, "output", "results", "paper_deliverables")

tms = pd.read_csv(os.path.join(PROC, "hdmap_s_scores_enrichment_8d.csv"), index_col=0)
topn = 20
herbs = set()
for d in tms.columns:
    herbs.update(tms[d].sort_values(ascending=False).head(topn).index.tolist())
herbs = sorted(herbs, key=lambda h: tms.loc[h].max(), reverse=True)
sub = tms.loc[herbs]

vmax = np.nanpercentile(sub.values, 98)
vmin = np.nanpercentile(sub.values, 2)

fig, ax = plt.subplots(figsize=(9, max(6, len(sub) * 0.32)))
im = ax.imshow(sub.values, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_xticks(range(len(sub.columns)))
ax.set_xticklabels(sub.columns, rotation=45, ha="right", fontsize=11)
ax.set_yticks(range(len(sub.index)))
ax.set_yticklabels(sub.index, fontsize=6)
ax.set_title("Figure 3 (8 diseases): cross-disease D-TMS heatmap (top-20 herbs per disease)",
             fontsize=11)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("D-TMS  (-log10 Fisher p)", fontsize=9)
plt.tight_layout()
dst = os.path.join(OUT, "figure3_tms_heatmap_8d.png")
plt.savefig(dst, dpi=150, bbox_inches="tight")
print("saved", dst, "shape", sub.shape)
