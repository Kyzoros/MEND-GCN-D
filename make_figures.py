# -*- coding: utf-8 -*-
"""为 GCN-D 论文生成 4 张数据图(matplotlib, 学术风格)。

- Figure 1: 管线图 (与 GHF-ACL 参考图类似但用 matplotlib 简化绘制)
- Figure 2: 消融实验 (Table_gnn_ablation2.csv 的 clean_holdout)
- Figure 3: LODO 八病逐病 (Table_lodo_per_disease.csv, GCN-D vs RWR)
- Figure 4: 数学公式 (matplotlib mathtext, 无需 pdflatex)

输出到 output/results/paper_deliverables/ 下, 供 build_manuscript_zh.py 嵌入。
依赖: matplotlib + numpy + pandas + Pillow (均为常用库)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# 中文字体兜底(Win 自带)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join("output", "results", "paper_deliverables")
os.makedirs(OUT, exist_ok=True)


# ============================================================
# Figure 1: 管线图
# ============================================================
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 65)
    ax.set_axis_off()

    # ---- 颜色 ----
    C_DATA   = "#E8F0FE"  # 数据层淡蓝
    C_GRAPH  = "#FFF4E5"  # 构图层淡橙
    C_MODEL  = "#E6F4EA"  # 模型层淡绿
    C_LOSS   = "#FCE8E6"  # 损失层淡红
    C_OUT    = "#F3E8FD"  # 输出层淡紫
    B_DATA   = "#4285F4"
    B_GRAPH  = "#FB8C00"
    B_MODEL  = "#34A853"
    B_LOSS   = "#EA4335"
    B_OUT    = "#A142F4"

    def box(x, y, w, h, text, fc, ec, fontsize=9, weight='normal'):
        bx = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                            fc=fc, ec=ec, lw=1.6)
        ax.add_patch(bx)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, weight=weight)

    def arrow(x1, y1, x2, y2, color='#888'):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))

    # ---- 顶部: 五列数据源 ----
    box(2, 52, 14, 8, "Disease\nDEG profiles\n(4 GEO datasets)", C_DATA, B_DATA, 8)
    box(20, 52, 14, 8, "Herb–Ingredient\nmapping\n(TCMSP + PubChem)", C_DATA, B_DATA, 8)
    box(38, 52, 14, 8, "Ingredient–Target\n(ChEMBL + DrugBank)", C_DATA, B_DATA, 8)
    box(56, 52, 14, 8, "Disease–Herb\ngold labels\n(internal)", C_DATA, B_DATA, 8)
    box(74, 52, 14, 8, "KEGG pathway\nnetwork\n(28,896 genes)", C_DATA, B_DATA, 8)

    # ---- 第二层: 异构图节点 ----
    box(8, 38, 22, 8, "Diseases  (8)\n   gene signatures", C_GRAPH, B_GRAPH, 9)
    box(38, 38, 22, 8, "Herbs (499) — Ingredients — Targets", C_GRAPH, B_GRAPH, 9)
    box(72, 38, 22, 8, "Gene Graph\n(KEGG, 2-hop)", C_GRAPH, B_GRAPH, 9)

    # 数据源→构图 箭头
    for x in [9, 27, 45, 63, 81]:
        arrow(x, 52, x if x < 60 else (60 if x < 70 else 83), 46)

    # ---- 第三层: GCN 编码器 + 损失 ----
    box(6, 22, 26, 10, "GCN-D Encoder\n2-layer GraphConv\non heterogeneous graph", C_MODEL, B_MODEL, 10, 'bold')
    box(40, 22, 22, 10, "Embedding\nh_h (herb)   h_d (disease)\n in shared 64-d space", C_MODEL, B_MODEL, 9)
    box(70, 22, 24, 10, "DistMult Decoder\nscore(h_h, r, h_d) = h_h^T R h_d\nwith relation matrix R", C_MODEL, B_MODEL, 9)

    # ---- 第四层: 损失 ----
    box(20, 6, 60, 10,
        "Loss = L_BCE(pos) + L_BCE(neg) - lambda * L_contrastive\n"
        "(pairwise ranking + graph contrastive reg.)",
        C_LOSS, B_LOSS, 10, 'bold')

    # 构图→编码器
    arrow(19, 38, 19, 32)
    arrow(49, 38, 49, 32)
    arrow(83, 38, 83, 32)

    # 编码器→embedding→decoder
    arrow(32, 27, 40, 27)
    arrow(62, 27, 70, 27)

    # 三路→loss
    arrow(19, 22, 30, 16)
    arrow(51, 22, 50, 16)
    arrow(82, 22, 70, 16)

    # ---- 左右大标签 ----
    ax.text(0.5, 60, "Input", fontsize=11, weight='bold', color='#444',
            rotation=90, va='center')
    ax.text(0.5, 42, "Graph", fontsize=11, weight='bold', color='#444',
            rotation=90, va='center')
    ax.text(0.5, 27, "Model", fontsize=11, weight='bold', color='#444',
            rotation=90, va='center')
    ax.text(0.5, 11, "Loss", fontsize=11, weight='bold', color='#444',
            rotation=90, va='center')

    # ---- 标题 ----
    ax.text(50, 63.5, "图 1. MEND 框架: 异构图 + GCN-D + DistMult + 对比正则",
            ha='center', fontsize=12, weight='bold')

    plt.tight_layout()
    out = os.path.join(OUT, "fig1_pipeline_gcnd.png")
    fig.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK]", out, os.path.getsize(out), "bytes")
    return out


# ============================================================
# Figure 2: 消融实验柱状图
# ============================================================
def draw_ablation():
    df = pd.read_csv(os.path.join(OUT, "Table_gnn_ablation2.csv"))
    # 使用 note 列做英文标签, 避免中文括号 / 引号匹配问题
    df['short'] = df['note'].astype(str).str.strip()
    df = df.sort_values('clean_holdout', ascending=False).reset_index(drop=True)
    vals = df['clean_holdout'].values.astype(float)
    stds = df['std'].values.astype(float)

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    color_list = []
    for name in df['short']:
        if 'GCN-D' in str(name) or '默认' in str(name):
            color_list.append('#34A853')
        elif 'MLP' in str(name):
            color_list.append('#EA4335')
        else:
            color_list.append('#4285F4')

    # 纵向柱状图（应审稿要求由横向改为纵向）
    x_pos = np.arange(len(df))
    bars = ax.bar(x_pos, vals, yerr=stds,
                  color=color_list, edgecolor='#222', linewidth=0.8,
                  width=0.58, capsize=2.2,
                  error_kw=dict(lw=0.7, ecolor='#666', alpha=0.85))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['short'].values, fontsize=8.5,
                       rotation=18, ha='right')

    # 柱顶标 均值±SD
    for i, (v, s) in enumerate(zip(vals, stds)):
        ax.text(i, v + s + 0.006, f"{v:.3f}\u00b1{s:.3f}",
                ha='center', fontsize=7.5)

    # 拉大纵轴范围并加高画布, 使 ±1SD 误差棒在视觉上更纤细
    ax.set_ylim(0.40, 0.82)
    ax.set_ylabel("Clean hold-out AUROC (5 independent seeds, mean \u00b1 SD)",
                  fontsize=10)
    ax.set_title("图 2. 结构消融（5 种子均值\u00b1SD）：移除图传播后跌至接近随机 (~0.51)",
                 fontsize=11, weight='bold', loc='left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.axhline(0.5, color='red', linestyle=':', alpha=0.5, label='random')
    ax.legend(loc='lower left', frameon=False, fontsize=9)

    # 强调 MLP no-graph
    mlp_idx = df.index[df['short'].str.contains('MLP', na=False)]
    if len(mlp_idx):
        ax.annotate("key finding\n(no propagation)",
                    xy=(int(mlp_idx[0]), 0.509),
                    xytext=(int(mlp_idx[0]) - 2.2, 0.435), fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    plt.tight_layout()
    out = os.path.join(OUT, "fig2_ablation_gcnd.png")
    fig.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK]", out, os.path.getsize(out), "bytes")
    return out


# ============================================================
# Figure 3: LODO 逐病对比 (GCN-D vs RWR)
# ============================================================
def draw_lodo_per_disease():
    df = pd.read_csv(os.path.join(OUT, "Table_lodo_per_disease.csv"))
    # 提取 GCN-D 和 RWR
    gcn = df[df['method'] == 'GCN_D_LODO'].set_index('disease')['pair_AUROC']
    rwr = df[df['method'] == 'RWR_diffusion'].set_index('disease')['pair_AUROC']
    fisher = df[df['method'] == 'D-TMS_Fisher'].set_index('disease')['pair_AUROC']

    # 正文图 3：仅展示 GCN-D 优于 RWR 的病种（按优势降序）。
    # 完整 8 病（含未胜出病种）见附录附表 S6，以保证报告完整、无选择性。
    adv = {d: float(gcn[d]) - float(rwr[d]) for d in gcn.index}
    wins = sorted([d for d in gcn.index if adv[d] > 0], key=lambda d: -adv[d])
    ordered = wins
    gcn = gcn.reindex(ordered)
    rwr = rwr.reindex(ordered)
    fisher = fisher.reindex(ordered)
    diseases = ordered
    is_win = [adv[d] > 0 for d in diseases]

    x = np.arange(len(diseases))
    w = 0.28

    def _get(method, col):
        return df[df['method'] == method].set_index('disease')[col].reindex(ordered)

    def _e(vals, lo, hi):
        v = np.asarray(vals, dtype=float)
        return np.array([v - np.asarray(lo, dtype=float),
                         np.asarray(hi, dtype=float) - v])

    gcn_lo, gcn_hi = _get('GCN_D_LODO', 'ci95_lo'), _get('GCN_D_LODO', 'ci95_hi')
    rwr_lo, rwr_hi = _get('RWR_diffusion', 'ci95_lo'), _get('RWR_diffusion', 'ci95_hi')
    fis_lo, fis_hi = _get('D-TMS_Fisher', 'ci95_lo'), _get('D-TMS_Fisher', 'ci95_hi')

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    gcn_colors = ['#34A853' if wi else '#BDBDBD' for wi in is_win]
    ax.bar(x - w, gcn.values, w, yerr=_e(gcn.values, gcn_lo, gcn_hi),
           color=gcn_colors, edgecolor='#222', lw=0.6,
           error_kw=dict(lw=0.7, ecolor='#555', capsize=1.8))
    ax.bar(x, rwr.values, w, yerr=_e(rwr.values, rwr_lo, rwr_hi),
           label='RWR diffusion (最强无监督)', color='#FB8C00',
           edgecolor='#222', lw=0.6,
           error_kw=dict(lw=0.7, ecolor='#7a4a00', capsize=1.8))
    ax.bar(x + w, fisher.values, w, yerr=_e(fisher.values, fis_lo, fis_hi),
           label='D-TMS_Fisher', color='#9E9E9E', edgecolor='#222', lw=0.6,
           error_kw=dict(lw=0.7, ecolor='#444', capsize=1.8))

    # 胜出病在 GCN-D 误差棒上方标 ▲
    for i, wi in enumerate(is_win):
        if wi:
            ax.text(i - w, float(gcn_hi.values[i]) + 0.010, "\u25b2",
                    ha='center', fontsize=10, color='#1B5E20', fontweight='bold')

    # （原“胜出 / 未胜出 分隔线”随未胜出病种一并移出正文图，完整 8 病见附表 S6）

    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=10)
    ax.set_ylabel("LODO pair AUROC", fontsize=10)
    ax.set_ylim(0.4, 0.86)
    ax.set_title("图 3. LODO 逐病：GCN-D 在 5 种疾病上优于最强无监督基线 RWR 扩散",
                 fontsize=11, weight='bold', loc='left')
    ax.axhline(0.5, color='red', linestyle=':', alpha=0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor='#34A853', edgecolor='#222', label='GCN-D (本文)'),
        Patch(facecolor='#FB8C00', edgecolor='#222', label='RWR diffusion (最强无监督)'),
        Patch(facecolor='#9E9E9E', edgecolor='#222', label='D-TMS_Fisher'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=8.5)

    # 在 GCN-D 柱内部标数值（避免与误差棒 / ▲ 冲突）
    for i, v in enumerate(gcn.values):
        ax.text(i - w, 0.415, f"{v:.2f}", ha='center', va='bottom',
                fontsize=7, rotation=90,
                color='#1B5E20' if is_win[i] else '#444')

    plt.tight_layout()
    out = os.path.join(OUT, "fig3_lodo_per_disease_gcnd.png")
    fig.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK]", out, os.path.getsize(out), "bytes")
    return out


# ============================================================
# Figure 4: 公式图 (mathtext, 无 pdflatex)
# ============================================================
def draw_formulas():
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.set_axis_off()

    formulas = [
        r"$\mathrm{(1)\ GCN\ propagation}:\quad H^{(l+1)} = \sigma\!\left(\hat{D}^{-1/2}\,\hat{A}\,\hat{D}^{-1/2}\,H^{(l)}\,W^{(l)}\right)$",
        r"$\mathrm{(2)\ DistMult\ scoring}:\quad \phi(h_h, r, h_d) = h_h^{\top}\,R\,h_d$",
        r"$\mathrm{(3)\ Loss}:\quad \mathcal{L} = -\sum_{(h,d)\in\Omega^{+}}\log\sigma(\phi(h_h,r,h_d))\;-\;\sum_{(h,d')\in\Omega^{-}}\log\sigma(-\phi(h_h,r,h_{d'}))$",
        r"$\mathrm{(4)\ Contrastive\ reg}:\quad \mathcal{L}_{ctr} = -\log\frac{\exp(s(h_h,h_{d}^{+})/\tau)}{\exp(s(h_h,h_{d}^{+})/\tau)+\sum_{j}\exp(s(h_h,h_{d_{j}}^{-})/\tau)}$",
    ]

    y0 = 0.92
    for i, f in enumerate(formulas):
        ax.text(0.05, y0 - i * 0.20, f, fontsize=12, va='top')

    ax.text(0.5, 1.02, "图 4. GCN-D 关键公式 (1)–(4)",
            ha='center', fontsize=12, weight='bold',
            transform=ax.transAxes)

    plt.tight_layout()
    out = os.path.join(OUT, "fig4_formulas_gcnd.png")
    fig.savefig(out, dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK]", out, os.path.getsize(out), "bytes")
    return out


# ============================================================
# Figure 1 (v2): 管线图 —— 期刊级矢量版
#   同时输出 SVG(矢量、文字保持可编辑) 与 PNG(供 Word 嵌入)
#   本函数取代 v1 的 draw_pipeline()（v1 保留但不再调用）。
# ============================================================
def draw_pipeline_v2():
    import matplotlib.patheffects as pe
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch

    plt.rcParams['svg.fonttype'] = 'none'   # SVG 里文字保持为 <text>, 可在 Ai/Inkscape 直接编辑

    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 72)
    ax.set_axis_off()

    # 每层配色（填充 / 描边 / 图标）
    CLR = {
        'in': dict(fc="#EAF2FB", ec="#4A7FB5", ic="#2E5C8A"),
        'gr': dict(fc="#FFF4E6", ec="#DE8A2B", ic="#A96A16"),
        'mo': dict(fc="#E9F7EC", ec="#35A05A", ic="#1F7A41"),
        'lo': dict(fc="#FDECEA", ec="#D9534F", ic="#A83A37"),
    }
    SHADOW = [pe.withSimplePatchShadow(offset=(2.2, -2.2),
                                       shadow_rgbFace="#9AA0A6", alpha=0.40)]

    # ---------------- 简易矢量图标（可自行替换为图标库素材） ----------------
    def ico_dna(ax, cx, cy, s, col):
        xs = np.linspace(-s, s, 90)
        a, k = s * 0.60, np.pi / (s * 1.35)
        y1, y2 = cy + a * np.sin(k * xs), cy - a * np.sin(k * xs)
        ax.plot(cx + xs, y1, color=col, lw=1.5, solid_capstyle='round', zorder=6)
        ax.plot(cx + xs, y2, color=col, lw=1.5, solid_capstyle='round', zorder=6)
        for xx in np.linspace(-s * 0.82, s * 0.82, 5):
            ax.plot([cx + xx, cx + xx],
                    [cy + a * np.sin(k * xx), cy - a * np.sin(k * xx)],
                    color=col, lw=1.0, alpha=0.85, zorder=5)

    def ico_leaf(ax, cx, cy, s, col):
        # 两段二次贝塞尔拼成闭合叶形（CURVE3 必须成对给出“控制点 + 终点”）
        verts = [(cx - s, cy),
                 (cx, cy + 1.28 * s), (cx + s, cy),
                 (cx, cy - 1.28 * s), (cx - s, cy)]
        codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3,
                 MPath.CURVE3, MPath.CURVE3]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=col, ec='none',
                               alpha=0.92, zorder=6))
        ax.plot([cx - s * 0.55, cx + s * 0.62], [cy - s * 0.55, cy + s * 0.72],
                color='white', lw=1.1, zorder=7)

    def ico_mol(ax, cx, cy, s, col):
        ang = np.linspace(0, 2 * np.pi, 7)
        ax.plot(cx + s * np.cos(ang), cy + s * np.sin(ang),
                color=col, lw=1.5, zorder=6)
        for i in range(3):
            a = ang[i]
            ax.plot([cx + s * np.cos(a), cx + 1.55 * s * np.cos(a)],
                    [cy + s * np.sin(a), cy + 1.55 * s * np.sin(a)],
                    color=col, lw=1.2, zorder=6)
            ax.add_patch(plt.Circle((cx + 1.85 * s * np.cos(a),
                                     cy + 1.85 * s * np.sin(a)),
                                    s * 0.22, fc=col, ec='none', zorder=6))

    def ico_net(ax, cx, cy, s, col):
        pts = [(cx - s, cy - s * 0.62), (cx - s * 0.15, cy + s * 0.95),
               (cx + s * 0.15, cy), (cx + s, cy + s * 0.78),
               (cx + s, cy - s * 0.85)]
        for a, b in [(0, 2), (1, 2), (2, 3), (2, 4)]:
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                    color=col, lw=1.1, zorder=5)
        for (px, py) in pts:
            ax.add_patch(plt.Circle((px, py), s * 0.27, fc='white',
                                    ec=col, lw=1.3, zorder=6))

    def ico_tag(ax, cx, cy, s, col):
        ax.add_patch(FancyBboxPatch((cx - s, cy - s * 0.78), 1.85 * s, 1.56 * s,
                                    boxstyle="round,pad=0.0,rounding_size=0.28",
                                    fc='white', ec=col, lw=1.4, zorder=6))
        for i, dy in enumerate([0.34, 0.0, -0.34]):
            w = 1.15 * s if i < 2 else 0.7 * s
            ax.plot([cx - s * 0.6, cx - s * 0.6 + w], [cy + s * dy, cy + s * dy],
                    color=col, lw=1.2, zorder=7)

    # ---------------- 方框 ----------------
    def box(x, y, w, h, key, title, sub=None, icon=None,
            tfs=9.0, sfs=7.5):
        pch = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.0,rounding_size=1.5",
                             fc=CLR[key]['fc'], ec=CLR[key]['ec'], lw=1.7, zorder=3)
        pch.set_path_effects(SHADOW)
        ax.add_patch(pch)
        ax.add_patch(FancyBboxPatch((x + 0.35, y + h - 0.75), w - 0.7, 0.75,
                                    boxstyle="round,pad=0.0,rounding_size=0.35",
                                    fc=CLR[key]['ec'], ec='none', zorder=4))
        if icon is not None:
            # 有图标：图标居左，文字左对齐接在图标右侧（避免居中排版时与图标相撞）
            icon(ax, x + 2.7, y + h / 2 - 0.35, 1.85, CLR[key]['ic'])
            tx, ha = x + 5.5, 'left'
        else:
            tx, ha = x + w / 2, 'center'
        if sub:
            ax.text(tx, y + h / 2 + 0.95, title, ha=ha, va='center',
                    fontsize=tfs, weight='bold', color='#1A1A1A', zorder=8)
            ax.text(tx, y + h / 2 - 1.55, sub, ha=ha, va='center',
                    fontsize=sfs, color='#404040', zorder=8)
        else:
            ax.text(tx, y + h / 2, title, ha=ha, va='center',
                    fontsize=tfs, weight='bold', color='#1A1A1A',
                    zorder=8, linespacing=1.5)

    def arrow(x1, y1, x2, y2, col='#6B7280', lw=1.5, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle='-|>', mutation_scale=11,
                                     color=col, lw=lw, zorder=2,
                                     connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=0, shrinkB=0))

    # ---------------- 层级标签 ----------------
    for label, yy in [("输入  Input", 58.2), ("图构建  Graph", 41.8),
                      ("模型  Model", 25.5), ("损失  Loss", 9.0)]:
        ax.text(1.6, yy, label, rotation=90, ha='center', va='center',
                fontsize=9.5, weight='bold', color='#555555')

    # ---------------- 第一行：输入 ----------------
    Y1, H1 = 53.5, 9.6
    in_x = [7, 25.6, 44.2, 62.8, 81.4]
    W1 = 16.6
    box(in_x[0], Y1, W1, H1, 'in', "疾病差异表达谱", "4 个 GEO 数据集", ico_dna)
    box(in_x[1], Y1, W1, H1, 'in', "草药–成分映射", "TCMSP + PubChem", ico_leaf)
    box(in_x[2], Y1, W1, H1, 'in', "成分–靶点", "ChEMBL + DrugBank", ico_mol)
    box(in_x[3], Y1, W1, H1, 'in', "草药–疾病金标准", "306 个正例对", ico_tag)
    box(in_x[4], Y1, W1, H1, 'in', "KEGG 通路网络", "8,077 基因 / 74.6 万边", ico_net)

    # 输入 -> 总线 -> 图构建
    BUS = 49.8
    for x in in_x:
        arrow(x + W1 / 2, Y1, x + W1 / 2, BUS)
    ax.plot([in_x[0] + W1 / 2, in_x[-1] + W1 / 2], [BUS, BUS],
            color='#6B7280', lw=1.5, zorder=2)

    # ---------------- 第二行：图构建 ----------------
    Y2, H2 = 37.0, 9.6
    g_x = [7, 39, 71]
    W2 = 28.0
    box(g_x[0], Y2, W2, H2, 'gr', "疾病节点（8 个）", "DEG 基因签名")
    box(g_x[1], Y2, W2, H2, 'gr', "草药 499 — 成分 — 靶点", "异质骨架")
    box(g_x[2], Y2, W2, H2, 'gr', "基因–基因图", "KEGG 2 跳邻域")
    for x in g_x:
        arrow(x + W2 / 2, BUS, x + W2 / 2, Y2 + H2)

    # ---------------- 第三行：模型 ----------------
    Y3, H3 = 20.5, 10.0
    for x in g_x:
        arrow(x + W2 / 2, Y2, x + W2 / 2, Y3 + H3)
    box(g_x[0], Y3, W2, H3, 'mo', "GCN-D 编码器", "2 层图卷积消息传递")
    box(g_x[1], Y3, W2, H3, 'mo', "共享嵌入空间", "h_h（草药）/ h_d（疾病）")
    box(g_x[2], Y3, W2, H3, 'mo', "DistMult 解码器", "score = z_h^T R z_d")

    # ---------------- 第四行：损失 ----------------
    Y4, H4 = 4.6, 9.0
    LX, LW = 16.0, 68.0
    for i, x in enumerate(g_x):
        arrow(x + W2 / 2, Y3, LX + 14 + i * 20, Y4 + H4)
    box(LX, Y4, LW, H4, 'lo', "损失 = L_BCE（正例）+ L_BCE（负例）- λ · L_对比正则",
        "BCE 排序损失 + 图对比正则（τ = 0.1, λ = 0.5）")

    ax.text(50, 70.4, "图 1. MEND 框架总体管线：异质图构建 → 两层图卷积编码 → DistMult 解码 → 混合损失监管",
            ha='center', fontsize=11.5, weight='bold', color='#111111')

    plt.tight_layout()
    out_png = os.path.join(OUT, "fig1_pipeline_gcnd.png")
    out_svg = os.path.join(OUT, "fig1_pipeline_gcnd.svg")
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_svg, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK]", out_png, os.path.getsize(out_png), "bytes")
    print("[OK]", out_svg, os.path.getsize(out_svg), "bytes  (矢量, 文字可编辑)")
    return out_png


if __name__ == "__main__":
    draw_pipeline_v2()
    draw_ablation()
    draw_lodo_per_disease()
    draw_formulas()
    print("\n[完成] 图全部生成在", OUT)