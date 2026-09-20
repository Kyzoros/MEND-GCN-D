"""
Method Comparison: Systems-Mapping vs 3 Baselines
===================================================
A. Target Overlap (Jaccard) - no network needed
B. Network Proximity - shortest path in PPI
C. Random Baseline - permuted TMS

Evaluated on AD validation set (25 pos / 38 neg herbs).
"""
import os, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# ---- 1. Load data ----
print("Loading data...")
t0 = time.time()

# Herb targets (from ingredient_targets_enhanced)
targets_df = pd.read_csv("data/processed/ingredient_targets_enhanced.csv")
herb_ing = pd.read_csv("data/processed/herb_ingredients_processed.csv")
ppi = pd.read_csv("data/processed/ppi_network_processed.csv")

# AD validation set
val = pd.read_csv("data/processed/ad_validation_set.csv")
pos_herbs = set(val[val["label"] == 1]["herb_id"])
neg_herbs = set(val[val["label"] == 0]["herb_id"])
labeled = list(pos_herbs | neg_herbs)
y_true = [1 if h in pos_herbs else 0 for h in labeled]

# AD disease genes (same filtering as ablation)
from data_loader import load_disease_profiles
disease_df = load_disease_profiles()
disease_df = disease_df[disease_df["disease_id"] == "AD"]

# Apply same strict filter as ablation
lfc_thresh = 0.5
pval_thresh = 0.001
sub = disease_df.copy()
if "logFC" in sub.columns:
    sub = sub[sub["logFC"].abs() > lfc_thresh]
if "P_value" in sub.columns:
    sub = sub[sub["P_value"] < pval_thresh]
disease_genes = set(sub["gene_symbol"].tolist())
print(f"  AD disease genes: {len(disease_genes)} (logFC>{lfc_thresh}, p<{pval_thresh})")

# Build herb -> targets mapping
herb_targets = defaultdict(set)
for hid in herb_ing["herb_id"].unique():
    ing_ids = set(herb_ing[herb_ing["herb_id"] == hid]["ingredient_id"])
    t = targets_df[targets_df["ingredient_id"].isin(ing_ids)]
    herb_targets[hid] = set(t["target_gene"].dropna())

print(f"  Herbs with targets: {len(herb_targets)}")
print(f"  Load time: {time.time()-t0:.1f}s")

# ---- 2. Systems-Mapping TMS scores (from latest ablation) ----
tms_df = pd.read_csv(
    "output/results/ablation/ablation_20260724_070536/full/tms_scores_full.csv",
    index_col=0
)
tms_col = tms_df.columns[0]
tms_scores = tms_df[tms_col].to_dict()

# ---- A. Jaccard Overlap ----
print("\nComputing Jaccard overlap...")
jaccard = {}
for hid, targets in herb_targets.items():
    if len(targets | disease_genes) > 0:
        jaccard[hid] = len(targets & disease_genes) / len(targets | disease_genes)
    else:
        jaccard[hid] = 0.0

# ---- B. Network Proximity ----
print("Computing network proximity...")

# Build PPI graph (undirected)
print("  Building PPI graph...")
import networkx as nx
G = nx.Graph()
for _, row in ppi.iterrows():
    G.add_edge(row["protein_A"], row["protein_B"])

print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Precompute shortest paths from disease genes to all reachable nodes
# (much faster than per-herb BFS)
print("  Computing shortest paths from disease genes...")
disease_genes_in_graph = disease_genes & set(G.nodes())
print(f"  Disease genes in graph: {len(disease_genes_in_graph)}/{len(disease_genes)}")

# Multi-source BFS: compute distance from all disease genes to all nodes at once
print("  Running multi-source BFS from disease genes...")
from collections import deque

dist_from_disease = {}
queue = deque()
for dg in disease_genes_in_graph:
    dist_from_disease[dg] = 0
    queue.append(dg)

while queue:
    node = queue.popleft()
    d = dist_from_disease[node]
    for neighbor in G[node]:
        if neighbor not in dist_from_disease:
            dist_from_disease[neighbor] = d + 1
            queue.append(neighbor)

print(f"  BFS covered {len(dist_from_disease)} nodes")

proximity = {}
n_herbs = len(herb_targets)
for i, (hid, targets) in enumerate(herb_targets.items()):
    targets_in_graph = targets & set(G.nodes())
    reachable = [dist_from_disease[tg] for tg in targets_in_graph if tg in dist_from_disease]

    if reachable:
        proximity[hid] = np.mean(reachable)
    else:
        proximity[hid] = np.inf

    if (i + 1) % 100 == 0:
        print(f"    {i+1}/{n_herbs} herbs done")

# Convert proximity: lower = better, so negate
# Also clip inf to max finite value
finite_prox = [v for v in proximity.values() if v < np.inf]
max_prox = max(finite_prox) if finite_prox else 100
proximity_score = {}
for hid, v in proximity.items():
    proximity_score[hid] = -min(v, max_prox)  # negate, lower dist = higher score

print(f"  Proximity done. Finite: {len(finite_prox)}, Infinite: {len(proximity)-len(finite_prox)}")

# ---- C. Random baseline (100 permutations) ----
print("Computing random baseline...")
rand_aurocs = []
np.random.seed(42)
scores_list = list(tms_scores.values())
for _ in range(100):
    perm = np.random.permutation(scores_list)
    rand_scores = {h: s for h, s in zip(tms_scores.keys(), perm)}
    rand_preds = [-rand_scores.get(h, 0) for h in labeled]
    rand_aurocs.append(roc_auc_score(y_true, rand_preds))
rand_auroc_mean = np.mean(rand_aurocs)
rand_auroc_std = np.std(rand_aurocs)

# ---- 3. Compute AUROC/AUPR for each method ----
print("\nComputing AUROC/AUPR...")

def evaluate(method_name, score_dict, lower_is_better=False):
    """Compute AUROC/AUPR. Set lower_is_better=True if smaller score = more positive."""
    preds_raw = [score_dict.get(h, 0) for h in labeled]
    preds = [-x if lower_is_better else x for x in preds_raw]
    try:
        auroc = roc_auc_score(y_true, preds)
    except:
        auroc = np.nan
    try:
        aupr = average_precision_score(y_true, preds)
    except:
        aupr = np.nan
    return auroc, aupr

results = {}

# Systems-Mapping (TMS: lower = better = more likely positive)
sm_auroc, sm_aupr = evaluate("Systems-Mapping", tms_scores, lower_is_better=True)
results["Systems-Mapping"] = {"AUROC": sm_auroc, "AUPR": sm_aupr, "type": "Our Method"}

# Jaccard (higher = better)
j_auroc, j_aupr = evaluate("Jaccard Overlap", jaccard, lower_is_better=False)
results["Jaccard Overlap"] = {"AUROC": j_auroc, "AUPR": j_aupr, "type": "Baseline"}

# Network Proximity (negated distance, higher = better)
p_auroc, p_aupr = evaluate("Network Proximity", proximity_score, lower_is_better=False)
results["Network Proximity"] = {"AUROC": p_auroc, "AUPR": p_aupr, "type": "Baseline"}

# Random
results["Random (mean)"] = {"AUROC": rand_auroc_mean, "AUPR": np.nan, "type": "Baseline"}

# Also add ablation baselines from the run
ablation_dir = "output/results/ablation/ablation_20260724_070536"
for sc in ["baseline", "no_drugai", "no_hdcti"]:
    em_path = os.path.join(ablation_dir, sc, "evaluation_metrics.json")
    if os.path.exists(em_path):
        with open(em_path) as f:
            em = json.load(f)
        m = em.get("metrics", {})
        sc_name = {"baseline": "Ablation: TCMSP-only",
                    "no_drugai": "Ablation: -DrugAI",
                    "no_hdcti": "Ablation: -HDCTI"}[sc]
        results[sc_name] = {"AUROC": m.get("auroc"), "AUPR": m.get("auprc"), "type": "Ablation"}

# ---- 4. Report ----
print("\n" + "=" * 70)
print("  METHOD COMPARISON RESULTS")
print("=" * 70)
print(f"{'Method':<30s} {'AUROC':>8s}  {'AUPR':>8s}  {'Category':>12s}")
print("-" * 70)

comp_rows = []
for name, r in results.items():
    auroc_str = f"{r['AUROC']:.4f}" if not np.isnan(r['AUROC']) else "N/A"
    aupr_str = f"{r['AUPR']:.4f}" if not np.isnan(r['AUPR']) else "N/A"
    print(f"{name:<30s} {auroc_str:>8s}  {aupr_str:>8s}  {r['type']:>12s}")
    comp_rows.append({
        "Method": name, "AUROC": r["AUROC"], "AUPR": r["AUPR"], "Category": r["type"]
    })

# Save
comp_df = pd.DataFrame(comp_rows)
out_dir = "output/method_comparison"
os.makedirs(out_dir, exist_ok=True)
comp_df.to_csv(os.path.join(out_dir, "method_comparison.csv"), index=False)

# ---- 5. Correlation analysis ----
print(f"\n--- Correlation with Systems-Mapping TMS ---")
common_herbs = set(jaccard.keys()) & set(proximity_score.keys()) & set(tms_scores.keys())
print(f"Common herbs: {len(common_herbs)}")
j_vals = [jaccard[h] for h in common_herbs]
p_vals = [proximity_score[h] for h in common_herbs]
t_vals = [tms_scores[h] for h in common_herbs]

from scipy.stats import spearmanr
j_corr, j_p = spearmanr(j_vals, t_vals)
p_corr, p_p = spearmanr(p_vals, t_vals)
print(f"  Jaccard vs TMS: Spearman r={j_corr:.3f} (p={j_p:.2e})")
print(f"  Proximity vs TMS: Spearman r={p_corr:.3f} (p={p_p:.2e})")

print(f"\nResults saved to: {out_dir}/method_comparison.csv")
print(f"Total time: {time.time()-t0:.0f}s")
