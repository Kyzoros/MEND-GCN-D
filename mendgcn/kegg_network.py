"""
KEGG Pathway Co-Membership Network Builder
===========================================
Builds a gene-gene adjacency matrix where edge weight = Jaccard index
of shared KEGG pathways between two genes.

Replaces PPI physical interactions with functional pathway relationships
as the propagation medium for Random Walk with Restart.

Usage:
    from kegg_network import build_kegg_adjacency
    adj, gene2idx = build_kegg_adjacency()

To cache (recommended after first build):
    adj, gene2idx = build_kegg_adjacency(cache_dir="data/processed")
"""
import os
import json
import time
import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import sys
# --- 目录重组后的路径引导 (MEND-GCN-D) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _MEND_SRC in (_HERE, _ROOT, os.path.join(_ROOT, "mendgcn")):
    if _MEND_SRC not in sys.path:
        sys.path.insert(0, _MEND_SRC)


def _load_kegg_pathway_genes():
    """Download or load cached full KEGG_2021_Human pathway gene sets."""
    import gseapy as gp

    lib = gp.get_library(name="KEGG_2021_Human", organism="human")
    # Filter out overly broad pathways (>500 genes) to avoid noise
    pathway_genes = {}
    for pw, genes in lib.items():
        if len(genes) <= 500:
            pathway_genes[pw] = set(genes)

    return pathway_genes


def _build_kegg_co_membership(pathway_genes, gene2idx, min_shared=1,
                               min_jaccard=0.0):
    """Build sparse co-membership adjacency from pathway gene sets.

    Parameters:
        pathway_genes: dict of pathway_name -> set of gene_symbols
        gene2idx: dict of gene -> index (from PPI network, for consistency)
        min_shared: minimum shared pathways to create an edge
        min_jaccard: minimum Jaccard index threshold
    """
    n = len(gene2idx)
    genes_in_kegg = set()
    for genes in pathway_genes.values():
        genes_in_kegg.update(genes)

    # Only process genes present in both KEGG and PPI gene2idx
    valid_genes = {g for g in genes_in_kegg if g in gene2idx}
    print(f"  [KEGG Net] Genes in KEGG: {len(genes_in_kegg)}")
    print(f"  [KEGG Net] Genes in KEGG & PPI gene2idx: {len(valid_genes)}")

    # Build gene -> pathways mapping for valid genes
    gene_to_pw = defaultdict(set)
    for pw, genes in pathway_genes.items():
        for g in genes & valid_genes:
            gene_to_pw[g].add(pw)

    # Build sparse co-membership matrix
    gene_list = sorted(valid_genes)
    g_to_pos = {g: i for i, g in enumerate(gene_list)}

    row_idx, col_idx, data = [], [], []
    n_genes = len(gene_list)

    t0 = time.time()
    for i in range(n_genes):
        g1 = gene_list[i]
        pw1 = gene_to_pw[g1]
        n1 = len(pw1)

        for j in range(i + 1, n_genes):
            g2 = gene_list[j]
            pw2 = gene_to_pw[g2]
            shared = len(pw1 & pw2)

            if shared < min_shared:
                continue

            union = len(pw1 | pw2)
            jaccard = shared / union if union > 0 else 0

            if jaccard < min_jaccard:
                continue

            row_idx.extend([i, j])
            col_idx.extend([j, i])
            data.extend([jaccard, jaccard])

    # Build sparse matrix using PPI gene2idx indices
    ppi_row, ppi_col, ppi_data = [], [], []
    for k in range(len(row_idx)):
        g_i = gene_list[row_idx[k]]
        g_j = gene_list[col_idx[k]]
        if g_i in gene2idx and g_j in gene2idx:
            ppi_row.append(gene2idx[g_i])
            ppi_col.append(gene2idx[g_j])
            ppi_data.append(data[k])

    adj = sp.coo_matrix(
        (ppi_data, (ppi_row, ppi_col)), shape=(n, n)
    ).tocsr()

    elapsed = time.time() - t0
    print(f"  [KEGG Net] Edges: {adj.nnz} (min_shared={min_shared}, "
          f"min_jaccard={min_jaccard})")
    print(f"  [KEGG Net] Build time: {elapsed:.1f}s")

    return adj


def _normalize_adjacency(adj):
    """Row-normalize adjacency matrix for RWR (random walk transition matrix)."""
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    d_inv = sp.diags(1.0 / row_sums)
    return d_inv @ adj


def build_kegg_adjacency(gene2idx=None, cache_dir=None,
                          min_shared=1, min_jaccard=0.0):
    """Build KEGG co-membership adjacency matrix.

    Parameters:
        gene2idx: reference gene->index mapping (from PPI, for consistency).
                  If None, builds one from KEGG genes only.
        cache_dir: directory to cache/load the adjacency matrix
        min_shared: minimum shared pathways for an edge
        min_jaccard: minimum Jaccard similarity threshold

    Returns:
        adj: row-normalized scipy CSR matrix
        gene2idx: gene -> index mapping used
    """
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "kegg_adjacency.npz")
        mapping_file = os.path.join(cache_dir, "kegg_gene2idx.json")

    # Try loading from cache
    if cache_file and os.path.exists(cache_file) and os.path.exists(mapping_file):
        print("[KEGG Net] Loading cached adjacency...")
        adj = sp.load_npz(cache_file)
        with open(mapping_file) as f:
            gene2idx = json.load(f)
        print(f"  [KEGG Net] Loaded: {adj.shape}, {adj.nnz} edges")
        return adj, gene2idx

    # Load KEGG pathway data
    print("[KEGG Net] Downloading KEGG pathway gene sets...")
    pathway_genes = _load_kegg_pathway_genes()
    print(f"  [KEGG Net] Pathways: {len(pathway_genes)}")

    # If no gene2idx provided, build from KEGG genes
    if gene2idx is None:
        all_genes = set()
        for genes in pathway_genes.values():
            all_genes.update(genes)
        gene2idx = {g: i for i, g in enumerate(sorted(all_genes))}

    # Build and normalize
    adj_raw = _build_kegg_co_membership(
        pathway_genes, gene2idx, min_shared, min_jaccard
    )
    adj_norm = _normalize_adjacency(adj_raw)

    # Cache
    if cache_file:
        sp.save_npz(cache_file, adj_norm)
        with open(mapping_file, "w") as f:
            json.dump(gene2idx, f)
        print(f"  [KEGG Net] Cached to {cache_dir}")

    return adj_norm, gene2idx
