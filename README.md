# MEND-GCN-D

A graph neural network for predicting therapeutic associations between herbs and diseases.

MEND-GCN-D scores every herb in a library against a given disease. It encodes herbs, their
chemical ingredients, target genes and diseases in a single heterogeneous graph, learns node
embeddings with a graph convolutional network (GCN), and scores herb–disease pairs with a
DistMult-style bilinear decoder. A classical unsupervised system-mapping pipeline is included
as a comparison baseline.

---

## Method

### Heterogeneous graph — `build_graph()` ([mendgcn/run_fair_comparison.py](mendgcn/run_fair_comparison.py))

| Item | Detail |
|------|--------|
| Nodes | herbs (H), ingredients (I), genes (G), diseases (D) |
| Edges | H–I (herb contains ingredient), I–G (ingredient targets gene), G–D (gene is differentially expressed in disease); optional gene–gene edges from KEGG pathways (`use_kegg`, controlled by `KEGG_MIN_SHARED`) |
| Adjacency | symmetrically normalised `A_sym = D^(-1/2) A D^(-1/2)` with self-loops |
| Node features | 6 dims: one-hot node type (4) + `sqrt(deg / deg_max)` + gene `|logFC|` from the disease expression profile |

### Encoder — class `GCN`

```
n_layers = 2 (default) :  h = ReLU(W1 (A_sym @ X));  z = Wout (A_sym @ h)
n_layers = 1           :  z = Wout ReLU(W1 X)          # MLP, no graph propagation (ablation)
```

Default dimensions: input 6 → hidden 64 → output 32. All layers are bias-free.

### Training — `train_gcn_model()` / `train_gcn()`

- **Positives**: herb–disease pairs with `label == 1` in the gold standard.
- **Negative sampling**: `k_neg = 5` negatives per disease per positive, drawn from the full
  herb set excluding all known positives of that disease (`build_exclude()`, to avoid leakage).
- **Loss**: binary cross-entropy, `L = -mean(log σ(pos)) - mean(log(1 - σ(neg)))`.
- **Optimiser**: Adam, `lr = 1e-2`; `manual_seed` fixed (42).
- **Decoder**: DistMult learnable diagonal weights `w`; `decoder="dot"` disables `w` (ablation).

### Scoring

`S = (z_h ⊙ w) @ z_dᵀ`, then standardised by its mean and standard deviation. `evaluate()`
reports AUROC under two conventions: `FULL` (negatives = all remaining herbs in the library)
and `labeled` (only herbs that carry an explicit `label` in the gold standard).

---

## Evaluation protocols

| Protocol | Setting | Script | Output table |
|----------|---------|--------|--------------|
| **P1** 5-fold transductive | positive pairs split into 5 folds, each held out once (seed 12345); pair-level AUROC / AUPRC / MRR / Hits@10 / Hits@20 with bootstrap 95% CI | [mendgcn/run_cv_final.py](mendgcn/run_cv_final.py) | `Table_cv_clean.csv` |
| **P2** leave-one-disease-out (inductive) | one disease node and its gene edges removed from the training graph; frozen weights are used to score the unseen disease | [experiments/run_gnn_lodo_coldstart.py](experiments/run_gnn_lodo_coldstart.py) `--stage lodo` | `Table_lodo.csv` (+ `_per_disease`) |
| **P3** cold-start herb (inductive) | 1/5 of herb nodes and their ingredient edges held out; unseen herbs are scored via ingredient → gene → disease propagation | [experiments/run_gnn_lodo_coldstart.py](experiments/run_gnn_lodo_coldstart.py) `--stage coldstart` | `Table_coldstart.csv` (+ `_per_fold`) |
| **Fair comparison** | 1/5 of positive pairs held out; Fisher enrichment, network proximity, target count and GCN-D evaluated on the same data and metric | [mendgcn/run_fair_comparison.py](mendgcn/run_fair_comparison.py) | `Table_fair_comparison.csv` |
| **Final baselines + clean holdout + ablation** | six methods incl. an MLP without graph propagation; epoch sweep and structural ablation | [mendgcn/run_fair_final.py](mendgcn/run_fair_final.py) | `Table_fair_final.csv`, `Table_holdout_clean.csv`, `Table_epoch_overfit.csv`, `Table_gnn_ablation2.csv` |
| **8-disease extension** | curated AUC on the four indication-derived diseases merged with independent validation | [experiments/run_8disease_curated_auc.py](experiments/run_8disease_curated_auc.py), [experiments/run_8disease_master_table.py](experiments/run_8disease_master_table.py) | `Table_8disease_curated_auc.csv`, `Table_8disease_master.csv` |

Default hyper-parameters for every GCN-D protocol: `lr = 1e-2, hid = 64, out = 32, k_neg = 5, n_layers = 2`.

---

## Comparison methods

| Method | Idea | Implementation |
|--------|------|----------------|
| Fisher enrichment (D-TMS) | significance of the overlap between a herb's target set and the disease gene set | `score_fisher()` ([mendgcn/run_fair_comparison.py](mendgcn/run_fair_comparison.py)) |
| Network proximity (Barabási) | degree-corrected z-score of the KEGG distance from disease genes to herb targets | `score_proximity()` (same file) |
| Target count | trivial baseline: number of targets, no disease specificity | `score_targetcount()` (same file) |
| RWR diffusion | random walk with restart seeded on disease genes; the steady-state herb probability is the score | `score_rwr()` ([mendgcn/run_fair_final.py](mendgcn/run_fair_final.py)) |
| MLP without graph propagation | GCN-D with `n_layers = 1`, isolating the contribution of the graph | [mendgcn/run_fair_final.py](mendgcn/run_fair_final.py) |

The unsupervised system-mapping pipeline is available as a baseline in
[baselines/main.py](baselines/main.py): RWR / thermal-diffusion perturbation signatures
(`perturbation_signature.py`) → HDmap-S TMS score with permutation test (`hdmap_s.py`) →
HDmap-M genetic-algorithm combination search (`hdmap_m.py`) → H–PG network evaluation
(`evaluation.py`, `evaluation_metrics.py`).

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml, requests.
Deep models additionally require `torch`; `joblib`, `numba`, `gseapy` and `seaborn` are optional
(see [requirements.txt](requirements.txt)).

---

## Data

This repository contains code only; the data files are not bundled. Place the following inputs
under `data/processed/` (all paths are defined in [config.py](config.py) and [config.yaml](config.yaml)):

| File | Content |
|------|---------|
| `herb_ingredients_processed.csv` | herb–ingredient relations |
| `ingredient_targets_enhanced_recovered.csv` | ingredient–target-gene relations (can be overridden with the `SM_TGT_FILE` environment variable) |
| `disease_profiles_processed.csv` | disease differential-expression profiles with columns `disease_id, gene_symbol, logFC, P_value` |
| `gold_standard_final_8d_supplemented.csv` | curated herb–disease labels (`label ∈ {0, 1}`) for eight diseases: AD, PAH, RA, T2D, HCC, CRC, ASTHMA, OA |
| KEGG gene–gene adjacency cache | built by [mendgcn/kegg_network.py](mendgcn/kegg_network.py) from KEGG pathway data |

Herb, ingredient and target information comes from TCMSP; the gene–gene network from KEGG;
disease expression profiles from GEO. The positives for HCC, CRC, ASTHMA and OA are extracted
from TCMBank indications.

---

## Usage

Run from the repository root:

```bash
# Fair-comparison table (4 methods × 8 diseases)
python mendgcn/run_fair_comparison.py

# P1: 5-fold transductive, pair-level evaluation
python mendgcn/run_cv_final.py --epochs 3000

# P2 / P3: leave-one-disease-out and cold-start herb (inductive)
python experiments/run_gnn_lodo_coldstart.py --stage lodo --epochs 3000
python experiments/run_gnn_lodo_coldstart.py --stage coldstart --epochs 3000

# Final baselines, clean holdout and ablation
python mendgcn/run_fair_final.py --epochs 3000 --seeds 0 1 2

# 8-disease extension
python experiments/run_8disease_curated_auc.py
python experiments/run_8disease_master_table.py

# Unsupervised system-mapping pipeline (baseline)
python baselines/main.py --disease PAH
```

---

## Outputs

All protocol scripts write to `output/results/paper_deliverables/` (see [config.py](config.py)):

- `Table_fair_comparison.csv`, `Table_fair_final.csv` — fair-comparison tables
- `Table_cv_clean.csv` — P1 pair-level metrics with bootstrap 95% CI
- `Table_lodo.csv`, `Table_coldstart.csv` — P2 / P3 inductive results
- `Table_holdout_clean.csv`, `Table_epoch_overfit.csv`, `Table_gnn_ablation2.csv` — holdout, epoch sweep and ablation
- `Table_8disease_curated_auc.csv`, `Table_8disease_master.csv` — 8-disease extension

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── config.py                     # configuration (data paths, graph / evaluation / enhancement settings)
├── config.yaml
├── mendgcn/                      # core library
│   ├── run_fair_comparison.py    # GCN class, build_graph, train_gcn, fair-comparison table
│   ├── run_cv_final.py           # P1: 5-fold transductive pair-level evaluation
│   ├── run_fair_final.py         # final baselines, clean holdout and ablation
│   ├── run_gnn_final.py          # epoch sweep / multi-seed runs
│   ├── data_loader.py            # data loading and enhancement integration
│   ├── kegg_network.py           # KEGG gene–gene adjacency and caching
│   ├── kegg_pathway_filter.py    # KEGG pathway filtering
│   ├── hdmap_s.py                # TMS score with permutation test (baseline)
│   ├── hdmap_m.py                # genetic-algorithm combination search (baseline)
│   ├── perturbation_signature.py # RWR / thermal-diffusion perturbation signatures (baseline)
│   ├── evaluation.py             # H–PG network evaluation (baseline)
│   ├── evaluation_metrics.py     # evaluation metrics (baseline)
│   ├── hdcti_adapter.py          # ingredient–target expansion adapter
│   ├── drugai_adapter.py         # ingredient–target direction prediction adapter
│   ├── experimental_mechanisms.py
│   ├── temporal_signature.py
│   └── transfer_learning.py
├── experiments/
│   ├── run_gnn_lodo_coldstart.py   # P2 leave-one-disease-out / P3 cold-start herb
│   ├── run_8disease_curated_auc.py
│   └── run_8disease_master_table.py
└── baselines/
    └── main.py                   # unsupervised system-mapping pipeline
```
