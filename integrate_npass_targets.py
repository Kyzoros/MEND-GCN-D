"""
Integrate NPASS v3.0 real ingredient->target associations into our target graph.
=====================================================================================
Inputs (all local):
  data/processed/ingredient_chembl_mapping_augmented.csv  (ingredient_id, npass_id)
  data/raw/NPASS/NPASS3.0_target.txt                      (target_id -> name/type/organism)
  data/raw/NPASS/NPASS3.0_activities.txt                  (np_id -> target_id, activity, organism)  [downloading]
  data/processed/ingredient_targets_enhanced_recovered.csv  (existing edges)

Join logic:
  our npass_id (NPCxxxxx) == NPASS np_id  -> direct.
  keep only target_type in {Individual protein, Single protein} AND organism == Homo sapiens.
  map target_id -> target_name (gene/protein symbol) as target_gene.
  confidence: quantitative binding (Kd/Ki/IC50/EC50/Kb)=0.9, else 0.75 (real experimental > HDCTI guess).

Output: data/processed/ingredient_targets_enhanced_npass.csv  (existing + NPASS, deduped)
Then re-run GNN: run_gnn_pilot_v3.py  (point --targets to it) to see AUROC lift.
"""
import os, sys
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

AUG   = 'data/processed/ingredient_chembl_mapping_augmented.csv'
TARGET = 'data/raw/NPASS/NPASS3.0_target.txt'
ACT   = 'data/raw/NPASS/NPASS3.0_activities.txt'
EXIST = 'data/processed/ingredient_targets_enhanced_recovered.csv'
OUT   = 'data/processed/ingredient_targets_enhanced_npass.csv'

if not os.path.exists(ACT) or os.path.getsize(ACT) < 1_000_000:
    print(f"ERROR: {ACT} not fully downloaded yet (size={os.path.getsize(ACT) if os.path.exists(ACT) else 0}). Wait for background curl.")
    sys.exit(1)

# our npass_id -> ingredient_id(s)
aug = pd.read_csv(AUG, dtype=str)
aug['npass_id'] = aug['npass_id'].fillna('')
np2ing = {}
for _, r in aug.iterrows():
    for nid in str(r['npass_id']).split(';'):
        nid = nid.strip()
        if nid:
            np2ing.setdefault(nid, []).append(str(r['ingredient_id']))
print(f'our ingredients with npass_id: {len(np2ing):,} distinct NPC ids')

# target_id -> meta
tgt = pd.read_csv(TARGET, sep='\t', dtype=str)
tgt['target_id'] = tgt['target_id'].fillna('')
tgt_meta = tgt[['target_id', 'target_type', 'target_name', 'target_organism']].copy()

GOOD_TYPES = {'Individual protein', 'Single protein'}
KEEP_ORG = 'Homo sapiens'

# activities
act = pd.read_csv(ACT, sep='\t', dtype=str, low_memory=False)
print(f'NPASS activities total: {len(act):,}')
act = act[act['np_id'].isin(set(np2ing.keys()))].copy()
print(f'  after np_id match to our ingredients: {len(act):,}')

# np_id -> ingredient_id pairs
npm = pd.DataFrame([(nid, ing) for nid, ings in np2ing.items() for ing in ings],
                   columns=['np_id', 'ingredient_id'])
act = act.merge(npm, on='np_id', how='inner')
act = act.merge(tgt_meta, on='target_id', how='left')
act = act[act['target_type'].isin(GOOD_TYPES) & (act['target_organism'] == KEEP_ORG)]
print(f'  after protein/human filter: {len(act):,} edges')

act['target_gene'] = act['target_name'].fillna('').str.strip()
act = act[act['target_gene'] != '']
act['base_confidence'] = act['activity_type'].apply(
    lambda x: 0.9 if str(x) in ('Kd', 'Ki', 'IC50', 'EC50', 'Kb') else 0.75)

new = pd.DataFrame({
    'ingredient_id': act['ingredient_id'],
    'target_gene': act['target_gene'],
    'action_type': 1,
    'source': 'NPASS',
    'action_source': 'NPASS_experimental',
    'drugai_conf': '',
    'exp_verified': 0,
    'base_confidence': act['base_confidence'],
    'confidence': act['base_confidence'],
})

ex = pd.read_csv(EXIST, dtype=str)
combined = pd.concat([ex, new], ignore_index=True)
before = len(combined)
combined = combined.drop_duplicates(subset=['ingredient_id', 'target_gene'], keep='first')
combined.to_csv(OUT, index=False)

print(f'\n=== NPASS integration ===')
print(f'new NPASS edges: {len(new):,}')
print(f'ingredients gaining NPASS targets: {new["ingredient_id"].nunique():,}')
print(f'merged {before:,} -> {len(combined):,} (dedup {before-len(combined)} overlaps)')
print(f'target-gene universe: {ex["target_gene"].nunique():,} -> {combined["target_gene"].nunique():,} '
      f'(+{combined["target_gene"].nunique()-ex["target_gene"].nunique()})')
print(f'saved -> {OUT}')
print('Next: run GNN with --targets ingredient_targets_enhanced_npass.csv')
