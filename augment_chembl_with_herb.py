"""
Path A: Augment our ingredient_chembl_mapping.csv with HERB v2.0 cross-DB IDs.

Strategy (dict-lookup, NO pandas merge -> no row duplication):
  1. Build HERB lookup dicts: InChIKey -> row, HERB Ingredient_id -> row, name/alias norm -> HERB Ingredient_id.
  2. For each of our 9,339 ingredients, find HERB row by InChIKey, else name, else alias.
  3. Add cross-DB IDs + fill missing SMILES/InChIKey (do NOT overwrite existing values).
  4. Save augmented file (one row per ingredient_id).

Output: data/processed/ingredient_chembl_mapping_augmented.csv
"""
import pandas as pd
import re

OUR = 'data/raw/ingredient_chembl_mapping.csv'
HERB = 'data/raw/HERB_2.0/HERB_ingredient_info_v2.txt'
NAMES = 'data/raw/herb_ingredients.csv'
OUT = 'data/processed/ingredient_chembl_mapping_augmented.csv'
CROSS = ['SymMap_id', 'TCMSP_id', 'NPASS_id', 'TCMID_id', 'HIT_id', 'TCM_ID_id', 'PubChem_id', 'DrugBank_id']


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


# ---- load ours ----
ours = pd.read_csv(OUR, dtype=str)
for c in ['inchikey', 'chembl_id', 'smiles']:
    ours[c] = ours[c].fillna('').str.strip()
print(f'our ingredients: {len(ours):,}')

try:
    names = pd.read_csv(NAMES, dtype=str, usecols=['ingredient_id', 'ingredient_name'])
    names['ingredient_name'] = names['ingredient_name'].fillna('').str.strip()
    name_map = names.groupby('ingredient_id')['ingredient_name'].first()
    ours['ingredient_name'] = ours['ingredient_id'].map(name_map).fillna('').astype(str)
except Exception as e:
    print(f'name file load failed: {e}; skipping name match')
    ours['ingredient_name'] = ''

# ---- load HERB v2.0 ----
herb = pd.read_csv(HERB, sep='\t', dtype=str, low_memory=False)
herb['InChIKey'] = herb['InChIKey'].fillna('').str.strip()
herb['Canonical_smiles'] = herb['Canonical_smiles'].fillna('').str.strip()
herb['Isomeric_smiles'] = herb['Isomeric_smiles'].fillna('').str.strip()
print(f'HERB v2.0 ingredients: {len(herb):,}')

# lookup dicts (keyed by HERB row index for O(1) access)
by_ik = {}
by_hid = {}
for idx, r in herb.iterrows():
    if r['InChIKey']:
        by_ik.setdefault(r['InChIKey'], idx)
    if r['Ingredient_id']:
        by_hid.setdefault(r['Ingredient_id'], idx)
by_name = {}
by_alias = {}
for idx, r in herb.iterrows():
    nn = norm(r['Ingredient_name'])
    if nn:
        by_name.setdefault(nn, idx)
    al = r['Ingredient_alias_name']
    if isinstance(al, str) and al:
        for part in al.split(';'):
            np_ = norm(part.strip())
            if np_ and np_ != 'na':
                by_alias.setdefault(np_, idx)
print(f'HERB lookup: ik={len(by_ik):,}, name={len(by_name):,}, alias={len(by_alias):,}')

def find_idx(row):
    ik = row['inchikey']
    if ik and ik in by_ik:
        return by_ik[ik], 'inchikey'
    nm = norm(row['ingredient_name'])
    if nm and nm in by_name:
        return by_name[nm], 'name'
    if nm and nm in by_alias:
        return by_alias[nm], 'alias'
    return None, 'none'

# ---- build augmented rows (one per our ingredient) ----
out_rows = []
match_counts = {'inchikey': 0, 'name': 0, 'alias': 0, 'none': 0}
for _, row in ours.iterrows():
    hb_idx, method = find_idx(row)
    match_counts[method] += 1
    hb = herb.loc[hb_idx] if hb_idx is not None else None
    rec = {
        'ingredient_id': row['ingredient_id'],
        'inchikey': row['inchikey'] if row['inchikey'] else (hb['InChIKey'] if hb is not None else ''),
        'chembl_id': row['chembl_id'],
        'smiles': row['smiles'],
        'ingredient_name': row['ingredient_name'],
    }
    if not rec['smiles'] and hb is not None:
        rec['smiles'] = hb['Isomeric_smiles'] if hb['Isomeric_smiles'] else hb['Canonical_smiles']
    for c in CROSS:
        rec[c.lower()] = (hb[c] if (hb is not None and isinstance(hb[c], str) and hb[c] != 'NA' and hb[c] != '') else '')
    rec['matched_by'] = method
    rec['herb_ingredient_name'] = (hb['Ingredient_name'] if hb is not None else '')
    out_rows.append(rec)

aug = pd.DataFrame(out_rows)
assert aug['ingredient_id'].is_unique, 'duplicate ingredient_id!'
aug.to_csv(OUT, index=False)
print(f'\n=== Saved {OUT} ({len(aug):,} rows, unique ingredient_id OK) ===')

print('\n=== match method ===')
print(pd.Series(match_counts))
print('\n=== cross-DB ID coverage ===')
for c in CROSS:
    cl = c.lower()
    n = (aug[cl].fillna('') != '').sum()
    print(f'  {cl:14s}: {n:,} ({n/len(aug)*100:.1f}%)')
before_s = (ours['smiles'] != '').sum()
after_s = (aug['smiles'].fillna('') != '').sum()
before_i = (ours['inchikey'] != '').sum()
after_i = (aug['inchikey'].fillna('') != '').sum()
print(f'\nSMILES: before {before_s:,} -> after {after_s:,} (+{after_s-before_s:,})')
print(f'InChIKey: before {before_i:,} -> after {after_i:,} (+{after_i-before_i:,})')
