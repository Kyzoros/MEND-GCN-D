"""
实验验证机制覆盖层 (办法B)
==========================
思路: 凡是有实验验证过的"某分子 激活/抑制 某靶标", 就直接采用实验结论,
      DrugAI 预测只负责补实验没覆盖到的空白。真值优先于预测。

数据来源
--------
1. DrugBank 机制数据集 (Dataset-of-activating-and-inhibiting-mechanisms.csv, 11229条)
   label: 1=激活(activating), 0=抑制(inhibiting)
2. data/raw/chembl_mechanisms.csv (若含真实数据; 目前仅2条示例)

匹配方式
--------
* exact    : RDKit canonical SMILES 完全一致。最严格。
* skeleton : InChIKey 首段(2D骨架), 忽略立体化学与盐型, 并取最大片段去盐。
             覆盖更高, 但把立体异构体视作等同 —— 对某些手性药物方向可能不成立。

!! 覆盖率实测 (184,848 条成分-靶标对) !!
    exact    :  37 条 (0.020%)
    skeleton : 144 条 (0.078%)
根本原因: DrugBank 收录的是已批准药物, 而本项目成分为天然产物, 分子重叠极小
(骨架层面仅 182/4010)。因此本模块**不是准确率杠杆**, 它的价值在于:
  (a) 正确性 —— 已知真值处不再让模型瞎猜;
  (b) 可扩展 —— 将来接入更多实验库(ChEMBL/BindingDB)时直接复用这套管线;
  (c) 提供一个独立的"预测 vs 实验"一致性体检指标。
真正提升方向准确率的是 DrugAI 模型本身的修复与重训 (见 retrain_drugai_rdkit.py)。
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False

DRUGBANK_MECHANISMS = (
    r"D:\Bio_information\DrugAI\DrugAI-main"
    r"\Dataset-of-activating-and-inhibiting-mechanisms.csv"
)
GOLD_CACHE_NAME = "experimental_mechanisms_gold.csv"


# ---------------------------------------------------------------- 分子标识

def canonical_smiles(smiles) -> Optional[str]:
    """RDKit canonical SMILES; 不可解析返回 None。"""
    if not _RDKIT_OK or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol) if mol is not None else None


def skeleton_key(smiles) -> Optional[str]:
    """InChIKey 首段 = 2D 骨架哈希 (忽略立体/同位素), 并先去盐取最大片段。"""
    if not _RDKIT_OK or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda f: f.GetNumHeavyAtoms())
        return Chem.MolToInchiKey(mol).split('-')[0]
    except Exception:
        return None


# ---------------------------------------------------------------- 金标准表

def _load_drugbank(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {'SMILES', 'gene_name', 'label'}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} 缺少列 {need - set(df.columns)}")
    return pd.DataFrame({
        'smiles': df['SMILES'].astype(str),
        'gene': df['gene_name'].astype(str).str.upper().str.strip(),
        # label 1=激活 -> +1 ; 0=抑制 -> -1
        'action': np.where(df['label'].astype(int) == 1, 1, -1),
        'evidence': 'DrugBank',
    })


def _load_chembl(path: str) -> Optional[pd.DataFrame]:
    """ChEMBL 机制表(可选)。需含 smiles/gene 以及 action_type 或 action 列。"""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    smi = cols.get('smiles') or cols.get('canonical_smiles')
    gene = cols.get('gene') or cols.get('gene_name') or cols.get('target_gene')
    act = cols.get('action') or cols.get('action_type') or cols.get('mechanism')
    if not (smi and gene and act):
        return None

    def to_action(v):
        s = str(v).strip().lower()
        if s in {'1', '+1', 'activator', 'agonist', 'activating', 'positive'}:
            return 1
        if s in {'-1', 'inhibitor', 'antagonist', 'inhibiting', 'negative', '0'}:
            return -1
        return 0

    res = pd.DataFrame({
        'smiles': df[smi].astype(str),
        'gene': df[gene].astype(str).str.upper().str.strip(),
        'action': df[act].map(to_action),
        'evidence': 'ChEMBL',
    })
    return res[res['action'] != 0]


def build_gold_table(processed_dir: str, raw_dir: str,
                     drugbank_path: str = DRUGBANK_MECHANISMS,
                     use_cache: bool = True) -> pd.DataFrame:
    """
    汇总所有实验机制数据 -> 唯一 (分子, 基因) -> action 表。

    标签冲突(同一分子对同一靶标既有激活又有抑制记录)按多数表决; 恰好平票则丢弃,
    因为"真值本身矛盾"时强行覆盖只会引入噪声。
    """
    cache = os.path.join(processed_dir, GOLD_CACHE_NAME)
    if use_cache and os.path.exists(cache):
        return pd.read_csv(cache)

    if not _RDKIT_OK:
        raise ImportError("需要 RDKit 才能构建实验机制金标准表")

    frames = []
    if os.path.exists(drugbank_path):
        frames.append(_load_drugbank(drugbank_path))
    ch = _load_chembl(os.path.join(raw_dir, "chembl_mechanisms.csv"))
    if ch is not None and len(ch):
        frames.append(ch)
    if not frames:
        return pd.DataFrame(columns=['canon', 'skel', 'gene', 'action', 'n_evidence', 'evidence'])

    raw = pd.concat(frames, ignore_index=True)

    # 分子标识只对唯一 SMILES 计算一次 (InChIKey 较慢)
    uniq = raw['smiles'].unique()
    canon_map = {s: canonical_smiles(s) for s in uniq}
    skel_map = {s: skeleton_key(s) for s in uniq}
    raw['canon'] = raw['smiles'].map(canon_map)
    raw['skel'] = raw['smiles'].map(skel_map)
    raw = raw[raw['canon'].notna()]

    rows = []
    for (canon, skel, gene), grp in raw.groupby(['canon', 'skel', 'gene'], dropna=False):
        votes = grp['action'].tolist()
        s = sum(votes)
        if s == 0:
            continue  # 平票: 实验证据自相矛盾, 不覆盖
        rows.append({
            'canon': canon, 'skel': skel, 'gene': gene,
            'action': 1 if s > 0 else -1,
            'n_evidence': len(votes),
            'evidence': ','.join(sorted(set(grp['evidence']))),
        })
    gold = pd.DataFrame(rows)
    os.makedirs(processed_dir, exist_ok=True)
    gold.to_csv(cache, index=False)
    return gold


# ---------------------------------------------------------------- 覆盖应用

def _ingredient_identifier_maps(mapping_path: str) -> Tuple[Dict, Dict]:
    """ingredient_id -> canonical smiles / skeleton"""
    mp = pd.read_csv(mapping_path)
    uniq = mp['smiles'].dropna().unique()
    canon_map = {s: canonical_smiles(s) for s in uniq}
    skel_map = {s: skeleton_key(s) for s in uniq}
    ids = mp['ingredient_id'].astype(str)
    return (
        dict(zip(ids, mp['smiles'].map(canon_map))),
        dict(zip(ids, mp['smiles'].map(skel_map))),
    )


def apply_experimental_override(df: pd.DataFrame, processed_dir: str, raw_dir: str,
                                match_mode: str = "skeleton",
                                verbose: bool = True) -> pd.DataFrame:
    """
    用实验验证的方向覆盖 action_type。

    被覆盖的行会置 confidence=1.0, action_source='Experimental',
    并新增 exp_verified 标记列, 便于下游区分"真值"与"预测"。
    """
    if not _RDKIT_OK:
        if verbose:
            print("[ExpMech] 未安装 RDKit, 跳过实验机制覆盖")
        return df
    if 'ingredient_id' not in df.columns or 'target_gene' not in df.columns:
        return df

    mapping_path = os.path.join(raw_dir, "ingredient_chembl_mapping.csv")
    if not os.path.exists(mapping_path):
        if verbose:
            print(f"[ExpMech] 缺少成分SMILES映射 {mapping_path}, 跳过")
        return df

    gold = build_gold_table(processed_dir, raw_dir)
    if gold.empty:
        if verbose:
            print("[ExpMech] 无可用实验机制数据, 跳过")
        return df

    canon_by_id, skel_by_id = _ingredient_identifier_maps(mapping_path)
    key_col = 'canon' if match_mode == 'exact' else 'skel'
    lookup = {(r[key_col], r['gene']): (int(r['action']), int(r['n_evidence']))
              for _, r in gold.iterrows() if pd.notna(r[key_col])}

    id_map = canon_by_id if match_mode == 'exact' else skel_by_id
    mol_keys = df['ingredient_id'].astype(str).map(id_map)
    genes = df['target_gene'].astype(str).str.upper().str.strip()

    if 'exp_verified' not in df.columns:
        df['exp_verified'] = 0

    n_hit = n_changed = n_agree = 0
    for idx, mk, gene in zip(df.index, mol_keys, genes):
        if pd.isna(mk):
            continue
        got = lookup.get((mk, gene))
        if got is None:
            continue
        action, _ = got
        n_hit += 1
        old = df.at[idx, 'action_type']
        if int(old) == action:
            n_agree += 1
        else:
            n_changed += 1
        df.at[idx, 'action_type'] = action
        df.at[idx, 'exp_verified'] = 1
        if 'confidence' in df.columns:
            df.at[idx, 'confidence'] = 1.0
        if 'action_source' in df.columns:
            df.at[idx, 'action_source'] = 'Experimental'

    if verbose:
        pct = n_hit / len(df) * 100 if len(df) else 0.0
        print(f"[ExpMech] 匹配方式={match_mode} | 金标准对 {len(lookup)} 条")
        print(f"[ExpMech] 命中 {n_hit}/{len(df)} 条 ({pct:.3f}%) -> 修正 {n_changed} 条, "
              f"原本已一致 {n_agree} 条")
        if n_hit:
            print(f"[ExpMech] 预测与实验一致率: {n_agree / n_hit * 100:.1f}% "
                  f"(可作为方向预测质量的独立体检)")
        print("[ExpMech] 注: 覆盖率低是因为草药成分多为天然产物, 与已批准药物重叠极小; "
              "本层保证真值优先, 但不是准确率主要来源")
    return df


if __name__ == '__main__':
    from config import PROCESSED_DIR, RAW_DIR
    g = build_gold_table(PROCESSED_DIR, RAW_DIR, use_cache=False)
    print(f"金标准表: {len(g)} 条唯一(分子,基因)对")
    print(g.head())
