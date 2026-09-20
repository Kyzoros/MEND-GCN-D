"""
构建Morgan指纹缓存 - 为Tanimoto相似度准备
一次性运行，生成 data/processed/morgan_fps.npz
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import os, sys

PROCESSED_DIR = 'data/processed'
CACHE_FILE = os.path.join(PROCESSED_DIR, 'morgan_fps.npz')

def main():
    # 加载SMILES数据
    sm_df = pd.read_csv('data/raw/ingredient_chembl_mapping.csv')
    hi_df = pd.read_csv('data/raw/herb_ingredients.csv')
    
    all_ing_ids = sorted(hi_df['ingredient_id'].unique())
    sm_lookup = {}
    for _, row in sm_df.iterrows():
        smi = str(row['smiles']).strip()
        if smi and smi.lower() != 'nan':
            sm_lookup[int(row['ingredient_id'])] = smi
    
    print(f'总成分: {len(all_ing_ids)}')
    print(f'有SMILES: {len(sm_lookup)}')
    
    # 计算Morgan指纹 (半径2, 2048位)
    ingredient_ids = []
    valid_indices = []
    fingerprints = []
    failed = 0
    
    for i, ing_id in enumerate(all_ing_ids):
        smiles = sm_lookup.get(int(ing_id))
        if not smiles:
            failed += 1
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed += 1
            continue
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fingerprints.append(fp)
        ingredient_ids.append(int(ing_id))
        valid_indices.append(i)
        
        if (i+1) % 1000 == 0:
            print(f'  进度: {i+1}/{len(all_ing_ids)}')
    
    n_valid = len(fingerprints)
    print(f'成功: {n_valid}, 失败: {failed}')
    
    # 预先构建二维numpy数组用于BulkTanimoto加速
    # 将BitVect转为2D numpy uint8数组: (n_valid, 2048)
    print('构建numpy指纹数组...')
    fps_array = np.zeros((n_valid,), dtype=object)
    for i, fp in enumerate(fingerprints):
        fps_array[i] = fp
    
    # 也构建二进制矩阵用于快速向量化计算
    fps_matrix = np.zeros((n_valid, 2048), dtype=np.uint8)
    for i, fp in enumerate(fingerprints):
        arr = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_matrix[i] = arr
    
    # 保存
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez_compressed(
        CACHE_FILE,
        ingredient_ids=np.array(ingredient_ids, dtype=np.int32),
        fps_matrix=fps_matrix,
        n_bits=2048,
        n_valid=n_valid
    )
    
    # 验证
    data = np.load(CACHE_FILE)
    print(f'\n缓存验证:')
    print(f'  ingredient_ids: {len(data["ingredient_ids"])}')
    print(f'  fps_matrix: {data["fps_matrix"].shape}')
    print(f'  n_bits: {data["n_bits"]}')
    print(f'  文件大小: {os.path.getsize(CACHE_FILE)/1024:.0f} KB')
    
    print(f'\n缓存已保存: {CACHE_FILE}')

if __name__ == '__main__':
    main()
