#!/usr/bin/env python3
"""
Preparar tiles 512x512 con overlap y crear splits train/val/test
para evaluación SkySense++.
"""
import os
import glob
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuración
FLOOD_MASKS_DIR = "data/processed/labels"
S1_DIR = "data/processed/sentinel1"
S2_DIR = "data/processed/sentinel2"
SPLITS_DIR = "data/splits"
TILE_SIZE = 512
OVERLAP = 64  # pixels
STRIDE = TILE_SIZE - OVERLAP

os.makedirs(S1_DIR, exist_ok=True)
os.makedirs(S2_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

def extract_tiles(image_path, output_dir, prefix, tile_size=TILE_SIZE, stride=STRIDE):
    """Extraer tiles de una imagen grande con overlap"""
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        profile = src.profile
        count = src.count
        
        # Calcular grid de tiles
        n_tiles_x = max(1, (width - tile_size) // stride + 1)
        n_tiles_y = max(1, (height - tile_size) // stride + 1)
        
        # Ajustar último tile para cubrir toda la imagen
        tile_paths = []
        
        for iy in range(n_tiles_y):
            for ix in range(n_tiles_x):
                x = min(ix * stride, width - tile_size)
                y = min(iy * stride, height - tile_size)
                
                window = Window(x, y, tile_size, tile_size)
                tile_data = src.read(window=window)
                
                # Solo guardar si tiene datos válidos (no todo nodata/zeros)
                if np.any(tile_data != src.nodata) and np.any(tile_data != 0):
                    tile_name = f"{prefix}_tile_{iy:03d}_{ix:03d}.tif"
                    tile_path = os.path.join(output_dir, tile_name)
                    
                    tile_profile = profile.copy()
                    tile_profile.update({
                        'width': tile_size,
                        'height': tile_size,
                        'transform': rasterio.windows.transform(window, src.transform),
                        'compress': 'deflate',
                        'predictor': 2
                    })
                    
                    with rasterio.open(tile_path, 'w', **tile_profile) as dst:
                        dst.write(tile_data)
                    
                    tile_paths.append({
                        'path': tile_name,
                        'window': [int(x), int(y), int(tile_size), int(tile_size)],
                        'source': os.path.basename(image_path)
                    })
        
        return tile_paths

def match_s1_s2_tiles(s1_tiles, s2_tiles, flood_tiles, max_time_diff_days=7):
    """
    Emparejar tiles S1 y S2 por ubicación espacial y proximidad temporal.
    Retorna lista de samples para SkySense++.
    """
    # Por simplicidad, asumir naming consistente: tile_YYY_XXX.tif
    matched = []
    
    # Crear dicts por coordenadas de tile
    s1_dict = {t['path'].replace('S1_', ''): t for t in s1_tiles}
    s2_dict = {t['path'].replace('S2_', ''): t for t in s2_tiles}
    flood_dict = {t['path'].replace('flood_', ''): t for t in flood_tiles}
    
    # Encontrar intersección
    common_keys = set(s1_dict.keys()) & set(s2_dict.keys()) & set(flood_dict.keys())
    
    for key in common_keys:
        matched.append({
            's1': s1_dict[key]['path'],
            's2': s2_dict[key]['path'],
            'flood': flood_dict[key]['path'],
            'tile_key': key
        })
    
    return matched

def create_splits(matched_samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                  stratify_by=None, random_state=42):
    """Crear splits train/val/test estratificados"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(matched_samples)
    indices = np.arange(n)
    
    # Primera división: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, random_state=random_state, 
        stratify=stratify_by if stratify_by is not None else None
    )
    
    # Segunda división: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, random_state=random_state,
        stratify=stratify_by[temp_idx] if stratify_by is not None else None
    )
    
    return {
        'train': [matched_samples[i] for i in train_idx],
        'val': [matched_samples[i] for i in val_idx],
        'test': [matched_samples[i] for i in test_idx]
    }

def write_split_files(splits, output_dir=SPLITS_DIR):
    """Escribir archivos de lista para SkySense++"""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, samples in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.txt")
        with open(out_path, 'w') as f:
            for s in samples:
                # Formato: s2_path s1_path flood_path class
                # Para 1-shot, class=1 (inundación)
                f.write(f"{s['s2']} {s['s1']} {s['flood']} 1\n")
        print(f"  {split_name}: {len(samples)} samples -> {out_path}")

def main():
    print("=" * 60)
    print("PREPARACIÓN DE TILES Y SPLITS - SKYSENSE++")
    print("=" * 60)
    
    # 1. Extraer tiles de máscaras de inundación (referencia espacial)
    flood_files = glob.glob(os.path.join(FLOOD_MASKS_DIR, "flood_*.tif"))
    print(f"Máscaras de inundación: {len(flood_files)}")
    
    all_flood_tiles = []
    for f in flood_files:
        tiles = extract_tiles(f, FLOOD_MASKS_DIR, "flood")
        all_flood_tiles.extend(tiles)
    
    print(f"Tiles de inundación extraídos: {len(all_flood_tiles)}")
    
    # 2. Extraer tiles S2 (si existen archivos fuente)
    s2_files = glob.glob("data/raw/S2_*.tif") + glob.glob("data/raw/S2_*/*.tif")
    all_s2_tiles = []
    for f in s2_files:
        tiles = extract_tiles(f, S2_DIR, "S2")
        all_s2_tiles.extend(tiles)
    
    print(f"Tiles S2 extraídos: {len(all_s2_tiles)}")
    
    # 3. Extraer tiles S1
    s1_files = glob.glob("data/raw/S1_*.tif") + glob.glob("data/raw/S1_*/*.tif")
    all_s1_tiles = []
    for f in s1_files:
        tiles = extract_tiles(f, S1_DIR, "S1")
        all_s1_tiles.extend(tiles)
    
    print(f"Tiles S1 extraídos: {len(all_s1_tiles)}")
    
    # 4. Emparejar
    matched = match_s1_s2_tiles(all_s1_tiles, all_s2_tiles, all_flood_tiles)
    print(f"Samples emparejados (S1+S2+Flood): {len(matched)}")
    
    if len(matched) == 0:
        print("[WARN] No hay samples emparejados. Verificar naming de archivos.")
        return
    
    # 5. Crear splits (estratificar por tile_key para evitar leakage espacial)
    tile_keys = [s['tile_key'] for s in matched]
    splits = create_splits(matched, stratify_by=tile_keys)
    
    # 6. Escribir archivos
    write_split_files(splits)
    
    # 7. Guardar metadata de splits
    meta = {
        'tile_size': TILE_SIZE,
        'overlap': OVERLAP,
        'stride': STRIDE,
        'splits': {k: len(v) for k, v in splits.items()},
        'total_samples': len(matched)
    }
    with open(os.path.join(SPLITS_DIR, 'splits_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n[OK] Splits guardados en {SPLITS_DIR}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")

if __name__ == "__main__":
    main()