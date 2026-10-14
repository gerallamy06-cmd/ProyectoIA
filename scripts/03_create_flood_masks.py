#!/usr/bin/env python3
"""
Crear máscaras de inundación usando índices espectrales (NDWI, MNDWI, AWEI)
a partir de imágenes Sentinel-2 L2A descargadas.
"""
import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from skimage.filters import threshold_otsu
from tqdm import tqdm
import json

# Configuración
S2_RAW_DIR = "data/raw/S2_*"
S1_RAW_DIR = "data/raw/S1_*"
PROCESSED_DIR = "data/processed"
LABELS_DIR = os.path.join(PROCESSED_DIR, "labels")
METADATA_DIR = "data/metadata"

os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Bandas Sentinel-2 L2A (orden en COG descargado de GEE)
# GEE exporta típicamente en orden: B2, B3, B4, B8, B11, B12
BAND_IDX = {
    'B2': 0,   # Blue 490nm
    'B3': 1,   # Green 560nm
    'B4': 2,   # Red 665nm
    'B8': 3,   # NIR 842nm
    'B11': 4,  # SWIR1 1610nm
    'B12': 5,  # SWIR2 2190nm
}

def compute_water_indices(img_stack):
    """
    Calcular índices de agua a partir de stack S2 (C, H, W)
    img_stack: array (6, H, W) con bandas [B2,B3,B4,B8,B11,B12]
    Retorna: dict con índices (H, W)
    """
    # Convertir a float32, escalar 0-1 (valores L2A son 0-10000 típicamente)
    img = img_stack.astype(np.float32) / 10000.0
    
    B3 = img[BAND_IDX['B3']]  # Green
    B8 = img[BAND_IDX['B8']]  # NIR
    B11 = img[BAND_IDX['B11']] # SWIR1
    B12 = img[BAND_IDX['B12']] # SWIR2
    
    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi = (B3 - B8) / (B3 + B8 + 1e-6)
    
    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    mndwi = (B3 - B11) / (B3 + B11 + 1e-6)
    
    # AWEI_sh = 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2)
    awei_sh = 4*(B3 - B11) - (0.25*B8 + 2.75*B12)
    
    return {
        'ndwi': ndwi,
        'mndwi': mndwi,
        'awei_sh': awei_sh
    }

def threshold_index(index, method='otsu'):
    """Umbralizar índice para obtener máscara binaria de agua"""
    if method == 'otsu':
        # Solo valores válidos (no NaN)
        valid = index[~np.isnan(index)]
        if len(valid) < 10:
            return np.zeros_like(index, dtype=bool)
        thresh = threshold_otsu(valid)
        return index > thresh
    elif method == 'zero':
        return index > 0
    else:
        raise ValueError(f"Método desconocido: {method}")

def consensus_water_mask(indices, method='otsu'):
    """
    Combinar múltiples índices por consenso (≥2 de 3 detectan agua)
    """
    masks = {}
    for name, idx in indices.items():
        masks[name] = threshold_index(idx, method)
    
    # Consenso: suma de máscaras >= 2
    consensus = (masks['ndwi'].astype(int) + 
                 masks['mndwi'].astype(int) + 
                 masks['awei_sh'].astype(int)) >= 2
    
    return consensus, masks

def clean_mask(mask, open_kernel=3, close_kernel=5):
    """Limpieza morfológica: opening + closing"""
    mask = mask.astype(bool)
    # Opening: eliminar ruido pequeño
    mask = ndimage.binary_opening(mask, structure=np.ones((open_kernel, open_kernel)))
    # Closing: rellenar huecos
    mask = ndimage.binary_closing(mask, structure=np.ones((close_kernel, close_kernel)))
    return mask

def detect_flood_change(water_t1, water_t2):
    """
    Detectar cambio de inundación: agua en t2 Y no agua en t1
    water_t1, water_t2: máscaras binarias (H, W)
    Retorna: máscara de cambio (1: nueva inundación, 0: sin cambio)
    """
    # Nueva agua = agua en t2 y no en t1
    flood = water_t2 & (~water_t1)
    # Eliminar pérdidas de agua (secado) - opcional
    drying = water_t1 & (~water_t2)
    return flood.astype(np.uint8), drying.astype(np.uint8)

def process_tile_pair(s2_path_t1, s2_path_t2, s1_path_t1=None, s1_path_t2=None, output_dir=LABELS_DIR):
    """Procesar un par de tiles (pre/post evento)"""
    base_name = os.path.basename(s2_path_t1).replace('.tif', '')
    # Asumir naming: S2_YYYYMMDD_tile_X.tif
    
    # Leer S2 t1
    with rasterio.open(s2_path_t1) as src:
        img_t1 = src.read()  # (6, H, W)
        profile = src.profile
    
    # Leer S2 t2
    with rasterio.open(s2_path_t2) as src:
        img_t2 = src.read()
    
    # Calcular índices para ambos tiempos
    indices_t1 = compute_water_indices(img_t1)
    indices_t2 = compute_water_indices(img_t2)
    
    # Máscaras de agua por consenso
    water_t1, _ = consensus_water_mask(indices_t1)
    water_t2, _ = consensus_water_mask(indices_t2)
    
    # Limpieza morfológica
    water_t1 = clean_mask(water_t1)
    water_t2 = clean_mask(water_t2)
    
    # Detectar cambio de inundación
    flood_mask, drying_mask = detect_flood_change(water_t1, water_t2)
    
    # Guardar máscara de inundación
    out_profile = profile.copy()
    out_profile.update({
        'dtype': 'uint8',
        'count': 1,
        'nodata': 0,
        'compress': 'deflate',
        'predictor': 2
    })
    
    flood_path = os.path.join(output_dir, f"flood_{base_name}.tif")
    with rasterio.open(flood_path, 'w', **out_profile) as dst:
        dst.write(flood_mask[np.newaxis, :, :])
    
    # Guardar también máscaras de agua individuales para análisis
    water_dir = os.path.join(PROCESSED_DIR, "water_masks")
    os.makedirs(water_dir, exist_ok=True)
    
    for name, mask in [('water_t1', water_t1), ('water_t2', water_t2), ('drying', drying_mask)]:
        w_path = os.path.join(water_dir, f"{name}_{base_name}.tif")
        with rasterio.open(w_path, 'w', **out_profile) as dst:
            dst.write(mask.astype(np.uint8)[np.newaxis, :, :])
    
    # Metadatos
    meta = {
        'tile': base_name,
        's2_t1': os.path.basename(s2_path_t1),
        's2_t2': os.path.basename(s2_path_t2),
        'flood_pixels': int(flood_mask.sum()),
        'total_pixels': flood_mask.size,
        'flood_ratio': float(flood_mask.sum() / flood_mask.size),
        'water_t1_pixels': int(water_t1.sum()),
        'water_t2_pixels': int(water_t2.sum()),
    }
    
    return meta, flood_path

def find_tile_pairs(s2_dir_pattern):
    """Encontrar pares temporales de tiles S2 (mismo tile, fechas consecutivas)"""
    files = sorted(glob.glob(s2_dir_pattern))
    # Agrupar por tile (naming: S2_Barranquilla_YYYY-MM-DD_N.tif)
    tiles_dict = {}
    for f in files:
        fname = os.path.basename(f)
        # Extraer tile_id y fecha
        # Naming: S2_Barranquilla_YYYY-MM-DD_N.tif
        fname_no_ext = fname.replace('.tif', '')
        parts = fname_no_ext.split('_')
        if len(parts) >= 4:
            # parts[0]='S2', parts[1]='Barranquilla', parts[2]='YYYY-MM-DD', parts[3]='N'
            date_str = parts[2].replace('-', '')  # Convert YYYY-MM-DD to YYYYMMDD
            tile_id = parts[3]  # The tile number (0, 1, etc.)
            if tile_id not in tiles_dict:
                tiles_dict[tile_id] = []
            tiles_dict[tile_id].append((date_str, f))
    
    # Crear pares consecutivos por tile
    pairs = []
    for tile_id, dated_files in tiles_dict.items():
        dated_files.sort(key=lambda x: x[0])
        for i in range(len(dated_files) - 1):
            pairs.append({
                'tile_id': tile_id,
                't1_date': dated_files[i][0],
                't1_path': dated_files[i][1],
                't2_date': dated_files[i+1][0],
                't2_path': dated_files[i+1][1],
            })
    return pairs

def main():
    print("=" * 60)
    print("CREACIÓN MÁSCARAS DE INUNDACIÓN - BARRANQUILLA")
    print("=" * 60)
    
    # Buscar archivos S2 descargados
    s2_files = glob.glob("data/raw/S2_*.tif") + glob.glob("data/raw/S2_*/*.tif")
    print(f"Archivos S2 encontrados: {len(s2_files)}")
    
    if len(s2_files) == 0:
        print("[WARN] No hay archivos S2 en data/raw/")
        print("  Ejecuta primero scripts/02_download_gee.py")
        print("  O coloca archivos manualmente en data/raw/")
        return
    
    # Encontrar pares temporales
    pairs = find_tile_pairs("data/raw/S2_*.tif")
    print(f"Pares temporales encontrados: {len(pairs)}")
    
    if len(pairs) == 0:
        print("[WARN] No se encontraron pares temporales válidos")
        return
    
    # Procesar cada par
    all_meta = []
    for i, pair in enumerate(tqdm(pairs, desc="Procesando pares")):
        try:
            meta, flood_path = process_tile_pair(
                pair['t1_path'], pair['t2_path']
            )
            meta['pair_index'] = i
            all_meta.append(meta)
            print(f"  [{i+1}/{len(pairs)}] {pair['tile_id']}: {meta['flood_pixels']} px inundación ({meta['flood_ratio']:.4%})")
        except Exception as e:
            print(f"  [ERROR] {pair['tile_id']}: {e}")
    
    # Guardar metadatos consolidados
    meta_path = os.path.join(METADATA_DIR, "flood_masks_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(all_meta, f, indent=2)
    
    print(f"\n[OK] Metadatos guardados: {meta_path}")
    print(f"[OK] Máscaras en: {LABELS_DIR}")
    
    # Estadísticas globales
    if all_meta:
        total_flood = sum(m['flood_pixels'] for m in all_meta)
        total_pixels = sum(m['total_pixels'] for m in all_meta)
        print(f"\nEstadísticas globales:")
        print(f"  Tiles procesados: {len(all_meta)}")
        print(f"  Píxeles inundación total: {total_flood:,}")
        print(f"  Ratio inundación global: {total_flood/total_pixels:.4%}")

if __name__ == "__main__":
    main()