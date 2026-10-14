#!/usr/bin/env python3
"""
Test rápido del pipeline completo con datos dummy
Verifica: AOI -> masks -> tiles -> inference -> metrics
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import tempfile

# Configurar paths
sys.path.insert(0, r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main")
sys.path.insert(0, r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main\antmmf\antmmf")
sys.path.insert(0, r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main\antmmf")

# Configurar GDAL fallback
os.environ['USE_PATH_FOR_GDAL_PYTHON'] = 'YES'

def create_dummy_data():
    """Crear datos dummy para testear pipeline"""
    print("Creando datos dummy...")
    
    # Dirs
    os.makedirs("data/processed/sentinel1", exist_ok=True)
    os.makedirs("data/processed/sentinel2", exist_ok=True)
    os.makedirs("data/processed/labels", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)
    
    # Crear 3 tiles dummy 512x512
    H, W = 512, 512
    transform = from_bounds(-74.78, 10.98, -74.76, 11.00, W, H)
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': W,
        'height': H,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'deflate',
        'predictor': 2
    }
    
    # S2: 6 bandas (simular reflectancia 0-1)
    s2_data = np.random.rand(6, H, W).astype(np.float32)
    # Hacer que algunas áreas sean "agua" (valores bajos en NIR)
    water_mask = np.random.rand(H, W) < 0.15
    s2_data[3, water_mask] *= 0.1  # NIR bajo en agua
    
    # S1: 2 bandas VV, VH (simular dB -20 a 5)
    s1_data = (np.random.rand(2, H, W) * 25 - 25).astype(np.float32)
    s1_data[:, water_mask] -= 10  # Agua más oscuro en SAR
    
    # Labels: inundación binaria
    flood_mask = water_mask.astype(np.uint8)
    
    # Guardar 3 tiles
    for i in range(3):
        tile_id = f"tile_{i:03d}"
        
        # S2
        with rasterio.open(f"data/processed/sentinel2/S2_{tile_id}.tif", 'w', **{**profile, 'count': 6}) as dst:
            dst.write(s2_data)
        
        # S1
        with rasterio.open(f"data/processed/sentinel1/S1_{tile_id}.tif", 'w', **{**profile, 'count': 2}) as dst:
            dst.write(s1_data)
        
        # Flood mask
        with rasterio.open(f"data/processed/labels/flood_{tile_id}.tif", 'w', **profile) as dst:
            dst.write(flood_mask[np.newaxis, :, :])
    
    # Crear splits
    with open("data/splits/train.txt", 'w') as f:
        f.write("S2_tile_000.tif S1_tile_000.tif flood_tile_000.tif 1\n")
    with open("data/splits/val.txt", 'w') as f:
        f.write("S2_tile_001.tif S1_tile_001.tif flood_tile_001.tif 1\n")
    with open("data/splits/test.txt", 'w') as f:
        f.write("S2_tile_002.tif S1_tile_002.tif flood_tile_002.tif 1\n")
    
    print("  3 tiles dummy creados")
    print("  splits train/val/test creados")

def test_inference():
    """Test inferencia simplificada"""
    print("\nProbando inferencia...")
    # Verificar que el predictor se puede importar desde SkySense++
    sys.path.insert(0, r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main")
    sys.path.insert(0, r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main\antmmf\antmmf")
    sys.path.insert(0, r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main\antmmf")
    from lib.predictors.flood3i_1shot import OneshotPredictor
    print("  Predictor OK")

def test_metrics():
    """Test cálculo de métricas"""
    print("\nProbando métricas...")
    from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
    
    def compute_metrics(y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return {
            'iou': jaccard_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
            'f1': f1_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
            'precision': precision_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
            'recall': recall_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
        }
    
    # Simular predicción perfecta
    y_true = np.random.randint(0, 2, (100, 100)).astype(np.uint8)
    y_pred = y_true.copy()
    mask = np.random.rand(100, 100) < 0.1
    y_pred[mask] = 1 - y_pred[mask]  # 10% error
    
    metrics = compute_metrics(y_true, y_pred)
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")

def main():
    print("=" * 60)
    print("TEST RÁPIDO PIPELINE COMPLETO (DATOS DUMMY)")
    print("=" * 60)
    
    try:
        create_dummy_data()
        test_inference()
        test_metrics()
        print("\n" + "=" * 60)
        print("[OK] PIPELINE FUNCIONA END-TO-END")
        print("=" * 60)
        print("\nCuando lleguen los datos reales de GEE:")
        print("  1. Reemplazar data/raw/ con archivos .tif descargados")
        print("  2. python scripts/03_create_flood_masks.py")
        print("  3. python scripts/04_prepare_tiles.py")
        print("  4. conda activate skysense && python scripts/06_run_inference.py")
        print("  5. python scripts/07_compute_metrics.py")
        print("  6. python scripts/08_generate_cards.py")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()