#!/usr/bin/env python3"
"""
Calcular métricas de evaluación desagregadas para Model Card (Mitchell et al. 2019)
IoU, F1, Precision, Recall por barrio, año, sensor, evento.
"""
import os
import glob
import json
import numpy as np
import rasterio
import pandas as pd
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import bootstrap
from tqdm import tqdm

# Configuración
PREDICTIONS_DIR = "outputs/predictions"
GROUND_TRUTH_DIR = "data/processed/labels"
SPLITS_DIR = "data/splits"
METRICS_DIR = "outputs/metrics"
FIGURES_DIR = "outputs/figures"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_prediction_gt_pairs(split='test'):
    """Cargar pares predicción-ground truth para un split"""
    split_file = os.path.join(SPLITS_DIR, f"{split}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    pairs = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                s2_path, s1_path, flood_path = parts[0], parts[1], parts[2]
                # Buscar predicción correspondiente
                pred_name = os.path.basename(flood_path).replace('flood_', 'pred_')
                pred_path = os.path.join(PREDICTIONS_DIR, pred_name)
                
                if os.path.exists(pred_path) and os.path.exists(flood_path):
                    pairs.append({
                        'pred': pred_path,
                        'gt': flood_path,
                        'tile': os.path.basename(flood_path)
                    })
    return pairs

def compute_metrics(y_true, y_pred):
    """Calcular métricas binarias"""
    # Aplanar
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Filtrar solo donde hay datos válidos (gt != nodata)
    valid = y_true_flat != 255  # asumir 255=nodata si aplica
    y_true_flat = y_true_flat[valid]
    y_pred_flat = y_pred_flat[valid]
    
    if len(y_true_flat) == 0:
        return None
    
    return {
        'iou': jaccard_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
        'f1': f1_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
        'precision': precision_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
        'recall': recall_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
        'accuracy': (y_true_flat == y_pred_flat).mean(),
        'tn': confusion_matrix(y_true_flat, y_pred_flat, labels=[0,1]).ravel()[0] if len(np.unique(y_true_flat)) > 1 else 0,
        'fp': confusion_matrix(y_true_flat, y_pred_flat, labels=[0,1]).ravel()[1] if len(np.unique(y_true_flat)) > 1 else 0,
        'fn': confusion_matrix(y_true_flat, y_pred_flat, labels=[0,1]).ravel()[2] if len(np.unique(y_true_flat)) > 1 else 0,
        'tp': confusion_matrix(y_true_flat, y_pred_flat, labels=[0,1]).ravel()[3] if len(np.unique(y_true_flat)) > 1 else 0,
        'n_pixels': len(y_true_flat),
        'flood_ratio_gt': y_true_flat.mean(),
        'flood_ratio_pred': y_pred_flat.mean(),
    }

def bootstrap_ci(metric_func, y_true, y_pred, n_resamples=1000, confidence_level=0.95):
    """Calcular intervalo de confianza por bootstrap"""
    data = (y_true.flatten(), y_pred.flatten())
    
    def stat(y_t, y_p):
        return metric_func(y_t, y_p, average='binary', zero_division=0)
    
    try:
        res = bootstrap(data, stat, n_resamples=n_resamples, confidence_level=confidence_level, 
                       random_state=42, paired=True)
        return res.confidence_interval.low, res.confidence_interval.high
    except:
        return np.nan, np.nan

def parse_tile_metadata(tile_name):
    """Extraer metadatos del nombre del tile: barrio, fecha, sensor, etc."""
    # Asumir naming: flood_tile_YYY_XXX.tif o similar
    # Adaptar según naming real
    meta = {
        'tile': tile_name,
        'barrio': 'unknown',
        'year': 'unknown',
        'sensor': 'fusion',  # s1, s2, fusion
        'event': 'unknown'
    }
    
    # Heurísticas basadas en coordenadas o naming
    # Si el tile_key tiene info de barrio
    if 'donjuan' in tile_name.lower() or 'don_juan' in tile_name.lower():
        meta['barrio'] = 'Don Juan'
    elif 'elsalado' in tile_name.lower() or 'el_salado' in tile_name.lower():
        meta['barrio'] = 'El Salado'
    
    # Extraer año si está en nombre
    import re
    year_match = re.search(r'(202[0-5])', tile_name)
    if year_match:
        meta['year'] = year_match.group(1)
    
    return meta

def evaluate_split(split='test'):
    """Evaluar un split completo con desagregación"""
    pairs = load_prediction_gt_pairs(split)
    print(f"Evaluando {split}: {len(pairs)} pares")
    
    results = []
    
    for pair in tqdm(pairs, desc=f"Computing {split} metrics"):
        # Cargar predicción y ground truth
        with rasterio.open(pair['pred']) as src:
            pred = src.read(1)
        with rasterio.open(pair['gt']) as src:
            gt = src.read(1)
        
        # Asegurar misma forma
        if pred.shape != gt.shape:
            # Resize pred a gt shape si necesario
            from skimage.transform import resize
            pred = resize(pred, gt.shape, order=0, preserve_range=True).astype(pred.dtype)
        
        # Binarizar (threshold 0.5 si probabilidades, o directo si ya binario)
        if pred.max() > 1:
            pred_bin = (pred > 127).astype(np.uint8)
        else:
            pred_bin = (pred > 0.5).astype(np.uint8)
        gt_bin = (gt > 0).astype(np.uint8)
        
        # Métricas globales del tile
        metrics = compute_metrics(gt_bin, pred_bin)
        if metrics is None:
            continue
        
        # Metadatos del tile
        tile_meta = parse_tile_metadata(pair['tile'])
        
        result = {**metrics, **tile_meta, 'split': split}
        results.append(result)
    
    return pd.DataFrame(results)

def disaggregate_metrics(df):
    """Calcular métricas desagregadas por factores"""
    factors = ['barrio', 'year', 'sensor', 'event', 'split']
    disagg_results = {}
    
    for factor in factors:
        if factor not in df.columns:
            continue
        
        grouped = df.groupby(factor).agg({
            'iou': ['mean', 'std', 'count'],
            'f1': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'n_pixels': 'sum',
            'flood_ratio_gt': 'mean',
        }).round(4)
        
        disagg_results[factor] = grouped
    
    # Interseccionales (barrio × year)
    if 'barrio' in df.columns and 'year' in df.columns:
        cross = df.groupby(['barrio', 'year']).agg({
            'iou': ['mean', 'std', 'count'],
            'f1': ['mean', 'std'],
        }).round(4)
        disagg_results['barrio_x_year'] = cross
    
    return disagg_results

def main():
    print("=" * 60)
    print("CÁLCULO DE MÉTRICAS DESAGREGADAS - MODEL CARD")
    print("=" * 60)
    
    # Verificar que existen predicciones
    pred_files = glob.glob(os.path.join(PREDICTIONS_DIR, "*.tif"))
    print(f"Archivos de predicción encontrados: {len(pred_files)}")
    
    if len(pred_files) == 0:
        print("[WARN] No hay predicciones en outputs/predictions/")
        print("  Ejecuta primero scripts/06_run_inference.py")
        
        # Crear CSV vacío con estructura esperada
        df_empty = pd.DataFrame(columns=[
            'iou', 'f1', 'precision', 'recall', 'accuracy',
            'tp', 'fp', 'fn', 'tn', 'n_pixels',
            'flood_ratio_gt', 'flood_ratio_pred',
            'barrio', 'year', 'sensor', 'event', 'split'
        ])
        df_empty.to_csv(os.path.join(METRICS_DIR, 'metrics_disaggregated.csv'), index=False)
        print(f"[INFO] Creado CSV vacío en {METRICS_DIR}/metrics_disaggregated.csv")
        return
    
    # Evaluar splits
    all_results = []
    for split in ['train', 'val', 'test']:
        if os.path.exists(os.path.join(SPLITS_DIR, f"{split}.txt")):
            df = evaluate_split(split)
            if len(df) > 0:
                all_results.append(df)
    
    if not all_results:
        print("[ERROR] No se pudieron calcular métricas")
        return
    
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Guardar resultados por tile
    df_all.to_csv(os.path.join(METRICS_DIR, 'metrics_per_tile.csv'), index=False)
    
    # Desagregar
    disagg = disaggregate_metrics(df_all)
    
    # Guardar cada desagregación
    for name, table in disagg.items():
        safe_name = name.replace(' ', '_').replace('×', 'x')
        table.to_csv(os.path.join(METRICS_DIR, f'metrics_{safe_name}.csv'))
        print(f"\n{name.upper()}:")
        print(table)
    
    # Resumen global
    global_metrics = {
        'iou_mean': df_all['iou'].mean(),
        'iou_std': df_all['iou'].std(),
        'f1_mean': df_all['f1'].mean(),
        'f1_std': df_all['f1'].std(),
        'precision_mean': df_all['precision'].mean(),
        'recall_mean': df_all['recall'].mean(),
        'n_tiles': len(df_all),
        'n_pixels_total': df_all['n_pixels'].sum(),
    }
    
    with open(os.path.join(METRICS_DIR, 'metrics_global.json'), 'w') as f:
        json.dump(global_metrics, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("RESUMEN GLOBAL")
    print(f"{'='*60}")
    for k, v in global_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Generar gráficas para Model Card
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. IoU por barrio
        if 'barrio' in df_all.columns:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # IoU por barrio
            sns.boxplot(data=df_all, x='barrio', y='iou', ax=axes[0,0])
            axes[0,0].set_title('IoU por Barrio')
            axes[0,0].set_ylabel('IoU')
            
            # F1 por año
            if 'year' in df_all.columns:
                sns.boxplot(data=df_all, x='year', y='f1', ax=axes[0,1])
                axes[0,1].set_title('F1-Score por Año')
                axes[0,1].set_ylabel('F1')
            
            # Precision vs Recall scatter
            axes[1,0].scatter(df_all['recall'], df_all['precision'], alpha=0.6)
            axes[1,0].plot([0,1], [0,1], 'r--', alpha=0.5)
            axes[1,0].set_xlabel('Recall')
            axes[1,0].set_ylabel('Precision')
            axes[1,0].set_title('Precision vs Recall')
            axes[1,0].set_xlim(0,1)
            axes[1,0].set_ylim(0,1)
            
            # Distribución flood ratio
            axes[1,1].hist(df_all['flood_ratio_gt'], bins=20, alpha=0.7, label='GT', density=True)
            axes[1,1].hist(df_all['flood_ratio_pred'], bins=20, alpha=0.7, label='Pred', density=True)
            axes[1,1].set_xlabel('Flood Ratio')
            axes[1,1].set_ylabel('Density')
            axes[1,1].set_title('Distribución Ratio Inundación')
            axes[1,1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'metrics_overview.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[OK] Gráficas guardadas en {FIGURES_DIR}/")
    except Exception as e:
        print(f"[WARN] No se pudieron generar gráficas: {e}")

if __name__ == "__main__":
    main()