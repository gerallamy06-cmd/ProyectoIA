#!/usr/bin/env python3
"""
Ejecutar inferencia 1-shot SkySense++ en datos de prueba Barranquilla.
Adaptado de tools/run_1shot.sh y lib/predictors/flood3i_1shot.py
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import rasterio

# Añadir paths de SkySense++
SKYSENSE_DIR = Path(r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main")
sys.path.insert(0, str(SKYSENSE_DIR))
sys.path.insert(0, str(SKYSENSE_DIR / "antmmf"))

# Configuración por defecto
DEFAULT_CONFIG = SKYSENSE_DIR / "configs" / "eval_skysense_pp_barranquilla.yml"
DEFAULT_WEIGHTS = SKYSENSE_DIR / "pretrain" / "skysensepp_release.ckpt"
TEST_SPLIT = "data/splits/test.txt"
OUTPUT_DIR = "outputs/predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_test_list(split_file):
    """Cargar lista de test tiles"""
    samples = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                samples.append({
                    's2': parts[0],
                    's1': parts[1],
                    'flood': parts[2],
                    'class': int(parts[3]) if len(parts) > 3 else 1
                })
    return samples

def load_tile_data(tile_info, data_root="data/processed"):
    """Cargar datos de un tile (S1, S2, target)"""
    # Rutas completas
    s2_path = os.path.join(data_root, "sentinel2", tile_info['s2'])
    s1_path = os.path.join(data_root, "sentinel1", tile_info['s1'])
    flood_path = os.path.join(data_root, "labels", tile_info['flood'])
    
    # Verificar existencia
    for p in [s2_path, s1_path, flood_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"No encontrado: {p}")
    
    # Leer datos
    with rasterio.open(s2_path) as src:
        s2 = src.read().astype(np.float32)  # (C, H, W)
    with rasterio.open(s1_path) as src:
        s1 = src.read().astype(np.float32)
    with rasterio.open(flood_path) as src:
        flood = src.read(1).astype(np.uint8)  # (H, W)
    
    return {
        's2': s2,
        's1': s1,
        'flood': flood,
        'tile_name': tile_info['s2']
    }

def preprocess_for_skysense(s2, s1, flood, config):
    """
    Preprocesar datos al formato esperado por SkySense++ 1-shot
    Basado en few_shot_flood3i_loader.py
    """
    # Obtener parámetros del config
    hr_size = tuple(config['task_attributes']['segmentation']['dataset_attributes']['few_shot_flood_segmentation']['image_size']['hr'])
    s2_size = tuple(config['task_attributes']['segmentation']['dataset_attributes']['few_shot_flood_segmentation']['image_size']['s2'])
    s1_size = tuple(config['task_attributes']['segmentation']['dataset_attributes']['few_shot_flood_segmentation']['image_size']['s1'])
    
    # Normalizar S2 (percentil 2-98 ya aplicado en tiling, pero re-normalizar)
    # ImageNet normalization para HR (primeras 3 bandas RGB)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    # HR: usar primeras 3 bandas (RGB) o B2,B3,B4
    if s2.shape[0] >= 3:
        hr = s2[:3]  # (3, H, W)
    else:
        hr = np.repeat(s2[:1], 3, axis=0)
    
    # Normalizar HR
    hr = hr / 10000.0  # L2A a 0-1
    hr = (hr - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
    
    # Resize HR a 512x512 si necesario
    if hr.shape[1:] != hr_size:
        from skimage.transform import resize
        hr_resized = np.zeros((3, *hr_size), dtype=np.float32)
        for c in range(3):
            hr_resized[c] = resize(hr[c], hr_size, order=3, preserve_range=True)
        hr = hr_resized
    
    # S2: todas las 6 bandas, resize a 16x16
    if s2.shape[0] >= 6:
        s2_data = s2[:6]
    else:
        s2_data = s2
    
    # Normalizar S2 (ya en 0-1 por percentil)
    s2_data = s2_data / 10000.0
    
    # Resize S2 a 16x16, mantener temporalidad (seq_len=1)
    from skimage.transform import resize
    s2_resized = np.zeros((6, 1, *s2_size), dtype=np.float32)
    for c in range(6):
        s2_resized[c, 0] = resize(s2_data[c], s2_size, order=3, preserve_range=True)
    
    # S1: 2 bandas (VV, VH), resize a 16x16
    s1_data = s1[:2] if s1.shape[0] >= 2 else s1
    # S1 en dB, normalizar a 0-1 aprox
    s1_data = np.clip((s1_data + 25) / 40, 0, 1)  # dB típico -25 a 15
    
    s1_resized = np.zeros((2, 1, *s1_size), dtype=np.float32)
    for c in range(2):
        s1_resized[c, 0] = resize(s1_data[c], s1_size, order=3, preserve_range=True)
    
    # Target: binario 0/1, resize a 512x512
    flood_resized = resize(flood, hr_size, order=0, preserve_range=True).astype(np.uint8)
    flood_resized = (flood_resized > 0).astype(np.uint8)
    
    return {
        'hr': hr,           # (3, 512, 512)
        's2': s2_resized,   # (6, 1, 16, 16)
        's1': s1_resized,   # (2, 1, 16, 16)
        'flood': flood_resized,  # (512, 512)
    }

def run_inference_simple(config_path, weights_path, test_split, output_dir, device='cuda'):
    """
    Inferencia simplificada usando el predictor de SkySense++.
    Nota: Requiere que el entorno SkySense++ esté configurado correctamente.
    """
    print(f"Config: {config_path}")
    print(f"Weights: {weights_path}")
    print(f"Test split: {test_split}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    
    # Verificar archivos
    for p in [config_path, weights_path, test_split]:
        if not os.path.exists(p):
            print(f"[ERROR] No encontrado: {p}")
            return False
    
    # Cargar config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Cargar test samples
    samples = load_test_list(test_split)
    print(f"Tiles de test: {len(samples)}")
    
    # Intentar importar predictor de SkySense++
    try:
        from antmmf.common.registry import registry
        from antmmf.common.report import Report
        from antmmf.structures import Sample, SampleList
        from antmmf.predictors.base_predictor import BasePredictor
        from antmmf.utils.timer import Timer
        from antmmf.predictors.build import build_predictor
        from antmmf.datasets.samplers import SequentialSampler
        from antmmf.common.build import build_config
        from antmmf.utils.flags import flags
        
        # Registrar predictor personalizado si no existe
        from lib.predictors.flood3i_1shot import OneshotPredictor
        registry.register_predictor("OneshotPredictor", OneshotPredictor)
        
        # Construir predictor
        sys.argv = ['run_inference', '--config', config_path, 'predictor_parameters.model_dir', weights_path]
        parser = flags.get_parser()
        args = parser.parse_args()
        
        # Build config
        config_obj = build_config(args)
        predictor = build_predictor(config_obj)
        
        # Load checkpoint
        predictor.load_checkpoint()
        predictor.model.to(device)
        predictor.model.eval()
        
        if config_obj.predictor_parameters.replace_speedup_op:
            from lib.utils.optim_utils import replace_speedup_op
            predictor.model = replace_speedup_op(predictor.model)
        
        print("[OK] Predictor cargado correctamente")
        
    except ImportError as e:
        print(f"[ERROR] No se pudo importar SkySense++: {e}")
        print("  Verifica que el entorno 'skysense' esté activado y las dependencias instaladas")
        return False
    except Exception as e:
        print(f"[ERROR] Error cargando predictor: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Procesar cada tile
    print("Iniciando inferencia...")
    for i, tile_info in enumerate(tqdm(samples, desc="Inferencia")):
        try:
            # Cargar y preprocesar
            tile_data = load_tile_data(tile_info)
            processed = preprocess_for_skysense(
                tile_data['s2'], tile_data['s1'], tile_data['flood'], config
            )
            
            # Convertir a tensores
            hr_tensor = torch.from_numpy(processed['hr']).unsqueeze(0).to(device)  # (1, 3, 512, 512)
            s2_tensor = torch.from_numpy(processed['s2']).unsqueeze(0).to(device)  # (1, 6, 1, 16, 16)
            s1_tensor = torch.from_numpy(processed['s1']).unsqueeze(0).to(device)  # (1, 2, 1, 16, 16)
            
            # Crear sample para predictor
            sample = Sample()
            sample.hr_img = hr_tensor
            sample.s2_img = s2_tensor
            sample.s1_img = s1_tensor
            sample.modality_idx = 7  # s2 + s1 + hr = 1+2+4=7
            sample.modality_flag_s2 = True
            sample.modality_flag_s1 = True
            sample.modality_flag_hr = True
            sample.dataset_name = 'barranquilla_flood'
            sample.task_type = 'test'
            
            # Inferencia
            with torch.no_grad():
                sample_list = SampleList([sample])
                sample_list = sample_list.to(device)
                report = predictor._forward_pass(sample_list)
                
                # Obtener logits
                logits = report.logits_hr  # (1, num_classes, H, W)
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
            
            # Guardar predicción
            pred_name = f"pred_{tile_info['s2']}"
            pred_path = os.path.join(output_dir, pred_name)
            
            with rasterio.open(os.path.join("data/processed/sentinel2", tile_info['s2'])) as src:
                profile = src.profile
            
            profile.update({
                'dtype': 'uint8',
                'count': 1,
                'nodata': 0,
                'compress': 'deflate'
            })
            
            with rasterio.open(pred_path, 'w', **profile) as dst:
                dst.write(pred.astype(np.uint8)[np.newaxis, :, :])
            
        except Exception as e:
            print(f"[ERROR] Tile {tile_info['s2']}: {e}")
            continue
    
    print(f"[OK] Inferencia completada. Predicciones en: {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="SkySense++ 1-shot inference Barranquilla")
    parser.add_argument('--config', default=str(DEFAULT_CONFIG), help='Config YAML path')
    parser.add_argument('--weights', default=str(DEFAULT_WEIGHTS), help='Model weights path')
    parser.add_argument('--split', default=TEST_SPLIT, help='Test split file')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    success = run_inference_simple(args.config, args.weights, args.split, args.output, args.device)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()