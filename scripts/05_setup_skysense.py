#!/usr/bin/env python3
"""
Setup entorno SkySense++ para inferencia 1-shot en Windows.
Incluye instalación de dependencias y descarga de pesos.
"""
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

# Rutas
SKYSENSE_DIR = Path(r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main")
PRETRAIN_DIR = SKYSENSE_DIR / "pretrain"
WEIGHTS_URL = "https://www.notion.so/SkySense-Checkpoints-a7fcff6ce29a4647a08c7fe416910509"  # Página Notion, no enlace directo

def run_cmd(cmd, cwd=None, shell=True):
    """Ejecutar comando y mostrar output"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, cwd=cwd, shell=shell, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0

def check_conda_env():
    """Verificar si existe entorno conda skysense"""
    result = subprocess.run("conda env list", shell=True, capture_output=True, text=True)
    return "skysense" in result.stdout

def create_conda_env():
    """Crear entorno conda con Python 3.8"""
    print("Creando entorno conda 'skysense' con Python 3.8...")
    return run_cmd("conda create -n skysense python=3.8 -y")

def install_pytorch():
    """Instalar PyTorch compatible"""
    print("Instalando PyTorch 1.13.1 + CUDA 11.7...")
    cmd = "conda activate skysense && pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117"
    return run_cmd(cmd)

def install_mmcv():
    """Instalar mmcv-full via mim (recomendado en Windows)"""
    print("Instalando mmcv-full via mim...")
    cmds = [
        "conda activate skysense && pip install -U openmim",
        "conda activate skysense && mim install mmcv-full==1.7.1"
    ]
    for cmd in cmds:
        if not run_cmd(cmd):
            return False
    return True

def install_mmseg():
    """Instalar mmsegmentation y mmcls"""
    print("Instalando mmsegmentation, mmcls, timm...")
    cmd = "conda activate skysense && pip install mmsegmentation==0.30.0 mmcls==0.25.0 timm==0.6.13"
    return run_cmd(cmd)

def install_gdal():
    """Instalar GDAL via conda (requerido en Windows)"""
    print("Instalando GDAL, libpysal, esda via conda-forge...")
    cmd = "conda activate skysense && conda install -c conda-forge gdal libpysal esda -y"
    return run_cmd(cmd)

def install_other_deps():
    """Otras dependencias"""
    print("Instalando otras dependencias...")
    cmd = "conda activate skysense && pip install scikit-image==0.19.3 gdal==3.4.0 numpy<2.0.0"
    return run_cmd(cmd)

def clone_antmmf():
    """Clonar AntMMF framework"""
    antmmf_dir = SKYSENSE_DIR / "antmmf"
    if antmmf_dir.exists():
        print("antmmf ya existe, actualizando...")
        return run_cmd("git pull", cwd=antmmf_dir)
    else:
        print("Clonando AntMMF...")
        return run_cmd("git clone https://github.com/alipay/Ant-Multi-Modal-Framework.git antmmf", cwd=SKYSENSE_DIR)

def download_weights():
    """Descargar pesos pre-entrenados"""
    print("=" * 60)
    print("DESCARGA DE PESOS SKYSENSE++")
    print("=" * 60)
    print(f"Directorio destino: {PRETRAIN_DIR}")
    print(f"Página de pesos: {WEIGHTS_URL}")
    print()
    print("IMPORTANTE: Los pesos NO se pueden descargar directamente via script.")
    print("Debes:")
    print("1. Abrir en navegador:", WEIGHTS_URL)
    print("2. Buscar 'skysensepp_release.ckpt' o similar")
    print("3. Descargar y colocar en:", PRETRAIN_DIR)
    print()
    
    # Verificar si ya existen
    weight_files = list(PRETRAIN_DIR.glob("*.ckpt")) + list(PRETRAIN_DIR.glob("*.pth"))
    if weight_files:
        print(f"[OK] Pesos encontrados: {[f.name for f in weight_files]}")
        return True
    else:
        print("[WARN] No hay pesos en pretrain/")
        print("  Por favor descarga manualmente y re-ejecuta este script")
        return False

def verify_installation():
    """Verificar que el entorno funciona"""
    print("Verificando instalación...")
    cmd = "conda activate skysense && python -c \"import torch; import mmcv; import mmseg; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('mmcv:', mmcv.__version__); print('mmseg:', mmseg.__version__)\""
    return run_cmd(cmd)

def create_config_barranquilla():
    """Crear config adaptado para Barranquilla basado en flood3i"""
    config_src = SKYSENSE_DIR / "configs" / "eval_skysense_pp_flood3i.yml"
    config_dst = SKYSENSE_DIR / "configs" / "eval_skysense_pp_barranquilla.yml"
    
    if not config_src.exists():
        print(f"[ERROR] Config base no encontrado: {config_src}")
        return False
    
    # Leer y adaptar
    import yaml
    with open(config_src) as f:
        config = yaml.safe_load(f)
    
    # Adaptaciones para Barranquilla
    config['task_attributes']['segmentation']['dataset_attributes']['few_shot_flood_segmentation'].update({
        'data_root_dir': 'eval_datasets/barranquilla_flood',
        'data_txt': 'eval_datasets/barranquilla_flood/val.txt',
        'img_dir': 'eval_datasets/barranquilla_flood/Images',
        'tgt_dir': 'eval_datasets/barranquilla_flood/Semantic_mask',
        # Ajustar tamaños según nuestros tiles (512x512)
        'image_size': {
            'hr': (512, 512),
            's2': (16, 16),
            's1': (16, 16),
            'anno': (512, 512)
        }
    })
    
    # Cambiar número de clases a 2 (fondo, inundación)
    config['model_attributes']['SkySensePP']['rec_head_hr']['num_classes'] = 2
    
    with open(config_dst, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"[OK] Config creado: {config_dst}")
    return True

def main():
    print("=" * 60)
    print("SETUP SKYSENSE++ PARA INFERENCIA 1-SHOT BARRANQUILLA")
    print("=" * 60)
    
    steps = [
        ("Verificar/crear entorno conda", check_conda_env, create_conda_env),
        ("Instalar PyTorch", None, install_pytorch),
        ("Instalar mmcv-full", None, install_mmcv),
        ("Instalar mmsegmentation", None, install_mmseg),
        ("Instalar GDAL + geo deps", None, install_gdal),
        ("Instalar otras deps", None, install_other_deps),
        ("Clonar AntMMF", None, clone_antmmf),
        ("Descargar pesos", None, download_weights),
        ("Crear config Barranquilla", None, create_config_barranquilla),
        ("Verificar instalación", None, verify_installation),
    ]
    
    for name, check_fn, action_fn in steps:
        print(f"\n--- {name} ---")
        if check_fn and check_fn():
            print(f"[SKIP] {name} ya completado")
            continue
        if not action_fn():
            print(f"[FAIL] {name} falló. Revisar errores arriba.")
            return False
    
    print("\n" + "=" * 60)
    print("[OK] SETUP COMPLETADO")
    print("=" * 60)
    print("Próximos pasos:")
    print("1. Activar entorno: conda activate skysense")
    print("2. Preparar datos en formato SkySense++ (eval_datasets/barranquilla_flood/)")
    print("3. Ejecutar inferencia: python scripts/06_run_inference.py")

if __name__ == "__main__":
    main()