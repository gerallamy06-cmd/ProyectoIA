#!/usr/bin/env python3
"""
Ejecutar inferencia 1-shot SkySense++ usando el predictor directo (como run_1shot.sh)
"""
import os
import sys
import subprocess
from pathlib import Path

# Configuración
SKYSENSE_DIR = Path(r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main")
CONFIG_PATH = SKYSENSE_DIR / "configs" / "eval_skysense_pp_barranquilla.yml"
WEIGHTS_PATH = SKYSENSE_DIR / "pretrain" / "skysensepp_release.ckpt"
DATASET_NAME = "barranquilla_flood"
SAVE_DIR = SKYSENSE_DIR / "eval" / f"{DATASET_NAME}_1shot" / "save"
GT_DIR = SKYSENSE_DIR / "eval_datasets" / DATASET_NAME / "Semantic_mask"
GT_LIST_PATH = SKYSENSE_DIR / "eval_datasets" / DATASET_NAME / "val.txt"

os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("INFERENCIA 1-SHOT SKYSENSE++ - BARRANQUILLA")
    print("=" * 60)
    print(f"Config: {CONFIG_PATH}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Save dir: {SAVE_DIR}")
    print(f"GT dir: {GT_DIR}")
    print(f"GT list: {GT_LIST_PATH}")
    
    # Verificar archivos
    for p in [CONFIG_PATH, WEIGHTS_PATH, GT_LIST_PATH]:
        if not p.exists():
            print(f"[ERROR] No encontrado: {p}")
            return False
    
    # Verificar GT dir
    if not GT_DIR.exists():
        print(f"[ERROR] GT dir no existe: {GT_DIR}")
        return False
    
    # Ejecutar predictor
    print("\nEjecutando predictor...")
    cmd = [
        sys.executable,
        str(SKYSENSE_DIR / "lib" / "predictors" / "flood3i_1shot.py"),
        "--model_path", str(WEIGHTS_PATH),
        "--config", str(CONFIG_PATH),
        "--save_dir", str(SAVE_DIR),
        "--seed", "0"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SKYSENSE_DIR) + os.pathsep + str(SKYSENSE_DIR / "antmmf") + os.pathsep + env.get("PYTHONPATH", "")
    
    print(f"Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SKYSENSE_DIR), env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Predictor falló:")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("[OK] Predictor completado")
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    
    # Copiar predicciones a outputs/predictions
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_files = list(SAVE_DIR.glob("*.png"))
    print(f"\nCopiando {len(pred_files)} predicciones a {output_dir}...")
    for pf in pred_files:
        dest = output_dir / pf.name
        import shutil
        shutil.copy2(pf, dest)
    
    print(f"[OK] Predicciones copiadas a {output_dir}")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)