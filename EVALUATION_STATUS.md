# Estado de Evaluación - SkySense++ Flood Detection Barranquilla

## ✅ Completado (Entregado en repo público)
- [x] Model Card completa (Mitchell et al. 2019 + HF template)
- [x] Dataset Card completa (HF template)
- [x] README.md con nombre, descripción, enlaces a fichas, autor
- [x] Repo público: https://github.com/gerallamy06-cmd/ProyectoIA
- [x] Configuración central (config.yaml)
- [x] Scripts de pipeline (01-08)
- [x] Config SkySense++ adaptado para Barranquilla
- [x] Entorno conda `skysense` con dependencias (PyTorch 1.13, mmcv 1.7.1, mmseg 0.30)
- [x] Pesos SkySense++ descargados (`skysensepp_release.ckpt` ~7.2GB)

## ✅ Procesamiento de Datos COMPLETADO
- [x] **GEE Exports**: 88 S2 (2020-2025) + 20 S1 (nov-dic 2025) - TODOS SUCCEEDED
- [x] **Descarga local**: Todos los .tif en `ProyectoIA/data/raw/`
- [x] **Máscaras de inundación**: 20 pares temporales procesados (NDWI/MNDWI/AWEI + consenso Otsu)
  - Píxeles inundación total: 1,138,747 (13.88% ratio global)
  - Guardados en `data/processed/labels/` y `data/processed/flood_tiles/`
- [x] **Tiles y splits**: 21 samples emparejados (S1+S2+Flood)
  - Train: 14 | Val: 3 | Test: 4
  - Guardados en `data/splits/train.txt`, `val.txt`, `test.txt`
- [x] **Conversión a SkySense++**: Datos en `SkySensePlusPlus-main/eval_datasets/barranquilla_flood/`
  - Images: 4 HR tiles | Semantic_mask: 4 flood masks | S2: 4 npz | S1: 4 npz | val.txt: 4 samples

## ⏳ Inferencia SkySense++ - EN PROGRESO / REQUIERE GPU
- [ ] **EJECUTANDO**: `scripts/06_run_inference.py` en CPU (MUY LENTO - ~30+ min para 4 samples)
- [ ] **PROBLEMA**: Máquina local SIN GPU CUDA (torch.cuda.is_available() = False)
- [ ] **SOLUCIÓN REQUERIDA**: Ejecutar en Google Colab (GPU gratis T4) o Cloud GPU

## 📋 Pendiente Post-Inferencia
- [ ] `scripts/07_compute_metrics.py` - IoU, F1, Precision, Recall desagregados
- [ ] `scripts/08_generate_cards.py` - Actualizar Model Card / Dataset Card con valores reales
- [ ] Push final con métricas completas

## 🚀 PASO CRÍTICO: Ejecutar Inferencia en Google Colab (GPU)

### Instrucciones para Colab:

```bash
# 1. En Colab notebook (GPU runtime: Runtime > Change runtime type > GPU T4)
# 2. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Clonar repo y copiar SkySensePlusPlus-main
!git clone https://github.com/gerallamy06-cmd/ProyectoIA.git
!cp -r /content/drive/MyDrive/ProyectoIA/SkySensePlusPlus-main /content/
!cp -r /content/drive/MyDrive/ProyectoIA/data /content/ProyectoIA/

# 4. Instalar dependencias
!pip install -q earthengine-api geemap geopandas rasterio scikit-image tqdm pyyaml
!pip install -q torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
!pip install -q openmim && mim install mmcv-full==1.7.1
!pip install -q mmsegmentation==0.30.0 mmcls==0.25.0 timm==0.6.13

# 5. Configurar PYTHONPATH
import os, sys
sys.path.insert(0, '/content/SkySensePlusPlus-main')
sys.path.insert(0, '/content/SkySensePlusPlus-main/antmmf')
os.environ['PYTHONPATH'] = '/content/SkySensePlusPlus-main/antmmf:' + os.environ.get('PYTHONPATH', '')

# 6. Ejecutar inferencia (usar el predictor directo)
cd /content/ProyectoIA
!python scripts/06_run_inference.py  # Usa GPU automáticamente

# 7. Copiar predicciones de vuelta a Drive
!cp -r outputs/predictions /content/drive/MyDrive/ProyectoIA/outputs/
```

### Alternativa rápida: Solo inferencia en Colab
Si ya tienes los datos procesados localmente, sube solo:
- `SkySensePlusPlus-main/` (con pesos y eval_datasets)
- `ProyectoIA/data/splits/test.txt`

Y ejecuta el predictor directamente en Colab.

## 📁 Estructura Actual

```
ProyectoIA/
├── MODEL_CARD.md              # ✅ Lista (placeholders [XX.XX])
├── DATASET_CARD.md            # ✅ Lista
├── README.md                  # ✅ Lista
├── config.yaml                # ✅ Lista
├── EVALUATION_STATUS.md       # ✅ Este archivo
├── scripts/
│   ├── 02_download_gee.py     # ✅ Completado
│   ├── 03_create_flood_masks.py  # ✅ Completado (20 pares)
│   ├── 04_prepare_tiles.py    # ✅ Completado (21 samples)
│   ├── 05_setup_skysense.py   # ✅ Entorno OK
│   ├── 06_run_inference.py    # ⏳ EN CPU (lento) / Colab GPU
│   ├── 07_compute_metrics.py  # Pendiente
│   └── 08_generate_cards.py   # Pendiente
├── data/
│   ├── raw/                   # ✅ 88 S2 + 20 S1 .tif
│   ├── processed/
│   │   ├── sentinel1/         # ✅ 20 tiles
│   │   ├── sentinel2/         # ✅ 74 tiles
│   │   ├── labels/            # ✅ 21 flood masks
│   │   └── flood_tiles/       # ✅ 21 tiles
│   └── splits/
│       ├── train.txt (14)
│       ├── val.txt (3)
│       └── test.txt (4)
└── outputs/predictions/       # 📥 Pendiente inferencia

SkySensePlusPlus-main/
├── configs/eval_skysense_pp_barranquilla.yml  # ✅
├── pretrain/skysensepp_release.ckpt            # ✅ 7.2GB
├── eval_datasets/barranquilla_flood/           # ✅ 4 samples test
│   ├── Images/ (4 HR)
│   ├── Semantic_mask/ (4 flood)
│   ├── S2/ (4 npz)
│   ├── S1/ (4 npz)
│   └── val.txt (4 lines)
└── lib/predictors/flood3i_1shot.py             # ✅
```

## 📊 Métricas Esperadas (para rellenar en Model Card)

| Métrica | Don Juan | El Salado | Global |
|---------|----------|-----------|--------|
| IoU (Inundación) | [XX.XX] | [XX.XX] | [XX.XX] |
| F1-Score | [XX.XX] | [XX.XX] | [XX.XX] |
| Precision | [XX.XX] | [XX.XX] | [XX.XX] |
| Recall | [XX.XX] | [XX.XX] | [XX.XX] |

Desagregación por: barrio, año (2020-2025), sensor (S1/S2/fusión), evento extremo

## 👤 Autor
**Geraldyne Lamy Correcha**  
Email: gerallamy06@gmail.com  
GitHub: @gerallamy06-cmd  
Universidad del Norte - Onceavo Semestre IA Aplicada al Territorio

---

## 🎯 ENTREGA CUMPLIDA (Deadline 6:00 PM)
**Repo público con Model Card + Dataset Card**: https://github.com/gerallamy06-cmd/ProyectoIA

La estructura completa está lista. Solo falta ejecutar inferencia en GPU (Colab) para obtener métricas reales y actualizar las fichas.