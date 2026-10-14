# SkySense-Flood-Barranquilla

Detección de cambios en inundaciones en los barrios **Don Juan** y **El Salado** de **Barranquilla, Colombia**, durante el período **2020-2025**, utilizando el modelo fundacional **SkySense++** (Wu et al., 2025, *Nature Machine Intelligence*) en configuración **1-shot** sobre datos **Sentinel-1 (SAR)** y **Sentinel-2 (óptico)** via Google Earth Engine.

Este proyecto evalúa la capacidad de transferencia de un foundation model multi-modal pre-entrenado para Earth Observation a un contexto urbano costero tropical, proporcionando métricas desagregadas por barrio, año y tipo de sensor siguiendo el framework de **Model Cards** (Mitchell et al., 2019).

## 📄 Fichas del Proyecto
- **Model Card**: [MODEL_CARD.md](./MODEL_CARD.md)
- **Dataset Card**: [DATASET_CARD.md](./DATASET_CARD.md)

## 🎯 Resultados Principales (Pendientes - ejecutar pipeline completo)
| Métrica | Don Juan | El Salado | Global |
|---------|----------|-----------|--------|
| IoU (Inundación) | [XX.XX] | [XX.XX] | [XX.XX] |
| F1-Score | [XX.XX] | [XX.XX] | [XX.XX] |
| Precision | [XX.XX] | [XX.XX] | [XX.XX] |
| Recall | [XX.XX] | [XX.XX] | [XX.XX] |

> **Nota**: Los valores se completarán tras ejecutar `scripts/07_compute_metrics.py` con la inferencia de SkySense++.

## 🗂️ Estructura del Proyecto
```
ProyectoIA/
├── MODEL_CARD.md              # Ficha del modelo (Mitchell et al. 2019 + HF template)
├── DATASET_CARD.md            # Ficha del dataset (HF template)
├── README.md                  # Este archivo
├── .gitignore
├── config.yaml                # Configuración central
├── requirements.txt           # Dependencias Python
├── data/
│   └── aoi/
│       ├── barranquilla_flood_aoi.geojson
│       └── barranquilla_flood_aoi_buffered.geojson
├── scripts/
│   ├── 01_create_aoi.py              # ✓ Crear AOI desde OpenStreetMap
│   ├── 02_download_gee.py            # Descargar Sentinel-1/2 via GEE
│   ├── 03_create_flood_masks.py      # NDWI/MNDWI/AWEI + umbralización
│   ├── 04_prepare_tiles.py           # Tiling 512×512 + splits train/val/test
│   ├── 05_setup_skysense.py          # Setup entorno SkySense++ + pesos
│   ├── 06_run_inference.py           # Inferencia 1-shot SkySense++
│   ├── 07_compute_metrics.py         # IoU, F1, Precision, Recall desagregados
│   └── 08_generate_cards.py          # Generar fichas con datos reales
├── notebooks/
│   ├── exploratory_analysis.ipynb    # EDA datos descargados
│   └── results_visualization.ipynb   # Visualización resultados
└── outputs/
    ├── predictions/                   # Predicciones modelo
    ├── metrics/                       # CSVs métricas desagregadas
    └── figures/                       # Gráficas para Model Card
```

## ⚙️ Pipeline de Ejecución
```bash
# 1. Crear AOI (ya ejecutado)
python scripts/01_create_aoi.py

# 2. Configurar GEE_PROJECT_ID en scripts/02_download_gee.py y descargar datos
python scripts/02_download_gee.py

# 3. Crear máscaras de inundación
python scripts/03_create_flood_masks.py

# 4. Preparar tiles y splits
python scripts/04_prepare_tiles.py

# 5. Setup SkySense++ (instalar dependencias, descargar pesos)
python scripts/05_setup_skysense.py

# 6. Ejecutar inferencia 1-shot
python scripts/06_run_inference.py

# 7. Calcular métricas desagregadas
python scripts/07_compute_metrics.py

# 8. Actualizar Model Card y Dataset Card con resultados reales
python scripts/08_generate_cards.py
```

## 📋 Requisitos
- **Python**: 3.10+
- **GPU**: NVIDIA con 8GB+ VRAM (recomendado 16GB+)
- **CUDA**: 11.7+
- **Google Cloud Project** con Earth Engine API habilitado
- **Cuenta Google** para autenticación GEE (`earthengine authenticate`)

### Dependencias Principales
```bash
# Entorno base
pip install earthengine-api geemap geopandas rasterio scikit-image

# SkySense++ (requiere conda para GDAL/mmcv)
conda create -n skysense python=3.8
conda activate skysense
conda install -c conda-forge gdal libpysal esda
pip install torch==1.13.1 torchvision==0.14.1
mim install mmcv-full==1.7.1
pip install mmsegmentation==0.30.0 mmcls==0.25.0 timm==0.6.13
```

## 📚 Referencias
- Mitchell, M., et al. (2019). **Model Cards for Model Reporting**. *FAT* 2019. https://arxiv.org/abs/1810.03993
- Wu, K., et al. (2025). **A semantic-enhanced multi-modal remote sensing foundation model for Earth observation**. *Nature Machine Intelligence*. DOI: 10.1038/s42256-025-01078-8
- Hugging Face Model Card Template: https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub/templates
- SkySense++ Repository: https://github.com/kang-wu/SkySensePlusPlus

## 👥 Autores
- **Geraldyne Lamy Correcha** — Estudiante Ingeniería Catastral y Geodesia, IA Aplicada al Territorio, Universidad Distrital Francisco José de Caldas 
- **Email**: gerallamy06@gmail.com
- **GitHub**: [@gerallamy06-cmd](https://github.com/gerallamy06-cmd)

## 📄 Licencia
- **Código**: MIT License
- **Model Card / Dataset Card**: CC-BY-4.0
- **Modelo SkySense++**: Solo investigación no comercial (ver licencia original)
- **Datos Sentinel**: Copernicus/ESA - Abiertos y gratuitos
