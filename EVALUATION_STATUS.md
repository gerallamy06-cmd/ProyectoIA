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

## ⏳ En Progreso / Pendiente

### 1. Datos Sentinel-1/2 (GEE → Google Drive → Local)
- [x] Export tasks enviados a Google Drive (20 S2 + 20 S1 imágenes, fechas recientes)
- [ ] **ESPERAR**: Completar exports en GEE (10-30 min)
- [ ] **DESCARGAR**: De Google Drive a `ProyectoIA/data/raw/`
- [ ] **NOTA**: Solo últimas 20 imágenes. Para 2020-2023 (La Niña, Iota) se necesitan más exports históricos

### 2. Procesamiento de Datos
- [ ] `scripts/03_create_flood_masks.py` - Crear máscaras inundación (NDWI/MNDWI/AWEI + consenso)
- [ ] `scripts/04_prepare_tiles.py` - Tiling 512×512 + splits train/val/test
- [ ] Convertir formato ProyectoIA → SkySense++ eval_datasets/

### 3. Inferencia SkySense++
- [ ] **REQUIERE GPU** (local: CUDA=False, sin GPU NVIDIA detectada)
- [ ] Opción A: Google Colab (GPU gratis T4 / Pro A100)
- [ ] Opción B: Cloud GPU (AWS/GCP/Azure)
- [ ] `scripts/06_run_inference.py` - Inferencia 1-shot

### 4. Métricas y Actualización
- [ ] `scripts/07_compute_metrics.py` - IoU, F1, Precision, Recall desagregados
- [ ] `scripts/08_generate_cards.py` - Actualizar Model Card / Dataset Card con valores reales
- [ ] Push final con métricas completas

## 🚀 Próximos Pasos Recomendados

### Para completar evaluación (requiere GPU):
1. **Descargar datos de Google Drive** a `ProyectoIA/data/raw/`
2. **Ejecutar pipeline local** (CPU para preprocesamiento):
   ```bash
   python scripts/03_create_flood_masks.py
   python scripts/04_prepare_tiles.py
   ```
3. **Subir a Google Colab** con GPU:
   - Subir carpeta `ProyectoIA/` a Google Drive
   - Abrir notebook en Colab
   - `!pip install earthengine-api geemap ...` (deps base)
   - `!conda create -n skysense python=3.8 && conda activate skysense`
   - `!conda install -c conda-forge gdal ...`
   - `!pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117`
   - `!mim install mmcv-full==1.7.1`
   - `!pip install mmsegmentation==0.30.0 mmcls==0.25.0 timm==0.6.13`
   - Copiar `SkySensePlusPlus-main/` a Colab
   - Ejecutar `scripts/06_run_inference.py` en Colab (GPU)
   - Descargar predicciones
4. **Local**: `scripts/07_compute_metrics.py` → `scripts/08_generate_cards.py` → push

### Para exportar más datos históricos (2020-2023):
Modificar `scripts/02_download_gee.py`:
```python
# Cambiar max_images=20 a max_images=100 o más
# O filtrar por fechas específicas de eventos:
# - La Niña 2020-2022: sep 2020 - mar 2022
# - Huracán Iota: nov 2020
# - Lluvias extremas 2022: may-jun 2022
# - Lluvias extremas 2024: abr-may 2024
```

## 📁 Estructura de Archivos Clave

```
ProyectoIA/
├── MODEL_CARD.md              # ✅ Lista (placeholders [XX.XX] en métricas)
├── DATASET_CARD.md            # ✅ Lista
├── README.md                  # ✅ Lista
├── config.yaml                # ✅ Lista
├── scripts/
│   ├── 02_download_gee.py     # ✅ Exports enviados a GDrive
│   ├── 03_create_flood_masks.py
│   ├── 04_prepare_tiles.py
│   ├── 05_setup_skysense.py   # ✅ Entorno OK, config creado
│   ├── 06_run_inference.py    # Listo (requiere GPU)
│   ├── 07_compute_metrics.py
│   └── 08_generate_cards.py
└── data/
    ├── raw/                   # 📥 Pendiente: descargar de GDrive
    ├── processed/             # 📥 Pendiente
    └── splits/                # 📥 Pendiente

SkySensePlusPlus-main/
├── configs/
│   ├── eval_skysense_pp_flood3i.yml
│   └── eval_skysense_pp_barranquilla.yml  # ✅ Creado
├── pretrain/
│   └── skysensepp_release.ckpt             # ✅ 7.2GB descargado
├── eval_datasets/
│   └── barranquilla_flood/                 # ✅ Estructura creada
│       ├── Images/
│       ├── Semantic_mask/
│       ├── S2/
│       └── S1/
└── lib/predictors/flood3i_1shot.py         # ✅ Predictor disponible
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