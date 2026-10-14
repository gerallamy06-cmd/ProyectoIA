---
tags:
  - earth-observation
  - flood-monitoring
  - sentinel-1
  - sentinel-2
  - barranquilla
  - colombia
  - time-series
  - change-detection
license: cc-by-4.0
dataset_name: Barranquilla-Flood-2020-2025
---

# Dataset Card: Datos de Inundación Barranquilla (Don Juan y El Salado) 2020-2025

## Resumen del Dataset
Serie temporal de imágenes **Sentinel-1 (SAR, VV+VH)** y **Sentinel-2 (L2A, 6 bandas: B2,B3,B4,B8,B11,B12)** para los barrios **Don Juan** y **El Salado**, **Barranquilla, Colombia**, durante el período **2020-2025**, con máscaras de cambio de inundación generadas semi-automáticamente mediante índices espectrales y validación visual experta.

Diseñado para evaluación de foundation models de Earth Observation (SkySense++) en detección de cambios de inundación urbana tropical.

## Detalles del Dataset
- **Curado por**: Geraldyne Lamy Correcha - Universidad del Norte
- **Financiado por**: Proyecto académico - IA Aplicada al Territorio (Onceavo Semestre)
- **Compartido por**: Geraldyne Lamy Correcha / @gerallamy06-cmd
- **Licencia**: CC-BY-4.0 (datos derivados) / Datos fuente: Copernicus/ESA (abiertos)
- **Idioma**: N/A (datos geoespaciales ráster)
- **DOI**: Pendiente / Repositorio GitHub

## Fuentes del Dataset (Dataset Sources)
- **Repositorio**: https://github.com/gerallamy06-cmd/ProyectoIA
- **Fuentes primarias**:
  - **Sentinel-2 L2A**: `COPERNICUS/S2_SR_HARMONIZED` (Google Earth Engine)
    - Bandas: B2(Blue), B3(Green), B4(Red), B8(NIR), B11(SWIR1), B12(SWIR2)
    - Resolución: 10m (B2,B3,B4,B8) / 20m (B11,B12) → reescalado a 10m
    - Período: 2020-01-01 a 2025-12-31
    - Filtro nubes: <30% (QA60 cloud mask)
  - **Sentinel-1 GRD**: `COPERNICUS/S1_GRD` (Google Earth Engine)
    - Polarización: VV + VH (IW mode)
    - Resolución: 10m
    - Período: 2020-01-01 a 2025-12-31
    - Procesamiento: Calibración, corrección terreno (SRTM), filtro Lee 5×5
- **AOI (Área de Interés)**: Barrios Don Juan y El Salado, Barranquilla
  - Fuente: OpenStreetMap + coordenadas aproximadas validadas
  - Archivo: `data/aoi/barranquilla_flood_aoi.geojson`
  - CRS: EPSG:4326 (WGS84)

## Usos (Uses)
### Uso Directo (Direct Use)
- Entrenamiento/evaluación de modelos de detección de cambios de inundación (change detection)
- Análisis de series temporales de cuerpos de agua urbanos en Caribe colombiano
- Investigación de variabilidad hidrológica extrema (La Niña, huracanes) en zona urbana
- Validación de foundation models EO (SkySense++, Prithvi, SatMAE, etc.) en dominio tropical
- Apoyo a gestión de riesgo de desastres y planificación urbana Barranquilla

### Fuera de Alcance (Out-of-Scope)
- Navegación marítima / operacional en tiempo real
- Análisis de calidad de agua (requiere bandas adicionales / in-situ)
- Modelado hidrológico detallado (requiere DEM de alta resolución, drenaje)
- Zonas fuera de los 2 barrios definidos sin validación adicional

## Estructura del Dataset (Dataset Structure)
```
data/
├── aoi/
│   ├── barranquilla_flood_aoi.geojson           # Límites barrios (EPSG:4326)
│   └── barranquilla_flood_aoi_buffered.geojson  # Con buffer 200m
├── raw/                                          # Descargas crudas GEE (gitignored)
│   ├── S2_Barranquilla_YYYY-MM-DD.tif
│   └── S1_Barranquilla_YYYY-MM-DD.tif
├── processed/                                    # Tiles listos para modelo
│   ├── sentinel1/                                # VV, VH por fecha (512×512)
│   ├── sentinel2/                                # B2,B3,B4,B8,B11,B12 por fecha
│   └── labels/                                   # Máscaras inundación (0/1)
├── splits/
│   ├── train.txt                                 # 70% tiles
│   ├── val.txt                                   # 15% tiles
│   └── test.txt                                  # 15% tiles
└── metadata/
    ├── acquisition_dates.csv                     # Fechas, nubosidad, órbita
    ├── flood_events.csv                          # Eventos conocidos (La Niña, Iota, etc.)
    └── processing_log.json                       # Parámetros preprocesamiento
```

### Formato de Datos Procesados (para SkySense++)
| Modalidad | Canales | Shape (por tile) | Tipo | Normalización |
|-----------|---------|------------------|------|---------------|
| **HR (S2 RGB+NIR)** | 4 | 512×512 | uint16/float32 | Percentil 2-98 → [0,1] |
| **S2 (6 bandas)** | 6 | T×16×16* | float32 | Idem + estandarización |
| **S1 (VV+VH)** | 2 | T×16×16* | float32 | dB → [0,1] (sigmoid) |
| **Target (inundación)** | 1 | 512×512 | uint8 | 0: no-inundación, 1: inundación |

*T = pasos temporales (seq_len=1 en config 1-shot, expandible)

### Archivo de lista (ej. `test.txt`)
```text
tile_001_s2_20200515_s1_20200514.tif 1
tile_002_s2_20211022_s1_20211021.tif 1
...
```
Formato: `nombre_archivo clase` (clase=1 para inundación en 1-shot)

## Creación del Dataset (Dataset Creation)
### Rationale (Curation Rationale)
- **Problema**: Barranquilla sufre inundaciones recurrentes por desbordamiento río Magdalena + lluvias extremas + marea
- **Brecha**: No existe dataset público de inundación urbana validado para Barranquilla
- **Barrios objetivo**: Don Juan (nororiente, denso, Vía 40) y El Salado (suroriente, Ciénaga Mallorquín) son altamente vulnerables
- **Período**: 2020-2025 cubre La Niña 2020-2022, Huracán Iota (nov-2020), eventos extremos 2022, 2024

### Recolección y Procesamiento (Data Collection and Processing)
1. **Búsqueda STAC/GEE**: Filtrado por AOI, fechas 2020-2025, nubosidad <30% (S2)
2. **Descarga**: Google Earth Engine → Google Drive → Local (GeoTIFF COG)
3. **Preprocesamiento Sentinel-1**:
   - Calibración radiométrica (sigma-nought)
   - Corrección terreno (Range-Doppler + SRTM 30m)
   - Filtro Lee adaptativo 5×5 (reducción speckle)
   - Co-registro a grilla S2 (10m, EPSG:4326)
   - Conversión a dB: `10*log10(sigma0)`
4. **Preprocesamiento Sentinel-2**:
   - Máscara nubes/cirros (QA60 bits 10,11)
   - Reescalado bandas 20m→10m (bilineal)
   - Recorte a AOI + tiling 512×512 overlap 64px
   - Normalización robusta: percentil 2-98 por banda por tile
5. **Fusión S1-S2**: Stack temporal `[S1_VV, S1_VH, S2_B2, B3, B4, B8, B11, B12]` → 8 canales
6. **Etiquetado (Ground Truth)**:
   - **Índices espectrales** (por tile S2):
     - NDWI = (Green - NIR) / (Green + NIR)
     - MNDWI = (Green - SWIR1) / (Green + SWIR1)
     - AWEI_sh = 4×(Green - SWIR1) - (0.25×NIR + 2.75×SWIR2)
   - **Umbralización**: Otsu automático por índice + consenso (≥2/3 índices)
   - **Detección cambio**: `inundación = agua_t2 AND NOT agua_t1` (pares pre/post evento)
   - **Validación visual**: Revisión experta 100% tiles test, 20% train/val
   - **Limpieza morfológica**: Opening 3×3 + Closing 5×5 (eliminar ruido)

### Productores de Datos Fuente (Source Data Producers)
- **ESA/Copernicus**: Proveedores datos Sentinel (política datos abiertos y gratuitos)
- **Google Earth Engine**: Plataforma hosting y API STAC
- **Microsoft Planetary Computer**: Alternativa acceso STAC (no usada en versión final)
- **IDEAM/IGAC/Alcaldía Barranquilla**: Datos hidrológicos/terreno referencia (no integrados directamente)

## Anotaciones (Annotations)
- **Proceso**: Semi-automático (índices espectrales) + validación manual experta
- **Herramientas**: Python (rasterio, geopandas, scikit-image), QGIS (validación visual)
- **Acuerdo inter-anotador**: Cohen's Kappa = [PENDIENTE - calcular en validación]
- **Clases**: Binaria (0: no-cambio/no-agua, 1: nueva inundación)
- **Información sensible**: No aplica

## Sesgos, Riesgos y Limitaciones (Bias, Risks, and Limitations)
| Sesgo/Limitación | Descripción | Impacto | Mitigación |
|------------------|-------------|---------|------------|
| **Nubosidad S2** | Caribe: >60% días nublados → gaps temporales grandes | Pérdida eventos, sesgo estación seca | Usar S1 como principal; fusar cuando S2 disponible |
| **Desbalance clases** | Inundación típicamente <5% píxeles | Métricas engañosas (accuracy alta) | Usar IoU, F1, Precision/Recall; weighted loss |
| **Cobertura espacial** | Solo 2 barrios (no toda Barranquilla) | No generalizable a otras zonas | Documentar explícitamente; ampliar en v2 |
| **Ground truth espectral** | Basado en índices, no in-situ / LiDAR | Errores en sombra, edificios, suelo húmedo | Validación visual; reportar incertidumbre |
| **Co-registro S1-S2** | Errores sub-píxel (~0.5-1 px) | Ruido en fusión, bordes difusos | Métricas por sensor separado + fusión |
| **Eventos extremos** | Sobrerepresentados en etiquetado | Sesgo hacia eventos grandes | Incluir períodos "normales" en test |
| **Definición inundación** | Binaria (sí/no) vs gradiente (profundidad) | Pérdida información severidad | v2: multi-clase (profundidad/velocidad) |

## Cita (Citation)
```bibtex
@misc{barranquilla_flood_2025,
  author = {[Tu Nombre]},
  title = {Barranquilla Flood Dataset: Don Juan y El Salado 2020-2025 (Sentinel-1/2)},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/gerallamy06-cmd/ProyectoIA},
  note = {CC-BY-4.0. Datos fuente: Copernicus/ESA via Google Earth Engine}
}
```

## Autores de la Dataset Card (Dataset Card Authors)
- **Nombre**: Geraldyne Lamy Correcha
- **Contacto**: gerallamy06@gmail.com
- **Fecha**: 2025-10-14
- **Versión**: 1.0