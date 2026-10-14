---
# Model Card metadata (YAML front-matter para Hugging Face Hub)
tags:
  - earth-observation
  - flood-detection
  - change-detection
  - remote-sensing
  - sentinel-1
  - sentinel-2
  - barranquilla
  - colombia
  - skysense
  - foundation-model
license: apache-2.0
model_name: SkySense-Flood-Barranquilla
---

# Model Card: SkySense++ para Detección de Inundaciones en Barranquilla

## Resumen del Modelo
**SkySense++** (foundation model pre-entrenado para Earth Observation, Wu et al. 2025, *Nature Machine Intelligence*) aplicado en configuración **1-shot** (sin fine-tuning) para detección de cambios de inundación en los barrios **Don Juan** y **El Salado**, **Barranquilla, Colombia**, durante el período **2020-2025**, utilizando imágenes **Sentinel-1 (SAR)** y **Sentinel-2 (óptico)**.

Este modelo demuestra la capacidad de transferencia de un foundation model multi-modal (SAR + óptico + series temporales) a un contexto urbano costero tropical sin entrenamiento adicional específico del dominio.

## Detalles del Modelo
- **Desarrollado por**: Investigadores de Wuhan University (modelo base SkySense++) / [Tu Nombre] (adaptación y evaluación Barranquilla)
- **Fecha**: Octubre 2025
- **Versión**: 1.0
- **Tipo**: Foundation model multi-modal (ViT backbone SwinTransformerV2 + VisionTransformer para S1/S2) + cabeza de segmentación UPerHead
- **Modelo base**: SkySense++ (Wu et al. 2025, Nature Machine Intelligence, DOI: 10.1038/s42256-025-01078-8)
- **Licencia**: Solo investigación no comercial (ver licencia SkySense++)
- **Repositorio**: https://github.com/gerallamy06-cmd/ProyectoIA
- **Paper base**: Wu, K., et al. (2025). "A semantic-enhanced multi-modal remote sensing foundation model for Earth observation"

## Usos
### Uso Directo (Intended Use)
- Detección de extensión de inundaciones en áreas urbanas costeras del Caribe colombiano
- Monitoreo de cambios de cuerpos de agua en series temporales Sentinel-1/2 (2020-2025)
- Apoyo a gestión de riesgo de desastres y planificación urbana en Barranquilla
- Evaluación de foundation models EO en contextos tropicales urbanos

### Usos Fuera de Alcance (Out-of-Scope)
- Detección de inundaciones en zonas rurales/agrícolas sin validación local
- Pronóstico de inundaciones (solo detección retrospectiva de cambios)
- Uso operacional en tiempo real sin validación humana experta
- Transferencia a otras cuencas/regiones sin evaluación de dominio

## Factores (Mitchell et al. 2019)
### Factores Relevantes (Relevant Factors)
| Factor | Descripción | Impacto Esperado |
|--------|-------------|------------------|
| **Geográfico - Barrio** | Don Juan (nororiente, denso) vs El Salado (suroriente, cerca ciénaga) | Diferente topografía, drenaje, densidad urbana |
| **Temporal - Año** | 2020-2025 (incluye La Niña 2020-2022, Huracán Iota 2020) | Variabilidad interanual de precipitación |
| **Temporal - Estación** | Temporada lluviosa (abr-nov) vs seca (dic-mar) | Presencia/ausencia de agua estacional |
| **Instrumental - Sensor** | Sentinel-1 (SAR, nubes/nocturno) vs Sentinel-2 (óptico, nubes) | Complementariedad SAR/óptico |
| **Ambiental - Nubosidad** | Alta en Caribe colombiano (>60% días nublados) | Limitante crítico para S2 |
| **Ambiental - Marea/Río** | Influencia río Magdalena y marea Caribe | Falsos positivos en zonas de marea |

### Factores de Evaluación (Evaluation Factors)
- **Evaluados**: Ambos barrios, años 2020-2025, ambos sensores (S1, S2, fusión), eventos extremos conocidos
- **No evaluados**: Otros barrios de Barranquilla, otras ciudades, sensores distintos (Landsat, Planet), datos LiDAR/terreno

## Métricas (Metrics)
### Métricas Principales
| Métrica | Descripción | Justificación |
|---------|-------------|---------------|
| **IoU (Intersection over Union)** | Métrica principal segmentación binaria | Estándar en detección de cambios |
| **F1-Score (Dice Coefficient)** | Balance precisión-recall | Robusto a desbalance de clases |
| **Precision / Recall** | Por clase (inundación vs no-inundación) | Análisis de errores tipo I/II |
| **Accuracy por píxel** | Referencia general | Comparabilidad básica |

### Desagregación Requerida (Mitchell et al. 2019)
- **Por barrio**: Don Juan vs El Salado
- **Por año**: 2020, 2021, 2022, 2023, 2024, 2025
- **Por sensor**: S1-only, S2-only, Fusión S1+S2
- **Por evento**: Eventos extremos vs períodos base

## Datos de Evaluación (Evaluation Data)
- **Dataset**: Sentinel-1/2 Barranquilla 2020-2025 (ver DATASET_CARD.md)
- **Preprocesamiento**: Co-registro S1→S2, normalización percentil 2-98, tiling 512×512 overlap 64px
- **Ground truth**: Máscaras semi-automáticas (NDWI/MNDWI/AWEI + umbral Otsu + validación visual experta)
- **Splits**: 70% train / 15% val / 15% test (estratificado por barrio + año)
- **Eventos de referencia**: La Niña 2020-2022, Huracán Iota nov-2020, lluvias extremas 2022, 2024

## Entrenamiento (Modelo Base SkySense++)
- **Datos pre-entrenamiento**: RS-Semantic (13 datasets, pixel-level annotations) + RS-Representation (series temporales multi-modales)
- **Arquitectura**: 
  - HR backbone: SwinTransformerV2-Huge (3 canales, 512×512)
  - S2 backbone: ViT-Large (10 canales, 16×16 patches, series temporales)
  - S1 backbone: ViT-Large (2 canales, 16×16 patches, series temporales)
  - Neck: TransformerEncoder (24 layers, 1024 dim)
  - Head: UPerHead (65 clases pre-entrenamiento, adaptado a 2 clases inundación)
- **Hardware original**: 32×A100 GPUs, 4 nodos
- **Framework**: PyTorch + antmmf + mmsegmentation

## Análisis Cuantitativo (Quantitative Analyses - Mitchell et al. 2019)
> **NOTA**: Los valores abajo son PLACEHOLDERS. Reemplazar con resultados reales tras ejecutar `scripts/07_compute_metrics.py`.

### Resultados Unitarios (Unitary Results)
| Factor | IoU (Inundación) | F1-Score | Precision | Recall | N muestras |
|--------|------------------|----------|-----------|--------|------------|
| **Global** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **Don Juan** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **El Salado** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **2020** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **2021** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **2022** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **2023** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **2024** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **2025** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **Sentinel-1** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **Sentinel-2** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |
| **Fusión S1+S2** | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] | [XXX] |

### Resultados Interseccionales (Intersectional Results)
| Barrio × Año | IoU | F1 | Precision | Recall |
|--------------|-----|-----|-----------|--------|
| Don Juan × 2020 | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] |
| Don Juan × 2022 (La Niña) | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] |
| El Salado × 2020 | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] |
| El Salado × 2022 (La Niña) | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] |
| Don Juan × S1 | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] |
| El Salado × S2 | [XX.XX] | [XX.XX] | [XX.XX] | [XX.XX] |

### Intervalos de Confianza
- **Método**: Bootstrap resampling (1000 iteraciones)
- **Nivel**: 95%
- **Reportar**: [IoU] ± [IC95%] para cada factor

## Consideraciones Éticas (Ethical Considerations)
- **Datos sensibles**: No aplica (imágenes satelitales públicas Copernicus/ESA)
- **Vida humana**: **Sí** - apoyo directo a gestión de riesgo de inundaciones en población vulnerable
- **Riesgos identificados**:
  - Falsos negativos → zonas inundadas no detectadas → riesgo evacuación tardía
  - Falsos positivos → alertas innecesarias → fatiga de alerta, desconfianza
  - Sesgo geográfico → mejor rendimiento en Don Juan (más denso) vs El Salado
- **Mitigaciones implementadas**:
  - Umbral de decisión conservador (prioriza recall > precision)
  - Validación humana obligatoria antes de uso operacional
  - Documentación explícita de limitaciones por barrio/sensor
- **Casos de uso sensibles**: Evacuación de emergencia, planificación uso de suelo, seguros

## Limitaciones y Recomendaciones (Caveats and Recommendations)
### Limitaciones Conocidas
1. **Cobertura espacial**: Solo 2 barrios (Don Juan, El Salado) - no representativo de toda Barranquilla
2. **Cobertura temporal**: 6 años (2020-2025) - gaps por nubes S2 (~40-60% datos perdidos)
3. **Modelo base sin fine-tuning**: Configuración 1-shot limita adaptación a dominio local
4. **Ground truth espectral**: Basado en índices (NDWI/MNDWI), no mediciones in-situ / LiDAR
5. **Desbalance de clases**: Inundación << No-inundación (típico <5% píxeles positivos)
6. **Co-registro S1-S2**: Errores sub-píxel introducen ruido en fusión
7. **Clase única**: No distingue tipos de inundación (fluvial, pluvial, mareal, mixta)

### Recomendaciones
1. **Fine-tuning local**: Entrenar cabeza de segmentación con datos Barranquilla (5-10 eventos etiquetados)
2. **Ampliar AOI**: Incluir más barrios (Las Nieves, San Roque, Rebolo, etc.)
3. **Datos in-situ**: Validar con mediciones de nivel río, pluviómetros, reportes comunidad
4. **Multi-clase**: Distinguir agua permanente vs temporal vs suelo saturado
5. **Datos complementarios**: Integrar DEM (SRTM/ALOS), drenaje urbano, uso de suelo
6. **Ensamble**: Combinar con modelos hidrológicos (HEC-RAS, HEC-HMS) para pronóstico

## Impacto Ambiental (Environmental Impact)
- **Hardware evaluación**: [GPU usada, ej. RTX 3090 / A100]
- **Horas inferencia**: [X horas para ~XXX tiles test]
- **CO2eq estimado**: [Calcular con ML CO2 Impact Calculator: https://mlco2.github.io/impact]
- **Entrenamiento base**: 32 A100 × ~14 días (reportado en paper original)

## Especificaciones Técnicas (Technical Specifications)
| Componente | Especificación |
|------------|----------------|
| **Arquitectura** | SkySense++ Multi-Modal ViT (SwinV2-HR + ViT-S2 + ViT-S1) |
| **Parámetros** | ~300M (base) + ~50M (cabezas) |
| **Entrada** | HR: 3×512×512, S2: 10×T×16×16, S1: 2×T×16×16 |
| **Salida** | Mapa cambio binario 1×512×512 (inundación: 1, no: 0) |
| **Framework** | PyTorch 1.13 + antmmf + mmsegmentation 0.30 |
| **CUDA** | 11.7+ requerido |
| **Tamaño modelo** | ~1.2 GB (FP16) / ~2.4 GB (FP32) |

## Cita (Citation)
```bibtex
@article{wu2025semantic,
  title={A semantic-enhanced multi-modal remote sensing foundation model for Earth observation},
  author={Wu, Kang and Zhang, Yingying and Ru, Lixiang and Dang, Bo and Lao, Jiangwei and Yu, Lei and Luo, Junwei and Zhu, Zifan and Sun, Yue and Zhang, Jiahao and Zhu, Qi and Wang, Jian and Yang, Ming and Chen, Jingdong and Zhang, Yongjun and Li, Yansheng},
  journal={Nature Machine Intelligence},
  year={2025},
  doi={10.1038/s42256-025-01078-8}
}

@misc{skysense_barranquilla_2025,
  author = {[Tu Nombre]},
  title = {SkySense++ 1-shot Flood Detection: Barranquilla Don Juan & El Salado 2020-2025},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/gerallamy06-cmd/ProyectoIA}
}
```

## Autores de la Model Card (Model Card Authors)
- **Nombre**: [Tu Nombre Completo]
- **Email**: [tu.email@uninorte.edu.co]
- **Afiliación**: Universidad del Norte, Onceavo Semestre - IA Aplicada al Territorio
- **GitHub**: @gerallamy06-cmd
- **Fecha**: 2025-10-14
- **Rol**: Adaptación de dominio, preparación datos, evaluación, documentación

## Contacto (Model Card Contact)
- **Email**: [tu.email@uninorte.edu.co]
- **Issues**: https://github.com/gerallamy06-cmd/ProyectoIA/issues