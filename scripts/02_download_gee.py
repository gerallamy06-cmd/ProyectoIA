#!/usr/bin/env python3
"""
Descargar datos Sentinel-1 y Sentinel-2 para AOI Barranquilla (2020-2025)
via Google Earth Engine usando geemap.
"""
import os
import ee
import geemap
import json

# ============================================================
# CONFIGURACIÓN - AJUSTAR ESTOS VALORES
# ============================================================
GEE_PROJECT_ID = "skysensebarranquilla"  # Google Cloud Project ID
AOI_PATH = "data/aoi/barranquilla_flood_aoi.geojson"
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
CLOUD_THRESHOLD = 30  # % nubes máximo para S2
OUTPUT_DIR = "data/raw"
EXPORT_SCALE = 10  # metros (resolución nativa S2)
# ============================================================

def initialize_gee():
    """Inicializar Earth Engine con project ID"""
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        print(f"[OK] GEE inicializado con project: {GEE_PROJECT_ID}")
        return True
    except Exception as e:
        print(f"[ERROR] GEE init failed: {e}")
        print("  Verifica GEE_PROJECT_ID en el script")
        return False

def load_aoi():
    """Cargar AOI desde GeoJSON"""
    with open(AOI_PATH) as f:
        aoi_json = json.load(f)
    # Convertir a ee.Geometry usando FeatureCollection para múltiples polígonos
    features = aoi_json['features']
    if len(features) == 1:
        aoi_ee = ee.Geometry(features[0]['geometry'])
    else:
        # Crear FeatureCollection y obtener geometría combinada
        fc = ee.FeatureCollection([ee.Feature(f) for f in features])
        aoi_ee = fc.geometry()
    print(f"[OK] AOI cargado: {len(features)} features")
    return aoi_ee

def get_sentinel2_collection(aoi, start, end, cloud_thresh):
    """Obtener colección Sentinel-2 L2A filtrada"""
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
          .select(["B2", "B3", "B4", "B8", "B11", "B12", "QA60"]))  # QA60 para máscara nubes
    return s2

def get_sentinel1_collection(aoi, start, end):
    """Obtener colección Sentinel-1 GRD filtrada"""
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))
    return s1

def add_cloud_mask_s2(image):
    """Añadir máscara de nubes usando QA60 (S2 L2A)"""
    qa = image.select('QA60')
    # Bits 10 y 11: nubes opacas y cirros
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask).select(["B2", "B3", "B4", "B8", "B11", "B12"])

def export_collection_to_drive(collection, prefix, aoi, scale=10, max_images=20):
    """Exportar colección a Google Drive como GeoTIFFs (limitado a max_images más recientes)"""
    # Ordenar por fecha descendente (más recientes primero) y limitar
    collection_sorted = collection.sort('system:time_start', False)
    img_list = collection_sorted.toList(max_images)
    count = img_list.size().getInfo()
    print(f"  Imágenes a exportar (últimas {max_images}): {min(count, max_images)}")
    
    for i in range(min(count, max_images)):
        img = ee.Image(img_list.get(i))
        date_str = img.date().format('YYYY-MM-dd').getInfo()
        desc = f"{prefix}_{date_str}_{i}"
        
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=desc,
            folder=f"GEE_Export_{prefix}",
            fileNamePrefix=desc,
            region=aoi,
            scale=scale,
            crs="EPSG:4326",
            maxPixels=1e13,
            fileFormat="GeoTIFF"
        )
        task.start()
        print(f"    Iniciado: {desc}")

def main():
    print("=" * 60)
    print("DESCARGA SENTINEL-1/2 BARRANQUILLA 2020-2025")
    print("=" * 60)
    
    if GEE_PROJECT_ID == "TU_PROJECT_ID_AQUI":
        print("[ERROR] Debes configurar GEE_PROJECT_ID en el script")
        print("  1. Ve a https://console.cloud.google.com/")
        print("  2. Crea/selecciona un proyecto")
        print("  3. Habilita Earth Engine API")
        print("  4. Copia el Project ID aquí")
        return
    
    if not initialize_gee():
        return
    
    aoi = load_aoi()
    
    # Sentinel-2
    print("\n[1/2] Sentinel-2 L2A...")
    s2 = get_sentinel2_collection(aoi, START_DATE, END_DATE, CLOUD_THRESHOLD)
    s2 = s2.map(add_cloud_mask_s2)
    s2_count = s2.size().getInfo()
    print(f"  Imágenes disponibles: {s2_count}")
    
    if s2_count > 0:
        export_collection_to_drive(s2, "S2_Barranquilla", aoi, scale=EXPORT_SCALE)
    
    # Sentinel-1
    print("\n[2/2] Sentinel-1 GRD...")
    s1 = get_sentinel1_collection(aoi, START_DATE, END_DATE)
    s1_count = s1.size().getInfo()
    print(f"  Imágenes disponibles: {s1_count}")
    
    if s1_count > 0:
        export_collection_to_drive(s1, "S1_Barranquilla", aoi, scale=EXPORT_SCALE)
    
    print("\n[OK] Tareas de exportación enviadas a Google Drive")
    print("  Revisa: https://drive.google.com/drive/folders/...")
    print("  Los archivos aparecerán en carpetas 'GEE_Export_S2_Barranquilla' y 'GEE_Export_S1_Barranquilla'")

if __name__ == "__main__":
    main()