#!/usr/bin/env python3
"""
Crear AOI (Area of Interest) para barrios Don Juan y El Salado, Barranquilla
usando coordenadas aproximadas conocidas + OSMnx para contexto.
"""
import os
import sys
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon

def create_aoi_from_coordinates():
    """Crear AOI usando bounding boxes conocidas de los barrios"""
    
    # Coordenadas aproximadas WGS84 (EPSG:4326) para Barranquilla
    # Don Juan: Nororiente, cerca Vía 40 / Cordialidad
    # El Salado: Suroriente, cerca Ciénaga de Mallorquín / Circunvalar
    
    # Estas son APROXIMADAS - DEBEN VALIDARSE CON DATOS OFICIALES (IGAC, IDEAM, Alcaldía)
    barrios_coords = {
        "Don Juan": {
            "minx": -74.785, "miny": 10.985,
            "maxx": -74.765, "maxy": 11.005,
            "descripcion": "Nororiente, entre Vía 40 y Cordialidad"
        },
        "El Salado": {
            "minx": -74.755, "miny": 10.945,
            "maxx": -74.730, "maxy": 10.970,
            "descripcion": "Suroriente, cerca Ciénaga Mallorquín"
        }
    }
    
    polys = []
    for nombre, coords in barrios_coords.items():
        geom = box(coords["minx"], coords["miny"], coords["maxx"], coords["maxy"])
        polys.append({
            "barrio": nombre,
            "descripcion": coords["descripcion"],
            "geometry": geom
        })
        print(f"{nombre}: {coords['descripcion']}")
        print(f"  Bounds: [{coords['minx']}, {coords['miny']}, {coords['maxx']}, {coords['maxy']}]")
        print(f"  Area aprox: {(coords['maxx']-coords['minx'])*(coords['maxy']-coords['miny']):.6f} deg²")
    
    aoi = gpd.GeoDataFrame(polys, crs="EPSG:4326")
    
    # Guardar
    output_path = "data/aoi/barranquilla_flood_aoi.geojson"
    aoi.to_file(output_path, driver="GeoJSON")
    
    # También versión con buffer de ~200m para capturar bordes
    aoi_buffered = aoi.copy()
    aoi_buffered.geometry = aoi_buffered.geometry.buffer(0.002)  # ~200m
    aoi_buffered.to_file("data/aoi/barranquilla_flood_aoi_buffered.geojson", driver="GeoJSON")
    
    print(f"\n[OK] AOI guardado en: {output_path}")
    print(f"[OK] AOI con buffer (200m) guardado")
    print(f"Barrios: {aoi['barrio'].tolist()}")
    print(f"CRS: {aoi.crs}")
    print(f"Bounds total: {aoi.total_bounds}")
    
    return aoi

def try_osm_search():
    """Intentar búsqueda en OSM como complemento"""
    try:
        import osmnx as ox
        ox.settings.log_console = False
        ox.settings.use_cache = True
        
        # Buscar el polígono de Barranquilla para contexto
        print("\nBuscando contexto: Barranquilla, Colombia...")
        barranquilla = ox.geocode_to_gdf("Barranquilla, Colombia")
        if not barranquilla.empty:
            barranquilla.to_file("data/aoi/barranquilla_city.geojson", driver="GeoJSON")
            print(f"[OK] Contexto Barranquilla guardado: {len(barranquilla)} geometrías")
    except Exception as e:
        print(f"[WARN] OSM search failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("CREANDO AOI: Don Juan y El Salado - Barranquilla")
    print("=" * 60)
    print("NOTA: Coordenadas son APROXIMADAS. Validar con shapefiles oficiales.")
    print()
    
    aoi = create_aoi_from_coordinates()
    try_osm_search()
    
    print("\n" + "=" * 60)
    print("PROXIMO PASO: Validar geometrias en QGIS / Google Earth")
    print("Y ajustar coordenadas en este script si es necesario.")
    print("=" * 60)