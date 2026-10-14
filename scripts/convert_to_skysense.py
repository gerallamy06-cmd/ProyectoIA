#!/usr/bin/env python3
"""
Convertir datos de ProyectoIA a formato SkySense++ eval_datasets/barranquilla_flood/
"""
import os
import shutil
import glob

# Rutas origen
S2_SRC = r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\ProyectoIA\data\processed\sentinel2"
S1_SRC = r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\ProyectoIA\data\processed\sentinel1"
FLOOD_SRC = r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\ProyectoIA\data\processed\flood_tiles"
LABELS_SRC = r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\ProyectoIA\data\processed\labels"

# Rutas destino SkySense++
DST_ROOT = r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\SkySensePlusPlus-main\eval_datasets\barranquilla_flood"
DST_IMAGES = os.path.join(DST_ROOT, "Images")
DST_MASKS = os.path.join(DST_ROOT, "Semantic_mask")
DST_S2 = os.path.join(DST_ROOT, "S2")
DST_S1 = os.path.join(DST_ROOT, "S1")

os.makedirs(DST_IMAGES, exist_ok=True)
os.makedirs(DST_MASKS, exist_ok=True)
os.makedirs(DST_S2, exist_ok=True)
os.makedirs(DST_S1, exist_ok=True)

# Leer test split
test_file = r"C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\ProyectoIA\data\splits\test.txt"
with open(test_file) as f:
    lines = [l.strip().split() for l in f if l.strip()]

print(f"Samples en test: {len(lines)}")

# Para SkySense++, necesitamos:
# - Images/ : HR (S2 RGB) - 3 bandas
# - Semantic_mask/ : máscara binaria (0/255)
# - S2/ : series temporales S2 (npz con 'image' y 'date')
# - S1/ : series temporales S1 (npz con 'image')
# - val.txt : lista de samples

# Copiar HR (usar primeras 3 bandas S2 como RGB)
s2_files = glob.glob(os.path.join(S2_SRC, "*.tif"))
print(f"S2 tiles disponibles: {len(s2_files)}")

# Para simplicidad: usar el primer tile S2 como HR para todos
if s2_files:
    import rasterio
    import numpy as np
    
    hr_src = s2_files[0]
    with rasterio.open(hr_src) as src:
        # Leer primeras 3 bandas (B2, B3, B4 -> RGB)
        hr_data = src.read([1, 2, 3])  # B2, B3, B4
        profile = src.profile
        profile.update({'count': 3})
    
    # Guardar HR para cada sample
    for i, (s2_name, s1_name, flood_name, cls) in enumerate(lines):
        hr_dst = os.path.join(DST_IMAGES, f"hr_{i:04d}.tif")
        with rasterio.open(hr_dst, 'w', **profile) as dst:
            dst.write(hr_data)

# Copiar máscaras de inundación
flood_files = glob.glob(os.path.join(FLOOD_SRC, "*.tif"))
for i, (s2_name, s1_name, flood_name, cls) in enumerate(lines):
    # Buscar flood tile correspondiente
    flood_src = os.path.join(FLOOD_SRC, flood_name)
    if os.path.exists(flood_src):
        flood_dst = os.path.join(DST_MASKS, f"flood_{i:04d}.png")
        with rasterio.open(flood_src) as src:
            data = src.read(1)
            # Convertir a 0/255
            data = (data > 0).astype(np.uint8) * 255
            profile = src.profile
            profile.update({'count': 1, 'dtype': 'uint8'})
        with rasterio.open(flood_dst, 'w', **profile) as dst:
            dst.write(data[np.newaxis, :, :])

# Copiar S2 tiles (convertir a npz)
for i, (s2_name, s1_name, flood_name, cls) in enumerate(lines):
    s2_src = os.path.join(S2_SRC, s2_name)
    if os.path.exists(s2_src):
        with rasterio.open(s2_src) as src:
            data = src.read()  # (C, H, W)
            # SkySense++ espera (C, T, H, W) - T=1 para 1-shot
            data = data[:, np.newaxis, :, :]  # (C, 1, H, W)
        npz_dst = os.path.join(DST_S2, f"s2_{i:04d}.npz")
        np.savez(npz_dst, image=data, date=np.array(['20200101']))

# Copiar S1 tiles
for i, (s2_name, s1_name, flood_name, cls) in enumerate(lines):
    s1_src = os.path.join(S1_SRC, s1_name)
    if os.path.exists(s1_src):
        with rasterio.open(s1_src) as src:
            data = src.read()
            data = data[:, np.newaxis, :, :]
        npz_dst = os.path.join(DST_S1, f"s1_{i:04d}.npz")
        np.savez(npz_dst, image=data)

# Crear val.txt para SkySense++
val_file = os.path.join(DST_ROOT, "val.txt")
with open(val_file, 'w') as f:
    for i in range(len(lines)):
        # Formato SkySense++: hr_path class (sin extensión)
        f.write(f"hr_{i:04d} 1\n")

print(f"[OK] Datos convertidos a {DST_ROOT}")
print(f"  Images: {len(glob.glob(os.path.join(DST_IMAGES, '*')))}")
print(f"  Semantic_mask: {len(glob.glob(os.path.join(DST_MASKS, '*')))}")
print(f"  S2: {len(glob.glob(os.path.join(DST_S2, '*')))}")
print(f"  S1: {len(glob.glob(os.path.join(DST_S1, '*')))}")
print(f"  val.txt creado")