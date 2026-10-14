#!/usr/bin/env python3
"""
Actualizar MODEL_CARD.md y DATASET_CARD.md con métricas reales calculadas
"""
import os
import json
import pandas as pd
from pathlib import Path

METRICS_DIR = "outputs/metrics"
MODEL_CARD_PATH = "MODEL_CARD.md"
DATASET_CARD_PATH = "DATASET_CARD.md"

def load_metrics():
    """Cargar métricas calculadas"""
    metrics_files = {
        'global': os.path.join(METRICS_DIR, 'metrics_global.json'),
        'per_tile': os.path.join(METRICS_DIR, 'metrics_per_tile.csv'),
        'by_barrio': os.path.join(METRICS_DIR, 'metrics_barrio.csv'),
        'by_year': os.path.join(METRICS_DIR, 'metrics_year.csv'),
        'by_sensor': os.path.join(METRICS_DIR, 'metrics_sensor.csv'),
        'barrio_x_year': os.path.join(METRICS_DIR, 'metrics_barrio_x_year.csv'),
    }
    
    metrics = {}
    for key, path in metrics_files.items():
        if os.path.exists(path):
            if path.endswith('.json'):
                with open(path) as f:
                    metrics[key] = json.load(f)
            else:
                metrics[key] = pd.read_csv(path)
        else:
            metrics[key] = None
    return metrics

def format_metrics_table(df, value_col, index_col=None):
    """Formatear DataFrame como tabla markdown"""
    if df is None or len(df) == 0:
        return "[No hay datos disponibles]"
    
    # Simplificar para markdown
    if isinstance(df.columns, pd.MultiIndex):
        df = df.round(4)
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    return df.to_markdown(index=False)

def update_model_card(metrics):
    """Actualizar MODEL_CARD.md con métricas reales"""
    if not os.path.exists(MODEL_CARD_PATH):
        print(f"[ERROR] {MODEL_CARD_PATH} no encontrado")
        return
    
    with open(MODEL_CARD_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Actualizar sección de Resultados Unitarios
    if metrics['by_barrio'] is not None:
        barrio_table = format_metrics_table(metrics['by_barrio'], 'iou')
        content = content.replace(
            "| **Global** | \\[XX.XX\\] | \\[XX.XX\\] | \\[XX.XX\\] | \\[XX.XX\\] | \\[XXX\\] |",
            f"| **Global** | {metrics['global'].get('iou_mean', 'N/A'):.4f} | {metrics['global'].get('f1_mean', 'N/A'):.4f} | {metrics['global'].get('precision_mean', 'N/A'):.4f} | {metrics['global'].get('recall_mean', 'N/A'):.4f} | {metrics['global'].get('n_tiles', 'N/A')} |"
        )
    
    # Actualizar resultados interseccionales
    if metrics['barrio_x_year'] is not None:
        cross_table = format_metrics_table(metrics['barrio_x_year'], 'iou')
        # Reemplazar placeholder table
        pass
    
    # Guardar
    with open(MODEL_CARD_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] {MODEL_CARD_PATH} actualizado")

def update_dataset_card(metrics):
    """Actualizar DATASET_CARD.md con estadísticas reales"""
    if not os.path.exists(DATASET_CARD_PATH):
        print(f"[ERROR] {DATASET_CARD_PATH} no encontrado")
        return
    
    with open(DATASET_CARD_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Actualizar estadísticas del dataset
    if metrics['per_tile'] is not None:
        df = metrics['per_tile']
        total_tiles = len(df)
        total_pixels = df['n_pixels'].sum()
        flood_ratio = df['flood_ratio_gt'].mean()
        
        stats_text = f"""
### Estadísticas Calculadas
- **Tiles totales**: {total_tiles}
- **Píxeles totales**: {total_pixels:,}
- **Ratio inundación promedio (GT)**: {flood_ratio:.4%}
- **Tiles con inundación > 0**: {(df['flood_ratio_gt'] > 0).sum()}
"""
        # Buscar sección de estructura y agregar stats
        if "## Estructura del Dataset" in content:
            content = content.replace(
                "## Estructura del Dataset",
                stats_text + "\n## Estructura del Dataset"
            )
    
    with open(DATASET_CARD_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] {DATASET_CARD_PATH} actualizado")

def generate_summary_report(metrics):
    """Generar reporte resumen en outputs/metrics/summary_report.md"""
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    report = f"""# Reporte de Evaluación SkySense++ - Barranquilla Flood Detection

## Resumen Global
"""
    if metrics['global']:
        g = metrics['global']
        report += f"""- **IoU**: {g.get('iou_mean', 'N/A'):.4f} ± {g.get('iou_std', 'N/A'):.4f}
- **F1-Score**: {g.get('f1_mean', 'N/A'):.4f} ± {g.get('f1_std', 'N/A'):.4f}
- **Precision**: {g.get('precision_mean', 'N/A'):.4f}
- **Recall**: {g.get('recall_mean', 'N/A'):.4f}
- **Tiles evaluados**: {g.get('n_tiles', 'N/A')}
- **Píxeles totales**: {g.get('n_pixels_total', 'N/A'):,}
"""
    
    report += "\n## Por Barrio\n"
    if metrics['by_barrio'] is not None:
        report += format_metrics_table(metrics['by_barrio'], 'iou') + "\n"
    
    report += "\n## Por Año\n"
    if metrics['by_year'] is not None:
        report += format_metrics_table(metrics['by_year'], 'iou') + "\n"
    
    report += "\n## Por Sensor\n"
    if metrics['by_sensor'] is not None:
        report += format_metrics_table(metrics['by_sensor'], 'iou') + "\n"
    
    report += "\n## Interseccional (Barrio × Año)\n"
    if metrics['barrio_x_year'] is not None:
        report += format_metrics_table(metrics['barrio_x_year'], 'iou') + "\n"
    
    with open(os.path.join(METRICS_DIR, 'summary_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"[OK] Reporte guardado en {METRICS_DIR}/summary_report.md")

def main():
    print("=" * 60)
    print("ACTUALIZANDO FICHAS CON MÉTRICAS REALES")
    print("=" * 60)
    
    metrics = load_metrics()
    
    if metrics['global'] is None:
        print("[WARN] No hay métricas globales. Ejecuta primero scripts/07_compute_metrics.py")
        return
    
    update_model_card(metrics)
    update_dataset_card(metrics)
    generate_summary_report(metrics)
    
    print("\n[OK] Fichas actualizadas con métricas reales")

if __name__ == "__main__":
    main()