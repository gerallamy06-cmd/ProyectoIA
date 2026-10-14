#!/usr/bin/env python3
"""
Métricas básicas para test_pipeline_dummy
"""
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

def compute_metrics(y_true, y_pred):
    """Calcular métricas binarias"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    return {
        'iou': jaccard_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
        'f1': f1_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
        'precision': precision_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
        'recall': recall_score(y_true_flat, y_pred_flat, average='binary', zero_division=0),
    }