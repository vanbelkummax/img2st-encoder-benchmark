#!/usr/bin/env python3
"""
Spatial transcriptomics prediction metrics.

This module centralizes all metric computation for H&E → ST prediction.
SSIM-ST implementation is preserved from the original compare_encoders.py.

Metrics (per GUARDRAILS.md):
- PRIMARY: MSE, SSIM-ST
- SECONDARY: Pearson r, Spearman ρ

Usage:
    from spatial_metrics import compute_metrics_for_gene, compute_all_metrics
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from typing import Dict, Optional


def compute_ssim_st(y_true: np.ndarray, y_pred: np.ndarray, grid_size: int = 32) -> float:
    """
    Compute SSIM-ST metric for sparse ST data.

    This is the CANONICAL implementation - do not modify without careful consideration.
    Moved verbatim from compare_encoders.py to centralize metric computation.

    For patch-level predictions (not per-bin), we use correlation as a proxy.
    Full SSIM-ST would require spatial bin-level predictions (32x32 grid per patch).

    Args:
        y_true: Ground truth expression values [n_spots]
        y_pred: Predicted expression values [n_spots]
        grid_size: Grid size for full SSIM-ST (unused for patch-level)

    Returns:
        SSIM-ST score (0 to 1, higher is better)
    """
    # For patch-level, use correlation as simplified spatial metric
    if np.std(y_true) < 1e-10 or np.std(y_pred) < 1e-10:
        return 0.0
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return max(0, r)  # Clamp negative correlations


def compute_metrics_for_gene(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid_size: int = 32,
) -> Dict[str, float]:
    """
    Compute all metrics for a single gene prediction.

    This is the standard interface for metric computation across all scripts.

    Args:
        y_true: Ground truth expression [n_spots]
        y_pred: Predicted expression [n_spots]
        grid_size: Grid size for SSIM-ST (unused at patch level)

    Returns:
        Dict with keys:
            - 'mse': Mean squared error (PRIMARY, lower is better)
            - 'ssim_st': SSIM-ST score (PRIMARY, higher is better)
            - 'pearson_r': Pearson correlation (SECONDARY)
            - 'spearman_rho': Spearman correlation (SECONDARY)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # MSE (PRIMARY)
    mse = mean_squared_error(y_true, y_pred)

    # SSIM-ST (PRIMARY) - uses the canonical implementation
    ssim_st = compute_ssim_st(y_true, y_pred, grid_size=grid_size)

    # Correlation metrics (SECONDARY)
    if np.std(y_true) < 1e-10 or np.std(y_pred) < 1e-10:
        pearson = 0.0
        spearman = 0.0
    else:
        pearson, _ = pearsonr(y_true, y_pred)
        spearman, _ = spearmanr(y_true, y_pred)

    return {
        'mse': mse,
        'ssim_st': ssim_st,
        'pearson_r': pearson,
        'spearman_rho': spearman,
    }


def summarize_metrics(metrics_list: list) -> Dict[str, Dict[str, float]]:
    """
    Summarize metrics across multiple genes.

    Args:
        metrics_list: List of dicts from compute_metrics_for_gene

    Returns:
        Dict with mean, std, median for each metric
    """
    if not metrics_list:
        return {}

    metric_names = ['mse', 'ssim_st', 'pearson_r', 'spearman_rho']
    summary = {}

    for name in metric_names:
        values = [m[name] for m in metrics_list if not np.isnan(m[name])]
        if values:
            summary[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'count': len(values),
            }
        else:
            summary[name] = {
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'count': 0,
            }

    return summary


def count_genes_above_threshold(
    metrics_list: list,
    metric: str = 'pearson_r',
    thresholds: list = [0.5, 0.8],
) -> Dict[float, int]:
    """
    Count genes with metric values above various thresholds.

    Args:
        metrics_list: List of dicts from compute_metrics_for_gene
        metric: Which metric to check
        thresholds: List of threshold values

    Returns:
        Dict mapping threshold -> count of genes above it
    """
    values = [m[metric] for m in metrics_list if not np.isnan(m[metric])]
    return {t: sum(v > t for v in values) for t in thresholds}


# Quick test
if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true * 0.8 + np.random.randn(100) * 0.3

    metrics = compute_metrics_for_gene(y_true, y_pred)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
