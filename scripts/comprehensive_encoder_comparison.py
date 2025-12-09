#!/usr/bin/env python3
"""
COMPREHENSIVE ENCODER COMPARISON

Run Ridge regression baseline on ALL available encoders with strict per-fold normalization.
This script evaluates:
- Single-scale encoders (256px, 512px)
- Multiscale concatenations
- Foundation models (Virchow2) vs standard (DenseNet, DINOv2)

Usage:
    python scripts/comprehensive_encoder_comparison.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Paths
FEATURES_DIR = Path('/home/user/img2st-baseline-comparison/features')
RESULTS_DIR = Path('/home/user/img2st-baseline-comparison/results')


def load_data():
    """Load all features and expression data."""
    # Load expression
    expr_data = np.load(FEATURES_DIR / 'all_genes_expression_aligned.npz', allow_pickle=True)
    expression = expr_data['expression']
    patient_ids = expr_data['patient_ids']

    with open(FEATURES_DIR / 'gene_names_all.json') as f:
        gene_names = json.load(f)

    # Load all encoder features
    encoders = {}

    # Single-scale encoders
    encoders['DenseNet-256'] = np.load(FEATURES_DIR / 'densenet_fm_features.npz')['features']
    encoders['DenseNet-512'] = np.load(FEATURES_DIR / 'densenet_512context.npz')['features']
    encoders['DINOv2-256'] = np.load(FEATURES_DIR / 'dinov2_features.npz')['features']
    encoders['DINOv2-512'] = np.load(FEATURES_DIR / 'dinov2_512context.npz')['features']
    encoders['Virchow2-256'] = np.load(FEATURES_DIR / 'virchow2_256_features.npz')['features']
    encoders['Virchow2-512'] = np.load(FEATURES_DIR / 'virchow2_512context.npz')['features']

    # Multiscale concatenations
    encoders['DenseNet-Multi'] = np.concatenate([
        encoders['DenseNet-256'], encoders['DenseNet-512']
    ], axis=1)

    encoders['DINOv2-Multi'] = np.concatenate([
        encoders['DINOv2-256'], encoders['DINOv2-512']
    ], axis=1)

    encoders['Virchow2-Multi'] = np.concatenate([
        encoders['Virchow2-256'], encoders['Virchow2-512']
    ], axis=1)

    # All-encoder concatenation (current best baseline)
    encoders['All-Concat'] = np.concatenate([
        encoders['DenseNet-256'], encoders['DenseNet-512'],
        encoders['DINOv2-256'], encoders['DINOv2-512']
    ], axis=1)

    # All-encoder with Virchow2
    encoders['All+Virchow2'] = np.concatenate([
        encoders['DenseNet-256'], encoders['DenseNet-512'],
        encoders['DINOv2-256'], encoders['DINOv2-512'],
        encoders['Virchow2-256'], encoders['Virchow2-512']
    ], axis=1)

    return encoders, expression, patient_ids, gene_names


def evaluate_strict_loocv(
    features: np.ndarray,
    expression: np.ndarray,
    patient_ids: np.ndarray,
    n_genes: int = 500,
    alpha: float = 1.0
) -> Tuple[np.ndarray, Dict]:
    """
    Evaluate with strict per-fold normalization.

    Args:
        features: (n_spots, n_features)
        expression: (n_spots, n_genes)
        patient_ids: (n_spots,)
        n_genes: Number of top-variance genes
        alpha: Ridge regularization

    Returns:
        correlations: (n_genes,) per-gene correlations
        metadata: Dict with summary stats
    """
    # Select top variance genes
    gene_vars = np.var(expression, axis=0)
    gene_indices = np.argsort(gene_vars)[-n_genes:]

    patients = np.unique(patient_ids)
    correlations = np.zeros(n_genes)

    for i, gene_idx in enumerate(gene_indices):
        y = expression[:, gene_idx]
        y_pred_all = []
        y_true_all = []

        for test_patient in patients:
            test_mask = patient_ids == test_patient
            train_mask = ~test_mask

            # STRICT: Scale features using ONLY training data
            feat_scaler = StandardScaler()
            X_train = feat_scaler.fit_transform(features[train_mask])
            X_test = feat_scaler.transform(features[test_mask])

            # STRICT: Scale expression using ONLY training data
            expr_scaler = StandardScaler()
            y_train = expr_scaler.fit_transform(y[train_mask].reshape(-1, 1)).ravel()
            y_test = expr_scaler.transform(y[test_mask].reshape(-1, 1)).ravel()

            # Train and predict
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_pred_all.extend(y_pred)
            y_true_all.extend(y_test)

        # Compute correlation
        if np.std(y_pred_all) > 1e-10 and np.std(y_true_all) > 1e-10:
            r, _ = stats.pearsonr(y_pred_all, y_true_all)
            correlations[i] = r if not np.isnan(r) else 0

    metadata = {
        'n_genes': n_genes,
        'mean_r': float(np.mean(correlations)),
        'median_r': float(np.median(correlations)),
        'std_r': float(np.std(correlations)),
        'genes_above_0.3': int(np.sum(correlations > 0.3)),
        'genes_above_0.5': int(np.sum(correlations > 0.5)),
        'genes_above_0.8': int(np.sum(correlations > 0.8)),
        'genes_above_0.9': int(np.sum(correlations > 0.9)),
    }

    return correlations, metadata


def evaluate_all_genes_strict(
    features: np.ndarray,
    expression: np.ndarray,
    patient_ids: np.ndarray,
    alpha: float = 1.0
) -> Tuple[np.ndarray, Dict]:
    """Evaluate ALL genes with strict protocol."""
    n_genes = expression.shape[1]
    patients = np.unique(patient_ids)
    correlations = np.zeros(n_genes)

    print(f"    Evaluating {n_genes} genes...")

    for gene_idx in tqdm(range(n_genes), desc="    All genes", leave=False):
        y = expression[:, gene_idx]
        y_pred_all = []
        y_true_all = []

        for test_patient in patients:
            test_mask = patient_ids == test_patient
            train_mask = ~test_mask

            feat_scaler = StandardScaler()
            X_train = feat_scaler.fit_transform(features[train_mask])
            X_test = feat_scaler.transform(features[test_mask])

            expr_scaler = StandardScaler()
            y_train = expr_scaler.fit_transform(y[train_mask].reshape(-1, 1)).ravel()
            y_test = expr_scaler.transform(y[test_mask].reshape(-1, 1)).ravel()

            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_pred_all.extend(y_pred)
            y_true_all.extend(y_test)

        if np.std(y_pred_all) > 1e-10 and np.std(y_true_all) > 1e-10:
            r, _ = stats.pearsonr(y_pred_all, y_true_all)
            correlations[gene_idx] = r if not np.isnan(r) else 0

    metadata = {
        'n_genes': n_genes,
        'mean_r': float(np.mean(correlations)),
        'median_r': float(np.median(correlations)),
        'std_r': float(np.std(correlations)),
        'genes_above_0.3': int(np.sum(correlations > 0.3)),
        'genes_above_0.5': int(np.sum(correlations > 0.5)),
        'genes_above_0.8': int(np.sum(correlations > 0.8)),
        'genes_above_0.9': int(np.sum(correlations > 0.9)),
    }

    return correlations, metadata


def main():
    print("=" * 70)
    print("COMPREHENSIVE ENCODER COMPARISON")
    print("Strict Per-Fold Normalization Protocol")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    encoders, expression, patient_ids, gene_names = load_data()

    print(f"Expression: {expression.shape}")
    print(f"Patients: {np.unique(patient_ids)}")
    print(f"Encoders: {list(encoders.keys())}")

    # Results storage
    results_hvg = []
    results_all = []

    # Evaluate each encoder
    for encoder_name, features in encoders.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {encoder_name}")
        print(f"  Feature dim: {features.shape[1]}")

        # HVG evaluation (500 genes)
        print("  Running HVG (500 genes)...")
        _, hvg_meta = evaluate_strict_loocv(
            features, expression, patient_ids, n_genes=500
        )
        hvg_meta['encoder'] = encoder_name
        hvg_meta['feature_dim'] = features.shape[1]
        results_hvg.append(hvg_meta)

        print(f"    Mean r: {hvg_meta['mean_r']:.4f}")
        print(f"    Genes r>0.5: {hvg_meta['genes_above_0.5']}")
        print(f"    Genes r>0.8: {hvg_meta['genes_above_0.8']}")

        # All genes evaluation (slower)
        print("  Running ALL genes (18085)...")
        _, all_meta = evaluate_all_genes_strict(
            features, expression, patient_ids
        )
        all_meta['encoder'] = encoder_name
        all_meta['feature_dim'] = features.shape[1]
        results_all.append(all_meta)

        print(f"    Mean r: {all_meta['mean_r']:.4f}")
        print(f"    Genes r>0.5: {all_meta['genes_above_0.5']}")
        print(f"    Genes r>0.8: {all_meta['genes_above_0.8']}")

    # Create results DataFrames
    df_hvg = pd.DataFrame(results_hvg)
    df_all = pd.DataFrame(results_all)

    # Sort by mean_r
    df_hvg = df_hvg.sort_values('mean_r', ascending=False)
    df_all = df_all.sort_values('mean_r', ascending=False)

    # Save results
    output_dir = RESULTS_DIR / 'encoder_comparison_strict'
    output_dir.mkdir(exist_ok=True)

    df_hvg.to_csv(output_dir / 'hvg_500_results.csv', index=False)
    df_all.to_csv(output_dir / 'all_genes_results.csv', index=False)

    # Print summary tables
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY: Top 500 HVGs (Strict Protocol)")
    print("=" * 70)
    print(df_hvg[['encoder', 'feature_dim', 'mean_r', 'median_r', 'genes_above_0.5', 'genes_above_0.8']].to_string(index=False))

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY: All 18,085 Genes (Strict Protocol)")
    print("=" * 70)
    print(df_all[['encoder', 'feature_dim', 'mean_r', 'median_r', 'genes_above_0.5', 'genes_above_0.8']].to_string(index=False))

    # Generate markdown report
    report = generate_markdown_report(df_hvg, df_all)
    with open(output_dir / 'ENCODER_COMPARISON_REPORT.md', 'w') as f:
        f.write(report)

    print(f"\n\nResults saved to {output_dir}/")
    print("  - hvg_500_results.csv")
    print("  - all_genes_results.csv")
    print("  - ENCODER_COMPARISON_REPORT.md")


def generate_markdown_report(df_hvg: pd.DataFrame, df_all: pd.DataFrame) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# Comprehensive Encoder Comparison Report")
    lines.append("")
    lines.append("**Protocol**: Strict per-fold normalization (no data leakage)")
    lines.append("**Dataset**: Visium HD CRC (222 patches, 3 patients)")
    lines.append("")

    # HVG results
    lines.append("## Top 500 Highly Variable Genes")
    lines.append("")
    lines.append("| Encoder | Feature Dim | Mean r | Median r | r > 0.5 | r > 0.8 |")
    lines.append("|---------|-------------|--------|----------|---------|---------|")

    for _, row in df_hvg.iterrows():
        lines.append(
            f"| {row['encoder']} | {row['feature_dim']} | "
            f"{row['mean_r']:.4f} | {row['median_r']:.4f} | "
            f"{row['genes_above_0.5']} ({100*row['genes_above_0.5']/500:.1f}%) | "
            f"{row['genes_above_0.8']} ({100*row['genes_above_0.8']/500:.1f}%) |"
        )

    lines.append("")
    lines.append("## All 18,085 Genes")
    lines.append("")
    lines.append("| Encoder | Feature Dim | Mean r | Median r | r > 0.5 | r > 0.8 |")
    lines.append("|---------|-------------|--------|----------|---------|---------|")

    for _, row in df_all.iterrows():
        lines.append(
            f"| {row['encoder']} | {row['feature_dim']} | "
            f"{row['mean_r']:.4f} | {row['median_r']:.4f} | "
            f"{row['genes_above_0.5']} ({100*row['genes_above_0.5']/18085:.1f}%) | "
            f"{row['genes_above_0.8']} ({100*row['genes_above_0.8']/18085:.1f}%) |"
        )

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")

    # Best encoder
    best_hvg = df_hvg.iloc[0]
    best_all = df_all.iloc[0]

    lines.append(f"- **Best HVG encoder**: {best_hvg['encoder']} (mean r = {best_hvg['mean_r']:.4f})")
    lines.append(f"- **Best all-genes encoder**: {best_all['encoder']} (mean r = {best_all['mean_r']:.4f})")

    # Multiscale vs single-scale
    lines.append("")
    lines.append("### Single-Scale vs Multi-Scale")
    lines.append("")

    for base in ['DenseNet', 'DINOv2', 'Virchow2']:
        single_256 = df_hvg[df_hvg['encoder'] == f'{base}-256']['mean_r'].values
        single_512 = df_hvg[df_hvg['encoder'] == f'{base}-512']['mean_r'].values
        multi = df_hvg[df_hvg['encoder'] == f'{base}-Multi']['mean_r'].values

        if len(single_256) > 0 and len(multi) > 0:
            delta = (multi[0] - single_256[0]) / single_256[0] * 100
            lines.append(f"- **{base}**: 256px r={single_256[0]:.3f}, Multi r={multi[0]:.3f} ({delta:+.1f}%)")

    lines.append("")
    lines.append("---")
    lines.append("*Generated with strict per-fold normalization protocol*")

    return "\n".join(lines)


if __name__ == '__main__':
    main()
