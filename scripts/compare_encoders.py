#!/usr/bin/env python3
"""
Compare encoder performance for H&E → ST prediction.

Compares DenseNet-ImageNet vs DINOv2-giant (and optionally UNI when available).
Uses LOOCV (leave-one-patient-out) ridge regression.

Metrics (per GUARDRAILS.md):
- PRIMARY: MSE, SSIM-ST (spatial structure preservation)
- SECONDARY: Pearson r, Spearman ρ

HARD GUARDRAILS (see GUARDRAILS.md):
1. REFUSES to load misaligned expression file (all_genes_expression.npz)
2. REQUIRES aligned expression file (*_aligned.npz)
3. Uses aligned 18k-gene panel as default "full gene set"
4. Imports metrics from spatial_metrics.py (canonical SSIM-ST implementation)

Usage:
    # Default: top 100 genes by variance
    python compare_encoders.py --output results/encoder_comparison.csv

    # Full 18k aligned gene panel (recommended)
    python compare_encoders.py --top_k 0 --output results/encoder_comparison_18k.csv

    # Custom gene list
    python compare_encoders.py --genes path/to/gene_list.txt --output results/custom.csv
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import metrics from centralized module (GUARDRAIL: use canonical SSIM-ST)
from spatial_metrics import compute_metrics_for_gene, summarize_metrics, count_genes_above_threshold


# =============================================================================
# HARD GUARDRAILS - DO NOT MODIFY
# =============================================================================

# GUARDRAIL 1: Explicitly refuse the misaligned expression file
# This file has known coordinate alignment issues (FCGBP r=0.06 vs r=0.90 aligned)
MISALIGNED_FILE = 'all_genes_expression.npz'
ALIGNED_FILE = 'all_genes_expression_aligned.npz'


def _check_expression_file_guard(path: Path) -> None:
    """
    HARD GUARD: Refuse to load misaligned expression data.

    The misaligned file (all_genes_expression.npz) had pixel vs array coordinate
    issues that caused FCGBP to have r=0.06 instead of r=0.90. This guard prevents
    accidental regression to misaligned data.
    """
    filename = path.name

    # Block the misaligned file
    if filename == MISALIGNED_FILE:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"HARD GUARD VIOLATION: Refusing to load misaligned expression file!\n"
            f"{'='*60}\n\n"
            f"File: {path}\n\n"
            f"This file has known coordinate alignment issues:\n"
            f"  - FCGBP achieves r=0.06 (wrong) instead of r=0.90 (correct)\n"
            f"  - See PROJECT_JOURNEY_REPORT.md for the full debugging story\n\n"
            f"Solution: Use the ALIGNED file instead:\n"
            f"  - {ALIGNED_FILE}\n"
            f"{'='*60}\n"
        )

    # Warn if not using the canonical aligned file
    if 'aligned' not in filename.lower():
        print(f"WARNING: Expression file '{filename}' may not be aligned.")
        print(f"         Recommended file: {ALIGNED_FILE}")


# =============================================================================
# Feature and Expression Loading
# =============================================================================

FEATURE_DIR = Path(__file__).parent.parent / 'features'
ENCODERS = {
    'densenet': 'densenet_fm_features.npz',
    'dinov2': 'dinov2_features.npz',
    # 'uni': 'uni_features.npz',  # Add when HuggingFace access granted
}


def load_features(encoder_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features for an encoder."""
    if encoder_name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder_name}. Available: {list(ENCODERS.keys())}")

    path = FEATURE_DIR / ENCODERS[encoder_name]
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")

    data = np.load(path)
    return data['features'], data['patient_ids'], data['patch_indices']


def load_expression(
    expression_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load ALIGNED expression data.

    GUARDRAIL: This function REFUSES to load the misaligned expression file.

    Args:
        expression_file: Path to expression file (default: aligned 18k panel)

    Returns:
        expression: Expression matrix [n_patches, n_genes]
        patient_ids: Patient IDs for each patch
        patch_indices: Patch indices
        gene_names: List of gene names
    """
    import json

    if expression_file is None:
        path = FEATURE_DIR / ALIGNED_FILE
    else:
        path = Path(expression_file)

    # GUARDRAIL: Check for misaligned file
    _check_expression_file_guard(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Expression file not found: {path}\n"
            f"Required: aligned 18k-gene panel ({ALIGNED_FILE})"
        )

    data = np.load(path)

    # Load gene names
    gene_names_path = FEATURE_DIR / 'gene_names_all.json'
    if not gene_names_path.exists():
        raise FileNotFoundError(f"Gene names not found: {gene_names_path}")

    with open(gene_names_path, 'r') as f:
        gene_names = json.load(f)

    return (
        data['expression'],
        data['patient_ids'],
        data['patch_indices'],
        gene_names
    )


def load_gene_list(gene_list_path: str) -> List[str]:
    """Load a custom gene list from a text file (one gene per line)."""
    with open(gene_list_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


# =============================================================================
# LOOCV Evaluation
# =============================================================================

def loocv_evaluate_gene(
    features: np.ndarray,
    expression: np.ndarray,
    patient_ids: np.ndarray,
    gene_idx: int,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluate prediction for one gene using LOOCV (leave-one-patient-out).

    Args:
        features: Feature matrix [n_patches, embed_dim]
        expression: Expression matrix [n_patches, n_genes]
        patient_ids: Patient IDs for each patch
        gene_idx: Index of gene to evaluate
        alpha: Ridge regularization parameter

    Returns:
        Dict with MSE, Pearson r, Spearman ρ, SSIM-ST
    """
    y = expression[:, gene_idx]
    unique_patients = np.unique(patient_ids)

    y_true_all = []
    y_pred_all = []

    for held_out in unique_patients:
        train_mask = patient_ids != held_out
        test_mask = patient_ids == held_out

        X_train, y_train = features[train_mask], y[train_mask]
        X_test, y_test = features[test_mask], y[test_mask]

        # Skip if no variance in training data
        if np.std(y_train) < 1e-10:
            continue

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Ridge regression
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    if len(y_true_all) == 0:
        return {'mse': np.nan, 'pearson_r': np.nan, 'spearman_rho': np.nan, 'ssim_st': np.nan}

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Use centralized metric computation (GUARDRAIL: canonical SSIM-ST)
    return compute_metrics_for_gene(y_true_all, y_pred_all)


# =============================================================================
# Encoder Comparison
# =============================================================================

def evaluate_encoder(
    encoder_name: str,
    expression: np.ndarray,
    patient_ids_expr: np.ndarray,
    gene_names: List[str],
    gene_indices: Optional[np.ndarray] = None,
    top_k: int = 0,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Evaluate an encoder on expression prediction.

    Args:
        encoder_name: Name of encoder
        expression: Expression matrix [n_patches, n_genes]
        patient_ids_expr: Patient IDs for expression
        gene_names: List of gene names
        gene_indices: Specific gene indices to evaluate (overrides top_k)
        top_k: Evaluate top-k genes by variance (0 = all, default)
        alpha: Ridge regularization

    Returns:
        DataFrame with metrics for each gene
    """
    print(f"\n=== Evaluating {encoder_name} ===")

    # Load features
    features, patient_ids_feat, patch_indices = load_features(encoder_name)
    print(f"Features shape: {features.shape}")

    # Verify alignment
    if len(patient_ids_feat) != len(patient_ids_expr):
        raise ValueError(f"Feature/expression size mismatch: {len(patient_ids_feat)} vs {len(patient_ids_expr)}")

    # Select genes to evaluate
    if gene_indices is not None:
        eval_gene_indices = gene_indices
    elif top_k > 0:
        # Select top-k genes by variance
        gene_vars = np.var(expression, axis=0)
        eval_gene_indices = np.argsort(gene_vars)[::-1][:top_k]
    else:
        # Evaluate ALL genes (aligned 18k panel)
        eval_gene_indices = np.arange(len(gene_names))

    n_genes = len(eval_gene_indices)
    print(f"Evaluating {n_genes} genes (top_k={top_k}, gene_indices={'provided' if gene_indices is not None else 'auto'})...")

    results = []
    for gene_idx in tqdm(eval_gene_indices, desc=f"Evaluating ({encoder_name})"):
        metrics = loocv_evaluate_gene(
            features, expression, patient_ids_expr, gene_idx, alpha=alpha
        )
        metrics['gene'] = gene_names[gene_idx]
        metrics['gene_idx'] = gene_idx
        metrics['encoder'] = encoder_name
        results.append(metrics)

    return pd.DataFrame(results)


def compare_encoders(
    encoders: List[str],
    output_path: str,
    gene_list_path: Optional[str] = None,
    top_k: int = 0,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Compare multiple encoders on expression prediction.

    Args:
        encoders: List of encoder names to compare
        output_path: Path to save results
        gene_list_path: Path to custom gene list (optional)
        top_k: Number of top genes by variance (0 = all, default)
        alpha: Ridge regularization parameter

    Returns:
        Combined results DataFrame
    """
    # Load expression (GUARDRAIL: uses aligned 18k panel)
    expression, patient_ids, patch_indices, gene_names = load_expression()
    print(f"Expression shape: {expression.shape}")
    print(f"Patient distribution: {dict(zip(*np.unique(patient_ids, return_counts=True)))}")
    print(f"Total genes in aligned panel: {len(gene_names)}")

    # Determine which genes to evaluate
    gene_indices = None
    gene_set_name = "18k_aligned"

    if gene_list_path is not None:
        # Custom gene list
        custom_genes = load_gene_list(gene_list_path)
        gene_indices = np.array([
            i for i, g in enumerate(gene_names) if g in custom_genes
        ])
        gene_set_name = f"custom_{len(gene_indices)}"
        print(f"Using custom gene list: {len(gene_indices)}/{len(custom_genes)} genes found")
    elif top_k > 0:
        gene_set_name = f"top{top_k}_by_variance"

    all_results = []

    for encoder_name in encoders:
        try:
            results = evaluate_encoder(
                encoder_name,
                expression,
                patient_ids,
                gene_names,
                gene_indices=gene_indices,
                top_k=top_k,
                alpha=alpha,
            )
            results['gene_set'] = gene_set_name
            all_results.append(results)
        except FileNotFoundError as e:
            print(f"Warning: Features not found for {encoder_name}, skipping: {e}")
        except Exception as e:
            print(f"Error evaluating {encoder_name}: {e}")
            raise

    if not all_results:
        raise RuntimeError("No encoders evaluated successfully!")

    combined = pd.concat(all_results, ignore_index=True)

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    # Print summary
    _print_summary(combined, encoders)

    return combined


def _print_summary(combined: pd.DataFrame, encoders: List[str]) -> None:
    """Print a summary of encoder comparison results."""
    print("\n" + "=" * 70)
    print("ENCODER COMPARISON SUMMARY")
    print("=" * 70)

    summary = combined.groupby('encoder').agg({
        'mse': ['mean', 'std', 'median'],
        'ssim_st': ['mean', 'std', 'median'],
        'pearson_r': ['mean', 'std', 'median'],
        'spearman_rho': ['mean', 'std', 'median'],
    }).round(4)

    print("\nPRIMARY METRICS (MSE↓, SSIM-ST↑):")
    print("-" * 70)
    for encoder in encoders:
        if encoder in summary.index:
            mse_mean = summary.loc[encoder, ('mse', 'mean')]
            mse_med = summary.loc[encoder, ('mse', 'median')]
            ssim_mean = summary.loc[encoder, ('ssim_st', 'mean')]
            ssim_med = summary.loc[encoder, ('ssim_st', 'median')]
            print(f"{encoder:20s}: MSE = {mse_mean:.4f} (med={mse_med:.4f}), "
                  f"SSIM-ST = {ssim_mean:.4f} (med={ssim_med:.4f})")

    print("\nSECONDARY METRICS (Pearson r, Spearman ρ):")
    print("-" * 70)
    for encoder in encoders:
        if encoder in summary.index:
            r_mean = summary.loc[encoder, ('pearson_r', 'mean')]
            r_med = summary.loc[encoder, ('pearson_r', 'median')]
            rho_mean = summary.loc[encoder, ('spearman_rho', 'mean')]
            rho_med = summary.loc[encoder, ('spearman_rho', 'median')]
            print(f"{encoder:20s}: r = {r_mean:.4f} (med={r_med:.4f}), "
                  f"ρ = {rho_mean:.4f} (med={rho_med:.4f})")

    # Count genes above thresholds
    print("\nGenes above Pearson r thresholds:")
    print("-" * 70)
    for encoder in encoders:
        enc_data = combined[combined['encoder'] == encoder]
        r_values = enc_data['pearson_r'].dropna()
        n_above_08 = (r_values > 0.8).sum()
        n_above_05 = (r_values > 0.5).sum()
        n_above_03 = (r_values > 0.3).sum()
        total = len(r_values)
        print(f"{encoder:20s}: r>0.8: {n_above_08}/{total}, "
              f"r>0.5: {n_above_05}/{total}, r>0.3: {n_above_03}/{total}")

    # Top genes by Pearson r
    print("\nTop 10 genes by Pearson r:")
    print("-" * 70)
    top_genes = combined.nlargest(10, 'pearson_r')[['gene', 'encoder', 'mse', 'ssim_st', 'pearson_r']]
    print(top_genes.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Compare encoders for H&E → ST prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate top 100 genes by variance (fast)
    python compare_encoders.py --top_k 100

    # Evaluate full aligned 18k-gene panel (recommended)
    python compare_encoders.py --top_k 0 --output results/encoder_comparison_18k.csv

    # Use custom gene list
    python compare_encoders.py --genes genes/top500.txt --output results/custom.csv
        """
    )
    parser.add_argument('--encoders', type=str, nargs='+', default=['densenet', 'dinov2'],
                       help='Encoders to compare (default: densenet dinov2)')
    parser.add_argument('--output', type=str, default='results/encoder_comparison.csv',
                       help='Output path for results')
    parser.add_argument('--genes', type=str, default=None,
                       help='Path to custom gene list (one gene per line)')
    parser.add_argument('--top_k', type=int, default=0,
                       help='Number of top genes by variance (0=all, default=0)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Ridge regularization parameter')

    args = parser.parse_args()

    compare_encoders(
        encoders=args.encoders,
        output_path=args.output,
        gene_list_path=args.genes,
        top_k=args.top_k,
        alpha=args.alpha,
    )


if __name__ == '__main__':
    main()
