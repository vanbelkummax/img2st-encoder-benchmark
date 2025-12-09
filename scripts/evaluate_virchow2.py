#!/usr/bin/env python3
"""
Evaluate Virchow2 pathology foundation model vs baseline encoders.

Virchow2 (PAIGE AI):
- ViT-H/14 (632M params) trained on 3.1M H&E WSIs
- Uses SwiGLU MLP layers
- Output: 2560d (1280 class token + 1280 mean patch token)
- Multi-magnification training includes 2um/pixel (Visium HD resolution)

Compares against DenseNet (ImageNet) and DINOv2 (natural images) baselines.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from spatial_metrics import compute_metrics_for_gene


FEATURES_DIR = Path('/home/user/img2st-baseline-comparison/features')
RESULTS_DIR = Path('/home/user/img2st-baseline-comparison/results')


def load_features():
    """Load all encoder features."""
    print("Loading features...")

    # Baseline encoders (256px)
    densenet_256 = np.load(FEATURES_DIR / 'densenet_fm_features.npz', allow_pickle=True)
    dinov2_256 = np.load(FEATURES_DIR / 'dinov2_features.npz', allow_pickle=True)

    # Virchow2 (256px and 512px context)
    virchow2_256 = np.load(FEATURES_DIR / 'virchow2_256_features.npz', allow_pickle=True)
    virchow2_512 = np.load(FEATURES_DIR / 'virchow2_512context.npz', allow_pickle=True)

    patient_ids = densenet_256['patient_ids']

    # Verify alignment
    for name, data in [('dinov2', dinov2_256), ('virchow2_256', virchow2_256), ('virchow2_512', virchow2_512)]:
        if not np.array_equal(patient_ids, data['patient_ids']):
            raise ValueError(f"Patient IDs don't match for {name}!")

    # StandardScaler for each encoder
    scaler = StandardScaler()

    encoders = {
        'densenet_256': scaler.fit_transform(densenet_256['features']),
        'dinov2_256': scaler.fit_transform(dinov2_256['features']),
        'virchow2_256': scaler.fit_transform(virchow2_256['features']),
        'virchow2_512': scaler.fit_transform(virchow2_512['features']),
    }

    # Concat baselines (256px only)
    concat_baseline = np.concatenate([
        scaler.fit_transform(densenet_256['features']),
        scaler.fit_transform(dinov2_256['features'])
    ], axis=1)
    encoders['concat_baseline'] = scaler.fit_transform(concat_baseline)

    # Concat with Virchow2 (256px)
    concat_virchow2 = np.concatenate([
        scaler.fit_transform(densenet_256['features']),
        scaler.fit_transform(dinov2_256['features']),
        scaler.fit_transform(virchow2_256['features'])
    ], axis=1)
    encoders['concat_with_virchow2'] = scaler.fit_transform(concat_virchow2)

    # Virchow2 multi-scale concat (256 + 512)
    virchow2_multiscale = np.concatenate([
        scaler.fit_transform(virchow2_256['features']),
        scaler.fit_transform(virchow2_512['features'])
    ], axis=1)
    encoders['virchow2_multiscale'] = scaler.fit_transform(virchow2_multiscale)

    print(f"Loaded encoders:")
    for name, feat in encoders.items():
        print(f"  {name}: {feat.shape}")

    # Expression
    print("\nLoading expression...")
    expr_data = np.load(FEATURES_DIR / 'all_genes_expression_aligned.npz', allow_pickle=True)

    with open(FEATURES_DIR / 'gene_names_all.json') as f:
        gene_names = json.load(f)

    return encoders, expr_data['expression'], gene_names, patient_ids


def evaluate_gene_loocv(features, expression, patient_ids, alpha=1.0):
    """Evaluate gene with LOOCV, returning Pearson r."""
    patients = np.unique(patient_ids)
    y_true_all = []
    y_pred_all = []

    for test_patient in patients:
        test_mask = patient_ids == test_patient
        train_mask = ~test_mask

        X_train, y_train = features[train_mask], expression[train_mask]
        X_test, y_test = features[test_mask], expression[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    if len(y_true_all) == 0:
        return np.nan

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    metrics = compute_metrics_for_gene(y_true, y_pred)
    return metrics['pearson_r']


def main():
    parser = argparse.ArgumentParser(description='Evaluate Virchow2 vs baselines')
    parser.add_argument('--top_k', type=int, default=0, help='Top K genes (0=all)')
    parser.add_argument('--output', type=str, default='results/virchow2_eval.csv')
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 70)
    print("VIRCHOW2 PATHOLOGY FM EVALUATION")
    print("=" * 70)

    encoders, expression, gene_names, patient_ids = load_features()

    n_genes = len(gene_names)
    if args.top_k > 0 and args.top_k < n_genes:
        gene_vars = np.var(expression, axis=0)
        eval_indices = np.argsort(gene_vars)[-args.top_k:][::-1]
        print(f"\nEvaluating top {args.top_k} genes by variance...")
    else:
        eval_indices = list(range(n_genes))
        print(f"\nEvaluating all {n_genes} genes...")

    results = []
    for gene_idx in tqdm(eval_indices, desc="Evaluating genes"):
        gene_name = gene_names[gene_idx]
        gene_expr = expression[:, gene_idx]

        row = {'gene': gene_name, 'gene_idx': gene_idx}

        for enc_name, features in encoders.items():
            r = evaluate_gene_loocv(features, gene_expr, patient_ids, args.alpha)
            row[f'{enc_name}_r'] = r

        results.append(row)

    df = pd.DataFrame(results)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    encoder_order = [
        'densenet_256', 'dinov2_256', 'virchow2_256', 'virchow2_512',
        'concat_baseline', 'virchow2_multiscale', 'concat_with_virchow2'
    ]

    for enc_name in encoder_order:
        col = f'{enc_name}_r'
        if col in df.columns:
            vals = df[col].dropna()
            print(f"\n{enc_name.upper()} ({len(vals)} genes):")
            print(f"  Mean r: {vals.mean():.4f}")
            print(f"  Median r: {vals.median():.4f}")
            print(f"  Genes r>0.5: {(vals > 0.5).sum()} ({(vals > 0.5).mean()*100:.1f}%)")
            print(f"  Genes r>0.8: {(vals > 0.8).sum()} ({(vals > 0.8).mean()*100:.1f}%)")

    # Key comparisons
    print("\n" + "-" * 70)
    print("KEY COMPARISONS")
    print("-" * 70)

    # Virchow2 vs DINOv2
    if 'dinov2_256_r' in df.columns and 'virchow2_256_r' in df.columns:
        dinov2 = df['dinov2_256_r']
        virchow2 = df['virchow2_256_r']
        print(f"\nVirchow2 vs DINOv2 (256px):")
        print(f"  DINOv2 mean r: {dinov2.mean():.4f}")
        print(f"  Virchow2 mean r: {virchow2.mean():.4f}")
        print(f"  Delta: {(virchow2 - dinov2).mean():+.4f}")
        print(f"  Virchow2 wins: {(virchow2 > dinov2).mean()*100:.1f}%")

    # Virchow2 multi-scale vs single
    if 'virchow2_256_r' in df.columns and 'virchow2_multiscale_r' in df.columns:
        v256 = df['virchow2_256_r']
        vmulti = df['virchow2_multiscale_r']
        print(f"\nVirchow2 multi-scale vs 256px:")
        print(f"  256px mean r: {v256.mean():.4f}")
        print(f"  Multi-scale mean r: {vmulti.mean():.4f}")
        print(f"  Delta: {(vmulti - v256).mean():+.4f}")

    # Adding Virchow2 to baseline concat
    if 'concat_baseline_r' in df.columns and 'concat_with_virchow2_r' in df.columns:
        base = df['concat_baseline_r']
        with_v = df['concat_with_virchow2_r']
        print(f"\nAdding Virchow2 to baseline concat:")
        print(f"  Baseline concat mean r: {base.mean():.4f}")
        print(f"  + Virchow2 mean r: {with_v.mean():.4f}")
        print(f"  Delta: {(with_v - base).mean():+.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return df


if __name__ == '__main__':
    main()
