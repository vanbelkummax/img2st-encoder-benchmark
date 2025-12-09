#!/usr/bin/env python3
"""
Extract expression for ALL genes aligned to existing patches.

FIXED VERSION - Uses same coordinate system as preprocess_visium_hd.py:
- Uses pxl_row_in_fullres/pxl_col_in_fullres (pixel coordinates)
- Scales by tissue_hires_scalef
- Groups into patches using PATCH_SIZE=256

This ensures alignment with the 50-gene preprocessed data.
"""

import sys
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from tqdm import tqdm

# Match preprocessing constants
PATCH_SIZE = 256  # pixels
GRID_SIZE = 32    # bins per patch


def load_expression_matrix(h5_path: str):
    """Load full expression matrix from h5 file."""
    with h5py.File(h5_path, 'r') as f:
        gene_names = [g.decode() for g in f['matrix']['features']['name'][:]]
        barcodes = [b.decode() for b in f['matrix']['barcodes'][:]]

        data = f['matrix']['data'][:]
        indices = f['matrix']['indices'][:]
        indptr = f['matrix']['indptr'][:]
        shape = f['matrix']['shape'][:]

        # Create sparse matrix (genes x barcodes)
        expr_sparse = sparse.csc_matrix((data, indices, indptr), shape=shape)

    return gene_names, barcodes, expr_sparse


def load_positions_and_scalefactors(data_dir: str, bin_size: str = '008um'):
    """Load positions and scale factors matching preprocessing."""
    pos_path = os.path.join(data_dir, 'binned_outputs', f'square_{bin_size}',
                            'spatial', 'tissue_positions.parquet')
    scale_path = os.path.join(data_dir, 'binned_outputs', f'square_{bin_size}',
                              'spatial', 'scalefactors_json.json')

    positions = pd.read_parquet(pos_path)

    with open(scale_path) as f:
        scalefactors = json.load(f)

    return positions, scalefactors


def compute_patch_assignments(positions: pd.DataFrame, scalefactors: dict):
    """
    Compute patch assignments using EXACT same logic as preprocessing.

    Key: Use pxl_row_in_fullres * scale, NOT array_row!
    """
    positions = positions.copy()

    # Get scale factor (same as preprocessing)
    scale = scalefactors.get('tissue_hires_scalef', 1.0)

    # Scale pixel coordinates (EXACT match to preprocessing lines 188-189)
    positions['pxl_row_scaled'] = (positions['pxl_row_in_fullres'] * scale).astype(int)
    positions['pxl_col_scaled'] = (positions['pxl_col_in_fullres'] * scale).astype(int)

    # Compute patch assignment (EXACT match to preprocessing lines 192-193)
    positions['patch_row'] = (positions['pxl_row_scaled'] // PATCH_SIZE).astype(int)
    positions['patch_col'] = (positions['pxl_col_scaled'] // PATCH_SIZE).astype(int)

    return positions, scale


def get_barcodes_for_patch(patch_row: int, patch_col: int,
                           positions_df: pd.DataFrame) -> list:
    """Get barcodes that fall within a patch using pixel-based coordinates."""
    mask = (
        (positions_df['patch_row'] == patch_row) &
        (positions_df['patch_col'] == patch_col) &
        (positions_df['in_tissue'] == 1)
    )
    return positions_df[mask]['barcode'].tolist()


def extract_patch_expression(barcodes: list, barcode_to_idx: dict,
                             expr_matrix: sparse.csc_matrix, n_genes: int):
    """Extract and aggregate expression for barcodes in a patch."""
    indices = [barcode_to_idx[b] for b in barcodes if b in barcode_to_idx]

    if len(indices) == 0:
        return np.zeros(n_genes)

    # expr_matrix is genes x barcodes (CSC format)
    patch_expr = expr_matrix[:, indices].toarray()  # [n_genes, n_barcodes]

    # Mean across barcodes (matches preprocessing aggregation approach)
    mean_expr = patch_expr.mean(axis=1)

    return mean_expr


def validate_alignment(positions_df: pd.DataFrame, patches: list, patient: str):
    """Validate that computed patches match preprocessed patches."""
    # Get unique patches from preprocessed data
    preproc_patches = set((p['patch_row'], p['patch_col']) for p in patches)

    # Get unique patches from our computation (in_tissue only)
    in_tissue = positions_df[positions_df['in_tissue'] == 1]
    computed_patches = set(zip(in_tissue['patch_row'], in_tissue['patch_col']))

    overlap = preproc_patches & computed_patches
    only_preproc = preproc_patches - computed_patches
    only_computed = computed_patches - preproc_patches

    print(f"  Alignment check for {patient}:")
    print(f"    Preprocessed patches: {len(preproc_patches)}")
    print(f"    Computed patches: {len(computed_patches)}")
    print(f"    Overlap: {len(overlap)}")
    print(f"    Only in preprocessed: {len(only_preproc)}")
    print(f"    Only in computed: {len(only_computed)}")

    if len(overlap) == len(preproc_patches):
        print(f"    ✓ PERFECT ALIGNMENT")
        return True
    else:
        print(f"    ✗ ALIGNMENT MISMATCH")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                       default='/mnt/x/img2st_rotation_demo/downloads/crc_hd',
                       help='Directory with raw Visium HD data')
    parser.add_argument('--patches_dir', type=str,
                       default='/mnt/x/img2st_rotation_demo/processed_crc_aligned',
                       help='Directory with processed patches')
    parser.add_argument('--output_dir', type=str,
                       default='/home/user/img2st-baseline-comparison/features',
                       help='Output directory')
    parser.add_argument('--bin_size', type=str, default='008um')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate alignment without extracting')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_expression = []
    all_patient_ids = []
    all_patch_indices = []
    gene_names = None
    all_aligned = True

    for patient in ['P1', 'P2', 'P5']:
        print(f"\n{'='*60}")
        print(f"Processing {patient}")
        print('='*60)

        # Paths
        patient_data_dir = os.path.join(args.data_dir, patient)
        h5_path = os.path.join(patient_data_dir, 'binned_outputs',
                               f'square_{args.bin_size}', 'filtered_feature_bc_matrix.h5')
        patches_path = os.path.join(args.patches_dir, patient, 'patches.npy')

        if not os.path.exists(h5_path):
            print(f"Missing h5 file for {patient}, skipping...")
            continue
        if not os.path.exists(patches_path):
            print(f"Missing patches file for {patient}, skipping...")
            continue

        # Load positions and scale factors
        print("Loading positions and scale factors...")
        positions, scalefactors = load_positions_and_scalefactors(patient_data_dir, args.bin_size)
        scale = scalefactors.get('tissue_hires_scalef', 1.0)
        print(f"  Scale factor: {scale:.6f}")
        print(f"  Barcodes: {len(positions)}")

        # Compute patch assignments using PIXEL coordinates (not array!)
        print("Computing patch assignments (pixel-based)...")
        positions, scale = compute_patch_assignments(positions, scalefactors)

        in_tissue = positions[positions['in_tissue'] == 1]
        print(f"  In-tissue barcodes: {len(in_tissue)}")
        print(f"  Pixel range: row [{in_tissue['pxl_row_scaled'].min()}, {in_tissue['pxl_row_scaled'].max()}]")
        print(f"  Pixel range: col [{in_tissue['pxl_col_scaled'].min()}, {in_tissue['pxl_col_scaled'].max()}]")
        print(f"  Patch range: row [{in_tissue['patch_row'].min()}, {in_tissue['patch_row'].max()}]")
        print(f"  Patch range: col [{in_tissue['patch_col'].min()}, {in_tissue['patch_col'].max()}]")

        # Load preprocessed patches
        patches = np.load(patches_path, allow_pickle=True)
        print(f"  Preprocessed patches: {len(patches)}")

        preproc_rows = [p['patch_row'] for p in patches]
        preproc_cols = [p['patch_col'] for p in patches]
        print(f"  Preproc patch range: row [{min(preproc_rows)}, {max(preproc_rows)}]")
        print(f"  Preproc patch range: col [{min(preproc_cols)}, {max(preproc_cols)}]")

        # Validate alignment
        aligned = validate_alignment(positions, patches, patient)
        if not aligned:
            all_aligned = False

        if args.validate_only:
            continue

        # Load expression
        print("Loading expression matrix...")
        patient_genes, barcodes, expr_sparse = load_expression_matrix(h5_path)
        print(f"  Genes: {len(patient_genes)}, Barcodes: {len(barcodes)}")

        if gene_names is None:
            gene_names = patient_genes
        else:
            if patient_genes != gene_names:
                print("  WARNING: Gene order differs!")

        # Create barcode -> index mapping
        barcode_to_idx = {b: i for i, b in enumerate(barcodes)}

        # Extract expression for each patch (using same order as preprocessed)
        print("Extracting expression for all genes...")
        patient_expression = []

        for i, patch in enumerate(tqdm(patches, desc=f"  {patient}")):
            barcodes_in_patch = get_barcodes_for_patch(
                patch['patch_row'], patch['patch_col'], positions
            )

            patch_expr = extract_patch_expression(
                barcodes_in_patch, barcode_to_idx, expr_sparse, len(gene_names)
            )

            patient_expression.append(patch_expr)
            all_patient_ids.append(patient)
            all_patch_indices.append(i)

        patient_expression = np.array(patient_expression)
        print(f"  Expression shape: {patient_expression.shape}")
        print(f"  Non-zero rate: {(patient_expression > 0).mean():.2%}")

        all_expression.append(patient_expression)

    if args.validate_only:
        print(f"\n{'='*60}")
        print("Validation Summary")
        print('='*60)
        if all_aligned:
            print("✓ ALL PATIENTS ALIGNED - Ready to extract")
        else:
            print("✗ ALIGNMENT ISSUES - Check coordinate system")
        return

    # Combine all patients
    all_expression = np.vstack(all_expression)

    print(f"\n{'='*60}")
    print("Combined Dataset")
    print('='*60)
    print(f"Expression shape: {all_expression.shape}")
    print(f"Patients: {len(set(all_patient_ids))}")
    print(f"Genes: {len(gene_names)}")

    # Log-transform and normalize
    print("\nNormalizing expression (log1p + z-score)...")
    expr_log = np.log1p(all_expression)

    # Z-score per gene
    expr_mean = expr_log.mean(axis=0, keepdims=True)
    expr_std = expr_log.std(axis=0, keepdims=True) + 1e-8
    expr_zscore = (expr_log - expr_mean) / expr_std

    # Save
    output_path = os.path.join(args.output_dir, 'all_genes_expression_aligned.npz')
    np.savez_compressed(
        output_path,
        expression=expr_zscore,
        expression_raw=all_expression,
        patient_ids=np.array(all_patient_ids),
        patch_indices=np.array(all_patch_indices),
    )

    # Save gene names
    gene_names_path = os.path.join(args.output_dir, 'gene_names_all.json')
    with open(gene_names_path, 'w') as f:
        json.dump(gene_names, f)

    print(f"\nSaved: {output_path}")
    print(f"Saved: {gene_names_path}")

    # Quick stats
    print("\nExpression Statistics:")
    print(f"  Total genes: {len(gene_names)}")
    nonzero_per_gene = (all_expression > 0).mean(axis=0)
    print(f"  Genes with >10% non-zero: {(nonzero_per_gene > 0.1).sum()}")
    print(f"  Genes with >25% non-zero: {(nonzero_per_gene > 0.25).sum()}")
    print(f"  Genes with >50% non-zero: {(nonzero_per_gene > 0.5).sum()}")


if __name__ == '__main__':
    main()
