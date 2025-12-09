#!/usr/bin/env python3
"""
Apply Gaussian Completion to Aligned CRC Data

Fills sparse bins using Gaussian-smoothed values from neighboring bins.
Creates new processed directory with completed labels while preserving
all other metadata (raw_label, norm_params, split_info, etc.).

Usage:
    python apply_gaussian_completion.py \
        --input_dir /mnt/x/img2st_rotation_demo/processed_crc_aligned \
        --output_dir /mnt/x/img2st_rotation_demo/processed_crc_completed_gaussian \
        --sigma 2.0 \
        --truncate 4.0

Implementation follows design document:
    docs/plans/2025-11-25-baseline-completion-comparison-design.md
"""

import argparse
import shutil
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter


def gaussian_completion(expr_zscore, sparse_mask, sigma=2.0, truncate=4.0):
    """
    Fill sparse bins with Gaussian-smoothed values.

    Args:
        expr_zscore: [bins, genes] z-scored expression on 32×32 grid
        sparse_mask: [bins, genes] boolean mask where True = unmeasured bin
                     (sourced from raw_counts == 0 before normalization)
        sigma: Gaussian kernel width (default: 2)
        truncate: Kernel truncation (default: 4.0 → 17×17 kernel)

    Returns:
        completed: [bins, genes] with sparse bins filled

    Note:
        sparse_mask should be computed from raw counts (pre-normalization) to
        distinguish true unmeasured bins from bins with measured zero expression.
        If raw count metadata unavailable, use expr_zscore == 0 as approximation.
    """
    n_bins, n_genes = expr_zscore.shape
    h = w = int(np.sqrt(n_bins))

    # Reshape to [H, W, genes] for 2D Gaussian filtering
    expr_grid = expr_zscore.reshape(h, w, n_genes)
    sparse_grid = sparse_mask.reshape(h, w, n_genes)

    # Apply Gaussian filter per gene (mode='reflect' for boundary handling)
    smoothed = np.zeros_like(expr_grid)
    for g in range(n_genes):
        smoothed[:, :, g] = gaussian_filter(
            expr_grid[:, :, g],
            sigma=sigma,
            mode='reflect',
            truncate=truncate
        )

    # Fill only sparse bins, preserve observed values
    completed_grid = np.where(sparse_grid, smoothed, expr_grid)

    # Flatten back to [bins, genes]
    return completed_grid.reshape(n_bins, n_genes)


def process_patches(patches, sigma, truncate):
    """Apply Gaussian completion to all patches and compute statistics.

    Args:
        patches: List of patch dictionaries with 'label' and 'raw_label'
        sigma: Gaussian kernel width
        truncate: Kernel truncation factor

    Returns:
        completed_patches: List with completed labels
        stats: Dictionary of completion statistics
    """
    completed_patches = []
    total_bins = 0
    total_filled = 0

    for patch in patches:
        # Extract data
        label = patch['label']  # [1024, 50] z-scored
        raw_label = patch['raw_label']  # [1024, 50] raw counts

        # Compute sparse mask from raw counts
        sparse_mask = (raw_label == 0)

        # Apply Gaussian completion
        completed_label = gaussian_completion(label, sparse_mask, sigma, truncate)

        # Count filled bins (any gene in bin was sparse)
        bins_filled = np.any(sparse_mask, axis=1).sum()
        total_bins += label.shape[0]
        total_filled += bins_filled

        # Create new patch with completed label
        new_patch = patch.copy()
        new_patch['label'] = completed_label
        completed_patches.append(new_patch)

    stats = {
        'total_bins': int(total_bins),
        'bins_filled': int(total_filled),
        'fill_rate': float(total_filled / total_bins) if total_bins > 0 else 0.0,
        'sigma': float(sigma),
        'truncate': float(truncate),
        'kernel_size': int(2 * np.ceil(sigma * truncate) + 1)
    }

    return completed_patches, stats


def main():
    parser = argparse.ArgumentParser(description="Apply Gaussian completion to CRC data")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory (processed_crc_aligned)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for completed data')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Gaussian kernel width (default: 2.0)')
    parser.add_argument('--truncate', type=float, default=4.0,
                        help='Kernel truncation factor (default: 4.0)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Gaussian Completion for CRC Aligned Data")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Sigma:  {args.sigma} (truncate={args.truncate})")
    kernel_size = int(2 * np.ceil(args.sigma * args.truncate) + 1)
    print(f"Kernel: {kernel_size}×{kernel_size} effective")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata files (unchanged)
    print("Copying metadata files...")
    for fname in ['gene_names.json', 'split_info.json']:
        src = input_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"  ✓ {fname}")

    # Process each patient
    patients = ['P1', 'P2', 'P5']
    all_stats = {}

    for patient in patients:
        print(f"\n{'─' * 70}")
        print(f"Processing {patient}...")
        print(f"{'─' * 70}")

        patient_input = input_dir / patient
        patient_output = output_dir / patient

        if not patient_input.exists():
            print(f"  ⚠ {patient} not found, skipping")
            continue

        patient_output.mkdir(parents=True, exist_ok=True)

        # Copy static files
        for fname in ['gene_names.json', 'metadata.json', 'norm_params.json']:
            src = patient_input / fname
            if src.exists():
                shutil.copy2(src, patient_output / fname)

        # Copy images directory
        img_src = patient_input / 'images'
        img_dst = patient_output / 'images'
        if img_src.exists():
            if img_dst.exists():
                shutil.rmtree(img_dst)
            shutil.copytree(img_src, img_dst)
            print(f"  ✓ Copied {len(list(img_src.glob('*.png')))} images")

        # Load and process patches
        patches_file = patient_input / 'patches.npy'
        patches = np.load(patches_file, allow_pickle=True).tolist()
        print(f"  Loaded {len(patches)} patches")

        # Apply Gaussian completion
        completed_patches, stats = process_patches(patches, args.sigma, args.truncate)
        all_stats[patient] = stats

        # Save completed patches
        np.save(patient_output / 'patches.npy', np.array(completed_patches, dtype=object))
        print(f"  ✓ Completed {stats['bins_filled']:,}/{stats['total_bins']:,} bins "
              f"({stats['fill_rate']*100:.1f}%)")

    # Recreate all_patches.npy, train_patches.npy, val_patches.npy
    print(f"\n{'─' * 70}")
    print("Recreating combined patch files...")
    print(f"{'─' * 70}")

    # Load split info
    with open(input_dir / 'split_info.json') as f:
        split_info = json.load(f)

    train_slides = split_info['train_slides']
    val_slides = split_info['val_slides']

    all_patches = []
    train_patches = []
    val_patches = []

    for patient in patients:
        patient_file = output_dir / patient / 'patches.npy'
        if patient_file.exists():
            patches = np.load(patient_file, allow_pickle=True).tolist()

            # Add slide identifier to each patch
            for patch in patches:
                patch['slide'] = patient

            all_patches.extend(patches)

            if patient in train_slides:
                train_patches.extend(patches)
            elif patient in val_slides:
                val_patches.extend(patches)

    # Save combined files
    np.save(output_dir / 'all_patches.npy', np.array(all_patches, dtype=object))
    np.save(output_dir / 'train_patches.npy', np.array(train_patches, dtype=object))
    np.save(output_dir / 'val_patches.npy', np.array(val_patches, dtype=object))

    print(f"  ✓ all_patches.npy: {len(all_patches)} patches")
    print(f"  ✓ train_patches.npy: {len(train_patches)} patches")
    print(f"  ✓ val_patches.npy: {len(val_patches)} patches")

    # Save completion statistics
    completion_stats = {
        'method': 'gaussian',
        'sigma': args.sigma,
        'truncate': args.truncate,
        'kernel_size': kernel_size,
        'per_patient': all_stats,
        'total': {
            'bins_filled': sum(s['bins_filled'] for s in all_stats.values()),
            'total_bins': sum(s['total_bins'] for s in all_stats.values()),
            'fill_rate': np.mean([s['fill_rate'] for s in all_stats.values()])
        }
    }

    with open(output_dir / 'completion_stats.json', 'w') as f:
        json.dump(completion_stats, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Completion Summary")
    print(f"{'=' * 70}")
    print(f"Method: Gaussian (σ={args.sigma}, truncate={args.truncate})")
    print(f"Kernel: {kernel_size}×{kernel_size}")
    print(f"Total bins filled: {completion_stats['total']['bins_filled']:,} / "
          f"{completion_stats['total']['total_bins']:,} "
          f"({completion_stats['total']['fill_rate']*100:.1f}%)")
    print(f"\nPer-patient fill rates:")
    for patient, stats in all_stats.items():
        print(f"  {patient}: {stats['fill_rate']*100:.1f}%")
    print(f"\n✓ Output saved to {output_dir}")
    print(f"✓ Statistics saved to {output_dir / 'completion_stats.json'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
