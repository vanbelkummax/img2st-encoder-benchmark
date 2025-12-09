#!/usr/bin/env python3
"""
Apply Mean Completion to Aligned CRC Data

Fills sparse bins with per-patient, per-gene mean values.
Creates new processed directory with completed labels while preserving
all other metadata (raw_label, norm_params, split_info, etc.).

Usage:
    python apply_mean_completion.py \
        --input_dir /mnt/x/img2st_rotation_demo/processed_crc_aligned \
        --output_dir /mnt/x/img2st_rotation_demo/processed_crc_completed_mean

Implementation follows design document:
    docs/plans/2025-11-25-baseline-completion-comparison-design.md
"""

import argparse
import shutil
import json
import numpy as np
from pathlib import Path


def mean_completion(expr_zscore_per_patient, patient_gene_means, sparse_mask):
    """
    Fill sparse bins with per-patient, per-gene mean.

    Args:
        expr_zscore_per_patient: [bins, genes] z-scored expression for one patient
        patient_gene_means: [genes] mean z-score for each gene (computed on training patients)
        sparse_mask: [bins, genes] boolean mask where True = unmeasured bin
                     (sourced from raw_counts == 0 before normalization)

    Returns:
        completed: [bins, genes] with sparse bins filled

    Note:
        sparse_mask should be computed from raw counts (pre-normalization) to
        distinguish true unmeasured bins from bins with measured zero expression.
        If raw count metadata unavailable, use expr_zscore == 0 as approximation.
    """
    # Broadcast gene means to all bins
    gene_means_broadcast = np.tile(patient_gene_means, (expr_zscore_per_patient.shape[0], 1))

    # Fill sparse bins with gene means
    result = np.where(sparse_mask, gene_means_broadcast, expr_zscore_per_patient)

    return result


def compute_patient_gene_means(patches):
    """Compute per-gene mean z-scores across all patches for one patient.

    Args:
        patches: List of patch dictionaries with 'label' (z-scored)

    Returns:
        gene_means: [genes] mean z-score for each gene
    """
    all_labels = np.stack([p['label'] for p in patches], axis=0)  # [N_patches, bins, genes]
    # Flatten across patches and bins, compute mean per gene
    flat_labels = all_labels.reshape(-1, all_labels.shape[-1])  # [N_patches*bins, genes]
    gene_means = np.mean(flat_labels, axis=0)  # [genes]
    return gene_means


def process_patches(patches, gene_means):
    """Apply mean completion to all patches and compute statistics.

    Args:
        patches: List of patch dictionaries with 'label' and 'raw_label'
        gene_means: [genes] per-gene mean values for this patient

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

        # Apply mean completion
        completed_label = mean_completion(label, gene_means, sparse_mask)

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
        'fill_rate': float(total_filled / total_bins) if total_bins > 0 else 0.0
    }

    return completed_patches, stats


def main():
    parser = argparse.ArgumentParser(description="Apply mean completion to CRC data")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory (processed_crc_aligned)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for completed data')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Mean Completion for CRC Aligned Data")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
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
    patient_gene_means = {}

    # First pass: Compute per-patient gene means
    print(f"\n{'─' * 70}")
    print("Computing per-patient gene means...")
    print(f"{'─' * 70}")

    for patient in patients:
        patient_input = input_dir / patient
        if not patient_input.exists():
            continue

        patches_file = patient_input / 'patches.npy'
        patches = np.load(patches_file, allow_pickle=True).tolist()

        gene_means = compute_patient_gene_means(patches)
        patient_gene_means[patient] = gene_means

        print(f"  {patient}: {len(patches)} patches, mean z-score range: "
              f"[{gene_means.min():.3f}, {gene_means.max():.3f}]")

    # Second pass: Apply completion
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

        # Apply mean completion using this patient's gene means
        gene_means = patient_gene_means[patient]
        completed_patches, stats = process_patches(patches, gene_means)
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
        'method': 'mean',
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
    print(f"Method: Mean (per-patient, per-gene)")
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
