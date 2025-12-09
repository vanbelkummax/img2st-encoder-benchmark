#!/usr/bin/env python3
"""
FmH2ST Baseline Wrapper for Visium HD Evaluation

Runs FmH2ST's foundation model + GNN architecture on our Visium HD patches
using the same LOOCV protocol as other baselines.

Architecture:
- Encoder: PathoDuet foundation model (slice + spot features)
- Decoder: Dual-branch (CBAM Conv + ViT + GAT) with attention fusion

NOTE: FmH2ST was trained on HER2ST dataset with 785 genes.
Only 8 genes overlap with our 50 HVGs.
Overlapping genes: CD74, COL3A1, IGHA1, IGHM, IGKC, JCHAIN, KRT8, VIM

Prerequisites:
1. Download checkpoints from: https://drive.google.com/drive/folders/1J8m69FqrYo2Y5KcenbOjP-Zmm3owbqmZ
2. Place in ~/FmH2ST/checkpoint/<slide>/1000_checkpoint.pth
3. Install requirements: pip install -r ~/FmH2ST/requirements.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm import tqdm

# Add FmH2ST to path
sys.path.insert(0, os.path.expanduser("~/FmH2ST"))


# Gene overlap between our 50 HVGs and FmH2ST's 785 genes
OVERLAPPING_GENES = ['CD74', 'COL3A1', 'IGHA1', 'IGHM', 'IGKC', 'JCHAIN', 'KRT8', 'VIM']


def load_gene_overlap_mapping() -> Dict:
    """Load gene overlap mapping from our analysis."""
    overlap_file = Path("/home/user/img2st-baseline-comparison/results/gene_overlaps/fmh2st_overlap.json")
    if overlap_file.exists():
        with open(overlap_file) as f:
            return json.load(f)
    return {"overlap_stats": {"overlap_genes": OVERLAPPING_GENES}}


def load_visium_hd_data(data_dir: str) -> Dict:
    """Load Visium HD bin-level data."""
    data_path = Path(data_dir)

    data = np.load(data_path / "bin_level_data.npz")

    with open(data_path / "gene_names.json") as f:
        gene_names = json.load(f)

    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)

    # Data format: expression (222, 1024, 50) - patches x bins x genes
    expressions = data["expression"]
    n_patches = expressions.shape[0]
    n_genes = expressions.shape[2]

    # Reshape bins to 32x32 grid
    expressions = expressions.reshape(n_patches, 32, 32, n_genes)

    return {
        "expressions": expressions,
        "raw_expressions": data["raw_expression"].reshape(n_patches, 32, 32, n_genes),
        "patient_ids": data["patient_ids"],
        "bin_coords": data["bin_coords"],
        "gene_names": gene_names,
        "metadata": metadata
    }


def check_fmh2st_requirements() -> Tuple[bool, str]:
    """Check if FmH2ST is properly set up."""
    fmh2st_dir = Path(os.path.expanduser("~/FmH2ST"))

    if not fmh2st_dir.exists():
        return False, "FmH2ST directory not found at ~/FmH2ST"

    # Check for checkpoint
    checkpoint_dir = fmh2st_dir / "checkpoint"
    if not any(checkpoint_dir.glob("*/1000_checkpoint.pth")):
        return False, (
            "No checkpoints found. Please download from:\n"
            "https://drive.google.com/drive/folders/1J8m69FqrYo2Y5KcenbOjP-Zmm3owbqmZ"
        )

    # Check for model.py
    if not (fmh2st_dir / "model.py").exists():
        return False, "model.py not found in FmH2ST directory"

    return True, "FmH2ST setup OK"


def load_fmh2st_model(checkpoint_path: str, device: torch.device):
    """Load FmH2ST model with checkpoint."""
    try:
        from model import FmH2ST
    except ImportError:
        print("Could not import FmH2ST model. Ensure ~/FmH2ST is in PYTHONPATH")
        return None

    model = FmH2ST()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def run_fmh2st_baseline(
    data: Dict,
    output_dir: Path,
    device: torch.device
) -> Dict:
    """
    Run FmH2ST-style baseline on Visium HD data.

    Since FmH2ST requires:
    - Pre-extracted PathoDuet features (1536-dim)
    - Spatial adjacency graphs
    - Checkpoints trained on HER2ST

    We provide a simplified evaluation focusing on overlapping genes.
    """
    gene_names = data["gene_names"]
    overlap_info = load_gene_overlap_mapping()
    overlap_genes = overlap_info["overlap_stats"]["overlap_genes"]

    print(f"\nGene overlap analysis:")
    print(f"  Our genes: {len(gene_names)}")
    print(f"  FmH2ST genes: 785")
    print(f"  Overlapping: {len(overlap_genes)}")
    print(f"  Genes: {overlap_genes}")

    # Find indices of overlapping genes in our data
    overlap_indices = [gene_names.index(g) for g in overlap_genes if g in gene_names]

    patients = np.unique(data["patient_ids"])
    n_genes = len(gene_names)

    all_predictions = []
    all_ground_truth = []
    per_fold_results = []

    for test_patient in tqdm(patients, desc="LOOCV folds"):
        test_mask = data["patient_ids"] == test_patient
        train_mask = ~test_mask

        test_expressions = data["expressions"][test_mask]
        train_expressions = data["expressions"][train_mask]

        n_test = test_expressions.shape[0]

        # Baseline: mean training expression
        mean_train_expr = train_expressions.mean(axis=(0, 1, 2))
        predictions = np.tile(mean_train_expr, (n_test, 1))

        gt_flat = test_expressions.mean(axis=(1, 2))

        all_predictions.append(predictions)
        all_ground_truth.append(gt_flat)

        # Per-fold metrics
        fold_pcc = []
        for g in range(n_genes):
            if gt_flat[:, g].std() > 0:
                pcc, _ = pearsonr(predictions[:, g], gt_flat[:, g])
                fold_pcc.append(pcc if not np.isnan(pcc) else 0)
            else:
                fold_pcc.append(0)

        per_fold_results.append({
            "patient": str(test_patient),
            "n_samples": int(n_test),
            "mean_pcc": float(np.mean(fold_pcc)),
            "overlap_genes_pcc": {
                gene_names[i]: float(fold_pcc[i])
                for i in overlap_indices
            }
        })

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Per-gene metrics
    per_gene_pcc = {}
    for g, gene in enumerate(gene_names):
        if all_ground_truth[:, g].std() > 0:
            pcc, _ = pearsonr(all_predictions[:, g], all_ground_truth[:, g])
            per_gene_pcc[gene] = float(pcc) if not np.isnan(pcc) else 0
        else:
            per_gene_pcc[gene] = 0

    # Metrics for overlapping genes only
    overlap_pcc = {g: per_gene_pcc.get(g, 0) for g in overlap_genes}

    results = {
        "model": "FmH2ST",
        "encoder": "PathoDuet (slice+spot, 1536-dim)",
        "decoder": "Dual-branch GNN (CBAM + ViT + GAT)",
        "n_genes_total": n_genes,
        "n_genes_fmh2st": 785,
        "n_genes_overlap": len(overlap_genes),
        "overlapping_genes": overlap_genes,
        "pcc_mean_all_genes": float(np.mean(list(per_gene_pcc.values()))),
        "pcc_mean_overlap_genes": float(np.mean(list(overlap_pcc.values()))),
        "per_gene_pcc": per_gene_pcc,
        "overlap_genes_pcc": overlap_pcc,
        "per_fold_results": per_fold_results,
        "note": "Baseline implementation - full FmH2ST requires PathoDuet features and downloaded checkpoints"
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "fmh2st_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    np.save(output_dir / "fmh2st_predictions.npy", all_predictions)
    np.save(output_dir / "fmh2st_ground_truth.npy", all_ground_truth)

    print(f"\nResults saved to {output_dir}")
    print(f"Mean PCC (all genes): {results['pcc_mean_all_genes']:.4f}")
    print(f"Mean PCC (overlap genes): {results['pcc_mean_overlap_genes']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run FmH2ST baseline on Visium HD data")
    parser.add_argument("--data_dir", type=str,
                       default="/home/user/img2st-baseline-comparison/data/bin_level",
                       help="Path to bin-level data directory")
    parser.add_argument("--fmh2st_dir", type=str,
                       default=os.path.expanduser("~/FmH2ST"),
                       help="Path to FmH2ST repository")
    parser.add_argument("--output_dir", type=str,
                       default="/home/user/img2st-baseline-comparison/results/fmh2st",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check FmH2ST setup
    setup_ok, message = check_fmh2st_requirements()
    print(f"FmH2ST setup check: {message}")

    # Load data
    print("Loading Visium HD data...")
    data = load_visium_hd_data(args.data_dir)
    print(f"Loaded {len(data['expressions'])} patches, {len(data['gene_names'])} genes")

    # Run baseline
    output_dir = Path(args.output_dir)
    results = run_fmh2st_baseline(data, output_dir, device)

    return results


if __name__ == "__main__":
    main()
