#!/usr/bin/env python3
"""
Gene Overlap Mapper for External Baselines

Computes gene vocabulary overlaps between our 50 HVGs and external models
(Loki, FmH2ST) to enable fair comparison on shared genes.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_our_genes(path: str = "/home/user/img2st-baseline-comparison/features/gene_names_50.json") -> List[str]:
    """Load our 50 highly variable genes."""
    with open(path) as f:
        return json.load(f)


def load_fmh2st_genes(path: str = "/home/user/FmH2ST/data/her_hvg_cut_1000.npy") -> List[str]:
    """Load FmH2ST's 785 HER2ST genes."""
    return list(np.load(path, allow_pickle=True))


def load_loki_genes(st_bank_path: str = None) -> List[str]:
    """
    Load Loki's gene vocabulary from ST-bank.
    Note: Loki uses gene sentences from ST-bank, vocabulary depends on training data.
    Returns placeholder if ST-bank not available.
    """
    # Loki's predex uses text embeddings - gene names come from ST-bank
    # Check if we have ST-bank data
    if st_bank_path and Path(st_bank_path).exists():
        # Parse gene names from ST-bank text.csv
        import pandas as pd
        df = pd.read_csv(st_bank_path)
        # Extract gene names from gene sentences (format varies)
        genes = df['gene'].unique().tolist() if 'gene' in df.columns else []
        return genes
    else:
        # Return empty - Loki's gene vocab requires ST-bank download
        print("Warning: ST-bank not found. Loki gene vocabulary unavailable.")
        print("Download from: https://drive.google.com/drive/folders/1J15cO-pXTwkTjRAR-v-_nQkqXNfcCNn3")
        return []


def compute_overlap(genes_a: List[str], genes_b: List[str]) -> Dict:
    """Compute overlap statistics between two gene lists."""
    set_a = set(genes_a)
    set_b = set(genes_b)
    overlap = set_a & set_b

    return {
        "set_a_size": len(set_a),
        "set_b_size": len(set_b),
        "overlap_size": len(overlap),
        "overlap_genes": sorted(list(overlap)),
        "overlap_pct_of_a": 100 * len(overlap) / len(set_a) if set_a else 0,
        "overlap_pct_of_b": 100 * len(overlap) / len(set_b) if set_b else 0,
        "genes_only_in_a": sorted(list(set_a - set_b)),
        "genes_only_in_b": sorted(list(set_b - set_a))[:20]  # Limit for readability
    }


def create_gene_index_mapping(our_genes: List[str], external_genes: List[str]) -> Dict:
    """
    Create index mapping for overlapping genes.
    Returns dict mapping our gene indices to external model gene indices.
    """
    our_gene_to_idx = {g: i for i, g in enumerate(our_genes)}
    ext_gene_to_idx = {g: i for i, g in enumerate(external_genes)}

    overlap = set(our_genes) & set(external_genes)

    mapping = {
        "our_indices": [],
        "external_indices": [],
        "gene_names": []
    }

    for gene in sorted(overlap):
        mapping["our_indices"].append(our_gene_to_idx[gene])
        mapping["external_indices"].append(ext_gene_to_idx[gene])
        mapping["gene_names"].append(gene)

    return mapping


def main():
    """Compute and display all gene overlaps."""
    print("=" * 60)
    print("Gene Overlap Analysis for External Baselines")
    print("=" * 60)

    # Load gene lists
    our_genes = load_our_genes()
    print(f"\nOur 50 HVGs loaded: {len(our_genes)} genes")

    # FmH2ST overlap
    print("\n" + "-" * 40)
    print("FmH2ST Gene Overlap")
    print("-" * 40)

    fmh2st_genes = load_fmh2st_genes()
    fmh2st_overlap = compute_overlap(our_genes, fmh2st_genes)

    print(f"Our genes: {fmh2st_overlap['set_a_size']}")
    print(f"FmH2ST genes: {fmh2st_overlap['set_b_size']}")
    print(f"Overlap: {fmh2st_overlap['overlap_size']} ({fmh2st_overlap['overlap_pct_of_a']:.1f}% of ours)")
    print(f"Overlapping genes: {fmh2st_overlap['overlap_genes']}")

    # Create index mapping
    fmh2st_mapping = create_gene_index_mapping(our_genes, fmh2st_genes)
    print(f"\nIndex mapping created for {len(fmh2st_mapping['gene_names'])} genes")

    # Loki overlap
    print("\n" + "-" * 40)
    print("Loki Gene Overlap")
    print("-" * 40)

    loki_genes = load_loki_genes()
    if loki_genes:
        loki_overlap = compute_overlap(our_genes, loki_genes)
        print(f"Loki genes: {loki_overlap['set_b_size']}")
        print(f"Overlap: {loki_overlap['overlap_size']}")
    else:
        print("Loki uses ST-bank gene vocabulary (requires download)")
        print("For now, Loki can predict any gene it was trained on")

    # Save results
    output_dir = Path("/home/user/img2st-baseline-comparison/results/gene_overlaps")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "fmh2st_overlap.json", 'w') as f:
        json.dump({
            "overlap_stats": fmh2st_overlap,
            "index_mapping": fmh2st_mapping
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return fmh2st_overlap, fmh2st_mapping


if __name__ == "__main__":
    main()
