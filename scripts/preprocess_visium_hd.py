#!/usr/bin/env python3
"""
Preprocess Visium HD data for Img2ST-Net training.

This script converts 10x Genomics Visium HD 2µm binned data into
training-ready patches with 32×32 gene expression grids.

Usage:
    python preprocess_visium_hd.py \
        --input_dir /path/to/visium_hd_data \
        --output_dir data/processed \
        --bin_size 2
"""

import argparse
import json
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
PATCH_SIZE = 256  # pixels
GRID_SIZE = 32    # bins per patch
NUM_GENES = 50    # top variable genes


def load_visium_hd_data(input_dir: Path, bin_size: int = 2):
    """Load Visium HD binned data."""
    bin_dir = input_dir / "binned_outputs" / f"square_{bin_size:03d}um"

    # Load filtered counts
    adata = sc.read_10x_h5(bin_dir / "filtered_feature_bc_matrix.h5")

    # Make var_names unique (10x data can have duplicate gene names)
    adata.var_names_make_unique()

    # Load tissue positions (parquet format for HD)
    positions = pd.read_parquet(
        bin_dir / "spatial" / "tissue_positions.parquet"
    )

    # Load scale factors
    with open(bin_dir / "spatial" / "scalefactors_json.json") as f:
        scalefactors = json.load(f)

    return adata, positions, scalefactors


def select_variable_genes(adata, n_genes: int = 50, fixed_gene_list: list = None):
    """Select top highly variable genes or use fixed gene list.

    IMPORTANT: Stores raw counts in adata.layers['raw'] before normalization
    so they can be retrieved later for 8µm aggregation.

    Args:
        adata: AnnData object
        n_genes: Number of genes to select (if no fixed list)
        fixed_gene_list: Optional list of gene names to use instead of HVG selection

    Returns:
        List of gene names (in order of fixed list or HVG ranking)
    """
    # Store raw counts BEFORE any normalization
    # This is critical for downstream 8µm aggregation which needs raw counts
    adata.layers['raw'] = adata.X.copy()

    # If fixed gene list provided, use those genes
    if fixed_gene_list is not None:
        # Filter to genes present in this sample
        available_genes = set(adata.var_names)
        selected_genes = [g for g in fixed_gene_list if g in available_genes]
        missing = set(fixed_gene_list) - set(selected_genes)
        if missing:
            print(f"  Warning: {len(missing)} genes not found: {list(missing)[:5]}...")
        print(f"  Using {len(selected_genes)} of {len(fixed_gene_list)} fixed genes")

        # Still normalize for downstream processing (but gene selection is fixed)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        return selected_genes

    # Default: HVG selection
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)

    hvg_mask = adata.var['highly_variable']
    gene_names = adata.var_names[hvg_mask].tolist()

    return gene_names


def zscore_normalize(expr_matrix, norm_params=None, save_params_path=None):
    """Z-score normalize gene expression.

    Args:
        expr_matrix: Expression matrix to normalize
        norm_params: Optional dict with 'mean' and 'std' arrays for global normalization
        save_params_path: Optional path to save computed mean/std (for training set)

    Returns:
        Normalized expression matrix, and optionally the computed params
    """
    if norm_params is not None:
        # Use pre-computed parameters (global normalization)
        mean = np.array(norm_params['mean']).reshape(1, -1)
        std = np.array(norm_params['std']).reshape(1, -1)
        std[std == 0] = 1
        print("  Using global normalization parameters")
    else:
        # Compute parameters from this sample (per-sample normalization)
        mean = expr_matrix.mean(axis=0, keepdims=True)
        std = expr_matrix.std(axis=0, keepdims=True)
        std[std == 0] = 1

    normalized = (expr_matrix - mean) / std

    # Save parameters if requested (for computing global normalization from training set)
    if save_params_path:
        params = {
            'mean': mean.flatten().tolist(),
            'std': std.flatten().tolist()
        }
        with open(save_params_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"  Saved normalization params to {save_params_path}")

    return normalized


def extract_patches(
    image_path: Path,
    positions: pd.DataFrame,
    expr_matrix: np.ndarray,
    raw_matrix: np.ndarray,
    obs_names: list,
    scalefactors: dict,
    output_dir: Path
):
    """Extract aligned H&E patches and gene expression grids.

    Args:
        image_path: Path to H&E image (can be hires or fullres)
        positions: DataFrame with barcode positions (index = barcode)
        expr_matrix: Z-score normalized expression matrix (rows aligned with obs_names)
        raw_matrix: Raw counts matrix for selected genes (rows aligned with obs_names)
        obs_names: List of barcodes in expr_matrix row order (from adata.obs_names)
        scalefactors: Visium HD scale factors
        output_dir: Output directory

    Returns:
        List of patch dicts with 'label' (z-scored) and 'raw_label' (raw counts)
    """

    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Determine scale factor based on which image we're using
    # Coordinates in tissue_positions are always in full resolution space
    if 'hires' in str(image_path):
        scale = scalefactors.get('tissue_hires_scalef', 1.0)
    elif 'lowres' in str(image_path):
        scale = scalefactors.get('tissue_lowres_scalef', 1.0)
    else:
        scale = 1.0  # Assume full resolution image

    print(f"  Image size: {img_width}x{img_height}, scale factor: {scale:.6f}")

    # Create barcode-to-row mapping for expression matrix lookup
    # This ensures we use the correct row regardless of positions DataFrame order
    barcode_to_row = {bc: i for i, bc in enumerate(obs_names)}

    # Group bins into patches
    positions = positions.copy()

    # Set barcode as index so iterrows() yields (barcode, row) pairs
    # This is critical: without this, iterrows() yields (numeric_index, row)
    # and barcode_to_row lookup fails for all barcodes
    if 'barcode' in positions.columns:
        positions = positions.set_index('barcode')

    # Scale coordinates from fullres to image space
    positions['pxl_row_scaled'] = (positions['pxl_row_in_fullres'] * scale).astype(int)
    positions['pxl_col_scaled'] = (positions['pxl_col_in_fullres'] * scale).astype(int)

    # Group into patches based on scaled coordinates
    positions['patch_row'] = (positions['pxl_row_scaled'] // PATCH_SIZE).astype(int)
    positions['patch_col'] = (positions['pxl_col_scaled'] // PATCH_SIZE).astype(int)

    patches = []
    patch_groups = positions.groupby(['patch_row', 'patch_col'])
    skipped_barcodes = 0

    for (pr, pc), group in tqdm(patch_groups, desc="Extracting patches"):
        if len(group) < GRID_SIZE * GRID_SIZE * 0.5:  # skip sparse patches
            continue

        # Extract image patch
        x0 = pc * PATCH_SIZE
        y0 = pr * PATCH_SIZE
        x1 = min(x0 + PATCH_SIZE, img_width)
        y1 = min(y0 + PATCH_SIZE, img_height)

        if x1 - x0 < PATCH_SIZE or y1 - y0 < PATCH_SIZE:
            continue  # skip edge patches

        patch_img = image.crop((x0, y0, x1, y1))

        # Build 32×32 expression grids (both normalized and raw)
        expr_grid = np.zeros((GRID_SIZE * GRID_SIZE, NUM_GENES), dtype=np.float32)
        raw_grid = np.zeros((GRID_SIZE * GRID_SIZE, NUM_GENES), dtype=np.float32)

        for barcode, row in group.iterrows():
            # Map to grid position using scaled coordinates
            local_row = int((row['pxl_row_scaled'] - y0) / (PATCH_SIZE / GRID_SIZE))
            local_col = int((row['pxl_col_scaled'] - x0) / (PATCH_SIZE / GRID_SIZE))

            if 0 <= local_row < GRID_SIZE and 0 <= local_col < GRID_SIZE:
                grid_idx = local_row * GRID_SIZE + local_col
                # Use barcode lookup to get correct expression row
                if barcode in barcode_to_row:
                    expr_row = barcode_to_row[barcode]
                    expr_grid[grid_idx] = expr_matrix[expr_row]
                    raw_grid[grid_idx] = raw_matrix[expr_row]
                else:
                    skipped_barcodes += 1

        # Save patch
        patch_id = f"patch_{pr:03d}_{pc:03d}"
        patch_path = output_dir / "images" / f"{patch_id}.png"
        patch_img.save(patch_path)

        patches.append({
            'img_path': f"./images/{patch_id}.png",
            'label': expr_grid,           # Z-score normalized (for training)
            'raw_label': raw_grid,         # Raw counts (for 8µm aggregation)
            'patch_row': pr,
            'patch_col': pc
        })

    if skipped_barcodes > 0:
        print(f"  Warning: {skipped_barcodes} barcodes in positions not found in expression matrix")

    return patches


def main():
    parser = argparse.ArgumentParser(description="Preprocess Visium HD data")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to Visium HD data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--bin_size', type=int, default=2,
                        help='Bin size in µm (default: 2)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to H&E image (auto-detected if not provided)')
    parser.add_argument('--gene_list', type=str, default=None,
                        help='Path to JSON file with fixed gene list (uses HVG if not provided)')
    parser.add_argument('--norm_params', type=str, default=None,
                        help='Path to JSON with mean/std for global normalization (computes per-sample if not provided)')
    parser.add_argument('--save_norm_params', action='store_true',
                        help='Save computed normalization params to output_dir/norm_params.json (use for training set)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # Load fixed gene list if provided
    fixed_genes = None
    if args.gene_list:
        with open(args.gene_list) as f:
            fixed_genes = json.load(f)
        print(f"Using fixed gene list with {len(fixed_genes)} genes from {args.gene_list}")

    print(f"Loading Visium HD data from {input_dir}...")
    adata, positions, scalefactors = load_visium_hd_data(input_dir, args.bin_size)
    print(f"  Loaded {adata.n_obs} spots, {adata.n_vars} genes")

    if fixed_genes:
        print(f"\nUsing fixed gene list ({len(fixed_genes)} genes)...")
        gene_names = select_variable_genes(adata, NUM_GENES, fixed_gene_list=fixed_genes)
    else:
        print(f"\nSelecting top {NUM_GENES} variable genes...")
        gene_names = select_variable_genes(adata, NUM_GENES)
    print(f"  Selected genes: {gene_names[:5]}...")

    # Extract expression matrix for selected genes (log-normalized from HVG selection)
    adata_subset = adata[:, gene_names]
    expr_matrix = adata_subset.X.toarray() if hasattr(adata_subset.X, 'toarray') else adata_subset.X

    # Extract RAW counts for selected genes (stored before normalization)
    # This is critical for 8µm aggregation which needs to sum counts, not z-scores
    raw_counts = adata.layers['raw'][:, adata.var_names.isin(gene_names)]
    raw_matrix = raw_counts.toarray() if hasattr(raw_counts, 'toarray') else raw_counts
    print(f"  Raw counts shape: {raw_matrix.shape}")

    print("\nZ-score normalizing...")
    # Load global normalization params if provided
    norm_params = None
    if args.norm_params:
        with open(args.norm_params) as f:
            norm_params = json.load(f)
        print(f"  Loaded normalization params from {args.norm_params}")

    # Determine if we should save params (for training set)
    save_params_path = None
    if args.save_norm_params and not args.norm_params:
        save_params_path = output_dir / "norm_params.json"
        print(f"  Will save computed norm params to {save_params_path}")

    expr_matrix = zscore_normalize(expr_matrix, norm_params=norm_params, save_params_path=save_params_path)

    # Find H&E image
    if args.image_path:
        image_path = Path(args.image_path)
    else:
        spatial_dir = input_dir / "binned_outputs" / f"square_{args.bin_size:03d}um" / "spatial"
        image_path = spatial_dir / "tissue_hires_image.png"
        if not image_path.exists():
            image_path = spatial_dir / "tissue_lowres_image.png"

    print(f"\nExtracting patches from {image_path}...")
    # Pass obs_names for barcode-based alignment between positions and expression matrix
    obs_names = list(adata_subset.obs_names)
    patches = extract_patches(
        image_path, positions, expr_matrix, raw_matrix, obs_names, scalefactors, output_dir
    )
    print(f"  Extracted {len(patches)} patches")

    # Save as numpy file
    np.save(output_dir / "patches.npy", patches, allow_pickle=True)

    # Save gene names
    with open(output_dir / "gene_names.json", 'w') as f:
        json.dump(gene_names, f, indent=2)

    # Save metadata
    metadata = {
        'bin_size_um': args.bin_size,
        'grid_size': GRID_SIZE,
        'patch_size_px': PATCH_SIZE,
        'num_genes': NUM_GENES,
        'num_patches': len(patches),
        'gene_names': gene_names,
        'gene_list_source': args.gene_list if args.gene_list else 'hvg_selection',
        'norm_params_source': args.norm_params if args.norm_params else ('computed_and_saved' if args.save_norm_params else 'per_sample'),
        'input_dir': str(input_dir)
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Output saved to {output_dir}")
    print(f"  - patches.npy: {len(patches)} training samples")
    print(f"    - 'label': Z-score normalized (for training)")
    print(f"    - 'raw_label': Raw counts (for 8µm aggregation)")
    print(f"  - gene_names.json: {NUM_GENES} genes")
    print(f"  - metadata.json: processing parameters")


if __name__ == '__main__':
    main()
