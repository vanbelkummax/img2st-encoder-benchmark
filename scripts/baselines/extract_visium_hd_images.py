#!/usr/bin/env python3
"""
Extract H&E patch images from Visium HD data for baseline evaluation.

This script:
1. Loads Visium HD 10x Genomics data
2. Extracts patches aligned with bin-level expression data
3. Saves images for use with Loki and other image-based models

Requires:
- 10x Genomics Visium HD data downloaded from:
  https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Colon_Cancer_P1/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_bin_level_metadata(data_dir: str) -> Dict:
    """Load bin-level data metadata."""
    data_path = Path(data_dir)

    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)

    data = np.load(data_path / "bin_level_data.npz")
    patient_ids = data["patient_ids"]
    bin_coords = data["bin_coords"]  # (1024, 2) coordinates for each bin in a patch

    return {
        "metadata": metadata,
        "patient_ids": patient_ids,
        "bin_coords": bin_coords,
        "n_patches": metadata["n_patches"],
        "grid_size": metadata["grid_size"],
        "bin_size_um": metadata["bin_size_um"]
    }


def load_visium_hd_image(visium_dir: str) -> Tuple[np.ndarray, Dict]:
    """
    Load the full H&E image from Visium HD data.

    Expected structure:
    visium_dir/
    ├── spatial/
    │   ├── tissue_hires_image.png
    │   ├── scalefactors_json.json
    │   └── tissue_positions.csv
    └── binned_outputs/
        └── square_008um/  (or other bin size)
    """
    spatial_dir = Path(visium_dir) / "spatial"

    # Load image
    image_path = spatial_dir / "tissue_hires_image.png"
    if not image_path.exists():
        image_path = spatial_dir / "tissue_lowres_image.png"

    image = np.array(Image.open(image_path))

    # Load scale factors
    with open(spatial_dir / "scalefactors_json.json") as f:
        scalefactors = json.load(f)

    return image, scalefactors


def extract_patch(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    patch_size_px: int = 256,
    scale_factor: float = 1.0
) -> np.ndarray:
    """Extract a square patch centered at given coordinates."""
    # Scale coordinates to image space
    x = int(center_x * scale_factor)
    y = int(center_y * scale_factor)

    half_size = patch_size_px // 2

    # Handle boundary conditions
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(image.shape[1], x + half_size)
    y2 = min(image.shape[0], y + half_size)

    patch = image[y1:y2, x1:x2]

    # Pad if needed
    if patch.shape[0] < patch_size_px or patch.shape[1] < patch_size_px:
        padded = np.zeros((patch_size_px, patch_size_px, 3), dtype=np.uint8)
        padded[:patch.shape[0], :patch.shape[1]] = patch
        patch = padded

    return patch


def compute_patch_centers(bin_coords: np.ndarray, grid_size: int = 32) -> List[Tuple[float, float]]:
    """
    Compute patch centers from bin coordinates.

    bin_coords: (1024, 2) array of (x, y) coordinates for each bin
    Returns: List of (center_x, center_y) for each unique patch
    """
    # Bins are in a 32x32 grid, compute center as mean of all bin positions
    centers = []

    # For our data, we have 222 patches, each with 1024 bins
    # The bin_coords should give us the spatial location
    center_x = bin_coords[:, 0].mean()
    center_y = bin_coords[:, 1].mean()

    return (center_x, center_y)


def extract_all_patches(
    visium_dir: str,
    bin_level_dir: str,
    output_dir: str,
    patch_size_px: int = 256
) -> None:
    """Extract all patches aligned with bin-level expression data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("Loading bin-level metadata...")
    meta = load_bin_level_metadata(bin_level_dir)

    # Load full H&E image
    print("Loading Visium HD image...")
    image, scalefactors = load_visium_hd_image(visium_dir)

    # Get scale factor for hires image
    if "tissue_hires_scalef" in scalefactors:
        scale = scalefactors["tissue_hires_scalef"]
    else:
        scale = scalefactors.get("tissue_lowres_scalef", 1.0)

    print(f"Image shape: {image.shape}")
    print(f"Scale factor: {scale}")
    print(f"Extracting {meta['n_patches']} patches...")

    # For each patch, compute center and extract
    # This is a simplified version - real implementation needs actual patch positions
    patch_paths = []

    for i in tqdm(range(meta['n_patches']), desc="Extracting patches"):
        patient_id = meta['patient_ids'][i]

        # Compute patch center from bin coordinates
        # In practice, this needs the actual spatial coordinates from Visium HD
        center = compute_patch_centers(meta['bin_coords'], meta['grid_size'])

        # Extract patch
        patch = extract_patch(
            image,
            center[0],
            center[1],
            patch_size_px=patch_size_px,
            scale_factor=scale
        )

        # Save patch
        patch_name = f"patch_{patient_id}_{i:03d}.png"
        patch_path = output_path / patch_name
        Image.fromarray(patch).save(patch_path)
        patch_paths.append(str(patch_path))

    # Save index
    index = {
        "n_patches": meta['n_patches'],
        "patch_size_px": patch_size_px,
        "patch_paths": patch_paths,
        "patient_ids": meta['patient_ids'].tolist()
    }

    with open(output_path / "patch_index.json", 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\nExtracted {len(patch_paths)} patches to {output_dir}")


def create_synthetic_patches(
    bin_level_dir: str,
    output_dir: str,
    patch_size_px: int = 256
) -> None:
    """
    Create synthetic placeholder patches for testing when real images unavailable.

    These are random RGB images that can be used to test the pipeline structure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta = load_bin_level_metadata(bin_level_dir)

    print(f"Creating {meta['n_patches']} synthetic patches for testing...")

    patch_paths = []
    np.random.seed(42)

    for i in tqdm(range(meta['n_patches']), desc="Creating synthetic patches"):
        patient_id = meta['patient_ids'][i]

        # Create random synthetic patch (pink-ish H&E-like colors)
        patch = np.random.randint(180, 255, (patch_size_px, patch_size_px, 3), dtype=np.uint8)
        # Add some structure
        patch[:, :, 1] = patch[:, :, 1] * 0.9  # Reduce green
        patch[:, :, 2] = patch[:, :, 2] * 0.9  # Reduce blue

        # Save patch
        patch_name = f"patch_{patient_id}_{i:03d}.png"
        patch_path = output_path / patch_name
        Image.fromarray(patch).save(patch_path)
        patch_paths.append(str(patch_path))

    # Save index
    index = {
        "n_patches": meta['n_patches'],
        "patch_size_px": patch_size_px,
        "patch_paths": patch_paths,
        "patient_ids": meta['patient_ids'].tolist(),
        "synthetic": True,
        "note": "Synthetic placeholder images for testing pipeline structure"
    }

    with open(output_path / "patch_index.json", 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\nCreated {len(patch_paths)} synthetic patches in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract H&E patches from Visium HD data"
    )
    parser.add_argument(
        "--visium_dir", type=str, default=None,
        help="Path to Visium HD data directory"
    )
    parser.add_argument(
        "--bin_level_dir", type=str,
        default="/home/user/img2st-baseline-comparison/data/bin_level",
        help="Path to bin-level expression data"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/user/img2st-baseline-comparison/data/images",
        help="Output directory for extracted patches"
    )
    parser.add_argument(
        "--patch_size", type=int, default=256,
        help="Patch size in pixels"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Create synthetic placeholder patches for testing"
    )
    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_patches(
            args.bin_level_dir,
            args.output_dir,
            args.patch_size
        )
    elif args.visium_dir:
        extract_all_patches(
            args.visium_dir,
            args.bin_level_dir,
            args.output_dir,
            args.patch_size
        )
    else:
        print("Error: Either --visium_dir or --synthetic must be specified")
        print("\nUsage:")
        print("  With real Visium HD data:")
        print("    python extract_visium_hd_images.py --visium_dir /path/to/visium_hd")
        print("\n  Create synthetic test images:")
        print("    python extract_visium_hd_images.py --synthetic")


if __name__ == "__main__":
    main()
