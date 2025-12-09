#!/usr/bin/env python3
"""
Extract PathoDuet features for Visium HD patches.

This script extracts 768-dimensional features from PathoDuet's H&E model
for use with FmH2ST baseline evaluation.

FmH2ST expects 1536-dimensional features (slice + spot concatenated).
We extract 768-dim features and duplicate them to match the expected format.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Import from local compatibility layer
from pathoduet_compat import vit_base, load_pathoduet_checkpoint


def load_pathoduet_model(
    checkpoint_path: str,
    device: str = "cuda"
) -> nn.Module:
    """
    Load PathoDuet H&E model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint_HE.pth
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Initialize model architecture
    model = vit_base(pretext_token=True, global_pool='avg')

    # Load checkpoint using compatibility function
    model = load_pathoduet_checkpoint(model, checkpoint_path)
    model = model.to(device)
    model.eval()

    print(f"Loaded PathoDuet model from {checkpoint_path}")
    return model


def preprocess_image(
    image_path: str,
    target_size: int = 224
) -> torch.Tensor:
    """
    Preprocess image for PathoDuet.

    Following PathoDuet paper: no normalization for pathology images.

    Args:
        image_path: Path to image file
        target_size: Target image size (224 for ViT-B/16)

    Returns:
        Preprocessed tensor of shape (1, 3, 224, 224)
    """
    img = Image.open(image_path).convert("RGB")

    # Resize to target size
    img = img.resize((target_size, target_size), Image.BILINEAR)

    # Convert to tensor [0, 1]
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0

    # Rearrange to CHW
    img_tensor = img_tensor.permute(2, 0, 1)

    # Add batch dimension
    return img_tensor.unsqueeze(0)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    image_paths: List[str],
    device: str = "cuda",
    batch_size: int = 32
) -> np.ndarray:
    """
    Extract features from a list of images.

    Args:
        model: PathoDuet model
        image_paths: List of image paths
        device: Device to use
        batch_size: Batch size for inference

    Returns:
        Features array of shape (N, 768)
    """
    all_features = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            tensor = preprocess_image(path)
            batch_tensors.append(tensor)

        # Stack batch
        batch = torch.cat(batch_tensors, dim=0).to(device)

        # Forward pass
        features_out, _ = model(batch)

        # Get CLS token features (index 1 after pretext token)
        # Or use global average pooling result
        if hasattr(model, 'global_pool') and model.global_pool == 'avg':
            # forward_head already does the pooling
            cls_features = model.forward_head(features_out, pre_logits=True)
        else:
            # Use CLS token (index 1)
            cls_features = features_out[:, 1, :]

        all_features.append(cls_features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def create_fmh2st_format_features(
    features: np.ndarray,
    output_dir: str,
    patch_names: List[str]
) -> None:
    """
    Save features in FmH2ST expected format.

    FmH2ST expects 1536-dim features (slice 768 + spot 768 concatenated).
    We duplicate the 768-dim features to create 1536-dim.

    Args:
        features: Array of shape (N, 768)
        output_dir: Output directory
        patch_names: List of patch identifiers
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Duplicate features to get 1536-dim (slice + spot)
    features_1536 = np.concatenate([features, features], axis=1)

    print(f"Saving {len(features)} features to {output_dir}")
    print(f"Feature dimension: {features.shape[1]} -> {features_1536.shape[1]} (duplicated)")

    for i, (feat, name) in enumerate(zip(features_1536, patch_names)):
        # Save in FmH2ST format: name_x_y.npy (we use 0x0 as placeholder coords)
        out_file = output_path / f"{name}_0x0.npy"
        np.save(out_file, feat)

    # Also save as single array for convenience
    np.save(output_path / "all_features.npy", features_1536)
    np.save(output_path / "patch_names.npy", np.array(patch_names))

    print(f"Saved {len(features)} individual feature files and all_features.npy")


def main():
    parser = argparse.ArgumentParser(
        description="Extract PathoDuet features for Visium HD patches"
    )
    parser.add_argument(
        "--images_dir", type=str,
        default="/home/user/img2st-baseline-comparison/data/images",
        help="Directory containing patch images"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.expanduser("~/PathoDuet/checkpoints/PathoDuet/checkpoint_HE.pth"),
        help="Path to PathoDuet H&E checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/user/img2st-baseline-comparison/data/pathoduet_features",
        help="Output directory for features"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for feature extraction"
    )
    args = parser.parse_args()

    # Check if images exist
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} does not exist")
        print("Run extract_visium_hd_images.py --synthetic first to create test images")
        return

    # Load patch index
    index_path = images_dir / "patch_index.json"
    if not index_path.exists():
        print(f"Error: patch_index.json not found in {images_dir}")
        return

    with open(index_path) as f:
        patch_index = json.load(f)

    image_paths = patch_index["patch_paths"]
    n_patches = len(image_paths)

    print(f"Found {n_patches} patches to process")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model
    model = load_pathoduet_model(args.checkpoint, args.device)

    # Extract features
    features = extract_features(
        model, image_paths, args.device, args.batch_size
    )

    print(f"Extracted features shape: {features.shape}")

    # Create patch names from paths
    patch_names = [Path(p).stem for p in image_paths]

    # Save in FmH2ST format
    create_fmh2st_format_features(features, args.output_dir, patch_names)

    # Also save metadata
    metadata = {
        "n_patches": n_patches,
        "feature_dim": 1536,  # After duplication
        "original_dim": 768,
        "model": "PathoDuet (checkpoint_HE.pth)",
        "checkpoint": args.checkpoint,
        "patch_names": patch_names
    }

    with open(Path(args.output_dir) / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nFeature extraction complete!")
    print(f"Features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
