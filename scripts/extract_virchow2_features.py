#!/usr/bin/env python3
"""
Extract Virchow2 features for Img2ST baseline comparison.

Virchow2: State-of-the-art pathology foundation model (ViT-H/14, 632M params)
- Trained on 3.1M H&E whole slide images
- Output: 2560-d embedding (1280 class token + 1280 mean patch token)
- Multi-magnification training includes 2µm/pixel (matches Visium HD)

Requirements:
    pip install timm>=0.9.11 huggingface_hub
    huggingface-cli login  # Must have Virchow2 access approved

Usage:
    python extract_virchow2_features.py
    python extract_virchow2_features.py --scale 512  # For 512px context
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image
from tqdm import tqdm


def load_virchow2_model(device: str = "cuda"):
    """Load Virchow2 model with proper initialization."""
    print("Loading Virchow2 model...")
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU
    )
    model = model.eval()
    model = model.to(device)

    # Get transforms
    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    print(f"Virchow2 loaded on {device}")
    return model, transforms


def extract_embedding(model, image_tensor, device: str = "cuda"):
    """Extract Virchow2 embedding from image tensor."""
    image_tensor = image_tensor.to(device)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(image_tensor)  # 1 x 261 x 1280

        class_token = output[:, 0]      # 1 x 1280
        patch_tokens = output[:, 5:]    # 1 x 256 x 1280 (skip register tokens 1-4)

        # Concatenate class token and mean patch token
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # 1 x 2560

    return embedding.cpu().numpy().astype(np.float32)


def load_patches(data_dir: Path, scale: int = 256):
    """Load patch images from data directory."""
    patch_dir = data_dir / "patches"

    # Load patch metadata
    with open(data_dir / "patch_metadata.json", "r") as f:
        metadata = json.load(f)

    patches = []
    patch_ids = []

    for patch_info in metadata["patches"]:
        patch_id = patch_info["patch_id"]
        # Construct path based on scale
        if scale == 256:
            img_path = patch_dir / f"{patch_id}.png"
        else:
            img_path = patch_dir / f"{patch_id}_{scale}px.png"

        if img_path.exists():
            patches.append(img_path)
            patch_ids.append(patch_id)

    return patches, patch_ids


def extract_features_from_npz(npz_path: Path, model, transforms, device: str = "cuda"):
    """Extract features from patches stored in NPZ format."""
    print(f"Loading patches from {npz_path}")
    data = np.load(npz_path)

    # Assuming patches are stored as 'patches' array
    if 'patches' in data:
        patches = data['patches']
    elif 'images' in data:
        patches = data['images']
    else:
        raise ValueError(f"No patch data found in {npz_path}")

    n_patches = len(patches)
    features = np.zeros((n_patches, 2560), dtype=np.float32)

    print(f"Extracting Virchow2 features for {n_patches} patches...")

    for i in tqdm(range(n_patches)):
        # Convert to PIL Image
        patch = patches[i]
        if patch.max() <= 1.0:
            patch = (patch * 255).astype(np.uint8)

        if patch.shape[0] == 3:  # CHW -> HWC
            patch = patch.transpose(1, 2, 0)

        img = Image.fromarray(patch)
        img_tensor = transforms(img).unsqueeze(0)

        features[i] = extract_embedding(model, img_tensor, device).squeeze()

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract Virchow2 features")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing patch data")
    parser.add_argument("--output", type=str, default="features/virchow2_features.npz",
                        help="Output NPZ file")
    parser.add_argument("--scale", type=int, default=256, choices=[256, 512],
                        help="Patch scale (256 or 512)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for extraction")
    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    device = args.device if torch.cuda.is_available() else "cpu"
    model, transforms = load_virchow2_model(device)

    # Check for existing feature files to get patch structure
    existing_features = Path("features/densenet_fm_features.npz")
    if existing_features.exists():
        # Use same patch order as existing features
        ref_data = np.load(existing_features)
        n_patches = ref_data['features'].shape[0] if 'features' in ref_data else len(ref_data['arr_0'])
        print(f"Using {n_patches} patches from reference features")

        # Load patches from global features file
        patch_features_path = Path("features/patch_features_global.npz")
        if patch_features_path.exists():
            patch_data = np.load(patch_features_path)
            if 'patches' in patch_data:
                patches = patch_data['patches']
            else:
                # Reconstruct from image files
                print("Reconstructing patches from image files...")
                patches = load_patches_from_images(data_dir, n_patches)
        else:
            patches = load_patches_from_images(data_dir, n_patches)
    else:
        raise FileNotFoundError("No reference features found. Run compute_patch_features.py first.")

    # Extract features
    features = np.zeros((len(patches), 2560), dtype=np.float32)

    print(f"Extracting Virchow2 features for {len(patches)} patches...")
    for i in tqdm(range(len(patches))):
        patch = patches[i]

        # Handle different formats
        if isinstance(patch, (str, Path)):
            img = Image.open(patch).convert('RGB')
        else:
            if patch.max() <= 1.0:
                patch = (patch * 255).astype(np.uint8)
            if patch.shape[0] == 3:
                patch = patch.transpose(1, 2, 0)
            img = Image.fromarray(patch)

        img_tensor = transforms(img).unsqueeze(0)
        features[i] = extract_embedding(model, img_tensor, device).squeeze()

    # Save features
    np.savez_compressed(
        output_path,
        features=features,
        encoder="virchow2",
        dim=2560,
        scale=args.scale
    )

    print(f"Saved Virchow2 features to {output_path}")
    print(f"Shape: {features.shape}")
    print(f"Feature dim: 2560 (1280 class + 1280 mean patch)")


def load_patches_from_images(data_dir: Path, n_patches: int):
    """Load patches from image files in data directory."""
    patch_dir = data_dir / "patches_256"
    if not patch_dir.exists():
        patch_dir = data_dir / "patches"

    patch_files = sorted(patch_dir.glob("*.png"))[:n_patches]

    patches = []
    for pf in patch_files:
        img = np.array(Image.open(pf).convert('RGB'))
        patches.append(img)

    return np.array(patches)


if __name__ == "__main__":
    main()
