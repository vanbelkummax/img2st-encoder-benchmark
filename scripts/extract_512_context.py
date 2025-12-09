#!/usr/bin/env python3
"""
Extract 512px context features from WSI hires images.

For each existing 256px patch, extracts a 512px region centered on it,
providing broader tissue context for prediction.

Uses same coordinate system as preprocess_visium_hd.py:
- patch_row, patch_col define 256px grid positions in hires image
- 512px context = 256px patch + 128px border on all sides

Usage:
    python extract_512_context.py --encoder densenet_imagenet
    python extract_512_context.py --encoder dinov2_giant --output features/dinov2_512context.npz
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from encoder_wrapper import get_encoder, list_encoders


# Data paths
DATA_DIR = Path('/mnt/x/img2st_rotation_demo/processed_crc_aligned')
RAW_DATA_DIR = Path('/mnt/x/img2st_rotation_demo/downloads/crc_hd')
PATIENTS = ['P1', 'P2', 'P5']

# Context configuration
PATCH_SIZE = 256        # Original patch size
CONTEXT_SIZE = 512      # Larger context window
ENCODER_INPUT = 224     # Encoder input size


class Context512Dataset(Dataset):
    """Dataset for loading 512px context regions from WSI."""

    def __init__(
        self,
        processed_dir: Path,
        raw_data_dir: Path,
        patients: List[str],
        context_size: int = CONTEXT_SIZE,
        patch_size: int = PATCH_SIZE,
        encoder_input: int = ENCODER_INPUT,
    ):
        """
        Args:
            processed_dir: Path to processed_crc_aligned directory
            raw_data_dir: Path to downloads/crc_hd directory with hires images
            patients: List of patient IDs
            context_size: Size of context window (512)
            patch_size: Size of original patches (256)
            encoder_input: Encoder input size (224)
        """
        self.processed_dir = Path(processed_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.context_size = context_size
        self.patch_size = patch_size
        self.encoder_input = encoder_input
        self.border = (context_size - patch_size) // 2  # 128px border

        self.samples = []

        # Load patches and map to hires images
        for patient in patients:
            # Load patches.npy
            patches_path = self.processed_dir / patient / 'patches.npy'
            if not patches_path.exists():
                print(f"Warning: {patches_path} not found, skipping {patient}")
                continue

            patches = np.load(patches_path, allow_pickle=True)

            # Find hires image
            hires_path = self.raw_data_dir / patient / 'binned_outputs' / 'square_008um' / 'spatial' / 'tissue_hires_image.png'
            if not hires_path.exists():
                print(f"Warning: {hires_path} not found, skipping {patient}")
                continue

            # Load hires image
            hires_img = Image.open(hires_path).convert('RGB')
            img_width, img_height = hires_img.size
            print(f"{patient}: hires image {img_width}x{img_height}, {len(patches)} patches")

            for idx, patch_data in enumerate(patches):
                patch_row = int(patch_data['patch_row'])
                patch_col = int(patch_data['patch_col'])

                # Calculate pixel coordinates of 256px patch
                x0 = patch_col * patch_size
                y0 = patch_row * patch_size

                # Calculate 512px context region (centered on patch)
                ctx_x0 = x0 - self.border
                ctx_y0 = y0 - self.border
                ctx_x1 = ctx_x0 + context_size
                ctx_y1 = ctx_y0 + context_size

                # Check if we can extract full context
                can_extract = (
                    ctx_x0 >= 0 and ctx_y0 >= 0 and
                    ctx_x1 <= img_width and ctx_y1 <= img_height
                )

                self.samples.append({
                    'patient': patient,
                    'patch_idx': idx,
                    'patch_row': patch_row,
                    'patch_col': patch_col,
                    'ctx_x0': max(0, ctx_x0),
                    'ctx_y0': max(0, ctx_y0),
                    'ctx_x1': min(img_width, ctx_x1),
                    'ctx_y1': min(img_height, ctx_y1),
                    'hires_path': str(hires_path),
                    'can_extract_full': can_extract,
                    'needs_padding': not can_extract,
                })

        # Preload images for efficiency
        self.images = {}
        for patient in patients:
            hires_path = self.raw_data_dir / patient / 'binned_outputs' / 'square_008um' / 'spatial' / 'tissue_hires_image.png'
            if hires_path.exists():
                self.images[patient] = Image.open(hires_path).convert('RGB')

        n_need_padding = sum(1 for s in self.samples if s['needs_padding'])
        print(f"\nLoaded {len(self.samples)} samples, {n_need_padding} need edge padding")

        # Transform: resize to encoder input
        self.transform = transforms.Compose([
            transforms.Resize((encoder_input, encoder_input)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        patient = sample['patient']

        # Get hires image
        img = self.images[patient]
        img_width, img_height = img.size

        # Calculate coordinates
        ctx_x0 = sample['patch_col'] * self.patch_size - self.border
        ctx_y0 = sample['patch_row'] * self.patch_size - self.border

        # Handle edge cases with padding
        if sample['needs_padding']:
            # Create padded image
            context = Image.new('RGB', (self.context_size, self.context_size), (255, 255, 255))

            # Calculate actual crop region (clamped to image bounds)
            crop_x0 = max(0, ctx_x0)
            crop_y0 = max(0, ctx_y0)
            crop_x1 = min(img_width, ctx_x0 + self.context_size)
            crop_y1 = min(img_height, ctx_y0 + self.context_size)

            # Crop from original
            cropped = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))

            # Paste into padded canvas at correct offset
            paste_x = max(0, -ctx_x0)
            paste_y = max(0, -ctx_y0)
            context.paste(cropped, (paste_x, paste_y))
        else:
            # Simple crop
            context = img.crop((ctx_x0, ctx_y0, ctx_x0 + self.context_size, ctx_y0 + self.context_size))

        # Transform
        tensor = self.transform(context)

        return {
            'image': tensor,
            'patient': sample['patient'],
            'patch_idx': sample['patch_idx'],
            'patch_row': sample['patch_row'],
            'patch_col': sample['patch_col'],
        }


def extract_512_context_features(
    encoder_name: str,
    output_path: str,
    processed_dir: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
    patients: Optional[List[str]] = None,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract 512px context features from WSI hires images.

    Args:
        encoder_name: Name of encoder from encoder_config.yaml
        output_path: Path to save output .npz file
        processed_dir: Path to processed data directory
        raw_data_dir: Path to raw data directory with hires images
        patients: List of patient IDs
        batch_size: Batch size for inference
        device: Device to use

    Returns:
        Dict with features and metadata
    """
    if processed_dir is None:
        processed_dir = DATA_DIR
    if raw_data_dir is None:
        raw_data_dir = RAW_DATA_DIR
    if patients is None:
        patients = PATIENTS
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("512px CONTEXT FEATURE EXTRACTION")
    print("=" * 70)
    print(f"  Encoder: {encoder_name}")
    print(f"  Context size: {CONTEXT_SIZE}px (256px patch + 128px border)")
    print(f"  Device: {device}")

    # Load encoder
    print(f"\nLoading encoder: {encoder_name}...")
    encoder = get_encoder(encoder_name, device=device)
    embed_dim = encoder.embed_dim
    print(f"  Embed dim: {embed_dim}")

    # Create dataset
    dataset = Context512Dataset(
        processed_dir=processed_dir,
        raw_data_dir=raw_data_dir,
        patients=patients,
    )

    if len(dataset) == 0:
        raise RuntimeError("No samples found!")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Extract features
    all_features = []
    all_patient_ids = []
    all_patch_indices = []
    all_patch_rows = []
    all_patch_cols = []

    print(f"\nExtracting features from {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"512px Context ({encoder_name})"):
            images = batch['image'].to(device)
            features = encoder(images, normalize=True)

            all_features.append(features.cpu().numpy())
            all_patient_ids.extend(batch['patient'])
            all_patch_indices.extend(batch['patch_idx'].numpy())
            all_patch_rows.extend(batch['patch_row'].numpy())
            all_patch_cols.extend(batch['patch_col'].numpy())

    # Concatenate
    features_array = np.concatenate(all_features, axis=0)
    patient_ids_array = np.array(all_patient_ids)
    patch_indices_array = np.array(all_patch_indices)
    patch_rows_array = np.array(all_patch_rows)
    patch_cols_array = np.array(all_patch_cols)

    print(f"\nFeature shape: {features_array.shape}")
    print(f"  Patients: {dict(zip(*np.unique(patient_ids_array, return_counts=True)))}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        features=features_array.astype(np.float32),
        patient_ids=patient_ids_array,
        patch_indices=patch_indices_array,
        patch_rows=patch_rows_array,
        patch_cols=patch_cols_array,
        encoder_name=encoder_name,
        embed_dim=embed_dim,
        context_size=CONTEXT_SIZE,
        patch_size=PATCH_SIZE,
    )

    print(f"\nSaved 512px context features to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return {
        'features': features_array,
        'patient_ids': patient_ids_array,
        'patch_indices': patch_indices_array,
    }


def main():
    parser = argparse.ArgumentParser(description='Extract 512px context features')
    parser.add_argument('--encoder', type=str, default='densenet_imagenet',
                       help='Encoder name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: features/{encoder}_512context.npz)')
    parser.add_argument('--processed_dir', type=str, default=str(DATA_DIR),
                       help='Path to processed data directory')
    parser.add_argument('--raw_data_dir', type=str, default=str(RAW_DATA_DIR),
                       help='Path to raw data directory with hires images')
    parser.add_argument('--patients', type=str, nargs='+', default=PATIENTS,
                       help='Patient IDs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device')
    parser.add_argument('--list', action='store_true',
                       help='List available encoders')

    args = parser.parse_args()

    if args.list:
        print("Available encoders:")
        for name, cfg in list_encoders().items():
            print(f"  {name}: {cfg['name']} (dim={cfg['embed_dim']})")
        return

    if args.output is None:
        args.output = f"features/{args.encoder}_512context.npz"

    extract_512_context_features(
        encoder_name=args.encoder,
        output_path=args.output,
        processed_dir=args.processed_dir,
        raw_data_dir=args.raw_data_dir,
        patients=args.patients,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == '__main__':
    main()
