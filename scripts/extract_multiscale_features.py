#!/usr/bin/env python3
"""
Extract multiscale features from H&E patches.

Two-scale extraction per patch:
- Local: 256x256 center crop (fine-grained cellular detail)
- Context: 512x512 full tile, downsampled to 224x224 (tissue context)

Uses same coordinate system as existing patches (GUARDRAIL).

Usage:
    python extract_multiscale_features.py --encoder densenet_imagenet
    python extract_multiscale_features.py --encoder dinov2_giant --output features/dinov2_multiscale.npz
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add scripts directory to path for encoder_wrapper
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from encoder_wrapper import get_encoder, list_encoders


# Data paths (same as extract_features_fm.py)
DATA_DIR = Path('/mnt/x/img2st_rotation_demo/processed_crc_aligned')
PATIENTS = ['P1', 'P2', 'P5']

# Multiscale configuration
# NOTE: Patches are 256x256, so we use 128px center crop for "local" (cell-level)
# and full 256px for "context" (tissue microenvironment)
LOCAL_CROP_SIZE = 128   # Center crop for local features (cell-level detail)
CONTEXT_SIZE = 256      # Full tile size for context (tissue neighborhood)
ENCODER_INPUT = 224     # Encoder input size


class MultiscalePatchDataset(Dataset):
    """Dataset for loading H&E patches with multiscale crops."""

    def __init__(
        self,
        data_dir: Path,
        patients: List[str],
        local_size: int = LOCAL_CROP_SIZE,
        context_size: int = CONTEXT_SIZE,
        encoder_input: int = ENCODER_INPUT,
    ):
        """
        Args:
            data_dir: Path to processed_crc_aligned directory
            patients: List of patient IDs
            local_size: Size of center crop for local features
            context_size: Size of full tile for context features
            encoder_input: Input size for encoder (resize target)
        """
        self.data_dir = Path(data_dir)
        self.patches = []
        self.local_size = local_size
        self.context_size = context_size
        self.encoder_input = encoder_input

        # Load patches from each patient
        for patient in patients:
            patient_dir = self.data_dir / patient
            patches_path = patient_dir / 'patches.npy'

            if not patches_path.exists():
                print(f"Warning: {patches_path} not found, skipping {patient}")
                continue

            patient_patches = np.load(patches_path, allow_pickle=True)

            for idx, patch_data in enumerate(patient_patches):
                img_path_rel = patch_data['img_path']
                if img_path_rel.startswith('./'):
                    img_path_rel = img_path_rel[2:]
                img_path = patient_dir / img_path_rel

                if img_path.exists():
                    self.patches.append({
                        'img_path': str(img_path),
                        'patient': patient,
                        'patch_idx': idx,
                    })

        print(f"Loaded {len(self.patches)} patches from {len(patients)} patients")

        # Transforms for each scale
        # Local: center crop to local_size, then resize to encoder_input
        self.local_transform = transforms.Compose([
            transforms.CenterCrop(local_size),
            transforms.Resize((encoder_input, encoder_input)),
            transforms.ToTensor(),
        ])

        # Context: resize full tile to encoder_input (preserves context)
        self.context_transform = transforms.Compose([
            transforms.Resize((encoder_input, encoder_input)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patch_info = self.patches[idx]

        # Load image
        img = Image.open(patch_info['img_path']).convert('RGB')

        # Apply transforms for each scale
        local_tensor = self.local_transform(img)
        context_tensor = self.context_transform(img)

        return {
            'local': local_tensor,
            'context': context_tensor,
            'patient': patch_info['patient'],
            'patch_idx': patch_info['patch_idx'],
        }


def extract_multiscale_features(
    encoder_name: str,
    output_path: str,
    data_dir: Optional[str] = None,
    patients: Optional[List[str]] = None,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract multiscale features from patches.

    Args:
        encoder_name: Name of encoder from encoder_config.yaml
        output_path: Path to save output .npz file
        data_dir: Path to processed data directory
        patients: List of patient IDs
        batch_size: Batch size for inference
        device: Device to use

    Returns:
        Dict with 'local', 'context', 'concat' features
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if patients is None:
        patients = PATIENTS
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Extracting multiscale features with {encoder_name}")
    print(f"  Local crop: {LOCAL_CROP_SIZE}px center → {ENCODER_INPUT}px")
    print(f"  Context: {CONTEXT_SIZE}px full → {ENCODER_INPUT}px")
    print(f"  Device: {device}")

    # Load encoder
    print(f"\nLoading encoder: {encoder_name}...")
    encoder = get_encoder(encoder_name, device=device)
    embed_dim = encoder.embed_dim
    print(f"  Embed dim: {embed_dim}")

    # Create dataset
    dataset = MultiscalePatchDataset(
        data_dir=data_dir,
        patients=patients,
        local_size=LOCAL_CROP_SIZE,
        context_size=CONTEXT_SIZE,
        encoder_input=ENCODER_INPUT,
    )

    if len(dataset) == 0:
        raise RuntimeError("No patches found!")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Extract features
    all_local = []
    all_context = []
    all_patient_ids = []
    all_patch_indices = []

    print(f"\nExtracting features from {len(dataset)} patches...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Multiscale ({encoder_name})"):
            local_imgs = batch['local'].to(device)
            context_imgs = batch['context'].to(device)

            # Forward pass for each scale
            local_feats = encoder(local_imgs, normalize=True)
            context_feats = encoder(context_imgs, normalize=True)

            all_local.append(local_feats.cpu().numpy())
            all_context.append(context_feats.cpu().numpy())
            all_patient_ids.extend(batch['patient'])
            all_patch_indices.extend(batch['patch_idx'].numpy())

    # Concatenate
    local_array = np.concatenate(all_local, axis=0)
    context_array = np.concatenate(all_context, axis=0)
    concat_array = np.concatenate([local_array, context_array], axis=1)

    patient_ids_array = np.array(all_patient_ids)
    patch_indices_array = np.array(all_patch_indices)

    print(f"\nFeature shapes:")
    print(f"  Local:   {local_array.shape}")
    print(f"  Context: {context_array.shape}")
    print(f"  Concat:  {concat_array.shape}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        local=local_array.astype(np.float32),
        context=context_array.astype(np.float32),
        concat=concat_array.astype(np.float32),
        patient_ids=patient_ids_array,
        patch_indices=patch_indices_array,
        encoder_name=encoder_name,
        embed_dim=embed_dim,
        local_crop_size=LOCAL_CROP_SIZE,
        context_size=CONTEXT_SIZE,
    )

    print(f"\nSaved multiscale features to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return {
        'local': local_array,
        'context': context_array,
        'concat': concat_array,
        'patient_ids': patient_ids_array,
        'patch_indices': patch_indices_array,
    }


def main():
    parser = argparse.ArgumentParser(description='Extract multiscale features')
    parser.add_argument('--encoder', type=str, default='densenet_imagenet',
                       help='Encoder name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: features/{encoder}_multiscale.npz)')
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR),
                       help='Path to processed data directory')
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
        args.output = f"features/{args.encoder}_multiscale.npz"

    extract_multiscale_features(
        encoder_name=args.encoder,
        output_path=args.output,
        data_dir=args.data_dir,
        patients=args.patients,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == '__main__':
    main()
