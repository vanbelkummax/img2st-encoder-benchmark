#!/usr/bin/env python3
"""
Extract frozen DenseNet features from all patches.

This creates a feature bank that can be used to quickly evaluate
gene predictability without retraining the full Img2ST model.

Output: NPZ file with patch features [N_patches, 1024] and metadata.
"""

import sys
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_DIR = SCRIPT_DIR.parent / 'model'
sys.path.insert(0, str(MODEL_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


class PatchDataset(Dataset):
    """Dataset for loading patches without expression labels."""

    def __init__(self, slide_dir, transform=None):
        self.transform = transform
        self.slide_dir = Path(slide_dir)
        patches_path = self.slide_dir / 'patches.npy'
        self.patches = np.load(patches_path, allow_pickle=True).tolist()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        item = self.patches[idx]
        img_rel_path = item['img_path'].lstrip('./')
        img_path = self.slide_dir / img_rel_path

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Get expression labels if available (for later use)
        label = np.array(item['label'])
        if label.ndim == 2 and label.shape[0] == 1024:
            label = label.reshape(32, 32, -1)

        return image, label, idx


class FeatureExtractor(nn.Module):
    """Frozen DenseNet121 feature extractor."""

    def __init__(self, pretrained_weights=None):
        super().__init__()

        # Load DenseNet121
        self.backbone = models.densenet121(weights='IMAGENET1K_V1')

        # Remove classifier
        self.features = self.backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Optionally load fine-tuned weights
        if pretrained_weights is not None:
            self._load_weights(pretrained_weights)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def _load_weights(self, checkpoint_path):
        """Load weights from Img2ST checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Extract backbone weights (img_encoder.backbone.*)
        backbone_weights = {}
        for k, v in state_dict.items():
            if 'img_encoder.backbone' in k:
                new_key = k.replace('img_encoder.backbone.', '')
                backbone_weights[new_key] = v

        if backbone_weights:
            self.backbone.load_state_dict(backbone_weights, strict=False)
            print(f"Loaded {len(backbone_weights)} backbone weights from checkpoint")

    def forward(self, x):
        """Extract global pooled features."""
        features = self.features(x)
        pooled = self.pool(features)
        return pooled.view(pooled.size(0), -1)  # [B, 1024]

    def forward_spatial(self, x):
        """Extract spatial feature maps (for per-bin analysis)."""
        features = self.features(x)  # [B, 1024, H, W]
        return features


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract patch features using frozen DenseNet')
    parser.add_argument('--data_dir', type=str,
                       default='/mnt/x/img2st_rotation_demo/processed_crc_aligned',
                       help='Directory containing processed patches')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Optional: path to Img2ST checkpoint for fine-tuned backbone')
    parser.add_argument('--output_dir', type=str,
                       default='/home/user/img2st-baseline-comparison/features',
                       help='Output directory for features')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for feature extraction')
    parser.add_argument('--spatial', action='store_true',
                       help='Extract spatial feature maps instead of global pooled')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load feature extractor
    print("Loading DenseNet121 feature extractor...")
    extractor = FeatureExtractor(pretrained_weights=args.model_path)
    extractor = extractor.to(device)
    extractor.eval()

    # Standard transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    all_features = []
    all_labels = []
    all_patient_ids = []
    all_patch_indices = []

    for patient in ['P1', 'P2', 'P5']:
        print(f"\n=== Processing {patient} ===")

        slide_dir = os.path.join(args.data_dir, patient)
        if not os.path.exists(slide_dir):
            print(f"Skipping {patient} - directory not found")
            continue

        dataset = PatchDataset(slide_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)

        print(f"Extracting features from {len(dataset)} patches...")

        patient_features = []
        patient_labels = []
        patient_indices = []

        with torch.no_grad():
            for images, labels, indices in tqdm(loader, desc=patient):
                images = images.to(device)

                if args.spatial:
                    # Spatial features [B, 1024, H, W]
                    features = extractor.forward_spatial(images)
                    # Interpolate to 32x32 to match expression grid
                    features = nn.functional.interpolate(
                        features, size=(32, 32), mode='bilinear', align_corners=False
                    )
                    features = features.cpu().numpy()
                else:
                    # Global pooled features [B, 1024]
                    features = extractor(images)
                    features = features.cpu().numpy()

                patient_features.append(features)
                patient_labels.append(labels.numpy())
                patient_indices.extend(indices.tolist())

        patient_features = np.concatenate(patient_features, axis=0)
        patient_labels = np.concatenate(patient_labels, axis=0)

        all_features.append(patient_features)
        all_labels.append(patient_labels)
        all_patient_ids.extend([patient] * len(patient_features))
        all_patch_indices.extend(patient_indices)

        print(f"  Features shape: {patient_features.shape}")
        print(f"  Labels shape: {patient_labels.shape}")

    # Combine all patients
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\n=== Combined Dataset ===")
    print(f"Features: {all_features.shape}")
    print(f"Labels: {all_labels.shape}")
    print(f"Patients: {len(set(all_patient_ids))}")

    # Save features
    suffix = '_spatial' if args.spatial else '_global'
    output_path = os.path.join(args.output_dir, f'patch_features{suffix}.npz')

    np.savez_compressed(
        output_path,
        features=all_features,
        labels=all_labels,
        patient_ids=np.array(all_patient_ids),
        patch_indices=np.array(all_patch_indices)
    )

    print(f"\nSaved: {output_path}")

    # Also load and save gene names
    gene_names_path = os.path.join(args.data_dir, 'gene_names.json')
    if os.path.exists(gene_names_path):
        with open(gene_names_path, 'r') as f:
            gene_names = json.load(f)
        gene_names_out = os.path.join(args.output_dir, 'gene_names_50.json')
        with open(gene_names_out, 'w') as f:
            json.dump(gene_names, f, indent=2)
        print(f"Saved gene names: {gene_names_out}")


if __name__ == '__main__':
    main()
