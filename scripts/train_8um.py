#!/usr/bin/env python3
"""
Train Img2ST-Net on 8µm Aggregated Data

Uses 8×8 grid (64 bins) aggregated from 2µm data.
Same training procedure as 2µm but with bin_num=64.

Usage:
    python train_8um.py \
        --data_dir data/processed_8um \
        --output_dir weights \
        --test_slide D \
        --exp_name colon8_lambda05
"""

import sys
import argparse
from pathlib import Path

# Add model directory to path (relative to script location)
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_DIR = SCRIPT_DIR.parent / 'model'
sys.path.insert(0, str(MODEL_DIR))

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model_extended import MultiBranchSpatialPredictorV2Extended, ImageSTContrastive

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 448
GRID_SIZE = 8  # 8×8 grid for 8µm
BIN_NUM = 64   # 8×8 = 64 bins
NUM_GENES = 50


class ColonHDDataset(Dataset):
    """Dataset for 8µm aggregated colon HD data."""

    def __init__(self, data_dir, slides, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.items = []

        for slide in slides:
            npy_path = self.data_dir / 'raw_setting' / '4' / f'{slide}.npy'
            data = np.load(npy_path, allow_pickle=True)
            for item in data:
                self.items.append({
                    'img_path': self.data_dir / item['img_path'].lstrip('./'),
                    'label': item['label']
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load image
        image = Image.open(item['img_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load expression label [64, 50]
        label = torch.tensor(item['label'], dtype=torch.float32)

        return image, label


def train_epoch(model, dataloader, optimizer, criterion, ctr_criterion, ctr_weight):
    """Train for one epoch with contrastive learning."""
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_ctr_loss = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass - model returns (img_pred, st_pred, img_ctr, st_ctr)
        img_pred, st_pred, img_ctr, st_ctr = model(images, labels)

        # Prediction loss (MSE)
        pred_loss = criterion(img_pred, labels)

        # Contrastive loss (image-ST alignment)
        if st_ctr is not None:
            ctr_loss = ctr_criterion(img_ctr, st_ctr)
        else:
            ctr_loss = torch.tensor(0.0, device=DEVICE)

        # Combined loss
        loss = pred_loss + ctr_weight * ctr_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_ctr_loss += ctr_loss.item() if isinstance(ctr_loss, torch.Tensor) else ctr_loss

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'pred_loss': total_pred_loss / n_batches,
        'ctr_loss': total_ctr_loss / n_batches
    }


def evaluate(model, dataloader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            img_pred, _, _, _ = model(images)
            loss = criterion(img_pred, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(
        description="Train Img2ST-Net on 8µm data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to 8µm preprocessed data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for model weights')
    parser.add_argument('--test_slide', type=str, required=True,
                        help='Held-out slide (A, B, C, or D)')
    parser.add_argument('--ctr_weight', type=float, default=0.5,
                        help='Contrastive loss weight')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--exp_name', type=str, default='colon8_lambda05',
                        help='Experiment name')
    args = parser.parse_args()

    print("=" * 70)
    print(f"Training 8µm Model (bin_num={BIN_NUM})")
    print("=" * 70)
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Test slide: {args.test_slide}")
    print(f"CTR weight: {args.ctr_weight}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {DEVICE}")

    # Data directories
    data_dir = Path(args.data_dir)
    weight_dir = Path(args.output_dir) / args.exp_name
    weight_dir.mkdir(parents=True, exist_ok=True)

    # Train/test split
    all_slides = ['A', 'B', 'C', 'D']
    train_slides = [s for s in all_slides if s != args.test_slide]
    test_slides = [args.test_slide]

    print(f"Train slides: {train_slides}")
    print(f"Test slide: {test_slides}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = ColonHDDataset(data_dir, train_slides, transform)
    test_dataset = ColonHDDataset(data_dir, test_slides, transform)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model (using extended version that supports 8×8 grid)
    model = MultiBranchSpatialPredictorV2Extended(
        bin_num=BIN_NUM,
        st_in_dim=NUM_GENES,
        pred_dim=NUM_GENES,
        ctr_dim=256
    ).to(DEVICE)

    # Contrastive loss
    ctr_criterion = ImageSTContrastive(temperature=0.07, patch_agg='mean')

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    best_test_loss = float('inf')
    print("\n" + "-" * 70)

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, ctr_criterion, args.ctr_weight)
        test_loss = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_metrics['loss']:.4f} (pred={train_metrics['pred_loss']:.4f}, ctr={train_metrics['ctr_loss']:.4f}) | "
              f"Test: {test_loss:.4f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_loss': test_loss,
                'config': {
                    'bin_num': BIN_NUM,
                    'grid_size': GRID_SIZE,
                    'ctr_weight': args.ctr_weight,
                    'test_slide': args.test_slide
                }
            }
            torch.save(checkpoint, weight_dir / f'model_best_{BIN_NUM}.pth')

    print("-" * 70)
    print(f"\nBest test loss: {best_test_loss:.4f}")
    print(f"Model saved: {weight_dir / f'model_best_{BIN_NUM}.pth'}")


if __name__ == '__main__':
    main()
