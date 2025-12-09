#!/usr/bin/env python3
"""
Train Img2ST-Net with Leave-One-Out Cross-Validation (LOOCV)

Performs 3-fold cross-validation with P1, P2, P5 slides.
Each fold uses one slide as test, remaining two as training.

Usage:
    python train_loocv.py \
        --data_dir /mnt/x/img2st_rotation_demo/processed_crc_aligned \
        --output_dir weights/loocv \
        --ctr_weight 0.0 \
        --epochs 200
"""

import sys
import argparse
from pathlib import Path

# Add model directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_DIR = SCRIPT_DIR.parent / 'model'
sys.path.insert(0, str(MODEL_DIR))

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
from model_extended import MultiBranchSpatialPredictorV2Extended, ImageSTContrastive


def compute_ssim_st(pred, gt):
    """Compute SSIM-ST metric for sparse ST data."""
    if pred.ndim == 2:
        n_bins, n_genes = pred.shape
        h = w = int(np.sqrt(n_bins))
        pred = pred.reshape(h, w, n_genes)
        gt = gt.reshape(h, w, n_genes)
    elif pred.ndim == 3 and pred.shape[0] != pred.shape[1]:
        pred = pred.reshape(-1, pred.shape[-1])
        gt = gt.reshape(-1, gt.shape[-1])
        ssim_scores = []
        for g in range(pred.shape[-1]):
            p, t = pred[:, g], gt[:, g]
            if np.std(t) > 0:
                r, _ = pearsonr(p, t)
                ssim_scores.append(max(0, r))
        return np.mean(ssim_scores) if ssim_scores else 0.0

    ssim_scores = []
    for g in range(pred.shape[-1]):
        pred_gene = pred[:, :, g]
        gt_gene = gt[:, :, g]
        if np.std(gt_gene) > 0:
            data_range = gt_gene.max() - gt_gene.min()
            if data_range > 0:
                score = ssim(pred_gene, gt_gene, data_range=data_range)
                ssim_scores.append(score)

    return np.mean(ssim_scores) if ssim_scores else 0.0


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 448
GRID_SIZE = 32
BIN_NUM = 1024
NUM_GENES = 50


class SlideDataset(Dataset):
    """Dataset for a single slide's patches."""

    def __init__(self, slide_dir, slide_name, transform=None):
        self.transform = transform
        self.slide_dir = Path(slide_dir)
        self.slide_name = slide_name

        # Load patches
        patches_path = self.slide_dir / 'patches.npy'
        self.patches = np.load(patches_path, allow_pickle=True).tolist()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        item = self.patches[idx]

        # Construct image path
        img_rel_path = item['img_path'].lstrip('./')
        img_path = self.slide_dir / img_rel_path

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load expression label
        label = torch.tensor(item['label'], dtype=torch.float32)

        return image, label


def train_epoch(model, dataloader, optimizer, criterion, ctr_criterion, ctr_weight, accum_steps=1):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_ctr_loss = 0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        img_pred, st_pred, img_ctr, st_ctr = model(images, labels)
        pred_loss = criterion(img_pred, labels)

        if st_ctr is not None:
            ctr_loss = ctr_criterion(img_ctr, st_ctr)
        else:
            ctr_loss = torch.tensor(0.0, device=DEVICE)

        loss = (pred_loss + ctr_weight * ctr_loss) / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        total_pred_loss += pred_loss.item()
        total_ctr_loss += ctr_loss.item() if isinstance(ctr_loss, torch.Tensor) else ctr_loss

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'pred_loss': total_pred_loss / n_batches,
        'ctr_loss': total_ctr_loss / n_batches
    }


def evaluate(model, dataloader, criterion):
    """Evaluate model with all metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            img_pred, _, _, _ = model(images)
            loss = criterion(img_pred, labels)
            total_loss += loss.item()

            all_preds.append(img_pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    mse = total_loss / len(dataloader)
    rmse = np.sqrt(mse)

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    if np.std(preds_flat) > 0 and np.std(labels_flat) > 0:
        r, _ = pearsonr(preds_flat, labels_flat)
        rho, _ = spearmanr(preds_flat, labels_flat)
    else:
        r = 0.0
        rho = 0.0

    ssim_st = compute_ssim_st(preds, labels)

    return {
        'mse': mse,
        'rmse': rmse,
        'pearson_r': r,
        'spearman_rho': rho,
        'ssim_st': ssim_st
    }


def train_fold(fold_name, train_datasets, test_dataset, args, output_dir):
    """Train a single LOOCV fold."""
    print(f"\n{'='*70}")
    print(f"FOLD: {fold_name}")
    print(f"{'='*70}")

    # Combine training datasets
    train_dataset = ConcatDataset(train_datasets)

    print(f"Training: {len(train_dataset)} patches")
    print(f"Testing: {len(test_dataset)} patches")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = MultiBranchSpatialPredictorV2Extended(
        bin_num=BIN_NUM,
        st_in_dim=NUM_GENES,
        pred_dim=NUM_GENES,
        ctr_dim=256
    ).to(DEVICE)

    ctr_criterion = ImageSTContrastive(temperature=0.07, patch_agg='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    best_test_mse = float('inf')  # MSE-first selection (lower is better)
    best_metrics = {}

    fold_dir = output_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-'*70}")

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer,
                                    criterion, ctr_criterion, args.ctr_weight,
                                    accum_steps=args.accum_steps)
        test_metrics = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Test r={test_metrics['pearson_r']:.4f}, "
              f"ρ={test_metrics['spearman_rho']:.4f}, "
              f"SSIM-ST={test_metrics['ssim_st']:.4f}")

        # Save best model (by MSE, following Img2ST paper and Phase 3 Priority 1)
        if test_metrics['mse'] < best_test_mse:
            best_test_mse = test_metrics['mse']
            best_metrics = test_metrics.copy()
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'test_metrics': test_metrics,
                'config': {
                    'bin_num': BIN_NUM,
                    'num_genes': NUM_GENES,
                    'ctr_weight': args.ctr_weight
                }
            }, fold_dir / 'model_best.pth')
            print(f"  -> New best model (MSE={best_test_mse:.4f}, r={test_metrics['pearson_r']:.4f})")

    print(f"{'-'*70}")
    print(f"Fold {fold_name} complete. Best MSE={best_test_mse:.4f}")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train with LOOCV")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed_crc_aligned directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for model weights')
    parser.add_argument('--ctr_weight', type=float, default=0.0,
                        help='Contrastive loss weight (lambda). Default 0.0 based on ablation showing MSE-only performs best.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs per fold. Default 200 based on extended training experiments.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--accum_steps', type=int, default=16,
                        help='Gradient accumulation steps')
    parser.add_argument('--exp_name', type=str, default='loocv_lambda025',
                        help='Experiment name')
    args = parser.parse_args()

    print("="*70)
    print("Leave-One-Out Cross-Validation (LOOCV)")
    print("="*70)
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Lambda: {args.ctr_weight}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Effective batch size: {args.batch_size * args.accum_steps}")
    print(f"Device: {DEVICE}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load all slide datasets
    slides = ['P1', 'P2', 'P5']
    datasets = {}
    for slide in slides:
        slide_dir = data_dir / slide
        if slide_dir.exists():
            datasets[slide] = SlideDataset(slide_dir, slide, transform)
            print(f"Loaded {slide}: {len(datasets[slide])} patches")
        else:
            print(f"Warning: {slide_dir} not found")

    if len(datasets) < 3:
        print("Error: Need all 3 slides (P1, P2, P5) for LOOCV")
        return

    # Define LOOCV folds
    folds = [
        ('test_P1', ['P2', 'P5'], 'P1'),
        ('test_P2', ['P1', 'P5'], 'P2'),
        ('test_P5', ['P1', 'P2'], 'P5'),
    ]

    # Run all folds
    all_results = {}

    for fold_name, train_slides, test_slide in folds:
        train_datasets = [datasets[s] for s in train_slides]
        test_dataset = datasets[test_slide]

        results = train_fold(fold_name, train_datasets, test_dataset, args, output_dir)
        all_results[fold_name] = results

    # Summary
    print("\n" + "="*70)
    print("LOOCV SUMMARY")
    print("="*70)

    metrics = ['pearson_r', 'spearman_rho', 'ssim_st', 'mse', 'rmse']
    summary = {}

    for metric in metrics:
        values = [all_results[fold][metric] for fold in all_results]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
        print(f"{metric}: {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f}")

    print(f"\nPer-fold results:")
    for fold_name, results in all_results.items():
        print(f"  {fold_name}: r={results['pearson_r']:.4f}, "
              f"ρ={results['spearman_rho']:.4f}, "
              f"SSIM-ST={results['ssim_st']:.4f}")

    # Save summary
    summary_path = output_dir / 'loocv_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'fold_results': {k: {m: float(v) for m, v in r.items()}
                            for k, r in all_results.items()},
            'summary': {m: {'mean': float(s['mean']), 'std': float(s['std'])}
                       for m, s in summary.items()},
            'config': {
                'ctr_weight': args.ctr_weight,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'accum_steps': args.accum_steps
            }
        }, f, indent=2)

    print(f"\nResults saved to {summary_path}")
    print("="*70)


if __name__ == '__main__':
    main()
