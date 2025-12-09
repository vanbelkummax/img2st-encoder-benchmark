#!/usr/bin/env python3
"""
Train Img2ST-Net on Aligned CRC Data with Slide-Level Split

Uses P2 (74 patches) for training, P5 (71 patches) for validation.
Both samples use identical 50-gene panel from master gene list.
8µm native bins with 32×32 grid (1024 bins).

Usage:
    python train_crc_aligned.py \
        --data_dir /mnt/x/img2st_rotation_demo/processed_crc_aligned \
        --output_dir weights/crc_aligned \
        --ctr_weight 0.25 \
        --epochs 50
"""

import sys
import argparse
from pathlib import Path

# Add model directory to path
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
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
from model_extended import MultiBranchSpatialPredictorV2Extended, ImageSTContrastive


def compute_ssim_st(pred, gt):
    """Compute SSIM-ST metric for sparse ST data (paper Section 3.3).

    Args:
        pred: Predictions [N, bins, genes] or [N, genes] flattened
        gt: Ground truth [N, bins, genes] or [N, genes] flattened

    Returns:
        Mean SSIM across all genes
    """
    # Reshape to [H, W, genes] if needed
    if pred.ndim == 2:
        n_bins, n_genes = pred.shape
        h = w = int(np.sqrt(n_bins))
        pred = pred.reshape(h, w, n_genes)
        gt = gt.reshape(h, w, n_genes)
    elif pred.ndim == 3 and pred.shape[0] != pred.shape[1]:
        # [N, bins, genes] -> combine all patches
        pred = pred.reshape(-1, pred.shape[-1])
        gt = gt.reshape(-1, gt.shape[-1])
        n_bins = pred.shape[0]
        h = w = int(np.sqrt(n_bins))
        # Can't reshape non-square, compute per-gene only
        ssim_scores = []
        for g in range(pred.shape[-1]):
            p, t = pred[:, g], gt[:, g]
            if np.std(t) > 0:
                # Use correlation as proxy for 1D data
                r, _ = pearsonr(p, t)
                ssim_scores.append(max(0, r))  # Clamp negative
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
GRID_SIZE = 32   # 32×32 grid for 8µm native
BIN_NUM = 1024   # 32×32 = 1024 bins
NUM_GENES = 50


class CRCAlignedDataset(Dataset):
    """Dataset for CRC aligned patches with slide-level split."""

    def __init__(self, patches_npy_path, transform=None):
        """
        Args:
            patches_npy_path: Path to patches.npy file
            transform: Image transforms
        """
        self.transform = transform
        self.data_dir = Path(patches_npy_path).parent

        # Load patches
        self.patches = np.load(patches_npy_path, allow_pickle=True).tolist()

        # Determine image directory based on slide
        # The img_path in patches is relative to the sample directory

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        item = self.patches[idx]

        # Construct full image path
        slide = item.get('slide', 'P2')
        img_rel_path = item['img_path'].lstrip('./')
        img_path = self.data_dir / slide / img_rel_path

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load expression label [1024, 50]
        label = torch.tensor(item['label'], dtype=torch.float32)

        return image, label


def train_epoch(model, dataloader, optimizer, criterion, ctr_criterion, ctr_weight, accum_steps=1):
    """Train for one epoch with contrastive learning and gradient accumulation.

    Args:
        accum_steps: Gradient accumulation steps (effective_batch = batch_size * accum_steps)
    """
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_ctr_loss = 0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        img_pred, st_pred, img_ctr, st_ctr = model(images, labels)

        # Prediction loss (MSE)
        pred_loss = criterion(img_pred, labels)

        # Contrastive loss
        if st_ctr is not None:
            ctr_loss = ctr_criterion(img_ctr, st_ctr)
        else:
            ctr_loss = torch.tensor(0.0, device=DEVICE)

        # Combined loss (scaled for gradient accumulation)
        loss = (pred_loss + ctr_weight * ctr_loss) / accum_steps

        loss.backward()

        # Step optimizer every accum_steps batches
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps  # Unscale for logging
        total_pred_loss += pred_loss.item()
        total_ctr_loss += ctr_loss.item() if isinstance(ctr_loss, torch.Tensor) else ctr_loss

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'pred_loss': total_pred_loss / n_batches,
        'ctr_loss': total_ctr_loss / n_batches
    }


def evaluate_with_correlation(model, dataloader, criterion):
    """Evaluate model with MSE, RMSE, Pearson r, Spearman rho, and SSIM-ST."""
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
    rmse = np.sqrt(mse)  # RMSE for interpretability

    # Compute correlations
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Flatten for overall correlation
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    # Avoid NaN if data has zero variance
    if np.std(preds_flat) > 0 and np.std(labels_flat) > 0:
        r, _ = pearsonr(preds_flat, labels_flat)
        rho, _ = spearmanr(preds_flat, labels_flat)  # Spearman (rank-based, robust to outliers)
    else:
        r = 0.0
        rho = 0.0

    # Compute SSIM-ST (paper's preferred metric)
    ssim_st = compute_ssim_st(preds, labels)

    return {
        'mse': mse,
        'rmse': rmse,
        'pearson_r': r,
        'spearman_rho': rho,
        'ssim_st': ssim_st
    }


def main():
    parser = argparse.ArgumentParser(description="Train on Aligned CRC Data")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed_crc_aligned directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for model weights')
    parser.add_argument('--ctr_weight', type=float, default=0.25,
                        help='Contrastive loss weight (lambda)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--exp_name', type=str, default='crc_aligned_lambda025',
                        help='Experiment name')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
    # Phase 2b: GNN arguments
    parser.add_argument('--use_gnn_head', action='store_true',
                        help='Enable GNN spatial head for neighbor context (Phase 2b)')
    parser.add_argument('--gnn_layers', type=int, default=2,
                        help='Number of GCN layers (default: 2)')
    parser.add_argument('--gnn_dropout', type=float, default=0.1,
                        help='GNN dropout rate (default: 0.1)')
    args = parser.parse_args()

    print("=" * 70)
    print("Training on Aligned CRC Data (Slide-Level Split)")
    print("=" * 70)
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Lambda (CTR weight): {args.ctr_weight}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.accum_steps} with {args.accum_steps}x accumulation)")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} = {BIN_NUM} bins")
    print(f"Genes: {NUM_GENES}")
    print(f"GNN Head: {'Enabled' if args.use_gnn_head else 'Disabled'}" +
          (f" ({args.gnn_layers} layers, dropout={args.gnn_dropout})" if args.use_gnn_head else ""))
    print(f"Device: {DEVICE}")

    # Directories
    data_dir = Path(args.data_dir)
    weight_dir = Path(args.output_dir) / args.exp_name
    weight_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets (slide-level split already done)
    train_dataset = CRCAlignedDataset(data_dir / 'train_patches.npy', transform)
    val_dataset = CRCAlignedDataset(data_dir / 'val_patches.npy', transform)

    print(f"\nSlide-level split:")
    print(f"  Training: {len(train_dataset)} patches (P2)")
    print(f"  Validation: {len(val_dataset)} patches (P5)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = MultiBranchSpatialPredictorV2Extended(
        bin_num=BIN_NUM,
        st_in_dim=NUM_GENES,
        pred_dim=NUM_GENES,
        ctr_dim=256,
        use_gnn=args.use_gnn_head,
        gnn_layers=args.gnn_layers,
        gnn_dropout=args.gnn_dropout
    ).to(DEVICE)

    # Contrastive loss
    ctr_criterion = ImageSTContrastive(temperature=0.07, patch_agg='mean')

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    best_val_mse = float('inf')  # MSE-first selection (lower is better)
    best_val_r = -1.0
    best_ssim_st = 0.0
    print("\n" + "-" * 70)

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer,
                                    criterion, ctr_criterion, args.ctr_weight,
                                    accum_steps=args.accum_steps)
        val_metrics = evaluate_with_correlation(model, val_loader, criterion)

        # Extract metrics from dict
        val_mse = val_metrics['mse']
        val_rmse = val_metrics['rmse']
        val_r = val_metrics['pearson_r']
        val_rho = val_metrics['spearman_rho']
        val_ssim = val_metrics['ssim_st']

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} "
              f"(pred={train_metrics['pred_loss']:.4f}, ctr={train_metrics['ctr_loss']:.4f}) | "
              f"Val MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, r={val_r:.4f}, ρ={val_rho:.4f}, SSIM-ST={val_ssim:.4f}")

        # Create checkpoint for current epoch
        current_checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_r': val_r,
            'val_rho': val_rho,
            'val_ssim_st': val_ssim,
            'config': {
                'bin_num': BIN_NUM,
                'grid_size': GRID_SIZE,
                'num_genes': NUM_GENES,
                'ctr_weight': args.ctr_weight,
                'accum_steps': args.accum_steps,
                'use_gnn': args.use_gnn_head,
                'gnn_layers': args.gnn_layers,
                'gnn_dropout': args.gnn_dropout
            }
        }

        # Save best model (by MSE, following Img2ST paper and Phase 3 Priority 1)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_r = val_r
            best_ssim_st = val_ssim
            torch.save(current_checkpoint, weight_dir / 'model_best.pth')
            print(f"  -> New best model saved (MSE={val_mse:.4f}, r={val_r:.4f}, SSIM-ST={val_ssim:.4f})")

        # Always save latest model (true current state)
        torch.save(current_checkpoint, weight_dir / 'model_latest.pth')

    print("-" * 70)
    print(f"\nTraining complete!")
    print(f"Best validation MSE: {best_val_mse:.4f}")
    print(f"Best validation Pearson r: {best_val_r:.4f}")
    print(f"Best validation SSIM-ST: {best_ssim_st:.4f}")
    print(f"Model saved: {weight_dir / 'model_best.pth'}")


if __name__ == '__main__':
    main()
