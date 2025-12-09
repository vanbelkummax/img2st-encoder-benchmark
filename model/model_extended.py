"""
Extended Img2ST-Net Model with Additional bin_num Support

Extends the original model to support bin_num=64 (8×8 grid)
for multi-scale spatial transcriptomics prediction.

Phase 2b: Adds optional GNN head for spatial context (8-connected grid graph).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
from functools import lru_cache


# -------------------------
# Spatial GCN for grid graphs (Phase 2b)
# -------------------------
@lru_cache(maxsize=4)
def build_grid_adjacency(grid_size: int, device: str = 'cpu') -> torch.Tensor:
    """
    Build normalized adjacency matrix for 8-connected 2D grid.
    Uses symmetric normalization: D^{-1/2} A D^{-1/2} for GCN.
    Cached per grid_size to avoid recomputation.
    """
    N = grid_size * grid_size
    edges = []

    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j
            # 8 neighbors: cardinal + diagonal
            neighbors = [
                (i-1, j), (i+1, j), (i, j-1), (i, j+1),  # cardinal
                (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)  # diagonal
            ]
            for ni, nj in neighbors:
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    neighbor_node = ni * grid_size + nj
                    edges.append((node, neighbor_node))

    # Build sparse adjacency with self-loops
    row = [e[0] for e in edges] + list(range(N))  # Add self-loops
    col = [e[1] for e in edges] + list(range(N))

    # Build dense adjacency (small enough for 32x32 = 1024 nodes)
    A = torch.zeros(N, N, dtype=torch.float32, device=device)
    for r, c in zip(row, col):
        A[r, c] = 1.0

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    D = A.sum(dim=1)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    A_norm = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)

    return A_norm


class SpatialGCNLayer(nn.Module):
    """Single GCN layer with optional bias."""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, D) node features
            A_norm: (N, N) normalized adjacency matrix
        Returns:
            (N, D') transformed features
        """
        # GCN propagation: A_norm @ X @ W
        return self.linear(A_norm @ x)


class SpatialGCN(nn.Module):
    """
    2-layer GCN for spatial context in 2D grids.
    Applied to feature maps after upsampling to target grid size.

    Input:  (B, C, H, W) feature map where H*W = grid_size^2
    Output: (B, C, H, W) enhanced features with spatial context

    Uses learnable gate (initialized small) to avoid disrupting pretrained features.
    """
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 512,
                 grid_size: int = 32, num_layers: int = 2, dropout: float = 0.1,
                 init_gate: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Build GCN layers
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            layers.append(SpatialGCNLayer(dims[i], dims[i+1]))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        # Residual projection if dimensions differ
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # Learnable gate: controls how much GNN output contributes
        # Initialize small (0.1) to avoid disrupting pretrained backbone features
        self.gate = nn.Parameter(torch.tensor(init_gate))

        # Cache for adjacency matrix
        self._A_norm = None
        self._A_device = None

    def _get_adjacency(self, device: torch.device) -> torch.Tensor:
        """Get cached adjacency matrix, moving to device if needed."""
        device_str = str(device)
        if self._A_norm is None or self._A_device != device_str:
            self._A_norm = build_grid_adjacency(self.grid_size, device_str)
            self._A_device = device_str
        return self._A_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            (B, C, H, W) enhanced features
        """
        B, C, H, W = x.shape
        assert H == W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        N = H * W

        # Reshape to (B, N, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)

        # Get normalized adjacency
        A_norm = self._get_adjacency(x.device)

        # Process each sample (adjacency is same for all)
        out_list = []
        for b in range(B):
            h = x_flat[b]  # (N, C)
            residual = self.residual(h)

            for i, layer in enumerate(self.layers):
                h = layer(h, A_norm)
                if i < len(self.layers) - 1:  # No activation after last layer
                    h = F.relu(h)
                    h = self.dropout(h)

            # Gated residual: gate controls GNN contribution (starts small to preserve pretrained features)
            h = self.norm(residual + self.gate * h)
            out_list.append(h)

        # Stack and reshape back to (B, C, H, W)
        out = torch.stack(out_list, dim=0)  # (B, N, out_dim)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, out_dim, H, W)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class MiniUNet(nn.Module):
    """
    Input:  (B,1024,14,14) feature map
    Output: (B,512,14,14) enhanced feature
    """
    def __init__(self, in_ch=1024, mid_ch=512, out_ch=512):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, mid_ch)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(mid_ch, mid_ch)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec  = ConvBlock(mid_ch + mid_ch, out_ch)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)
        x  = torch.cat([x4, x1], dim=1)
        return self.dec(x)


class MultiBranchSpatialPredictorV2Extended(nn.Module):
    """
    Extended model with additional bin_num support (64 for 8×8 grid).

    Supported bin_num values:
    - 1024: 32×32 (2µm resolution)
    - 196:  14×14
    - 64:   8×8   (8µm resolution) [NEW]
    - 49:   7×7
    - 1:    1×1   (global)

    Phase 2b: Optional GNN head for spatial context (8-connected grid graph).
    """
    def __init__(self,
                 bin_num: int,
                 st_in_dim: Optional[int],
                 pred_dim: int = 300,
                 ctr_dim: int = 256,
                 densenet_weights: str = "IMAGENET1K_V1",
                 use_gnn: bool = False,
                 gnn_layers: int = 2,
                 gnn_dropout: float = 0.1):
        super().__init__()

        # Backbone: DenseNet121
        try:
            densenet = models.densenet121(weights=getattr(models.DenseNet121_Weights, densenet_weights))
        except Exception:
            densenet = models.densenet121(weights=densenet_weights)
        self.backbone = densenet.features

        self.unet = MiniUNet(1024, 512, 512)

        # Extended bin_num support
        self.bin_num = bin_num
        if bin_num == 1024:      # 32×32 (2µm)
            self.resize_to_grid = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
            self.hw = (32, 32)
        elif bin_num == 196:     # 14×14
            self.resize_to_grid = nn.Identity()
            self.hw = (14, 14)
        elif bin_num == 64:      # 8×8 (8µm) - NEW
            self.resize_to_grid = nn.AdaptiveAvgPool2d((8, 8))
            self.hw = (8, 8)
        elif bin_num == 49:      # 7×7
            self.resize_to_grid = nn.AdaptiveAvgPool2d((7, 7))
            self.hw = (7, 7)
        elif bin_num == 1:       # 1×1
            self.resize_to_grid = nn.AdaptiveAvgPool2d((1, 1))
            self.hw = (1, 1)
        else:
            raise ValueError(f"Unsupported bin_num={bin_num}. Supported: 1, 49, 64, 196, 1024.")

        # Optional GNN for spatial context (Phase 2b)
        self.use_gnn = use_gnn
        if use_gnn:
            grid_size = self.hw[0]  # Assumes square grid
            self.spatial_gnn = SpatialGCN(
                in_dim=512, hidden_dim=512, out_dim=512,
                grid_size=grid_size, num_layers=gnn_layers, dropout=gnn_dropout
            )

        # Image branch heads
        self.image_pred_head = nn.Sequential(
            nn.Conv2d(512, 512, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, pred_dim, 1)
        )
        self.image_ctr_head = nn.Sequential(
            nn.Conv2d(512, 512, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, ctr_dim, 1)
        )

        # ST branch
        self.has_st = st_in_dim is not None
        if self.has_st:
            hidden = max(pred_dim * 2, 256)
            self.st_shared = nn.Sequential(
                nn.Linear(st_in_dim, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.st_pred_head = nn.Linear(hidden, pred_dim)
            self.st_ctr_head  = nn.Linear(hidden, ctr_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.unet, self.image_pred_head, self.image_ctr_head]:
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                    if mod.bias is not None: nn.init.zeros_(mod.bias)
                elif isinstance(mod, nn.BatchNorm2d):
                    nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)
        if self.has_st:
            for mod in self.st_shared.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    if mod.bias is not None: nn.init.zeros_(mod.bias)
            nn.init.xavier_uniform_(self.st_pred_head.weight)
            nn.init.zeros_(self.st_pred_head.bias)
            nn.init.xavier_uniform_(self.st_ctr_head.weight)
            nn.init.zeros_(self.st_ctr_head.bias)

    @staticmethod
    def _to_seq(x_2d: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_2d.shape
        return x_2d.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def forward(self, image: torch.Tensor, st: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        # Image stream
        feat = self.backbone(image)
        feat = self.unet(feat)
        feat = self.resize_to_grid(feat)

        # Optional GNN for spatial context (Phase 2b)
        if self.use_gnn:
            feat = self.spatial_gnn(feat)  # (B,512,H,W) enhanced with neighbor context

        img_pred_map = self.image_pred_head(feat)
        img_pred = self._to_seq(img_pred_map)

        img_ctr_map = self.image_ctr_head(feat)
        img_ctr = self._to_seq(img_ctr_map)

        st_pred = st_ctr = None
        if self.has_st and st is not None:
            shared = self.st_shared(st)
            st_pred = self.st_pred_head(shared)
            st_ctr  = self.st_ctr_head(shared)

        return img_pred, st_pred, img_ctr, st_ctr


class ImageSTContrastive(nn.Module):
    """Contrastive loss between image and ST embeddings."""
    def __init__(self, temperature: float = 0.07, normalize: bool = True, patch_agg: str = 'mean'):
        super().__init__()
        assert patch_agg in ['mean', 'patch']
        self.tau = temperature
        self.normalize = normalize
        self.patch_agg = patch_agg

    def _norm(self, x):
        return F.normalize(x, dim=-1) if self.normalize else x

    def forward(self, img_ctr: torch.Tensor, st_ctr: torch.Tensor) -> torch.Tensor:
        if self.patch_agg == 'mean':
            img_vec = self._norm(img_ctr.mean(dim=1))
            st_vec  = self._norm(st_ctr.mean(dim=1))
            logits_i2s = (img_vec @ st_vec.t()) / self.tau
            logits_s2i = (st_vec @ img_vec.t()) / self.tau
            targets = torch.arange(img_vec.size(0), device=img_ctr.device)
            return 0.5 * (F.cross_entropy(logits_i2s, targets) +
                          F.cross_entropy(logits_s2i, targets))
        else:
            B, P, D = img_ctr.shape
            img_flat = self._norm(img_ctr.reshape(B * P, D))
            st_flat  = self._norm(st_ctr.reshape(B * P, D))
            logits_i2s = (img_flat @ st_flat.t()) / self.tau
            logits_s2i = (st_flat @ img_flat.t()) / self.tau
            targets = torch.arange(B * P, device=img_ctr.device)
            return 0.5 * (F.cross_entropy(logits_i2s, targets) +
                          F.cross_entropy(logits_s2i, targets))
