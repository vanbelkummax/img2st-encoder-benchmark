import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
from functools import lru_cache

# -------------------------
# Basic U-Net style blocks
# -------------------------
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
    Structure: encoder -> downsample -> deeper encoder -> upsample -> decoder with skip connection
    """
    def __init__(self, in_ch=1024, mid_ch=512, out_ch=512):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, mid_ch)  # 14x14
        self.pool = nn.MaxPool2d(2)           # downsample: 14->7
        self.enc2 = ConvBlock(mid_ch, mid_ch) # 7x7
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec  = ConvBlock(mid_ch + mid_ch, out_ch)  # concat skip connection
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)
        x  = torch.cat([x4, x1], dim=1)
        return self.dec(x)  # (B,512,14,14)


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
        # Batch processing: reshape to (B*N, C), apply batched GCN, reshape back
        # Note: For efficiency with fixed graph, we process samples independently
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


# -------------------------
# Main model with Image + ST branches
# -------------------------
class MultiBranchSpatialPredictorV2(nn.Module):
    """
    Returns:
      img_pred: (B, bin, pred_dim)  # prediction stream for regression tasks
      st_pred:  (B, bin, pred_dim) or None
      img_ctr:  (B, bin, ctr_dim)   # contrastive embedding stream
      st_ctr:   (B, bin, ctr_dim) or None
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
        # Backbone: DenseNet121 (feature extractor only)
        try:
            densenet = models.densenet121(weights=getattr(models.DenseNet121_Weights, densenet_weights))
        except Exception:
            densenet = models.densenet121(weights=densenet_weights)
        self.backbone = densenet.features  # output: (B,1024,14,14)

        self.unet = MiniUNet(1024, 512, 512)

        # Match target bin_num (grid size of output patches)
        self.bin_num = bin_num
        if bin_num == 1024:  # 32x32 (upsample from 14x14)
            self.resize_to_grid = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
            self.hw = (32, 32)
        elif bin_num == 196:   # 14x14
            self.resize_to_grid = nn.Identity(); self.hw = (14, 14)
        elif bin_num == 49:  # 7x7
            self.resize_to_grid = nn.AdaptiveAvgPool2d((7, 7)); self.hw = (7, 7)
        elif bin_num == 1:   # 1x1
            self.resize_to_grid = nn.AdaptiveAvgPool2d((1, 1)); self.hw = (1, 1)
        else:
            raise ValueError("Unsupported bin_num. Only support 1, 49, 196, 1024.")

        # Optional GNN for spatial context (Phase 2b)
        self.use_gnn = use_gnn
        if use_gnn:
            grid_size = self.hw[0]  # Assumes square grid
            self.spatial_gnn = SpatialGCN(
                in_dim=512, hidden_dim=512, out_dim=512,
                grid_size=grid_size, num_layers=gnn_layers, dropout=gnn_dropout
            )

        # Image branch: prediction head
        self.image_pred_head = nn.Sequential(
            nn.Conv2d(512, 512, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, pred_dim, 1)
        )
        # Image branch: contrastive head
        self.image_ctr_head = nn.Sequential(
            nn.Conv2d(512, 512, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, ctr_dim, 1)
        )

        # ST branch (shared layers + dual heads)
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
        # Initialize new layers; DenseNet already has pretrained weights
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
            nn.init.xavier_uniform_(self.st_pred_head.weight); nn.init.zeros_(self.st_pred_head.bias)
            nn.init.xavier_uniform_(self.st_ctr_head.weight);  nn.init.zeros_(self.st_ctr_head.bias)

    @staticmethod
    def _to_seq(x_2d: torch.Tensor) -> torch.Tensor:
        """Convert feature map (B, C, H, W) -> patch sequence (B, H*W, C)."""
        B, C, H, W = x_2d.shape
        return x_2d.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def forward(self, image: torch.Tensor, st: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        # Image stream
        feat = self.backbone(image)          # (B,1024,14,14)
        feat = self.unet(feat)               # (B,512,14,14)
        feat = self.resize_to_grid(feat)     # (B,512,H,W) where H*W=bin_num

        # Optional GNN for spatial context (Phase 2b)
        if self.use_gnn:
            feat = self.spatial_gnn(feat)    # (B,512,H,W) enhanced with neighbor context

        # Prediction output (for regression/supervised tasks)
        img_pred_map = self.image_pred_head(feat)   # (B,pred_dim,H,W)
        img_pred = self._to_seq(img_pred_map)       # (B,bin,pred_dim)

        # Contrastive embedding output
        img_ctr_map = self.image_ctr_head(feat)     # (B,ctr_dim,H,W)
        img_ctr = self._to_seq(img_ctr_map)         # (B,bin,ctr_dim)

        st_pred = st_ctr = None
        if self.has_st and st is not None:
            # st: (B,bin,st_in_dim)
            shared = self.st_shared(st)             # (B,bin,hidden)
            st_pred = self.st_pred_head(shared)     # (B,bin,pred_dim)
            st_ctr  = self.st_ctr_head(shared)      # (B,bin,ctr_dim)

        return img_pred, st_pred, img_ctr, st_ctr


# -------------------------
# Symmetric InfoNCE loss
# -------------------------
class ImageSTContrastive(nn.Module):
    """
    Contrastive loss between image and ST embeddings (InfoNCE).
    Modes:
      - patch_agg='mean': aggregate patch embeddings -> (B,D), do sample-level contrast
      - patch_agg='patch': contrast at patch level (requires patch order alignment)
    """
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
            # Sample-level: mean over patches
            img_vec = self._norm(img_ctr.mean(dim=1))  # (B,D)
            st_vec  = self._norm(st_ctr.mean(dim=1))   # (B,D)
            logits_i2s = (img_vec @ st_vec.t()) / self.tau
            logits_s2i = (st_vec @ img_vec.t()) / self.tau
            targets = torch.arange(img_vec.size(0), device=img_ctr.device)
            return 0.5 * (F.cross_entropy(logits_i2s, targets) +
                          F.cross_entropy(logits_s2i, targets))
        else:
            # Patch-level: flatten all patches across batch
            B, P, D = img_ctr.shape
            img_flat = self._norm(img_ctr.reshape(B * P, D))
            st_flat  = self._norm(st_ctr.reshape(B * P, D))
            logits_i2s = (img_flat @ st_flat.t()) / self.tau
            logits_s2i = (st_flat @ img_flat.t()) / self.tau
            targets = torch.arange(B * P, device=img_ctr.device)
            return 0.5 * (F.cross_entropy(logits_i2s, targets) +
                          F.cross_entropy(logits_s2i, targets))
