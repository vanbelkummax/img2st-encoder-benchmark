#!/usr/bin/env python3
"""
PathoDuet model compatibility layer for modern timm versions.

This module provides a compatible implementation of PathoDuet's
VisionTransformerMoCo that works with timm>=1.0.
"""

import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer

# Handle different timm versions
try:
    from timm.layers import PatchEmbed
    from timm.layers.helpers import to_2tuple
except ImportError:
    from timm.models.layers import PatchEmbed
    from timm.models.layers.helpers import to_2tuple


class VisionTransformerMoCo(VisionTransformer):
    """PathoDuet's VisionTransformerMoCo adapted for modern timm."""

    def __init__(self, pretext_token=True, stop_grad_conv1=False, **kwargs):
        # Remove incompatible kwargs for newer timm
        kwargs.pop('num_classes', None)

        super().__init__(**kwargs)

        # inserting a new token
        self.num_prefix_tokens = getattr(self, 'num_prefix_tokens', 1) + (1 if pretext_token else 0)
        self.pretext_token = nn.Parameter(torch.ones(1, 1, self.embed_dim)) if pretext_token else None

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # Weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.cls_token, std=1e-6)
        if self.pretext_token is not None:
            nn.init.normal_(self.pretext_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            if self.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                if self.patch_embed.proj.bias is not None:
                    self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')

        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w),
                            torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # Tokens: [pretext][cls]
        n_tokens = 2 if self.pretext_token is not None else 1
        pe_token = torch.zeros([1, n_tokens, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def _pos_embed(self, x):
        # Add pretext token and cls token
        B = x.shape[0]

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        if self.pretext_token is not None:
            x = torch.cat((self.pretext_token.expand(B, -1, -1), x), dim=1)

        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, ref=None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits=False):
        # Global average pooling (excluding cls and pretext tokens)
        if hasattr(self, 'global_pool') and self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        else:
            x = x[:, 1]  # CLS token

        if hasattr(self, 'fc_norm'):
            x = self.fc_norm(x)

        return x if pre_logits else self.head(x)

    def forward(self, x, ref=None):
        x_out = self.forward_features(x, ref)
        x = self.forward_head(x_out)
        return x_out, x


def vit_base(**kwargs):
    """Create ViT-Base model compatible with PathoDuet checkpoints."""
    model = VisionTransformerMoCo(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def load_pathoduet_checkpoint(model, checkpoint_path, strict=False):
    """Load PathoDuet checkpoint with compatibility handling."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Remove head weights (different sizes)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}

    # Load with strict=False to handle any remaining mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Missing keys: {len(missing)} (expected for head layer)")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    return model


if __name__ == "__main__":
    # Test the model
    model = vit_base(pretext_token=True, global_pool='avg')
    print(f"Model created: embed_dim={model.embed_dim}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out, _ = model(x)
        feat = model.forward_head(out, pre_logits=True)
    print(f"Output shape: {feat.shape}")
    print("Model test passed!")
