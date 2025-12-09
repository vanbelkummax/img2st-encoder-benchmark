#!/usr/bin/env python3
"""
Generic encoder wrapper for H&E patch feature extraction.

Supports:
- DenseNet121 (torchvision, ImageNet pretrained)
- UNI (HuggingFace, MahmoodLab)
- Virchow/Virchow2 (HuggingFace, Paige AI)
- Prov-GigaPath (HuggingFace, Microsoft)
- DINOv2 (torch hub, fallback)

Usage:
    encoder = get_encoder('uni')
    features = encoder(batch_of_images)  # [B, embed_dim]
"""

import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any


def load_encoder_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load encoder configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'encoder_config.yaml'

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from file or environment."""
    # Check environment variable first
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if token:
        return token

    # Check token file
    token_path = os.path.expanduser('~/.huggingface_token')
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            return f.read().strip()

    # Check HuggingFace CLI cache
    hf_token_path = os.path.expanduser('~/.cache/huggingface/token')
    if os.path.exists(hf_token_path):
        with open(hf_token_path, 'r') as f:
            return f.read().strip()

    return None


class DenseNetEncoder(nn.Module):
    """DenseNet121 encoder with ImageNet pretrained weights."""

    def __init__(self, weights: str = 'IMAGENET1K_V1'):
        super().__init__()
        from torchvision import models

        self.backbone = models.densenet121(weights=weights)
        self.features = self.backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_dim = 1024

        # Freeze
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features)
        return pooled.view(pooled.size(0), -1)  # [B, 1024]


class UNIEncoder(nn.Module):
    """UNI encoder from MahmoodLab (ViT-L/16)."""

    def __init__(self, model_name: str = 'MahmoodLab/UNI', token: Optional[str] = None):
        super().__init__()
        import timm

        if token is None:
            token = get_hf_token()

        # Login to HuggingFace if token available
        if token:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)

        # Load UNI model
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True
        )
        self.model.eval()
        self.embed_dim = 1024

        # Freeze
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # [B, 1024]


class VirchowEncoder(nn.Module):
    """Virchow encoder from Paige AI (ViT-H/14)."""

    def __init__(self, model_name: str = 'paige-ai/Virchow', token: Optional[str] = None):
        super().__init__()
        import timm

        if token is None:
            token = get_hf_token()

        if token:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)

        # Determine model variant
        if 'Virchow2' in model_name:
            hf_name = "hf-hub:paige-ai/Virchow2"
            # Virchow2 uses SwiGLU MLP layers - must specify at creation
            from timm.layers import SwiGLUPacked
            self.model = timm.create_model(
                hf_name,
                pretrained=True,
                dynamic_img_size=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU
            )
        else:
            hf_name = "hf-hub:paige-ai/Virchow"
            self.model = timm.create_model(
                hf_name,
                pretrained=True,
                dynamic_img_size=True
            )
        self.model.eval()
        self.is_virchow2 = 'Virchow2' in model_name
        # Virchow2: 2560d (class + mean patch), Virchow: 1280d
        self.embed_dim = 2560 if self.is_virchow2 else 1280

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_virchow2:
            # Virchow2 outputs [B, 261, 1280] - 1 class + 4 register + 256 patch tokens
            output = self.model.forward_features(x)  # [B, 261, 1280]
            class_token = output[:, 0]  # [B, 1280]
            patch_tokens = output[:, 5:]  # [B, 256, 1280] - skip register tokens
            embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)  # [B, 2560]
            return embedding
        else:
            return self.model(x)  # [B, 1280]


class GigaPathEncoder(nn.Module):
    """Prov-GigaPath encoder from Microsoft (ViT-G)."""

    def __init__(self, model_name: str = 'prov-gigapath/prov-gigapath', token: Optional[str] = None):
        super().__init__()
        import timm

        if token is None:
            token = get_hf_token()

        if token:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)

        self.model = timm.create_model(
            "hf-hub:prov-gigapath/prov-gigapath",
            pretrained=True
        )
        self.model.eval()
        self.embed_dim = 1536

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # [B, 1536]


class DINOv2Encoder(nn.Module):
    """DINOv2 encoder from Meta (fallback, no auth required)."""

    def __init__(self, model_name: str = 'dinov2_vitg14'):
        super().__init__()

        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()

        # Embed dim depends on model size
        if 'vitg' in model_name:
            self.embed_dim = 1536
        elif 'vitl' in model_name:
            self.embed_dim = 1024
        elif 'vitb' in model_name:
            self.embed_dim = 768
        else:
            self.embed_dim = 384  # vits

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EncoderWrapper(nn.Module):
    """
    Unified wrapper for all encoders.

    Handles:
    - Loading correct encoder class
    - Normalization
    - Device management
    """

    def __init__(self, encoder_name: str, config: Optional[Dict] = None):
        super().__init__()

        if config is None:
            full_config = load_encoder_config()
            config = full_config['encoders'].get(encoder_name)
            if config is None:
                raise ValueError(f"Unknown encoder: {encoder_name}. "
                               f"Available: {list(full_config['encoders'].keys())}")

        self.config = config
        self.encoder_name = encoder_name

        # Load the actual encoder
        self.encoder = self._load_encoder()
        self.embed_dim = self.encoder.embed_dim

        # Store normalization params
        self.register_buffer('mean', torch.tensor(config['normalize']['mean']).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(config['normalize']['std']).view(1, 3, 1, 1))

    def _load_encoder(self) -> nn.Module:
        """Load the appropriate encoder based on config."""
        source = self.config['source']
        model_name = self.config.get('model_name', '')

        if source == 'torchvision':
            if 'densenet' in model_name.lower():
                return DenseNetEncoder(weights=self.config.get('weights', 'IMAGENET1K_V1'))
            else:
                raise ValueError(f"Unsupported torchvision model: {model_name}")

        elif source == 'huggingface':
            if 'UNI' in model_name:
                return UNIEncoder(model_name=model_name)
            elif 'Virchow' in model_name:
                return VirchowEncoder(model_name=model_name)
            elif 'gigapath' in model_name.lower():
                return GigaPathEncoder(model_name=model_name)
            else:
                raise ValueError(f"Unsupported HuggingFace model: {model_name}")

        elif source == 'torch_hub':
            if 'dinov2' in model_name.lower():
                return DINOv2Encoder(model_name=model_name)
            else:
                raise ValueError(f"Unsupported torch hub model: {model_name}")

        else:
            raise ValueError(f"Unsupported source: {source}")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet-style normalization."""
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input images [B, 3, H, W], values in [0, 1]
            normalize: Whether to apply normalization (default True)

        Returns:
            Features [B, embed_dim]
        """
        if normalize:
            x = self.normalize(x)
        return self.encoder(x)


def get_encoder(encoder_name: str, device: Optional[str] = None) -> EncoderWrapper:
    """
    Get an encoder by name.

    Args:
        encoder_name: Name from encoder_config.yaml (e.g., 'uni', 'densenet_imagenet')
        device: Device to load model on (default: auto-detect)

    Returns:
        EncoderWrapper ready for feature extraction
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = EncoderWrapper(encoder_name)
    encoder = encoder.to(device)
    encoder.eval()

    return encoder


def list_encoders() -> Dict[str, Dict]:
    """List all available encoders and their configs."""
    config = load_encoder_config()
    return config['encoders']


# Quick test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='densenet_imagenet',
                       help='Encoder to test')
    parser.add_argument('--list', action='store_true', help='List available encoders')
    args = parser.parse_args()

    if args.list:
        print("Available encoders:")
        for name, cfg in list_encoders().items():
            print(f"  {name}: {cfg['name']} (dim={cfg['embed_dim']})")
    else:
        print(f"Testing encoder: {args.encoder}")
        encoder = get_encoder(args.encoder)
        print(f"  Embed dim: {encoder.embed_dim}")
        print(f"  Device: {next(encoder.parameters()).device}")

        # Test forward pass
        dummy = torch.randn(2, 3, 224, 224).to(next(encoder.parameters()).device)
        with torch.no_grad():
            out = encoder(dummy)
        print(f"  Output shape: {out.shape}")
        print("  Success!")
