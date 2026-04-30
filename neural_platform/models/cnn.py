"""
NeuralForge — Convolutional Neural Network Model
Supports custom conv stacks or pretrained backbone transfer learning.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from neural_platform.core.config import ModelConfig
from neural_platform.core.registry import registry, MODEL
from neural_platform.models.base import BaseModel, make_activation, make_fc_head


def _build_conv_block(in_channels: int, cfg: dict) -> tuple[nn.Sequential, int]:
    """Build a single conv block from a config dict."""
    out_channels = cfg.get("out_channels", 32)
    kernel_size = cfg.get("kernel_size", 3)
    stride = cfg.get("stride", 1)
    padding = cfg.get("padding", 1)
    activation = cfg.get("activation", "relu")
    batch_norm = cfg.get("batch_norm", True)
    pool = cfg.get("pool", False)
    pool_size = cfg.get("pool_size", 2)
    dropout = cfg.get("dropout", 0.0)

    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(make_activation(activation))
    if pool:
        layers.append(nn.MaxPool2d(pool_size))
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))

    return nn.Sequential(*layers), out_channels


def _compute_flat_size(in_channels: int, h: int, w: int, conv_layers: list) -> int:
    """Compute flattened feature map size after all conv blocks."""
    ch, cur_h, cur_w = in_channels, h, w
    for cfg in conv_layers:
        stride = cfg.get("stride", 1)
        kernel_size = cfg.get("kernel_size", 3)
        padding = cfg.get("padding", 1)
        pool = cfg.get("pool", False)
        pool_size = cfg.get("pool_size", 2)

        cur_h = math.floor((cur_h + 2 * padding - kernel_size) / stride + 1)
        cur_w = math.floor((cur_w + 2 * padding - kernel_size) / stride + 1)
        if pool:
            cur_h = math.floor(cur_h / pool_size)
            cur_w = math.floor(cur_w / pool_size)
        ch = cfg.get("out_channels", 32)

    return ch * cur_h * cur_w


@registry.register(MODEL, "cnn")
class CNN(BaseModel):
    """
    Convolutional Neural Network.

    Supports:
      - Custom conv stack defined in config
      - Pretrained backbone (resnet18, resnet50, vgg16, efficientnet_b0) with optional fine-tuning
      - Configurable FC head
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        arch = config.cnn

        if arch.backbone:
            self._build_from_backbone(arch)
        else:
            self._build_custom(arch)

    def _build_custom(self, arch):
        """Build from scratch using conv_layers config."""
        blocks = []
        in_ch = arch.input_channels
        for layer_cfg in arch.conv_layers:
            block, in_ch = _build_conv_block(in_ch, layer_cfg)
            blocks.append(block)
        self.conv_backbone = nn.Sequential(*blocks)

        flat_size = _compute_flat_size(
            arch.input_channels, arch.input_height, arch.input_width, arch.conv_layers
        )
        self.fc_head, out_features = make_fc_head(arch.fc_layers, flat_size)
        self.classifier = nn.Linear(out_features, arch.output_size)
        self._use_backbone = False

    def _build_from_backbone(self, arch):
        """Build using a pretrained torchvision backbone."""
        import torchvision.models as tv_models

        backbone_map = {
            "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.DEFAULT, 512),
            "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.DEFAULT, 2048),
            "resnet101": (tv_models.resnet101, tv_models.ResNet101_Weights.DEFAULT, 2048),
            "vgg16": (tv_models.vgg16, tv_models.VGG16_Weights.DEFAULT, 4096),
            "efficientnet_b0": (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.DEFAULT, 1280),
            "mobilenet_v3_small": (tv_models.mobilenet_v3_small, tv_models.MobileNet_V3_Small_Weights.DEFAULT, 576),
        }
        name = arch.backbone.lower()
        if name not in backbone_map:
            raise ValueError(f"Unknown backbone '{arch.backbone}'. Available: {list(backbone_map.keys())}")

        factory, weights, feature_dim = backbone_map[name]
        backbone = factory(weights=weights if arch.pretrained else None)

        # Remove the classification head — keep feature extractor only
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier"):
            backbone.classifier = nn.Identity()

        if arch.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.conv_backbone = backbone
        self.fc_head, out_features = make_fc_head(arch.fc_layers, feature_dim)
        self.classifier = nn.Linear(out_features, arch.output_size)
        self._use_backbone = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images of shape (batch, channels, H, W)
        Returns:
            Logits of shape (batch, output_size)
        """
        features = self.conv_backbone(x)
        if not self._use_backbone:
            features = features.view(features.size(0), -1)
        if self.fc_head:
            features = self.fc_head(features)
        return self.classifier(features)
