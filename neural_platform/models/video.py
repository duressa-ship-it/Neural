"""
NeuralForge — Video3DCNN (experimental)

A simple 3D CNN over `(batch, channels, time, height, width)` clips.
Useful for frame-stacked classification tasks; not a substitute for
purpose-built video architectures (I3D, SlowFast, MViT) — those should
land via `pretrained:` later.

This is the *recognized but minimal* end of the modality spectrum: the
data loader resamples clips to a fixed `num_frames`, the model accepts
them as a 5-D tensor, and the validator marks `model.type=video_cnn` as
experimental in the Builder.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from neural_platform.core.config import ModelConfig
from neural_platform.core.registry import registry, MODEL
from neural_platform.models.base import BaseModel, make_fc_head


@registry.register(MODEL, "video_cnn")
class Video3DCNN(BaseModel):
    """
    3D-conv classifier for short video clips.

    Input shape:  `(batch, channels, time, height, width)`
    Output shape: `(batch, output_size)`
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        arch = config.video_cnn

        layers: List[nn.Module] = []
        prev = arch.input_channels
        for spec in arch.conv_layers:
            out_c = int(spec["out_channels"])
            k     = int(spec.get("kernel_size", 3))
            stride = int(spec.get("stride", 1))
            pad   = int(spec.get("padding", k // 2))
            layers.append(nn.Conv3d(prev, out_c, kernel_size=k, stride=stride, padding=pad))
            layers.append(nn.BatchNorm3d(out_c))
            layers.append(nn.ReLU(inplace=True))
            if spec.get("pool", True):
                layers.append(nn.MaxPool3d(kernel_size=2))
            prev = out_c
        layers.append(nn.AdaptiveAvgPool3d((1, 4, 4)))
        self.conv_stack = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, arch.input_channels, arch.num_frames,
                                 arch.input_height, arch.input_width)
            feat = self.conv_stack(dummy)
            flat = feat.view(1, -1).size(1)

        self.head, last = make_fc_head(arch.fc_layers, flat)
        self.output_layer = nn.Linear(last, arch.output_size)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dim() == 4:
            # (B, T, H, W) → assume single channel
            clip = clip.unsqueeze(1)
        feats = self.conv_stack(clip).flatten(start_dim=1)
        feats = self.head(feats)
        return self.output_layer(feats)
