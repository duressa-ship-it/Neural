"""
NeuralForge — Temporal Convolutional Network (Bai et al. 2018)

Stacks of dilated causal 1D convolutions with residual connections. Often
the right default for time-series classification or sequence-to-one
forecasting — long effective receptive field, no recurrent unroll, works
well on irregular sampling.

Input  shape: `(batch, timesteps, channels)` — same convention as the RNN.
Output shape: `(batch, output_size)` after pooling, or `(batch, T, output_size)` with output_mode='all'.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_platform.core.config import ModelConfig
from neural_platform.core.registry import registry, MODEL
from neural_platform.models.base import BaseModel


class _Chomp1d(nn.Module):
    """Remove right padding to make the convolution causal."""
    def __init__(self, chomp_size: int): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size else x


class _TemporalBlock(nn.Module):
    """Two dilated causal convs + residual, à la Bai 2018."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel, padding=padding, dilation=dilation)),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


@registry.register(MODEL, "tcn")
class TCN(BaseModel):
    """Temporal Convolutional Network with residual blocks and exponentially growing dilations."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        arch = config.tcn

        layers = []
        in_ch = arch.input_size
        for i, out_ch in enumerate(arch.channels):
            layers.append(_TemporalBlock(
                in_ch, out_ch, kernel=arch.kernel_size,
                dilation=2 ** i, dropout=arch.dropout,
            ))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

        # Resolve output_mode (with backwards-compat for the older `pooling` field)
        self.output_mode = arch.output_mode or arch.pooling
        self.output_layer = nn.Linear(in_ch, arch.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — TCN wants (B, C, T)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # tolerate single-channel sequences
        x = x.transpose(1, 2)
        feats = self.network(x)                         # (B, C', T)

        if self.output_mode == "last":
            pooled = feats[:, :, -1]
        elif self.output_mode == "mean":
            pooled = feats.mean(dim=2)
        elif self.output_mode == "max":
            pooled, _ = feats.max(dim=2)
        elif self.output_mode == "all":
            return self.output_layer(feats.transpose(1, 2))   # (B, T, output)
        else:
            pooled = feats[:, :, -1]
        return self.output_layer(pooled)
