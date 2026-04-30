"""
NeuralForge — Feedforward / MLP Model
Fully-connected network configurable via MLPConfig.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_platform.core.config import ModelConfig
from neural_platform.core.registry import registry, MODEL
from neural_platform.models.base import BaseModel, make_activation, make_fc_head


@registry.register(MODEL, "mlp")
class MLP(BaseModel):
    """
    Multi-Layer Perceptron (Feedforward Network).

    Configurable via MLPConfig:
      - Arbitrary depth and width via hidden_layers
      - Per-layer activation, dropout, and batch norm
      - Custom output activation
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        arch = config.mlp

        # Build hidden layers
        self.hidden, out_features = make_fc_head(arch.hidden_layers, arch.input_size)

        # Output layer
        self.output_layer = nn.Linear(out_features, arch.output_size)
        self.output_activation = make_activation(arch.output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, input_size)
        Returns:
            Tensor of shape (batch, output_size)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.output_layer(x)
        return self.output_activation(x)
