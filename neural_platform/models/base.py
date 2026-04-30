"""
NeuralForge Base Model
All framework-agnostic model interface and PyTorch base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from neural_platform.core.config import ModelConfig


class BaseModel(nn.Module, ABC):
    """
    Base class for all NeuralForge PyTorch models.
    Wraps nn.Module and adds save/load, parameter counting, and config binding.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_config = config

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass — must be implemented by subclasses."""
        ...

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience inference method (eval mode, no grad)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable model summary dict."""
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        return {
            "name": self.model_config.name,
            "type": self.model_config.type.value,
            "framework": self.model_config.framework.value,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
        }

    def save(self, path: str | Path, extra: Optional[Dict] = None) -> None:
        """Save model weights and config to a .pt checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "model_config": self.model_config.model_dump(),
            **(extra or {}),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BaseModel":
        """Load a model from a checkpoint file (any registered model type)."""
        from neural_platform.frameworks.pytorch_adapter import _ensure_models_registered
        from neural_platform.core.registry import registry, MODEL
        _ensure_models_registered()
        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        config = ModelConfig.model_validate(payload["model_config"])
        model_cls = registry.get(MODEL, config.type.value)
        model = model_cls(config)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model


def make_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "mish": nn.Mish(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(dim=-1),
        "leaky_relu": nn.LeakyReLU(0.01),
        "elu": nn.ELU(),
        "none": nn.Identity(),
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(activations.keys())}")
    return activations[key]


def make_fc_head(layers_config, in_features: int) -> Tuple[nn.Sequential, int]:
    """
    Build a fully-connected head from a list of LayerConfig objects.
    Returns (nn.Sequential, out_features).
    """
    from neural_platform.core.config import LayerConfig

    layers = []
    current = in_features
    for lc in layers_config:
        layers.append(nn.Linear(current, lc.size))
        if lc.batch_norm:
            layers.append(nn.BatchNorm1d(lc.size))
        layers.append(make_activation(lc.activation))
        if lc.dropout > 0:
            layers.append(nn.Dropout(lc.dropout))
        current = lc.size
    return nn.Sequential(*layers), current
