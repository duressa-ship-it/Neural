"""
NeuralForge — TabularNet

Production-grade tabular learner. Differentiates from MLP by handling:

* **Categorical features** via learned embeddings — one nn.Embedding per
  feature, sized via the standard `min(50, ceil(cardinality**0.56))` rule
  unless the user pins `embed_dim`.
* **Mixed dtypes** in a single batch: numeric features are concatenated to
  the flattened embeddings before the dense stack.
* **Missing values**: handled at the dataset/preprocessor level; the model
  expects already-imputed inputs.

Input shape: dict
    {
      "numeric":      Tensor (B, N_num)         — N_num = len(numeric_features)
      "categorical":  LongTensor (B, N_cat)     — N_cat = len(categorical_features)
                                                   columns ordered as in config
    }
or, when there are no categorical features, just a Tensor (B, N_num).

Output shape: (B, output_size).
"""

from __future__ import annotations

import math
from typing import Dict, List, Union

import torch
import torch.nn as nn

from neural_platform.core.config import ModelConfig
from neural_platform.core.registry import registry, MODEL
from neural_platform.models.base import BaseModel, make_activation, make_fc_head


def _suggest_embed_dim(cardinality: int) -> int:
    """fastai/Howard's rule of thumb."""
    return min(50, max(2, int(math.ceil(cardinality ** 0.56))))


@registry.register(MODEL, "tabular")
class TabularNet(BaseModel):
    """Tabular network with categorical embeddings + numeric concat + dense stack."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        arch = config.tabular

        # Build embeddings
        self.embeddings = nn.ModuleList()
        self.cat_names: List[str] = []
        embed_total = 0
        for spec in arch.categorical_features:
            name = spec["name"]
            cardinality = int(spec["cardinality"])
            embed_dim = int(spec.get("embed_dim") or _suggest_embed_dim(cardinality))
            self.embeddings.append(nn.Embedding(cardinality, embed_dim))
            self.cat_names.append(name)
            embed_total += embed_dim

        n_numeric = len(arch.numeric_features)
        in_dim = embed_total + n_numeric
        if in_dim == 0:
            raise ValueError(
                "TabularNet has no input features — set numeric_features and/or "
                "categorical_features."
            )

        # Optional batch-norm over numeric features (helps when scales differ)
        self.numeric_bn = nn.BatchNorm1d(n_numeric) if n_numeric > 0 else None

        self.hidden, last = make_fc_head(arch.hidden_layers, in_dim)
        self.output_layer = nn.Linear(last, arch.output_size)
        self.output_activation = make_activation(arch.output_activation)

    def forward(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(x, dict):
            numeric = x.get("numeric")
            categorical = x.get("categorical")
        else:
            numeric, categorical = x, None

        parts = []
        if self.embeddings and categorical is not None:
            for i, emb in enumerate(self.embeddings):
                parts.append(emb(categorical[:, i]))
        if numeric is not None and numeric.shape[-1] > 0:
            parts.append(self.numeric_bn(numeric) if self.numeric_bn is not None else numeric)

        feats = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        feats = self.hidden(feats)
        return self.output_activation(self.output_layer(feats))
