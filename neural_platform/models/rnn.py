"""
NeuralForge — Recurrent Neural Network Model
Supports LSTM, GRU, and vanilla RNN with bidirectional support.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_platform.core.config import ModelConfig, RNNVariant
from neural_platform.core.registry import registry, MODEL
from neural_platform.models.base import BaseModel, make_fc_head


_CELL_MAP = {
    RNNVariant.LSTM: nn.LSTM,
    RNNVariant.GRU: nn.GRU,
    RNNVariant.VANILLA: nn.RNN,
}


@registry.register(MODEL, "rnn")
class RNN(BaseModel):
    """
    Recurrent Neural Network (LSTM / GRU / vanilla RNN).

    Supports:
      - All three RNN cell types
      - Bidirectional encoding
      - Multiple stacked layers with dropout
      - Output from last timestep, all timesteps, or mean pooling
      - Optional fully-connected head after RNN
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        arch = config.rnn

        rnn_cls = _CELL_MAP[arch.variant]
        self.rnn = rnn_cls(
            input_size=arch.input_size,
            hidden_size=arch.hidden_size,
            num_layers=arch.num_layers,
            batch_first=True,
            bidirectional=arch.bidirectional,
            dropout=arch.dropout if arch.num_layers > 1 else 0.0,
        )

        direction_mult = 2 if arch.bidirectional else 1
        rnn_out_dim = arch.hidden_size * direction_mult

        self.output_mode = arch.output_mode
        self.fc_head, out_features = make_fc_head(arch.fc_layers, rnn_out_dim)
        self.output_layer = nn.Linear(out_features, arch.output_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            lengths: Optional tensor of actual sequence lengths (batch,) for packing.
        Returns:
            Tensor of shape (batch, output_size)
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, hidden = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, hidden = self.rnn(x)

        # Pool over time dimension
        if self.output_mode == "last":
            if isinstance(hidden, tuple):  # LSTM returns (h_n, c_n)
                features = hidden[0][-1]  # last layer, all batch
            else:
                features = hidden[-1]
        elif self.output_mode == "mean":
            features = out.mean(dim=1)
        else:  # "all"
            features = out.reshape(out.size(0), -1)

        if self.fc_head:
            features = self.fc_head(features)
        return self.output_layer(features)
