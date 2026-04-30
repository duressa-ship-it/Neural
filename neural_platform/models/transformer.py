"""
NeuralForge — Transformer Model
Encoder-only (BERT-style) or encoder-decoder architecture,
plus fine-tuning support via HuggingFace pretrained models.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from neural_platform.core.config import ModelConfig
from neural_platform.core.registry import registry, MODEL
from neural_platform.models.base import BaseModel, make_fc_head


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


class _HuggingFaceWrapper(BaseModel):
    """Thin wrapper around a HuggingFace pretrained model for fine-tuning."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from transformers import AutoModel

        arch = config.transformer
        self.encoder = AutoModel.from_pretrained(arch.use_pretrained)
        hf_hidden = self.encoder.config.hidden_size
        self.output_mode = arch.output_mode
        # Projection head
        self.head = nn.Linear(hf_hidden, arch.output_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                token_type_ids: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        # token_type_ids is only meaningful for some HF models (BERT-style);
        # pass it through when present, ignore for others.
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        if self.output_mode == "cls":
            pooled = hidden[:, 0]
        elif self.output_mode == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = hidden.mean(1)
        else:
            pooled = hidden[:, 0]
        return self.head(pooled)


@registry.register(MODEL, "transformer")
class TransformerModel(BaseModel):
    """
    Transformer model (encoder-only or encoder-decoder).

    When use_pretrained is set, wraps a HuggingFace model for fine-tuning.
    Otherwise, builds a from-scratch transformer with configurable depth,
    width, attention heads, and positional encoding.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        arch = config.transformer

        # Delegate to HuggingFace wrapper if pretrained model specified
        if arch.use_pretrained:
            self._impl = _HuggingFaceWrapper(config)
            self._is_hf = True
            return

        self._is_hf = False

        # Token embedding
        self.token_embedding = nn.Embedding(arch.vocab_size, arch.d_model)
        self.embed_scale = math.sqrt(arch.d_model)

        # Positional encoding
        if arch.positional_encoding == "learned":
            self.pos_encoding = LearnedPositionalEncoding(arch.d_model, arch.max_seq_len, arch.dropout)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(arch.d_model, arch.max_seq_len, arch.dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=arch.d_model,
            nhead=arch.num_heads,
            dim_feedforward=arch.d_ff,
            dropout=arch.dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=arch.num_encoder_layers
        )

        # Optional decoder
        if arch.num_decoder_layers > 0:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=arch.d_model,
                nhead=arch.num_heads,
                dim_feedforward=arch.d_ff,
                dropout=arch.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=arch.num_decoder_layers
            )
        else:
            self.decoder = None

        self.norm = nn.LayerNorm(arch.d_model)
        self.output_mode = arch.output_mode
        self.output_layer = nn.Linear(arch.d_model, arch.output_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        **extra_tokenizer_kwargs,        # noqa: ARG002  — accept & ignore
    ) -> torch.Tensor:
        """
        Args:
            input_ids:          (batch, seq_len) token indices
            attention_mask:     (batch, seq_len) 1=real token, 0=pad (optional)
            decoder_input_ids:  (batch, tgt_len) for encoder-decoder mode
            **extra_tokenizer_kwargs:
                Silently absorbed. HuggingFace tokenizers emit additional
                fields like ``token_type_ids`` for BERT-style models that
                this from-scratch encoder doesn't use; we tolerate them so
                the dataset loader doesn't have to filter.
        Returns:
            Tensor of shape (batch, output_size)
        """
        if self._is_hf:
            return self._impl(input_ids, attention_mask)

        # Convert padding mask: True = ignore position
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0  # (batch, seq_len)

        x = self.token_embedding(input_ids) * self.embed_scale
        x = self.pos_encoding(x)
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        memory = self.norm(memory)

        if self.decoder is not None and decoder_input_ids is not None:
            tgt = self.token_embedding(decoder_input_ids) * self.embed_scale
            tgt = self.pos_encoding(tgt)
            out = self.decoder(tgt, memory)
            out = self.norm(out)
            return self.output_layer(out)

        # Pool encoder output
        if self.output_mode == "cls":
            pooled = memory[:, 0]
        elif self.output_mode == "mean":
            if src_key_padding_mask is not None:
                mask = (~src_key_padding_mask).float().unsqueeze(-1)
                pooled = (memory * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = memory.mean(1)
        else:  # "all" — return all tokens (caller handles)
            return self.output_layer(memory)

        return self.output_layer(pooled)
