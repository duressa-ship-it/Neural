"""
NeuralForge — Audio Models

Two execution paths share one config:
  * `use_spectrogram=False` — 1D convolutions directly over the raw waveform.
  * `use_spectrogram=True`  — short-time Fourier transform → mel spectrogram
                               → 2D convolutions (the standard production
                               approach for speech / music classification).

Optional fine-tuning wrapper around HuggingFace audio models (wav2vec2,
HuBERT, etc.) when `pretrained` is set.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from neural_platform.core.config import ModelConfig
from neural_platform.core.registry import registry, MODEL
from neural_platform.models.base import BaseModel, make_fc_head


class _MelSpectrogram(nn.Module):
    """Lazy-import torchaudio mel-spectrogram module so torch is enough to import the model."""
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int):
        super().__init__()
        try:
            import torchaudio.transforms as TT
        except ImportError as exc:
            raise ImportError(
                "Audio models with use_spectrogram=true require torchaudio. "
                "Install with: pip install torchaudio"
            ) from exc
        self.melspec = TT.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        )
        self.amplitude_to_db = TT.AmplitudeToDB()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, samples) → mel: (batch, n_mels, time_frames)
        return self.amplitude_to_db(self.melspec(waveform))


class _HuggingFaceAudioWrapper(BaseModel):
    """Wraps a HuggingFace audio model (wav2vec2, hubert, …) with a classification head."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "audio_cnn.pretrained requires the `transformers` package."
            ) from exc
        arch = config.audio_cnn
        self.encoder = AutoModel.from_pretrained(arch.pretrained)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, arch.output_size)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(waveform)
        # Most audio backbones return last_hidden_state of shape (B, T, D)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.head(pooled)


@registry.register(MODEL, "audio_cnn")
class AudioCNN(BaseModel):
    """
    Audio classifier — either:
      * Spectrogram path (default): waveform → mel-spec → 2D conv stack → FC head.
      * Waveform path: 1D conv stack directly over raw audio samples.

    Input shape: `(batch, samples)` Tensor of floats normalized to [-1, 1].
    Output: `(batch, output_size)` logits.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        arch = config.audio_cnn

        # Pretrained wrapper short-circuit
        if arch.pretrained:
            self._impl = _HuggingFaceAudioWrapper(config)
            self._is_pretrained = True
            return
        self._is_pretrained = False
        self._use_spectrogram = arch.use_spectrogram

        if arch.use_spectrogram:
            self.frontend = _MelSpectrogram(
                sample_rate=arch.sample_rate,
                n_fft=arch.n_fft,
                hop_length=arch.hop_length,
                n_mels=arch.n_mels,
            )
            self.conv_stack = self._build_2d_stack(arch.conv_channels)
        else:
            self.frontend = None
            self.conv_stack = self._build_1d_stack(arch.conv_channels)

        # Compute the flattened size with a dummy forward pass at init time.
        with torch.no_grad():
            dummy_samples = int(arch.sample_rate * arch.duration_secs)
            dummy = torch.zeros(1, dummy_samples)
            feat = self._extract_features(dummy)
            flat_size = feat.view(1, -1).size(1)

        self.head, last_fc = make_fc_head(arch.fc_layers, flat_size)
        self.output_layer = nn.Linear(last_fc, arch.output_size)

    @staticmethod
    def _build_1d_stack(channels: List[int]) -> nn.Sequential:
        layers, prev = [], 1
        for c in channels:
            layers.extend([
                nn.Conv1d(prev, c, kernel_size=9, stride=4, padding=4),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            prev = c
        return nn.Sequential(*layers)

    @staticmethod
    def _build_2d_stack(channels: List[int]) -> nn.Sequential:
        layers, prev = [], 1
        for c in channels:
            layers.extend([
                nn.Conv2d(prev, c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            prev = c
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        return nn.Sequential(*layers)

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run the convolutional frontend, return shape (B, F)."""
        if self._use_spectrogram:
            spec = self.frontend(waveform).unsqueeze(1)        # (B, 1, n_mels, T)
            return self.conv_stack(spec)
        return self.conv_stack(waveform.unsqueeze(1))           # (B, 1, samples)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._is_pretrained:
            return self._impl(waveform)
        feats = self._extract_features(waveform)
        feats = feats.flatten(start_dim=1)
        feats = self.head(feats)
        return self.output_layer(feats)
