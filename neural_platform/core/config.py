"""
NeuralForge Configuration System
All experiment parameters are defined here as Pydantic models.
Supports loading from YAML or JSON files.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Framework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


class ModelType(str, Enum):
    MLP = "mlp"
    CNN = "cnn"
    RNN = "rnn"
    TRANSFORMER = "transformer"
    AUDIO_CNN = "audio_cnn"
    TCN = "tcn"
    TABULAR = "tabular"
    VIDEO_CNN = "video_cnn"
    HF_PIPELINE = "hf_pipeline"


class RNNVariant(str, Enum):
    LSTM = "lstm"
    GRU = "gru"
    VANILLA = "rnn"


class Optimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


class Scheduler(str, Enum):
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    WARMUP_COSINE = "warmup_cosine"
    NONE = "none"


class LossFunction(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    BCE = "bce"
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    NLL = "nll"
    CTC = "ctc"


class DataSource(str, Enum):
    CSV = "csv"
    IMAGE_FOLDER = "image_folder"
    HUGGINGFACE = "huggingface"
    SYNTHETIC = "synthetic"
    NUMPY = "numpy"
    CUSTOM = "custom"


class Task(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEQUENCE_TO_SEQUENCE = "seq2seq"
    LANGUAGE_MODEL = "language_model"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    TEXT_CLASSIFICATION = "text_classification"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class LayerConfig(BaseModel):
    """Config for a single layer in an MLP or custom stack."""
    size: int = Field(..., description="Number of neurons / channels")
    activation: str = Field("relu", description="Activation: relu, gelu, tanh, sigmoid, silu, none")
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout probability after this layer")
    batch_norm: bool = Field(False, description="Apply batch normalization after activation")


class MLPConfig(BaseModel):
    """Feedforward / MLP-specific parameters."""
    input_size: int = Field(..., description="Input feature dimension")
    hidden_layers: List[LayerConfig] = Field(
        default_factory=lambda: [LayerConfig(size=128), LayerConfig(size=64)],
        description="List of hidden layer configs"
    )
    output_size: int = Field(..., description="Output dimension (num classes or regression dim)")
    output_activation: str = Field("none", description="Final activation: softmax, sigmoid, none")


class CNNConfig(BaseModel):
    """Convolutional network parameters."""
    input_channels: int = Field(3, description="Number of input channels (1=grayscale, 3=RGB)")
    input_height: int = Field(224, description="Input image height in pixels")
    input_width: int = Field(224, description="Input image width in pixels")
    conv_layers: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "pool": True},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "pool": True},
            {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "pool": True},
        ],
        description="Convolutional layer configurations"
    )
    fc_layers: List[LayerConfig] = Field(
        default_factory=lambda: [LayerConfig(size=512, dropout=0.5)],
        description="Fully-connected head layers"
    )
    output_size: int = Field(..., description="Number of output classes / regression dim")
    backbone: Optional[str] = Field(None, description="Optional pretrained backbone: resnet18, resnet50, vgg16, efficientnet_b0")
    pretrained: bool = Field(True, description="Load pretrained weights for backbone")
    freeze_backbone: bool = Field(False, description="Freeze backbone weights during training")


class RNNConfig(BaseModel):
    """Recurrent network parameters (LSTM / GRU / vanilla RNN)."""
    variant: RNNVariant = Field(RNNVariant.LSTM, description="RNN cell type: lstm, gru, rnn")
    input_size: int = Field(..., description="Input feature dimension per timestep")
    hidden_size: int = Field(256, description="Hidden state dimension")
    num_layers: int = Field(2, ge=1, le=16, description="Number of stacked RNN layers")
    bidirectional: bool = Field(False, description="Use bidirectional RNN")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout between RNN layers")
    output_size: int = Field(..., description="Output dimension")
    output_mode: Literal["last", "all", "mean"] = Field("last", description="Which timestep outputs to use")
    fc_layers: List[LayerConfig] = Field(
        default_factory=list,
        description="Optional FC head after RNN"
    )


class TransformerConfig(BaseModel):
    """Transformer / attention model parameters."""
    vocab_size: int = Field(30522, description="Vocabulary size (for embedding layer)")
    max_seq_len: int = Field(512, description="Maximum sequence length")
    d_model: int = Field(256, description="Model / embedding dimension")
    num_heads: int = Field(8, description="Number of attention heads (d_model must be divisible)")
    num_encoder_layers: int = Field(6, ge=1, description="Number of transformer encoder layers")
    num_decoder_layers: int = Field(0, ge=0, description="Number of decoder layers (0 = encoder-only)")
    d_ff: int = Field(1024, description="Feed-forward inner dimension")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout throughout transformer")
    output_size: int = Field(..., description="Output dimension (num classes, vocab size, etc.)")
    output_mode: Literal["cls", "mean", "all"] = Field("cls", description="How to pool encoder outputs")
    positional_encoding: Literal["sinusoidal", "learned"] = Field("sinusoidal", description="Positional encoding type")
    use_pretrained: Optional[str] = Field(None, description="HuggingFace model name to fine-tune, e.g. bert-base-uncased")

    @field_validator("num_heads")
    @classmethod
    def heads_must_divide_d_model(cls, v, info):
        d_model = info.data.get("d_model", 256)
        if d_model % v != 0:
            raise ValueError(f"num_heads ({v}) must divide d_model ({d_model}) evenly")
        return v


class AudioCNNConfig(BaseModel):
    """1D-CNN over raw waveform, with optional mel-spectrogram preprocessing."""
    sample_rate: int = Field(16000, description="Expected sample rate after resample")
    duration_secs: float = Field(2.0, description="Pad/truncate clips to this length")
    use_spectrogram: bool = Field(True, description="If true, convert waveform to mel-spectrogram before convs (2D path)")
    n_mels: int = Field(64, description="Mel bins (when use_spectrogram=true)")
    n_fft: int = Field(1024, description="STFT window")
    hop_length: int = Field(256, description="STFT hop")
    conv_channels: List[int] = Field(default_factory=lambda: [32, 64, 128, 256])
    fc_layers: List[LayerConfig] = Field(default_factory=lambda: [LayerConfig(size=128, dropout=0.3)])
    output_size: int = Field(..., description="Number of output classes")
    pretrained: Optional[str] = Field(None, description="Optional HuggingFace audio model name (e.g. 'facebook/wav2vec2-base')")


class TCNConfig(BaseModel):
    """Temporal Convolutional Network for 1D time-series."""
    input_size: int = Field(..., description="Channels per timestep")
    output_size: int = Field(..., description="Output dimension")
    channels: List[int] = Field(default_factory=lambda: [64, 64, 64, 64], description="Per-block channel widths")
    kernel_size: int = Field(3, ge=2, description="Causal conv kernel")
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    output_mode: Literal["last", "mean", "all"] = Field("last", description="Pool the temporal axis with the last step / mean / leave as-is")
    pooling: Literal["last", "mean", "max"] = Field("last", description="(Deprecated alias for output_mode)")


class TabularConfig(BaseModel):
    """Tabular learner with first-class categorical embeddings + missing-value handling."""
    numeric_features: List[str] = Field(default_factory=list, description="Names of numeric columns")
    categorical_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {name, cardinality, embed_dim?} per categorical column",
    )
    output_size: int = Field(..., description="Number of classes / regression dim")
    hidden_layers: List[LayerConfig] = Field(
        default_factory=lambda: [LayerConfig(size=256, dropout=0.2, batch_norm=True),
                                 LayerConfig(size=128, dropout=0.2, batch_norm=True)],
    )
    output_activation: str = Field("none")
    impute_strategy: Literal["zero", "mean", "median"] = Field("mean", description="How to handle missing values")


class HFPipelineConfig(BaseModel):
    """Universal wrapper around any HuggingFace pretrained model.

    Pairs a `pipeline_task` (e.g. ``automatic-speech-recognition``) with a
    `pretrained` model id (e.g. ``openai/whisper-tiny``). At training time
    we route through the right ``transformers.Auto*`` class based on the
    task. At inference time the same task drives the right
    ``transformers.pipeline()`` invocation.

    This is how NeuralForge supports the long tail of HF tasks (ASR,
    summarization, VQA, image-captioning, image-segmentation, …) without
    hand-coding an architecture per task. Trade-off: full fine-tuning may
    require more compute than the from-scratch CNN/Transformer baselines.
    """
    pretrained: str = Field(..., description="HuggingFace model id, e.g. 'openai/whisper-tiny'")
    output_size: Optional[int] = Field(None, description="Number of classes for classification tasks. Inferred from the HF config when omitted.")
    freeze_backbone: bool = Field(False, description="Freeze encoder weights and train only the head")
    revision: Optional[str] = Field(None, description="HuggingFace model revision / git ref")
    trust_remote_code: bool = Field(False, description="Some HF models require this flag to load custom architectures")


class VideoCNNConfig(BaseModel):
    """3D CNN over (T, C, H, W) frame stacks. Experimental — basic architecture only."""
    input_channels: int = Field(3, description="Channels per frame")
    num_frames: int = Field(16, description="Frames per clip (uniformly sampled)")
    input_height: int = Field(112, description="Frame height after resize")
    input_width: int = Field(112, description="Frame width after resize")
    conv_layers: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "pool": True},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "pool": True},
            {"out_channels": 128, "kernel_size": 3, "stride": 1, "pool": True},
        ],
    )
    fc_layers: List[LayerConfig] = Field(default_factory=lambda: [LayerConfig(size=256, dropout=0.5)])
    output_size: int = Field(..., description="Number of action / video classes")


class ModelConfig(BaseModel):
    """Top-level model configuration."""
    type: ModelType = Field(..., description="Model family: mlp, cnn, rnn, transformer, audio_cnn, tcn, tabular, video_cnn")
    name: str = Field("model", description="Human-readable model name")
    framework: Framework = Field(Framework.PYTORCH, description="Training framework: pytorch, tensorflow, jax")

    # Exactly one of these is populated depending on type
    mlp: Optional[MLPConfig] = None
    cnn: Optional[CNNConfig] = None
    rnn: Optional[RNNConfig] = None
    transformer: Optional[TransformerConfig] = None
    audio_cnn: Optional[AudioCNNConfig] = None
    tcn: Optional[TCNConfig] = None
    tabular: Optional[TabularConfig] = None
    video_cnn: Optional[VideoCNNConfig] = None
    hf_pipeline: Optional[HFPipelineConfig] = None

    @model_validator(mode="after")
    def check_arch_config_present(self) -> ModelConfig:
        arch_map = {
            ModelType.MLP: self.mlp,
            ModelType.CNN: self.cnn,
            ModelType.RNN: self.rnn,
            ModelType.TRANSFORMER: self.transformer,
            ModelType.AUDIO_CNN: self.audio_cnn,
            ModelType.TCN: self.tcn,
            ModelType.TABULAR: self.tabular,
            ModelType.VIDEO_CNN: self.video_cnn,
            ModelType.HF_PIPELINE: self.hf_pipeline,
        }
        if arch_map[self.type] is None:
            raise ValueError(
                f"Model type is '{self.type.value}' but the corresponding "
                f"'{self.type.value}' config block is missing."
            )
        return self

    def get_arch_config(self):
        return {
            ModelType.MLP: self.mlp,
            ModelType.CNN: self.cnn,
            ModelType.RNN: self.rnn,
            ModelType.TRANSFORMER: self.transformer,
            ModelType.AUDIO_CNN: self.audio_cnn,
            ModelType.TCN: self.tcn,
            ModelType.TABULAR: self.tabular,
            ModelType.VIDEO_CNN: self.video_cnn,
            ModelType.HF_PIPELINE: self.hf_pipeline,
        }[self.type]


class OptimizerConfig(BaseModel):
    """Optimizer settings."""
    type: Optimizer = Field(Optimizer.ADAMW, description="Optimizer algorithm")
    lr: float = Field(1e-3, gt=0, description="Learning rate")
    weight_decay: float = Field(1e-4, ge=0, description="L2 weight decay")
    momentum: float = Field(0.9, ge=0, description="Momentum (SGD only)")
    betas: List[float] = Field([0.9, 0.999], description="Adam beta1 and beta2")
    eps: float = Field(1e-8, gt=0, description="Adam epsilon")
    grad_clip: Optional[float] = Field(None, description="Max gradient norm for clipping (None = disabled)")


class SchedulerConfig(BaseModel):
    """Learning rate scheduler settings."""
    type: Scheduler = Field(Scheduler.COSINE, description="LR scheduler type")
    warmup_steps: int = Field(0, ge=0, description="Warmup steps for warmup_cosine scheduler")
    step_size: int = Field(10, ge=1, description="Step size for StepLR")
    gamma: float = Field(0.1, gt=0, description="Decay factor for step/exponential schedulers")
    patience: int = Field(5, ge=1, description="Patience for ReduceLROnPlateau")
    min_lr: float = Field(1e-7, ge=0, description="Minimum learning rate floor")
    t_max: Optional[int] = Field(None, description="T_max for CosineAnnealingLR (defaults to num_epochs)")


class TrainingConfig(BaseModel):
    """Training loop configuration."""
    task: Task = Field(Task.CLASSIFICATION, description="Coarse task category — affects loss function and metrics. Kept for backwards compatibility; prefer `pipeline_task` for fine-grained HF-aligned tasks.")
    pipeline_task: Optional[str] = Field(
        None,
        description="HuggingFace pipeline_tag, e.g. 'audio-classification', 'automatic-speech-recognition', 'image-classification', 'visual-question-answering'. See core.tasks.Task for the full enum. When set, drives validator + dataset-loader + model-architecture decisions; when omitted, falls back to the coarse `task` field above.",
    )
    loss: LossFunction = Field(LossFunction.CROSS_ENTROPY, description="Loss function")
    num_epochs: int = Field(50, ge=1, description="Maximum number of training epochs")
    batch_size: int = Field(32, ge=1, description="Training batch size")
    val_batch_size: Optional[int] = Field(None, description="Validation batch size (defaults to batch_size)")
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    early_stopping_patience: Optional[int] = Field(10, description="Stop if val loss doesn't improve for N epochs. None = disabled")
    mixed_precision: bool = Field(False, description="Enable AMP mixed-precision training (PyTorch only)")
    accumulation_steps: int = Field(1, ge=1, description="Gradient accumulation steps")
    num_workers: int = Field(4, ge=0, description="DataLoader worker processes")
    seed: int = Field(42, description="Random seed for reproducibility")
    device: str = Field("auto", description="Device: auto, cpu, cuda, cuda:0, mps")
    checkpoint_every: int = Field(5, ge=1, description="Save checkpoint every N epochs")
    keep_best_only: bool = Field(True, description="Only keep the best checkpoint by val loss")
    log_every: int = Field(10, ge=1, description="Log training metrics every N batches")


class DataConfig(BaseModel):
    """Dataset and preprocessing configuration."""
    source: DataSource = Field(DataSource.SYNTHETIC, description="Data source type")
    path: Optional[str] = Field(None, description="Path to data (CSV file, image folder, etc.)")
    dataset_name: Optional[str] = Field(None, description="HuggingFace dataset name, e.g. 'mnist', 'imdb'")
    dataset_config: Optional[str] = Field(None, description="HuggingFace dataset sub-configuration (e.g. 'asr' for the 'superb' dataset)")
    dataset_split: str = Field("train", description="HuggingFace dataset split")
    target_column: Optional[str] = Field(None, description="Target column name for CSV data")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns (None = all except target)")
    text_column: Optional[str] = Field(None, description="Text column for NLP tasks")
    label_column: Optional[str] = Field(None, description="Label column for NLP tasks")
    val_split: float = Field(0.2, ge=0.0, lt=1.0, description="Fraction of data for validation")
    test_split: float = Field(0.0, ge=0.0, lt=1.0, description="Fraction for test set (0 = no test split)")
    max_samples: Optional[int] = Field(None, description="Limit dataset size (useful for debugging)")
    transforms: Optional[Dict[str, Any]] = Field(None, description="Transform configuration dict")
    # Synthetic data options
    synthetic_n_samples: int = Field(1000, description="Number of synthetic samples")
    synthetic_n_features: int = Field(10, description="Number of synthetic features")
    synthetic_n_classes: int = Field(2, description="Number of synthetic classes")
    synthetic_noise: float = Field(0.1, description="Noise level for synthetic data")


class DeployConfig(BaseModel):
    """Inference server deployment configuration.

    Binds to localhost by default — exposing the inference server to
    other machines is opt-in (set `host: 0.0.0.0` in your config or pass
    `--host 0.0.0.0` to `neural serve`). The CLI prints a security
    warning when you do.
    """
    host: str = Field("127.0.0.1", description="Server bind host (default localhost-only)")
    port: int = Field(8080, ge=1024, le=65535, description="Server bind port")
    workers: int = Field(1, ge=1, description="Number of uvicorn workers")
    reload: bool = Field(False, description="Hot-reload (dev only)")
    checkpoint: Optional[str] = Field(None, description="Path to checkpoint to serve (defaults to best)")
    max_batch_size: int = Field(32, ge=1, description="Maximum inference batch size")
    timeout_seconds: float = Field(30.0, gt=0, description="Request timeout in seconds")


class ExperimentConfig(BaseModel):
    """
    Top-level experiment configuration.
    This is the single object you pass to Trainer.
    """
    name: str = Field("experiment", description="Experiment name (used for logging and checkpoint dirs)")
    description: Optional[str] = Field(None, description="Optional description of the experiment")
    tags: List[str] = Field(default_factory=list, description="Optional tags for filtering experiments")
    output_dir: str = Field("runs", description="Root directory for experiment outputs")
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    deploy: DeployConfig = Field(default_factory=DeployConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "training.log"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a YAML or JSON file.

    Args:
        path: Path to a .yaml, .yml, or .json config file.

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the config fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml or .json")

    return ExperimentConfig.model_validate(data)


def save_config(config: ExperimentConfig, path: Union[str, Path]) -> None:
    """Save an ExperimentConfig to a YAML or JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump()
    with open(path, "w") as f:
        if path.suffix in (".yaml", ".yml"):
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            json.dump(data, f, indent=2)
