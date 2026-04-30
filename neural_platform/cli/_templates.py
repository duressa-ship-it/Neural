"""
Config template generators for `neural init`.
Returns pre-filled config dicts for each model type.
"""

from __future__ import annotations


def get_template_config(model_type: str, name: str, output_dir: str, framework: str) -> dict:
    generators = {
        "mlp":         _mlp_template,
        "cnn":         _cnn_template,
        "rnn":         _rnn_template,
        "transformer": _transformer_template,
        "audio_cnn":   _audio_cnn_template,
        "tcn":         _tcn_template,
        "tabular":     _tabular_template,
        "video_cnn":   _video_cnn_template,
    }
    gen = generators.get(model_type, _mlp_template)
    return gen(name, output_dir, framework)


SUPPORTED_MODEL_TYPES = (
    "mlp", "cnn", "rnn", "transformer",
    "audio_cnn", "tcn", "tabular", "video_cnn",
)


def _base(name, output_dir, framework, model_type, description):
    return {
        "name": name,
        "description": description,
        "tags": [],
        "output_dir": output_dir,
        "model": {"type": model_type, "name": name, "framework": framework},
        "training": {
            "task": "classification",
            "loss": "cross_entropy",
            "num_epochs": 50,
            "batch_size": 64,
            "optimizer": {
                "type": "adamw",
                "lr": 1e-3,
                "weight_decay": 1e-4,
            },
            "scheduler": {
                "type": "cosine",
            },
            "early_stopping_patience": 10,
            "mixed_precision": False,
            "seed": 42,
            "device": "auto",
            "checkpoint_every": 5,
        },
        "data": {
            "source": "synthetic",
            "val_split": 0.2,
        },
        "deploy": {
            "host": "0.0.0.0",
            "port": 8080,
        },
    }


def _mlp_template(name, output_dir, framework):
    cfg = _base(name, output_dir, framework, "mlp", "Feedforward MLP experiment")
    cfg["model"]["mlp"] = {
        "input_size": 10,
        "hidden_layers": [
            {"size": 256, "activation": "relu", "dropout": 0.3, "batch_norm": True},
            {"size": 128, "activation": "relu", "dropout": 0.2, "batch_norm": True},
            {"size": 64,  "activation": "relu", "dropout": 0.1, "batch_norm": False},
        ],
        "output_size": 2,
        "output_activation": "none",
    }
    cfg["data"]["synthetic_n_samples"] = 5000
    cfg["data"]["synthetic_n_features"] = 10
    cfg["data"]["synthetic_n_classes"] = 2
    return cfg


def _cnn_template(name, output_dir, framework):
    cfg = _base(name, output_dir, framework, "cnn", "CNN image classification experiment")
    cfg["training"]["task"] = "image_classification"
    cfg["model"]["cnn"] = {
        "input_channels": 3,
        "input_height": 32,
        "input_width": 32,
        "conv_layers": [
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "batch_norm": True, "pool": True},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "batch_norm": True, "pool": True},
            {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "batch_norm": True, "pool": True},
        ],
        "fc_layers": [
            {"size": 256, "activation": "relu", "dropout": 0.5, "batch_norm": False}
        ],
        "output_size": 10,
        "backbone": None,
        "pretrained": False,
        "freeze_backbone": False,
    }
    cfg["data"]["source"] = "huggingface"
    cfg["data"]["dataset_name"] = "cifar10"
    cfg["data"]["transforms"] = {
        "image": {
            "resize": [32, 32],
            "normalize": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.247, 0.243, 0.261]},
            "augment": True,
        }
    }
    return cfg


def _rnn_template(name, output_dir, framework):
    cfg = _base(name, output_dir, framework, "rnn", "LSTM sequence classification experiment")
    cfg["training"]["task"] = "classification"
    cfg["model"]["rnn"] = {
        "variant": "lstm",
        "input_size": 32,
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.2,
        "output_size": 2,
        "output_mode": "last",
        "fc_layers": [
            {"size": 64, "activation": "relu", "dropout": 0.3, "batch_norm": False}
        ],
    }
    cfg["data"]["synthetic_n_samples"] = 2000
    cfg["data"]["synthetic_n_features"] = 32
    cfg["data"]["synthetic_n_classes"] = 2
    return cfg


def _transformer_template(name, output_dir, framework):
    cfg = _base(name, output_dir, framework, "transformer", "Transformer text classification (IMDB)")
    cfg["training"]["task"] = "text_classification"
    cfg["training"]["loss"] = "cross_entropy"
    cfg["training"]["batch_size"] = 32
    cfg["training"]["optimizer"]["lr"] = 2e-5
    cfg["model"]["transformer"] = {
        "vocab_size": 30522,
        "max_seq_len": 128,
        "d_model": 256,
        "num_heads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 0,
        "d_ff": 512,
        "dropout": 0.1,
        "output_size": 2,
        "output_mode": "cls",
        "positional_encoding": "sinusoidal",
        "use_pretrained": None,
    }
    cfg["data"]["source"] = "huggingface"
    cfg["data"]["dataset_name"] = "imdb"
    cfg["data"]["text_column"] = "text"
    cfg["data"]["label_column"] = "label"
    cfg["data"]["max_samples"] = 5000  # IMDB is large — keep first try fast
    cfg["data"]["transforms"] = {"text": {"tokenizer": "bert-base-uncased", "max_length": 128}}
    return cfg


def _audio_cnn_template(name, output_dir, framework):
    cfg = _base(name, output_dir, framework, "audio_cnn", "Audio classification (SUPERB keyword spotting)")
    cfg["training"]["task"] = "classification"
    cfg["training"]["batch_size"] = 32
    cfg["training"]["optimizer"]["lr"] = 1e-3
    cfg["model"]["audio_cnn"] = {
        "sample_rate": 16000,
        "duration_secs": 1.0,
        "use_spectrogram": True,
        "n_mels": 64,
        "n_fft": 1024,
        "hop_length": 256,
        "conv_channels": [32, 64, 128, 256],
        "fc_layers": [{"size": 128, "activation": "relu", "dropout": 0.3, "batch_norm": False}],
        "output_size": 12,
        "pretrained": None,
    }
    cfg["data"]["source"] = "huggingface"
    cfg["data"]["dataset_name"] = "speech_commands"
    cfg["data"]["dataset_split"] = "train"
    cfg["data"]["max_samples"] = 5000
    return cfg


def _tcn_template(name, output_dir, framework):
    cfg = _base(name, output_dir, framework, "tcn", "TCN time-series classification (synthetic sine waves)")
    cfg["training"]["task"] = "classification"
    cfg["training"]["batch_size"] = 64
    cfg["model"]["tcn"] = {
        "input_size": 1,
        "output_size": 4,
        "channels": [64, 64, 64, 64],
        "kernel_size": 3,
        "dropout": 0.1,
        "output_mode": "last",
        "pooling": "last",
    }
    # Synthetic time-series: 4 wave classes
    cfg["data"]["synthetic_n_samples"] = 4000
    cfg["data"]["synthetic_n_features"] = 64    # treated as timesteps
    cfg["data"]["synthetic_n_classes"] = 4
    return cfg


def _tabular_template(name, output_dir, framework):
    cfg = _base(name, output_dir, framework, "tabular", "Tabular learner with categorical embeddings")
    cfg["training"]["task"] = "classification"
    cfg["model"]["tabular"] = {
        "numeric_features": ["feat_0", "feat_1", "feat_2", "feat_3"],
        "categorical_features": [
            {"name": "cat_0", "cardinality": 10},
            {"name": "cat_1", "cardinality": 50, "embed_dim": 16},
        ],
        "output_size": 2,
        "hidden_layers": [
            {"size": 256, "activation": "relu", "dropout": 0.2, "batch_norm": True},
            {"size": 128, "activation": "relu", "dropout": 0.2, "batch_norm": True},
        ],
        "output_activation": "none",
        "impute_strategy": "mean",
    }
    cfg["data"]["source"] = "csv"
    cfg["data"]["path"] = "data/tabular.csv"
    cfg["data"]["target_column"] = "label"
    return cfg


def _video_cnn_template(name, output_dir, framework):
    cfg = _base(name, output_dir, framework, "video_cnn", "Video classification (3D CNN, experimental)")
    cfg["training"]["task"] = "classification"
    cfg["training"]["batch_size"] = 8       # videos are heavy
    cfg["model"]["video_cnn"] = {
        "input_channels": 3,
        "num_frames": 16,
        "input_height": 112,
        "input_width": 112,
        "conv_layers": [
            {"out_channels": 32,  "kernel_size": 3, "stride": 1, "pool": True},
            {"out_channels": 64,  "kernel_size": 3, "stride": 1, "pool": True},
            {"out_channels": 128, "kernel_size": 3, "stride": 1, "pool": True},
        ],
        "fc_layers": [{"size": 256, "activation": "relu", "dropout": 0.5, "batch_norm": False}],
        "output_size": 10,
    }
    cfg["data"]["source"] = "huggingface"
    cfg["data"]["dataset_name"] = "ucf101"  # placeholder — user picks one with Video features
    cfg["data"]["max_samples"] = 1000
    return cfg
