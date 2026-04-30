"""
NeuralForge Data Transforms
Config-driven preprocessing for images, tabular data, and text sequences.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_transforms(transforms_cfg: Optional[Dict[str, Any]], mode: str = "train"):
    """
    Build a transform pipeline from a config dict.

    transforms_cfg keys:
        image:
            resize: [H, W]
            normalize: {mean: [...], std: [...]}
            augment: bool  (train only)
            center_crop: int
            grayscale: bool
        tabular:
            normalize: bool
            scaler: 'standard' | 'minmax' | 'robust'
        text:
            tokenizer: str  (HuggingFace model name)
            max_length: int
            padding: bool
            truncation: bool

    Returns a callable transform.
    """
    if transforms_cfg is None:
        return None

    if "image" in transforms_cfg:
        return _build_image_transforms(transforms_cfg["image"], mode)
    if "tabular" in transforms_cfg:
        return _build_tabular_transforms(transforms_cfg["tabular"])
    if "text" in transforms_cfg:
        return _build_text_transforms(transforms_cfg["text"])
    return None


def _build_image_transforms(cfg: Dict, mode: str):
    """Build torchvision transform pipeline."""
    import torchvision.transforms as T

    ops = []
    resize = cfg.get("resize", None)
    grayscale = cfg.get("grayscale", False)
    center_crop = cfg.get("center_crop", None)
    normalize = cfg.get("normalize", None)
    augment = cfg.get("augment", True) and mode == "train"

    if grayscale:
        ops.append(T.Grayscale())
    if resize:
        ops.append(T.Resize(resize))
    if augment:
        ops.extend([
            T.RandomHorizontalFlip(),
            T.RandomCrop(center_crop or (resize[0] if resize else 32), padding=4),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    elif center_crop:
        ops.append(T.CenterCrop(center_crop))

    ops.append(T.ToTensor())

    if normalize:
        mean = normalize.get("mean", [0.485, 0.456, 0.406])
        std = normalize.get("std", [0.229, 0.224, 0.225])
        ops.append(T.Normalize(mean=mean, std=std))

    return T.Compose(ops)


def _build_tabular_transforms(cfg: Dict):
    """Return sklearn scaler for tabular data."""
    scaler_type = cfg.get("scaler", "standard")
    if not cfg.get("normalize", True):
        return None
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler_type, StandardScaler)()
    except ImportError:
        return None


def _build_text_transforms(cfg: Dict):
    """Build a HuggingFace tokenizer wrapper."""
    tokenizer_name = cfg.get("tokenizer", "bert-base-uncased")
    max_length = cfg.get("max_length", 128)
    padding = cfg.get("padding", True)
    truncation = cfg.get("truncation", True)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize(texts):
            return tokenizer(
                texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors="pt",
            )

        return tokenize
    except ImportError:
        raise ImportError(
            "transformers is required for text transforms. "
            "Install with: pip install transformers"
        )
