"""
NeuralForge Data Loader
Unified data loading from CSV, image folders, HuggingFace datasets, numpy arrays,
and synthetic data. Returns standard PyTorch DataLoaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from neural_platform.core.config import DataConfig, DataSource
from neural_platform.data.transforms import build_transforms


# ---------------------------------------------------------------------------
# Custom Dataset wrappers
# ---------------------------------------------------------------------------

class CSVDataset(Dataset):
    """Dataset from a CSV file with configurable target and feature columns."""

    def __init__(self, path: str, target_col: str, feature_cols=None, transform=None):
        import pandas as pd
        df = pd.read_csv(path)
        if feature_cols:
            X = df[feature_cols].values
        else:
            X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long if y.dtype.kind in "iub" else torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]


class ImageFolderDataset(Dataset):
    """Thin wrapper around torchvision ImageFolder with optional transforms."""

    def __init__(self, root: str, transform=None):
        from torchvision.datasets import ImageFolder
        self._ds = ImageFolder(root, transform=transform)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]

    @property
    def classes(self):
        return self._ds.classes


def inspect_hf_features(hf_dataset) -> dict:
    """
    Categorize a HuggingFace dataset's columns by inferred modality.
    Thin wrapper around `core.hf_introspect.inspect_dataset`.
    """
    from neural_platform.core.hf_introspect import inspect_dataset
    info = inspect_dataset(hf_dataset)
    # Backwards-compatible "all_columns" alias for older callers
    info["all_columns"] = info["columns"]
    return info


class HuggingFaceTextDataset(Dataset):
    """
    HuggingFace text dataset → PyTorch Dataset.

    When `tokenizer` is provided, yields `(input_ids/attention_mask dict,
    label_tensor)`; otherwise yields `(raw_string, label_tensor)`.
    """

    def __init__(self, hf_dataset, text_col: str, label_col: Optional[str],
                 tokenizer=None, max_length: int = 128):
        self.ds = hf_dataset
        self.text_col = text_col
        self.label_col = label_col
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.classes = _hf_class_names(hf_dataset, label_col)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        text = item[self.text_col]
        label = item[self.label_col] if self.label_col else 0
        label = int(label) if not isinstance(label, int) else label

        if self.tokenizer:
            encoded = self.tokenizer(
                text, max_length=self.max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            inputs = {k: v.squeeze(0) for k, v in encoded.items()}
            return inputs, torch.tensor(label, dtype=torch.long)
        return text, torch.tensor(label, dtype=torch.long)


class HuggingFaceAudioDataset(Dataset):
    """
    HuggingFace audio dataset → PyTorch Dataset.

    Reads `audio_col` (a `datasets.Audio` feature, decoded to a dict with
    `array` + `sampling_rate`) and an optional `label_col`. Resamples to
    `target_sample_rate` and pads/truncates to `target_samples` so every
    item has the same shape — mandatory for default DataLoader collation.

    Yields `(waveform_tensor, label_tensor)` where waveform is shape
    `(target_samples,)`.
    """

    def __init__(self, hf_dataset, audio_col: str, label_col: Optional[str],
                 target_sample_rate: int = 16000, duration_secs: float = 2.0):
        self.ds = hf_dataset
        self.audio_col = audio_col
        self.label_col = label_col
        self.target_sample_rate = target_sample_rate
        self.target_samples = int(target_sample_rate * duration_secs)
        self.classes = _hf_class_names(hf_dataset, label_col)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        audio = item[self.audio_col]
        wav, sr = _coerce_waveform(audio, self.target_sample_rate)
        if wav.dim() > 1:                              # (channels, samples) → mono
            wav = wav.mean(dim=0)

        # Resample if needed
        if sr != self.target_sample_rate:
            try:
                import torchaudio.functional as AF
                wav = AF.resample(wav.unsqueeze(0), sr, self.target_sample_rate).squeeze(0)
            except Exception:
                # Fall back: linear interp via torch.nn.functional.interpolate
                wav = torch.nn.functional.interpolate(
                    wav.view(1, 1, -1),
                    scale_factor=self.target_sample_rate / max(sr, 1),
                    mode="linear", align_corners=False,
                ).view(-1)

        # Pad / truncate to fixed length
        if wav.size(0) < self.target_samples:
            wav = torch.nn.functional.pad(wav, (0, self.target_samples - wav.size(0)))
        elif wav.size(0) > self.target_samples:
            wav = wav[:self.target_samples]

        label = item[self.label_col] if self.label_col else 0
        label = int(label) if not isinstance(label, int) else label
        return wav, torch.tensor(label, dtype=torch.long)


class HuggingFaceVideoDataset(Dataset):
    """
    HuggingFace video dataset → PyTorch Dataset (experimental).

    Each item is uniformly subsampled to `num_frames` frames, each resized
    to `(input_height, input_width)`. Yields `(clip, label)` where clip is
    shape `(channels, num_frames, height, width)`.

    Tolerates several HF representations: `datasets.Video` (frames generator),
    a list of `datasets.Image` (frame stack), or a path to a video file.
    """

    def __init__(self, hf_dataset, video_col: str, label_col: Optional[str],
                 num_frames: int = 16, input_height: int = 112, input_width: int = 112):
        self.ds = hf_dataset
        self.video_col = video_col
        self.label_col = label_col
        self.num_frames = num_frames
        self.input_height = input_height
        self.input_width = input_width
        self.classes = _hf_class_names(hf_dataset, label_col)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        from torchvision import transforms as T
        from PIL import Image
        item = self.ds[idx]
        video = item[self.video_col]
        frames = self._extract_frames(video)
        if not frames:
            raise RuntimeError(f"Could not extract frames from video item {idx}")

        # Uniformly sample num_frames
        if len(frames) >= self.num_frames:
            step = len(frames) / self.num_frames
            frames = [frames[int(i * step)] for i in range(self.num_frames)]
        else:
            # Repeat last frame to pad
            frames = frames + [frames[-1]] * (self.num_frames - len(frames))

        transform = T.Compose([T.Resize((self.input_height, self.input_width)), T.ToTensor()])
        tensors = []
        for f in frames:
            if isinstance(f, np.ndarray):
                f = Image.fromarray(f.astype(np.uint8))
            elif not hasattr(f, "convert"):
                continue
            tensors.append(transform(f.convert("RGB")))
        clip = torch.stack(tensors, dim=1)  # (C, T, H, W)

        label = item[self.label_col] if self.label_col else 0
        label = int(label) if not isinstance(label, int) else label
        return clip, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _extract_frames(video) -> list:
        # List of PIL frames (most common when Sequence(Image))
        if isinstance(video, list):
            return video
        # HF Video feature decodes lazily; iterate
        if hasattr(video, "__iter__") and not hasattr(video, "convert"):
            try:
                return list(video)
            except Exception:
                return []
        # Path to video file → torchvision.io.read_video if available
        if isinstance(video, (str, dict)):
            path = video if isinstance(video, str) else video.get("path")
            if not path:
                return []
            try:
                from torchvision.io import read_video
                tensor, _, _ = read_video(path, pts_unit="sec")  # (T, H, W, C) uint8
                from PIL import Image
                return [Image.fromarray(tensor[i].numpy()) for i in range(tensor.size(0))]
            except Exception:
                return []
        return []


class HuggingFaceImageDataset(Dataset):
    """
    HuggingFace image dataset → PyTorch Dataset.

    Reads `image_col` (a PIL Image, numpy array, or dict with bytes) and
    `label_col` (int or ClassLabel). Applies the configured torchvision
    transform to the image. Produces `(image_tensor, label_tensor)`.

    If `label_col` is None (e.g. unlabeled image set), yields a constant
    label of 0 — the model can ignore it for unsupervised setups, or the
    user can switch to a supervised dataset.
    """

    def __init__(self, hf_dataset, image_col: str, label_col: Optional[str], transform=None):
        self.ds = hf_dataset
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.classes = _hf_class_names(hf_dataset, label_col)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item[self.image_col]
        # HF Image features decode to PIL.Image automatically when accessed
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        elif isinstance(img, dict) and "bytes" in img:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        elif isinstance(img, np.ndarray):
            from PIL import Image
            img = Image.fromarray(img.astype(np.uint8))
        # else: assume it's already a tensor

        if self.transform is not None:
            img = self.transform(img)
        else:
            # Minimal fallback: ToTensor
            from torchvision.transforms import ToTensor
            img = ToTensor()(img) if hasattr(img, "convert") or isinstance(img, np.ndarray) else img

        label = item[self.label_col] if self.label_col else 0
        label = int(label) if not isinstance(label, int) else label
        return img, torch.tensor(label, dtype=torch.long)


def _hf_class_names(hf_dataset, label_col: Optional[str]) -> Optional[List[str]]:
    from neural_platform.core.hf_introspect import class_names_for
    return class_names_for(hf_dataset, label_col)


def _coerce_waveform(audio, default_sr: int):
    """
    Turn a HuggingFace `Audio` cell into `(waveform_tensor, sample_rate)`.

    HF can return audio in *several* shapes depending on which decoders the
    user has installed:
      * `{'array': np.ndarray<float>, 'sampling_rate': int, 'path': str}`
        — the happy path, when soundfile/torchaudio is available.
      * `{'array': np.ndarray<object>, 'sampling_rate': int, 'bytes': bytes, ...}`
        — when the array couldn't be decoded; we need to decode `bytes`/`path`.
      * `{'bytes': bytes, 'path': str}` — raw, never decoded.
      * a bare ndarray, list of floats, or a string path.

    Always returns a float32 1D tensor and an integer sample rate.
    """
    sr: int = default_sr
    arr = None
    bytes_blob = None
    path = None
    cell_keys: List[str] = []

    if isinstance(audio, dict):
        cell_keys = list(audio.keys())
        sr = int(audio.get("sampling_rate") or default_sr)
        arr = audio.get("array")
        bytes_blob = audio.get("bytes")
        path = audio.get("path")
    elif isinstance(audio, str):
        # Some HF datasets ship the audio column as just a filesystem path
        # (string), not a dict — common for older Common Voice / Mozilla
        # Spanish speech datasets.
        path = audio
    elif isinstance(audio, (bytes, bytearray)):
        bytes_blob = bytes(audio)
    else:
        arr = audio

    # Path 1: numeric array (np.ndarray / list / torch.Tensor)
    if arr is not None and not isinstance(arr, (bytes, str)):
        wav = _array_to_float_tensor(arr)
        if wav is not None:
            return wav, sr

    # Path 2: filesystem path that actually exists → decode it directly.
    # This is the common case for HF Audio cells when the array is already
    # cached to disk but `array` decoding hasn't happened yet (some torchcodec
    # / torchaudio version mismatches end up here).
    import os as _os
    if path and isinstance(path, str) and _os.path.exists(path):
        try:
            wav, sr2 = _decode_audio_blob(None, path)
            return wav, sr2 or sr
        except Exception:
            pass  # fall through and try bytes

    # Path 3: raw bytes
    if bytes_blob:
        try:
            wav, sr2 = _decode_audio_blob(bytes_blob, None)
            return wav, sr2 or sr
        except Exception:
            pass

    # Path 4: a path we couldn't validate, or one we missed above — last try.
    if path and isinstance(path, str):
        try:
            wav, sr2 = _decode_audio_blob(None, path)
            return wav, sr2 or sr
        except Exception as e:
            raise RuntimeError(_audio_decode_error_msg(cell_keys, path, str(e)))

    # Nothing usable — emit a debuggable error.
    raise RuntimeError(_audio_decode_error_msg(cell_keys, path, None))


def _audio_decode_error_msg(cell_keys, path, last_err) -> str:
    """Build a debuggable error showing exactly what was in the audio cell."""
    parts = ["Audio cell could not be decoded."]
    if cell_keys:
        # Show keys + a hint about which fields are populated
        keys_repr = ", ".join(repr(k) for k in cell_keys)
        parts.append(f"Cell keys present: [{keys_repr}].")
    else:
        parts.append("Cell was not a dict.")
    if path:
        parts.append(f"Path field: {path!r} "
                     f"({'exists' if _path_exists(path) else 'NOT on disk'}).")
    if last_err:
        parts.append(f"Last decoder error: {last_err}")
    parts.append("")
    parts.append(
        "Hints:\n"
        "  • pip install torchaudio soundfile torchcodec  # most common fix\n"
        "  • For HuggingFace datasets that ship `bytes` only, soundfile is enough.\n"
        "  • For .mp3 / .ogg, torchaudio + ffmpeg or torchcodec are usually needed.\n"
        "  • If your dataset uses streaming, also: pip install fsspec[http]"
    )
    return "\n".join(parts)


def _path_exists(p) -> bool:
    try:
        import os as _os
        return _os.path.exists(p)
    except Exception:
        return False


def _array_to_float_tensor(arr) -> Optional["torch.Tensor"]:
    """
    Best-effort coercion of an unknown numeric container to float32 tensor.
    Returns None if the array is genuinely non-numeric (object dtype with
    non-numeric items) so the caller can try the bytes/path decode path.
    """
    if isinstance(arr, torch.Tensor):
        return arr.detach().to(torch.float32)
    np_arr = np.asarray(arr)

    # Common case: numpy float/int array — fast path
    if np_arr.dtype != object:
        try:
            return torch.tensor(np_arr, dtype=torch.float32)
        except Exception:
            return None

    # Object dtype: usually a list of numbers wrapped, or list of arrays.
    # Try strict cast, then per-element cast.
    try:
        return torch.tensor(np.asarray(arr, dtype=np.float32))
    except Exception:
        pass
    try:
        flat = []
        for v in np_arr.flat:
            if hasattr(v, "tolist"):
                flat.extend(np.asarray(v, dtype=np.float32).flatten().tolist())
            else:
                flat.append(float(v))
        return torch.tensor(flat, dtype=torch.float32)
    except Exception:
        return None


def _decode_audio_blob(blob: Optional[bytes], path: Optional[str]):
    """
    Decode raw audio bytes / a file path to (waveform_1d_tensor, sample_rate).
    Tries torchaudio first, then soundfile, then librosa. Raises if none work.
    """
    # torchaudio handles both bytes (via BytesIO) and paths
    try:
        import io
        import torchaudio
        if blob:
            wav, sr = torchaudio.load(io.BytesIO(blob))
        else:
            wav, sr = torchaudio.load(path)
        # wav is (channels, samples) — caller will do the mono reduce
        return wav.to(torch.float32), int(sr)
    except Exception:
        pass

    # soundfile
    try:
        import io
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(blob) if blob else path, dtype="float32")
        return torch.tensor(data, dtype=torch.float32), int(sr)
    except Exception:
        pass

    # librosa as last resort (slow but very compatible)
    try:
        import io
        import librosa
        data, sr = librosa.load(io.BytesIO(blob) if blob else path, sr=None, mono=False)
        return torch.tensor(data, dtype=torch.float32), int(sr)
    except Exception as e:
        raise RuntimeError(f"All decoders failed: {e}")


# Backwards-compat shim — the older name kept for any external callers.
HuggingFaceDataset = HuggingFaceTextDataset


class NumericTabularDataset(Dataset):
    """
    HF dataset → flat numeric (B, N) tensor + (B,) label tensor.

    Used as the fallback for any HF dataset that's just typed numeric
    columns (sensor dumps, recommendation logs, classic tabular benchmarks).
    Missing values become 0; user can subclass to swap in a real imputer.
    """

    def __init__(self, hf_dataset, feature_cols: list, label_col):
        self.ds = hf_dataset
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.classes = _hf_class_names(hf_dataset, label_col)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        x = [_safe_float(item.get(c)) for c in self.feature_cols]
        y = item.get(self.label_col, 0) if self.label_col else 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(int(_safe_float(y)), dtype=torch.long)


class NumericRowAsSequenceDataset(Dataset):
    """
    Tabular HF dataset → sequence-shaped tensor for RNN/TCN models.

    `reshape="sequence"` → returns `(timesteps=N, features=1)` so the model
    sees N timesteps of 1 channel. This is the right interpretation when
    the user clusters columns like `domain_a_seq_38..46` and wants a
    causal model to learn patterns over them.
    """

    def __init__(self, hf_dataset, feature_cols: list, label_col, reshape: str = "sequence"):
        self.ds = hf_dataset
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.reshape = reshape
        self.classes = _hf_class_names(hf_dataset, label_col)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        x = [_safe_float(item.get(c)) for c in self.feature_cols]
        y = item.get(self.label_col, 0) if self.label_col else 0
        if self.reshape == "sequence":
            # (timesteps, channels=1)
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        else:
            t = torch.tensor(x, dtype=torch.float32)
        return t, torch.tensor(int(_safe_float(y)), dtype=torch.long)


class NumericSequenceDataset(Dataset):
    """
    Each row contains a single typed Sequence(Value(numeric)) column —
    e.g. `signal: [0.12, 0.34, ..., 0.99]`. Yields `(seq_tensor (T,1), label)`.
    """

    def __init__(self, hf_dataset, seq_col, label_col):
        self.ds = hf_dataset
        self.seq_col = seq_col
        self.label_col = label_col
        self.classes = _hf_class_names(hf_dataset, label_col)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        seq = item[self.seq_col] or []
        y = item.get(self.label_col, 0) if self.label_col else 0
        t = torch.tensor([float(v) for v in seq], dtype=torch.float32).unsqueeze(-1)
        return t, torch.tensor(int(_safe_float(y)), dtype=torch.long)


def _safe_float(v) -> float:
    """Coerce mixed types to float, replacing None/NaN/non-numeric with 0."""
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v) if v == v else 0.0   # NaN check
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


class NumpyDataset(Dataset):
    """Dataset from numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long if y.dtype.kind in "iub" else torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _make_synthetic(cfg: DataConfig) -> Dataset:
    from sklearn.datasets import make_classification, make_regression

    n, n_feat, n_cls = cfg.synthetic_n_samples, cfg.synthetic_n_features, cfg.synthetic_n_classes
    noise = cfg.synthetic_noise

    if n_cls == 1:
        X, y = make_regression(n_samples=n, n_features=n_feat, noise=noise * 10, random_state=42)
        y = y.astype(np.float32).reshape(-1, 1)
    else:
        X, y = make_classification(
            n_samples=n, n_features=n_feat, n_classes=n_cls,
            n_informative=max(2, n_feat // 2), random_state=42
        )
    X = X.astype(np.float32)
    return NumpyDataset(X, y)


# ---------------------------------------------------------------------------
# Built-in datasets (via torchvision)
# ---------------------------------------------------------------------------

_TORCHVISION_DATASETS = {
    "mnist", "fashionmnist", "fashion_mnist", "cifar10", "cifar100", "svhn"
}

def _load_torchvision(name: str, train: bool, transform=None) -> Dataset:
    import torchvision.datasets as tvd

    kwargs = dict(root=".cache/datasets", download=True, transform=transform)
    name_lower = name.lower().replace("-", "").replace("_", "")

    if name_lower == "mnist":
        return tvd.MNIST(train=train, **kwargs)
    elif name_lower in ("fashionmnist", "fashionmnist"):
        return tvd.FashionMNIST(train=train, **kwargs)
    elif name_lower == "cifar10":
        return tvd.CIFAR10(train=train, **kwargs)
    elif name_lower == "cifar100":
        return tvd.CIFAR100(train=train, **kwargs)
    raise ValueError(f"Unknown torchvision dataset: {name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: DataConfig,
    training_cfg=None,
    model_type: str = "mlp",
    model_cfg=None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Build train, val (and optionally test) DataLoaders from a DataConfig.

    For HuggingFace text datasets paired with transformer models, this wires
    up an appropriate tokenizer automatically — either the one named in
    cfg.transforms['text'] or, if not specified, a default
    `bert-base-uncased` (or the pretrained model's own tokenizer).

    Returns:
        (train_loader, val_loader, test_loader)  — val and test may be None
    """
    batch_size = training_cfg.batch_size if training_cfg else 32
    val_batch = training_cfg.val_batch_size or batch_size if training_cfg else batch_size
    num_workers = training_cfg.num_workers if training_cfg else 0

    transform_train = build_transforms(cfg.transforms, mode="train")
    transform_val = build_transforms(cfg.transforms, mode="val")

    # --- Load the raw dataset ---
    if cfg.source == DataSource.SYNTHETIC:
        dataset = _make_synthetic(cfg)
        train_ds, val_ds, test_ds = _split_dataset(dataset, cfg)

    elif cfg.source == DataSource.CSV:
        if not cfg.path:
            raise ValueError(
                "data.path is required for CSV source. "
                "Set it to the path of your .csv file."
            )
        if not Path(cfg.path).exists():
            raise FileNotFoundError(f"CSV file not found: {cfg.path}")
        if not cfg.target_column:
            raise ValueError("data.target_column is required for CSV source.")
        dataset = CSVDataset(cfg.path, cfg.target_column, cfg.feature_columns, transform_train)
        train_ds, val_ds, test_ds = _split_dataset(dataset, cfg)

    elif cfg.source == DataSource.IMAGE_FOLDER:
        if not cfg.path:
            raise ValueError(
                "data.path is required for image_folder source. "
                "Point it at a directory of class subfolders."
            )
        root = Path(cfg.path)
        if not root.exists():
            raise FileNotFoundError(f"image_folder path not found: {cfg.path}")
        if not root.is_dir():
            raise ValueError(f"image_folder path must be a directory: {cfg.path}")
        # Expect train/val subdirs, or split from single dir
        if (root / "train").exists():
            train_ds = ImageFolderDataset(str(root / "train"), transform_train)
            val_ds = ImageFolderDataset(str(root / "val"), transform_val) if (root / "val").exists() else None
            test_ds = ImageFolderDataset(str(root / "test"), transform_val) if (root / "test").exists() else None
        else:
            dataset = ImageFolderDataset(str(root), transform_train)
            train_ds, val_ds, test_ds = _split_dataset(dataset, cfg)

    elif cfg.source == DataSource.HUGGINGFACE:
        train_ds, val_ds, test_ds = _build_hf_dataloaders(
            cfg, model_type, model_cfg, transform_train, transform_val,
        )

    elif cfg.source == DataSource.NUMPY:
        if not cfg.path:
            raise ValueError(
                "data.path is required for numpy source — point at a .npz with X/y arrays. "
                "Or call build_dataloaders_from_arrays() directly with in-memory arrays."
            )
        if not Path(cfg.path).exists():
            raise FileNotFoundError(f"numpy file not found: {cfg.path}")
        npz = np.load(cfg.path, allow_pickle=False)
        X = npz["X"] if "X" in npz else npz[npz.files[0]]
        y = npz["y"] if "y" in npz else npz[npz.files[-1]]
        dataset = NumpyDataset(X, y, transform_train)
        train_ds, val_ds, test_ds = _split_dataset(dataset, cfg)

    else:
        raise ValueError(f"Unsupported data source: {cfg.source}")

    if cfg.max_samples and cfg.source not in (DataSource.HUGGINGFACE,):
        train_ds = _subsample(train_ds, int(cfg.max_samples * (1 - cfg.val_split - cfg.test_split)))

    import torch
    _pin = torch.cuda.is_available()  # only beneficial (and supported) on CUDA

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=_pin, drop_last=False,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=val_batch, shuffle=False, num_workers=num_workers, pin_memory=_pin)
        if val_ds else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=val_batch, shuffle=False, num_workers=num_workers)
        if test_ds else None
    )

    return train_loader, val_loader, test_loader


def build_dataloaders_from_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build DataLoaders directly from numpy arrays."""
    train_ds = NumpyDataset(X_train, y_train)
    val_ds = NumpyDataset(X_val, y_val) if X_val is not None else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_ds else None
    return train_loader, val_loader


def _split_dataset(dataset, cfg: DataConfig):
    """Split a dataset into train / val / test."""
    n = len(dataset)
    n_test = int(n * cfg.test_split)
    n_val = int(n * cfg.val_split)
    n_train = n - n_val - n_test

    if n_test > 0:
        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )
    elif n_val > 0:
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        test_ds = None
    else:
        train_ds = dataset
        val_ds = None
        test_ds = None

    return train_ds, val_ds, test_ds


def _subsample(dataset, n: int):
    """Randomly subsample a dataset to n items."""
    if n >= len(dataset):
        return dataset
    indices = torch.randperm(len(dataset))[:n].tolist()
    from torch.utils.data import Subset
    return Subset(dataset, indices)


def _build_hf_dataloaders(cfg: DataConfig, model_type: str, model_cfg,
                          transform_train, transform_val):
    """
    HuggingFace branch — feature-aware. Inspects the dataset's `features`
    dict to decide whether it's image, text, or numeric, then builds the
    appropriate Dataset wrapper. Falls through to torchvision for the
    well-known short names (cifar10, mnist, etc.) so we keep the cheap
    cached loaders.
    """
    name = cfg.dataset_name
    if not name:
        raise ValueError(
            "data.dataset_name is required for huggingface source. "
            "Try 'mnist', 'cifar10', or 'imdb' to start."
        )

    # Short-circuit for the torchvision builtins
    name_lower = name.lower().replace("-", "").replace("_", "")
    if name_lower in _TORCHVISION_DATASETS:
        train_ds = _load_torchvision(name, train=True, transform=transform_train)
        try:
            val_ds = _load_torchvision(name, train=False, transform=transform_val)
            return train_ds, val_ds, None
        except Exception:
            return _split_dataset(train_ds, cfg)

    # Real HuggingFace path
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace dataset support requires the `datasets` package. "
            "Install with: pip install datasets"
        ) from exc

    try:
        load_args = (name,) if not cfg.dataset_config else (name, cfg.dataset_config)
        hf_ds = load_dataset(*load_args, split=cfg.dataset_split)
    except Exception as exc:
        from neural_platform.core.hf_introspect import parse_available_configs
        choices = parse_available_configs(str(exc))
        if choices:
            raise RuntimeError(
                f"Dataset '{name}' has multiple sub-configurations. Pick one "
                f"and set `data.dataset_config` in your config. "
                f"Available: {choices}"
            ) from exc
        raise RuntimeError(
            f"Failed to load HuggingFace dataset '{name}'"
            f"{f' (config={cfg.dataset_config})' if cfg.dataset_config else ''}"
            f" (split={cfg.dataset_split}): {exc}\n"
            f"Hint: try `neural inspect {name}` to see available splits/configurations, "
            f"or pick a different dataset_split (e.g. 'train', 'validation', 'test')."
        ) from exc

    if cfg.max_samples:
        hf_ds = hf_ds.select(range(min(cfg.max_samples, len(hf_ds))))

    schema = inspect_hf_features(hf_ds)
    user_text  = (cfg.text_column or "").strip() or None
    user_label = (cfg.label_column or "").strip() or None

    has_images   = bool(schema["image_columns"])
    has_text     = bool(schema["text_columns"])
    has_audio    = bool(schema.get("audio_columns"))
    has_video    = bool(schema.get("video_columns"))
    has_sequence = bool(schema.get("sequence_columns"))

    # ---- AUDIO ----
    if model_type in ("audio_cnn",) or (has_audio and model_type not in ("transformer", "cnn")):
        if not has_audio:
            raise ValueError(
                f"Dataset '{name}' has no audio column. Available: {schema['columns']}. "
                "If your model is `audio_cnn`, the dataset needs a `datasets.Audio` feature."
            )
        audio_col = _pick_column(None, schema["audio_columns"], hf_ds, kind="audio")
        label_col = _pick_column(user_label,
                                  schema["label_columns"] + schema["numeric_columns"],
                                  hf_ds, kind="label", optional=True)
        sr = getattr(getattr(model_cfg, "audio_cnn", None), "sample_rate", 16000)
        dur = getattr(getattr(model_cfg, "audio_cnn", None), "duration_secs", 2.0)
        ds = HuggingFaceAudioDataset(
            hf_ds, audio_col=audio_col, label_col=label_col,
            target_sample_rate=sr, duration_secs=dur,
        )
        return _split_dataset(ds, cfg)

    # ---- VIDEO ----
    if model_type in ("video_cnn",) or (has_video and model_type not in ("transformer", "cnn")):
        if not has_video:
            raise ValueError(
                f"Dataset '{name}' has no video column. Available: {schema['columns']}. "
                "If your model is `video_cnn`, the dataset needs a `datasets.Video` feature "
                "or a Sequence(Image) frame stack."
            )
        video_col = _pick_column(None, schema["video_columns"], hf_ds, kind="video")
        label_col = _pick_column(user_label,
                                  schema["label_columns"] + schema["numeric_columns"],
                                  hf_ds, kind="label", optional=True)
        v = getattr(model_cfg, "video_cnn", None)
        ds = HuggingFaceVideoDataset(
            hf_ds, video_col=video_col, label_col=label_col,
            num_frames=getattr(v, "num_frames", 16),
            input_height=getattr(v, "input_height", 112),
            input_width=getattr(v, "input_width", 112),
        )
        return _split_dataset(ds, cfg)

    # ---- IMAGE ----
    if model_type == "cnn" or (has_images and not has_text and not has_audio):
        image_col = _pick_column(None, schema["image_columns"], hf_ds, kind="image")
        if image_col is None:
            raise ValueError(
                f"Dataset '{name}' has no image column. Available: {schema['columns']}.\n"
                f"Either pick a different dataset, or set data.source='image_folder' if you "
                f"have local images."
            )
        label_col = _pick_column(user_label,
                                  schema["label_columns"] + schema["numeric_columns"],
                                  hf_ds, kind="label", optional=True)
        ds = HuggingFaceImageDataset(
            hf_ds, image_col=image_col, label_col=label_col,
            transform=transform_train or _default_image_transform(model_cfg),
        )
        return _split_dataset(ds, cfg)

    # ---- TEXT ----
    if model_type == "transformer" or (has_text and not has_images):
        text_col = _pick_column(user_text, schema["text_columns"], hf_ds, kind="text")
        if text_col is None:
            raise ValueError(
                f"Dataset '{name}' has no string column to use as text. "
                f"Available columns: {schema['columns']}. "
                f"Image columns: {schema['image_columns']}, "
                f"audio columns: {schema['audio_columns']}, "
                f"video columns: {schema['video_columns']}.\n"
                "Switch the model to match the dataset's modality."
            )
        label_col = _pick_column(user_label,
                                  schema["label_columns"] + schema["numeric_columns"],
                                  hf_ds, kind="label", optional=True)
        tokenizer = _resolve_text_tokenizer(cfg, model_cfg, model_type)
        max_length = (cfg.transforms or {}).get("text", {}).get("max_length", 128)
        ds = HuggingFaceTextDataset(hf_ds, text_col, label_col,
                                     tokenizer=tokenizer, max_length=max_length)
        return _split_dataset(ds, cfg)

    # ---- TIME-SERIES (typed Sequence(Value(int/float))) ----
    if model_type in ("tcn", "rnn") and has_sequence:
        seq_col = schema["sequence_columns"][0]
        label_col = _pick_column(user_label, schema["label_columns"], hf_ds, kind="label", optional=True)
        ds = NumericSequenceDataset(hf_ds, seq_col, label_col)
        return _split_dataset(ds, cfg)

    # ---- NUMERIC / TABULAR fallback (works for mlp / tabular / rnn / tcn) ----
    # Many real datasets (recommendation logs, sensor dumps) have only typed
    # numeric columns. We can train any of MLP / Tabular / RNN / TCN on this
    # by reshaping each row to the model's expected layout.
    has_numeric = bool(schema["numeric_columns"]) or bool(schema["label_columns"])
    pattern_groups = schema.get("pattern_sequence_groups", []) or []

    if has_numeric and model_type in ("mlp", "tabular", "rnn", "tcn", "transformer"):
        label_candidates = schema["label_columns"] or schema["numeric_columns"]
        label_col = _pick_column(user_label, label_candidates, hf_ds, kind="label")

        # Derive feature columns: prefer pattern-grouped sequence columns when
        # the model is sequence-aware (rnn/tcn), otherwise use all non-label
        # numerics.
        if model_type in ("rnn", "tcn") and pattern_groups:
            # Use the longest pattern group as the time dimension
            best = max(pattern_groups, key=lambda g: g["length"])
            feature_cols = [c for c in best["columns"] if c != label_col]
            ds = NumericRowAsSequenceDataset(
                hf_ds, feature_cols=feature_cols, label_col=label_col,
                reshape="sequence",  # → (T, 1) per sample
            )
        elif model_type in ("rnn", "tcn"):
            # No pattern groups — treat each row as a 1-step sequence
            feature_cols = [c for c in schema["numeric_columns"] if c != label_col]
            ds = NumericRowAsSequenceDataset(
                hf_ds, feature_cols=feature_cols, label_col=label_col,
                reshape="sequence",
            )
        else:
            feature_cols = [c for c in schema["numeric_columns"] if c != label_col]
            ds = NumericTabularDataset(hf_ds, feature_cols=feature_cols, label_col=label_col)

        if not feature_cols:
            raise ValueError(
                f"Dataset '{name}' has no usable feature columns after removing the "
                f"label column '{label_col}'. Pick a different dataset or set "
                f"data.label_column to a different column."
            )
        return _split_dataset(ds, cfg)

    # No path matched — actionable error with full schema
    raise ValueError(
        f"Couldn't auto-pick a modality for dataset '{name}' with model '{model_type}'.\n"
        f"Detected schema:\n"
        f"  numeric: {len(schema['numeric_columns'])} cols (first 10: {schema['numeric_columns'][:10]})\n"
        f"  text:    {schema['text_columns']}\n"
        f"  image:   {schema['image_columns']}\n"
        f"  audio:   {schema['audio_columns']}\n"
        f"  video:   {schema['video_columns']}\n"
        f"  label:   {schema['label_columns']}\n"
        f"  sequence-pattern groups: {[g['prefix'] for g in pattern_groups]}\n"
        f"  other:   {schema['other_columns']}\n"
        f"Pick a model that matches (suggested: mlp / tabular for dense numeric data, "
        f"rnn / tcn for sequence patterns, cnn for image, transformer for text), or "
        f"set data.label_column / data.text_column explicitly."
    )


def _pick_column(user_choice: Optional[str], candidates: list, hf_ds, kind: str,
                 optional: bool = False) -> Optional[str]:
    """
    Resolve a column: user pick wins (with validation); otherwise pick the
    first candidate; otherwise None (if optional) or raise (if not).
    """
    cols = set(getattr(hf_ds, "column_names", []) or [])
    if user_choice:
        if user_choice not in cols:
            raise ValueError(
                f"data.{kind}_column '{user_choice}' is not in the dataset. "
                f"Available columns: {sorted(cols)}. "
                f"Likely {kind} candidates: {candidates}. "
                f"Leave the field blank to let NeuralForge auto-detect."
            )
        return user_choice
    if candidates:
        return candidates[0]
    if optional:
        return None
    raise ValueError(
        f"Could not auto-detect a {kind} column. "
        f"Available columns: {sorted(cols)}. "
        f"Set data.{kind}_column explicitly."
    )


def _default_image_transform(model_cfg):
    """Minimal image transform used when the user didn't supply one."""
    try:
        from torchvision import transforms as T
    except ImportError:
        return None
    if model_cfg is not None and getattr(model_cfg, "cnn", None) is not None:
        h = model_cfg.cnn.input_height
        w = model_cfg.cnn.input_width
        return T.Compose([T.Resize((h, w)), T.ToTensor()])
    return T.ToTensor()


def _autodetect_text_column(hf_ds) -> Optional[str]:
    """Pick the most likely text field from a HuggingFace dataset's columns."""
    candidates = ("text", "sentence", "content", "review", "body", "tweet", "question")
    cols = set(hf_ds.column_names)
    for c in candidates:
        if c in cols:
            return c
    return None


def _resolve_text_tokenizer(cfg: DataConfig, model_cfg, model_type: str):
    """
    Decide which tokenizer to use for HF text datasets.

    Order of preference:
      1. cfg.transforms.text.tokenizer
      2. model.transformer.use_pretrained (matching tokenizer)
      3. bert-base-uncased default
    Returns None if `transformers` is not installed (caller must handle the
    string-input case).
    """
    requested = None
    if cfg.transforms and isinstance(cfg.transforms, dict):
        requested = cfg.transforms.get("text", {}).get("tokenizer")

    if not requested and model_cfg is not None and model_type == "transformer":
        if model_cfg.transformer and model_cfg.transformer.use_pretrained:
            requested = model_cfg.transformer.use_pretrained

    requested = requested or "bert-base-uncased"

    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(requested)
    except ImportError:
        return None
    except Exception:
        # Fall back to the default if the requested name fails to load
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception:
            return None


def get_class_names(loader: DataLoader) -> Optional[List[str]]:
    """
    Try to recover human-readable class names from a built dataloader.
    Returns a list of strings indexed by class id, or None if not derivable.
    """
    if loader is None:
        return None
    ds = loader.dataset
    # Unwrap Subset
    while hasattr(ds, "dataset") and not hasattr(ds, "classes"):
        ds = ds.dataset
    classes = getattr(ds, "classes", None)
    if classes:
        return list(classes)
    # torchvision _ds wrapping
    inner = getattr(ds, "_ds", None)
    if inner is not None and hasattr(inner, "classes"):
        return list(inner.classes)
    return None
