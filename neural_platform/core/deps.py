"""
Modality-aware dependency probe.

Every model type / data source has a small set of optional Python packages
it leans on. When one is missing, the user typically hits a confusing
`ImportError` deep inside a DataLoader worker. This module gives us:

  * `requirements_for(model_type, data_source)` — the list of packages
    that are *required* (hard) and *recommended* (soft) for a given config.
  * `check_dependencies(model_type, data_source)` — actually probes whether
    each is importable, returns a structured report.
  * `format_report(report)` — pretty-print for the CLI.

Wired into:
  * `core/validator.py` — surfaces missing required deps as validation
    errors before training starts.
  * `cli/commands.py` — `neural deps` command runs a full audit.
  * `web/app.py` — `/api/deps` for the dashboard's Settings page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

@dataclass
class Dep:
    """One Python package the platform may need."""
    package:    str       # the name passed to `pip install`
    importable: str       # the module name to `import` to verify install
    purpose:    str       # short, user-facing description
    required:   bool = True


# Per-model-type dependencies. PyTorch is always required (the core platform
# can't run without it), so we list it under "core" and don't repeat it.
MODEL_DEPS: Dict[str, List[Dep]] = {
    "mlp":        [],
    "cnn":        [Dep("torchvision", "torchvision", "image transforms + pretrained backbones")],
    "rnn":        [],
    "transformer": [
        Dep("transformers", "transformers", "HuggingFace tokenizers + pretrained encoders", required=False),
    ],
    "audio_cnn": [
        Dep("torchaudio",   "torchaudio",   "audio decoding + resampling + mel-spec"),
        Dep("torchcodec",   "torchcodec",   "robust HF Audio feature decoding", required=False),
        Dep("soundfile",    "soundfile",    "WAV/FLAC fallback decoder", required=False),
        Dep("transformers", "transformers", "for `pretrained: facebook/wav2vec2-base` etc.", required=False),
    ],
    "tcn":      [],
    "tabular":  [],
    "video_cnn": [
        Dep("torchvision", "torchvision",   "video frame decoding via torchvision.io"),
        Dep("av",          "av",            "PyAV — required by some torchvision video readers", required=False),
        Dep("decord",      "decord",        "alternate video reader", required=False),
    ],
    "hf_pipeline": [
        Dep("transformers", "transformers", "the universal HF model wrapper depends on this"),
        Dep("accelerate",   "accelerate",   "speeds up HF model loading and training", required=False),
    ],
}

# Per-data-source dependencies.
DATA_DEPS: Dict[str, List[Dep]] = {
    "synthetic":     [Dep("scikit-learn", "sklearn", "make_classification / make_regression")],
    "csv":           [Dep("pandas", "pandas", "CSV loading + column selection")],
    "image_folder":  [
        Dep("torchvision", "torchvision",   "ImageFolder + image transforms"),
        Dep("Pillow",      "PIL",           "image loading"),
    ],
    "huggingface":   [Dep("datasets", "datasets", "HuggingFace dataset loading")],
    "numpy":         [],
}

# Modality-specific HF dataset feature decoders. When an HF dataset has an
# Audio/Video/Image feature, datasets needs codecs to decode it.
HF_FEATURE_DEPS: Dict[str, List[Dep]] = {
    "audio_feature": [
        Dep("soundfile", "soundfile", "decode HF Audio bytes", required=False),
        Dep("torchcodec", "torchcodec", "decode HF Audio bytes", required=False),
    ],
    "video_feature": [
        Dep("av", "av", "decode HF Video frames via PyAV", required=False),
    ],
}


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

@dataclass
class DepStatus:
    package:    str
    importable: str
    purpose:    str
    required:   bool
    installed:  bool
    version:    Optional[str] = None


@dataclass
class DepReport:
    statuses:           List[DepStatus] = field(default_factory=list)
    missing_required:   List[DepStatus] = field(default_factory=list)
    missing_optional:   List[DepStatus] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required


def requirements_for(model_type: str, data_source: Optional[str] = None) -> List[Dep]:
    """All Dep entries that apply to a given model_type + data_source pair.
    Deduped by `package` so torchvision (used by both CNN and image_folder)
    isn't listed twice."""
    seen: set = set()
    out: List[Dep] = []
    for dep in MODEL_DEPS.get(model_type, []):
        if dep.package in seen:
            continue
        seen.add(dep.package); out.append(dep)
    if data_source:
        for dep in DATA_DEPS.get(data_source, []):
            if dep.package in seen:
                continue
            seen.add(dep.package); out.append(dep)
    return out


def check_dependencies(model_type: str, data_source: Optional[str] = None) -> DepReport:
    """Probe every required + recommended package; return a structured report."""
    report = DepReport()
    for dep in requirements_for(model_type, data_source):
        installed, version = _probe(dep.importable)
        status = DepStatus(
            package=dep.package, importable=dep.importable, purpose=dep.purpose,
            required=dep.required, installed=installed, version=version,
        )
        report.statuses.append(status)
        if not installed:
            (report.missing_required if dep.required else report.missing_optional).append(status)
    return report


def check_all() -> Dict[str, DepReport]:
    """Probe every modality / data source — used by `neural deps`."""
    out: Dict[str, DepReport] = {}
    for mtype in MODEL_DEPS.keys():
        out[f"model:{mtype}"] = check_dependencies(mtype)
    for src in DATA_DEPS.keys():
        out[f"data:{src}"] = check_dependencies("mlp", src)  # mlp has no deps, isolates source
    return out


def _probe(module: str):
    """Try to import a module; return `(installed, version_or_None)`."""
    try:
        import importlib
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", None)
        return True, version
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_report(report: DepReport, indent: str = "  ") -> str:
    """Pretty multi-line summary for the terminal."""
    lines = []
    for s in report.statuses:
        marker = "✓" if s.installed else ("✗" if s.required else "○")
        version = f" v{s.version}" if s.version else ""
        tag = "" if s.required else " (optional)"
        lines.append(f"{indent}{marker} {s.package:<14}{version} — {s.purpose}{tag}")
    return "\n".join(lines) if lines else f"{indent}(no extra deps)"


def install_command(report: DepReport, only_required: bool = False) -> Optional[str]:
    """Return a single `pip install …` line for the missing packages, or None."""
    targets = report.missing_required[:]
    if not only_required:
        targets += report.missing_optional
    if not targets:
        return None
    return "pip install " + " ".join(sorted({d.package for d in targets}))
