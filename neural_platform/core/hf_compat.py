"""
NeuralForge — HuggingFace transformers compatibility shim.

Older ``trust_remote_code`` repositories (Qwen-Audio, some early
LLaVA forks, several 2023-era custom architectures) import names
from ``transformers`` that have since been moved or renamed. The
canonical example:

    # In the model's modeling_*.py shipped on the Hub:
    from transformers.generation import DisjunctiveConstraint

After transformers 4.37 this lives at
``transformers.generation.beam_constraints.DisjunctiveConstraint``
and the legacy import path raises ``ImportError``. The downstream
helper packages (``transformers_stream_generator``) also fail this
way. References:

  * https://github.com/huggingface/transformers/issues/41623
  * https://github.com/huggingface/transformers/issues/28649
  * https://github.com/hiyouga/LlamaFactory/issues/7920

This module restores the old import paths **as aliases** so older
remote code keeps working without us pinning transformers to a
version that wouldn't support newer models (Gemma-3, Whisper-v3,
etc.). It's purely additive — we never overwrite an existing symbol.

Call :func:`install_compat_shims` once at process startup (the
inference server's lifespan hook already does this). Successive
calls are no-ops.

The shim list is intentionally small and explicit. We don't try to
auto-discover every renamed symbol — that would be brittle and
mask real bugs. When users hit a new ImportError from a remote-code
model, we add a single line here.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Dict, List, Tuple

logger = logging.getLogger("neuralforge.hf_compat")


# (old_module_path, attr_name, new_module_path) tuples — the shim
# imports `attr_name` from `new_module_path` and re-binds it on
# `old_module_path` so legacy imports succeed.
_SHIMS: List[Tuple[str, str, str]] = [
    # generation.* moved to generation.beam_constraints.* around 4.37
    ("transformers.generation", "DisjunctiveConstraint",
     "transformers.generation.beam_constraints"),
    ("transformers.generation", "PhrasalConstraint",
     "transformers.generation.beam_constraints"),
    ("transformers.generation", "Constraint",
     "transformers.generation.beam_constraints"),
    # SampleOutput was removed in 4.37; some `trust_remote_code`
    # models still try to import it from transformers.generation_utils.
    ("transformers.generation_utils", "SampleOutput",
     "transformers.generation.utils"),
    ("transformers.generation_utils", "GreedySearchOutput",
     "transformers.generation.utils"),
    ("transformers.generation_utils", "BeamSearchOutput",
     "transformers.generation.utils"),
]


_installed = False


def install_compat_shims() -> Dict[str, str]:
    """Install all known compatibility shims. Idempotent.

    Returns a small dict mapping ``"<old_module>.<attr>"`` → status,
    where status is ``"shimmed"`` (we restored a missing symbol),
    ``"present"`` (no shim needed), or ``"unavailable"`` (the new
    location doesn't expose it either — transformers may have removed
    the symbol entirely). The result is mainly useful for tests and
    /info diagnostics; the side effect is the only thing that matters
    in practice.
    """
    global _installed
    out: Dict[str, str] = {}
    if _installed:
        return out

    try:
        import transformers   # noqa: F401
    except ImportError:
        # No transformers, no shims to install.
        _installed = True
        return out

    for old_module_path, attr, new_module_path in _SHIMS:
        key = f"{old_module_path}.{attr}"
        try:
            old_mod = importlib.import_module(old_module_path)
        except ImportError:
            out[key] = "unavailable"
            continue
        if hasattr(old_mod, attr):
            out[key] = "present"
            continue
        # Symbol missing — try to import it from the new location and
        # bind it on the old module so legacy imports succeed.
        try:
            new_mod = importlib.import_module(new_module_path)
            target = getattr(new_mod, attr)
        except (ImportError, AttributeError):
            out[key] = "unavailable"
            continue
        try:
            setattr(old_mod, attr, target)
            out[key] = "shimmed"
            logger.debug("hf_compat: shimmed %s -> %s.%s",
                         key, new_module_path, attr)
        except Exception:
            out[key] = "unavailable"

    _installed = True
    return out


def reset_for_testing() -> None:
    """Reset the idempotency latch — only intended for the test suite
    so successive ``install_compat_shims`` calls can be observed."""
    global _installed
    _installed = False
    # Also drop the symbols we may have shimmed so tests can verify
    # the absent → shimmed transition.
    for old_module_path, attr, _ in _SHIMS:
        mod = sys.modules.get(old_module_path)
        if mod is not None and hasattr(mod, attr):
            # Only delete if it looks like our shim (the canonical class
            # would re-import on next access; here we just clear so the
            # next install_compat_shims call has work to do).
            try:
                delattr(mod, attr)
            except Exception:
                pass
