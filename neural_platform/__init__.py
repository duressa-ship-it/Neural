"""
NeuralForge — Multi-framework Neural Network Platform
Build, train, and deploy neural networks via config, CLI, or web UI.
"""

__version__ = "0.4.2"
__author__ = "NeuralForge"

def __getattr__(name):
    """Lazy imports — avoid loading torch at import time."""
    if name in ("ExperimentConfig", "load_config"):
        from neural_platform.core.config import ExperimentConfig, load_config
        return {"ExperimentConfig": ExperimentConfig, "load_config": load_config}[name]
    if name == "registry":
        from neural_platform.core.registry import registry
        return registry
    raise AttributeError(f"module 'neural_platform' has no attribute {name!r}")

__all__ = ["ExperimentConfig", "load_config", "registry"]
