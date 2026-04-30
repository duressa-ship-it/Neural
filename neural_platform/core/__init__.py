from neural_platform.core.config import ExperimentConfig, load_config
from neural_platform.core.registry import registry
from neural_platform.core.experiment import ExperimentTracker

def __getattr__(name):
    if name == "Trainer":
        from neural_platform.core.trainer import Trainer
        return Trainer
    if name == "Evaluator":
        from neural_platform.core.evaluator import Evaluator
        return Evaluator
    raise AttributeError(f"module 'neural_platform.core' has no attribute {name!r}")

__all__ = ["ExperimentConfig", "load_config", "registry", "Trainer", "Evaluator", "ExperimentTracker"]
