"""
NeuralForge — JAX/Flax Framework Adapter
Provides graceful availability check; full implementation loaded only when JAX is installed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from neural_platform.core.config import ExperimentConfig
from neural_platform.frameworks.base import FrameworkAdapter


def _check_jax():
    try:
        import jax  # noqa: F401
        import flax  # noqa: F401
        return True
    except ImportError:
        return False


class JAXAdapter(FrameworkAdapter):
    """
    JAX/Flax training backend.
    Install with: pip install neural-platform[jax]
    """

    def __init__(self, config: ExperimentConfig):
        if not _check_jax():
            raise ImportError(
                "JAX and Flax are not installed. "
                "Install with: pip install neural-platform[jax]"
            )
        super().__init__(config)
        import jax
        import flax
        self.jax = jax
        self.flax = flax

    def get_device(self):
        return self.jax.devices()[0]

    def build_model(self) -> Any:
        raise NotImplementedError(
            "JAX adapter: model construction not yet implemented. "
            "Use the PyTorch adapter or contribute a Flax model implementation."
        )

    def build_optimizer(self, model: Any) -> Any:
        try:
            import optax
            opt_cfg = self.config.training.optimizer
            return optax.adamw(opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
        except ImportError:
            raise ImportError("Install optax: pip install optax")

    def build_scheduler(self, optimizer: Any) -> Optional[Any]:
        return None

    def build_loss(self) -> Any:
        raise NotImplementedError("JAX adapter: loss functions not yet implemented.")

    def train_step(self, model, batch, optimizer, loss_fn, scaler=None) -> Tuple[float, Dict]:
        raise NotImplementedError("JAX adapter: train_step not yet implemented.")

    def eval_step(self, model, batch, loss_fn) -> Tuple[float, Dict]:
        raise NotImplementedError("JAX adapter: eval_step not yet implemented.")

    def save_checkpoint(self, model, optimizer, path: str, extra: Dict) -> None:
        raise NotImplementedError("JAX adapter: save_checkpoint not yet implemented.")

    def load_checkpoint(self, path: str) -> Tuple[Any, Dict]:
        raise NotImplementedError("JAX adapter: load_checkpoint not yet implemented.")


def get_adapter(config: ExperimentConfig) -> FrameworkAdapter:
    """Factory: return the appropriate adapter for the configured framework."""
    framework = config.model.framework.value
    if framework == "pytorch":
        from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter
        return PyTorchAdapter(config)
    elif framework == "tensorflow":
        return TensorFlowAdapter(config)
    elif framework == "jax":
        return JAXAdapter(config)
    raise ValueError(f"Unknown framework: '{framework}'")
