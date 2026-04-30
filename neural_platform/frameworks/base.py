"""
NeuralForge Framework Adapter Interface
All framework adapters implement this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Tuple

from neural_platform.core.config import ExperimentConfig


class FrameworkAdapter(ABC):
    """
    Abstract interface for a training framework backend.
    Each framework (PyTorch, TF, JAX) implements this to provide
    unified training/inference semantics to the Trainer.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    @abstractmethod
    def build_model(self) -> Any:
        """Build and return the model for this framework."""
        ...

    @abstractmethod
    def build_optimizer(self, model: Any) -> Any:
        """Build and return the optimizer."""
        ...

    @abstractmethod
    def build_scheduler(self, optimizer: Any) -> Optional[Any]:
        """Build and return the LR scheduler (or None)."""
        ...

    @abstractmethod
    def build_loss(self) -> Any:
        """Build and return the loss function."""
        ...

    @abstractmethod
    def train_step(
        self,
        model: Any,
        batch: Any,
        optimizer: Any,
        loss_fn: Any,
        scaler: Optional[Any] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Run a single training step.
        Returns (loss_value, metrics_dict).
        """
        ...

    @abstractmethod
    def eval_step(
        self,
        model: Any,
        batch: Any,
        loss_fn: Any,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Run a single evaluation step.
        Returns (loss_value, metrics_dict).
        """
        ...

    @abstractmethod
    def save_checkpoint(self, model: Any, optimizer: Any, path: str, extra: Dict) -> None:
        """Save model + optimizer state to disk."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> Tuple[Any, Dict]:
        """Load model from checkpoint. Returns (model, metadata_dict)."""
        ...

    @abstractmethod
    def get_device(self) -> Any:
        """Return the device/accelerator being used."""
        ...

    def framework_name(self) -> str:
        return self.__class__.__name__.replace("Adapter", "").lower()
