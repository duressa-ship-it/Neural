"""
NeuralForge Component Registry
A central registry for models, optimizers, schedulers, losses, and transforms.
Components register themselves with decorators; the Trainer resolves them by name.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """
    A generic component registry that maps string keys to classes or factory functions.

    Usage:
        @registry.register("model", "mlp")
        class MLP(nn.Module): ...

        model_cls = registry.get("model", "mlp")
        model = model_cls(**params)
    """

    def __init__(self):
        self._stores: Dict[str, Dict[str, Any]] = {}

    def _get_store(self, category: str) -> Dict[str, Any]:
        if category not in self._stores:
            self._stores[category] = {}
        return self._stores[category]

    def register(self, category: str, name: str) -> Callable:
        """Decorator to register a class or function under category/name."""
        def decorator(obj: Any) -> Any:
            store = self._get_store(category)
            if name in store:
                raise ValueError(
                    f"'{name}' is already registered under '{category}'. "
                    f"Existing: {store[name].__name__}, New: {obj.__name__}"
                )
            store[name] = obj
            return obj
        return decorator

    def get(self, category: str, name: str) -> Any:
        """Retrieve a registered component by category and name."""
        store = self._get_store(category)
        if name not in store:
            available = list(store.keys())
            raise KeyError(
                f"No '{name}' registered under '{category}'. "
                f"Available: {available}"
            )
        return store[name]

    def list(self, category: str) -> Dict[str, Any]:
        """List all registered components in a category."""
        return dict(self._get_store(category))

    def categories(self) -> list:
        """List all registered categories."""
        return list(self._stores.keys())

    def has(self, category: str, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._get_store(category)


# Global singleton registry
registry = Registry()


# Category constants for type-safe access
MODEL = "model"
OPTIMIZER = "optimizer"
SCHEDULER = "scheduler"
LOSS = "loss"
TRANSFORM = "transform"
DATASET = "dataset"
CALLBACK = "callback"
METRIC = "metric"
