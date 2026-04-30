"""Framework adapter factory."""

from neural_platform.core.config import ExperimentConfig
from neural_platform.frameworks.base import FrameworkAdapter


def get_adapter(config: ExperimentConfig) -> FrameworkAdapter:
    """Return the appropriate framework adapter for the configured framework."""
    framework = config.model.framework.value
    if framework == "pytorch":
        from neural_platform.frameworks.pytorch_adapter import PyTorchAdapter
        return PyTorchAdapter(config)
    elif framework == "tensorflow":
        from neural_platform.frameworks.tensorflow_adapter import TensorFlowAdapter
        return TensorFlowAdapter(config)
    elif framework == "jax":
        from neural_platform.frameworks.jax_adapter import JAXAdapter
        return JAXAdapter(config)
    raise ValueError(f"Unknown framework: '{framework}'. Choose: pytorch, tensorflow, jax")
