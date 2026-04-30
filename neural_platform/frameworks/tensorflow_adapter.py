"""
NeuralForge — TensorFlow/Keras Framework Adapter
Provides graceful availability check; full implementation loaded only when TF is installed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from neural_platform.core.config import ExperimentConfig
from neural_platform.frameworks.base import FrameworkAdapter


def _check_tf():
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except ImportError:
        return False


class TensorFlowAdapter(FrameworkAdapter):
    """
    TensorFlow/Keras training backend.
    Install tensorflow extra: pip install neural-platform[tensorflow]
    """

    def __init__(self, config: ExperimentConfig):
        if not _check_tf():
            raise ImportError(
                "TensorFlow is not installed. "
                "Install it with: pip install neural-platform[tensorflow]"
            )
        super().__init__(config)
        import tensorflow as tf
        self.tf = tf

    def get_device(self):
        gpus = self.tf.config.list_physical_devices("GPU")
        return gpus[0] if gpus else "CPU:0"

    def build_model(self) -> Any:
        """Build a Keras model from config."""
        import tensorflow as tf
        from tensorflow import keras

        cfg = self.config.model
        arch_type = cfg.type.value

        if arch_type == "mlp":
            arch = cfg.mlp
            layers = [keras.Input(shape=(arch.input_size,))]
            for lc in arch.hidden_layers:
                layers_list = [keras.layers.Dense(lc.size, activation=lc.activation)]
                if lc.batch_norm:
                    layers_list.append(keras.layers.BatchNormalization())
                if lc.dropout > 0:
                    layers_list.append(keras.layers.Dropout(lc.dropout))
            model = keras.Sequential([
                keras.Input(shape=(arch.input_size,)),
                *[keras.layers.Dense(lc.size, activation=lc.activation) for lc in arch.hidden_layers],
                keras.layers.Dense(arch.output_size),
            ])
            return model

        raise NotImplementedError(
            f"TensorFlow adapter does not yet support model type '{arch_type}'. "
            "Use the PyTorch adapter for full model support."
        )

    def build_optimizer(self, model: Any) -> Any:
        import tensorflow as tf
        opt_cfg = self.config.training.optimizer
        lr = opt_cfg.lr
        optimizers = {
            "adam": tf.keras.optimizers.Adam(lr),
            "adamw": tf.keras.optimizers.AdamW(lr, weight_decay=opt_cfg.weight_decay),
            "sgd": tf.keras.optimizers.SGD(lr, momentum=opt_cfg.momentum),
            "rmsprop": tf.keras.optimizers.RMSprop(lr),
        }
        return optimizers.get(opt_cfg.type.value, tf.keras.optimizers.Adam(lr))

    def build_scheduler(self, optimizer: Any) -> Optional[Any]:
        return None  # Keras handles scheduling via callbacks

    def build_loss(self) -> Any:
        import tensorflow as tf
        loss_map = {
            "cross_entropy": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "bce": tf.keras.losses.BinaryCrossentropy(from_logits=True),
            "mse": tf.keras.losses.MeanSquaredError(),
            "mae": tf.keras.losses.MeanAbsoluteError(),
        }
        return loss_map.get(self.config.training.loss.value, tf.keras.losses.SparseCategoricalCrossentropy())

    def train_step(self, model, batch, optimizer, loss_fn, scaler=None) -> Tuple[float, Dict]:
        import tensorflow as tf
        inputs, targets = batch
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = loss_fn(targets, outputs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return float(loss.numpy()), {"loss": float(loss.numpy())}

    def eval_step(self, model, batch, loss_fn) -> Tuple[float, Dict]:
        inputs, targets = batch
        outputs = model(inputs, training=False)
        loss = loss_fn(targets, outputs)
        return float(loss.numpy()), {"loss": float(loss.numpy())}

    def save_checkpoint(self, model, optimizer, path: str, extra: Dict) -> None:
        model.save(path)

    def load_checkpoint(self, path: str) -> Tuple[Any, Dict]:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        return model, {}
