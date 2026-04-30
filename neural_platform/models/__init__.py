def __getattr__(name):
    if name == "MLP":
        from neural_platform.models.mlp import MLP
        return MLP
    if name == "CNN":
        from neural_platform.models.cnn import CNN
        return CNN
    if name == "RNN":
        from neural_platform.models.rnn import RNN
        return RNN
    if name == "TransformerModel":
        from neural_platform.models.transformer import TransformerModel
        return TransformerModel
    raise AttributeError(f"module 'neural_platform.models' has no attribute {name!r}")

__all__ = ["MLP", "CNN", "RNN", "TransformerModel"]
