"""
NeuralForge — PyTorch Framework Adapter
Full PyTorch implementation of the FrameworkAdapter interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from neural_platform.core.config import (
    ExperimentConfig, LossFunction, Optimizer, Scheduler, Task
)
from neural_platform.core.registry import registry, MODEL
from neural_platform.frameworks.base import FrameworkAdapter


def _resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto', 'cpu', 'cuda', 'mps', 'cuda:N' to a torch.device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = cfg.optimizer
    if opt.type == Optimizer.ADAM:
        return torch.optim.Adam(params, lr=opt.lr, betas=tuple(opt.betas),
                                eps=opt.eps, weight_decay=opt.weight_decay)
    elif opt.type == Optimizer.ADAMW:
        return torch.optim.AdamW(params, lr=opt.lr, betas=tuple(opt.betas),
                                 eps=opt.eps, weight_decay=opt.weight_decay)
    elif opt.type == Optimizer.SGD:
        return torch.optim.SGD(params, lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    elif opt.type == Optimizer.RMSPROP:
        return torch.optim.RMSprop(params, lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.type == Optimizer.ADAGRAD:
        return torch.optim.Adagrad(params, lr=opt.lr, weight_decay=opt.weight_decay)
    raise ValueError(f"Unknown optimizer: {opt.type}")


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg) -> Optional[Any]:
    sched_cfg = cfg.scheduler
    num_epochs = cfg.num_epochs
    t_max = sched_cfg.t_max or num_epochs

    if sched_cfg.type == Scheduler.NONE:
        return None
    elif sched_cfg.type == Scheduler.COSINE:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=sched_cfg.min_lr
        )
    elif sched_cfg.type == Scheduler.STEP:
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma
        )
    elif sched_cfg.type == Scheduler.EXPONENTIAL:
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=sched_cfg.gamma
        )
    elif sched_cfg.type == Scheduler.PLATEAU:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=sched_cfg.patience,
            factor=sched_cfg.gamma, min_lr=sched_cfg.min_lr
        )
    elif sched_cfg.type == Scheduler.WARMUP_COSINE:
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=sched_cfg.warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=t_max - sched_cfg.warmup_steps, eta_min=sched_cfg.min_lr)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[sched_cfg.warmup_steps])
    raise ValueError(f"Unknown scheduler: {sched_cfg.type}")


def _build_loss(cfg: ExperimentConfig) -> nn.Module:
    loss_type = cfg.training.loss
    if loss_type == LossFunction.CROSS_ENTROPY:
        return nn.CrossEntropyLoss()
    elif loss_type == LossFunction.BCE:
        return nn.BCEWithLogitsLoss()
    elif loss_type == LossFunction.MSE:
        return nn.MSELoss()
    elif loss_type == LossFunction.MAE:
        return nn.L1Loss()
    elif loss_type == LossFunction.HUBER:
        return nn.HuberLoss()
    elif loss_type == LossFunction.NLL:
        return nn.NLLLoss()
    raise ValueError(f"Unknown loss: {loss_type}")


def _compute_metrics(preds: torch.Tensor, targets: torch.Tensor, task: Task) -> Dict[str, float]:
    """Compute task-appropriate metrics."""
    metrics = {}
    with torch.no_grad():
        if task in (Task.CLASSIFICATION, Task.IMAGE_CLASSIFICATION, Task.TEXT_CLASSIFICATION):
            if preds.dim() > 1:
                predicted = preds.argmax(dim=1)
            else:
                predicted = (torch.sigmoid(preds) > 0.5).long()
            correct = (predicted == targets).float()
            metrics["accuracy"] = correct.mean().item()
        elif task in (Task.REGRESSION,):
            mse = ((preds - targets.float()) ** 2).mean()
            metrics["mse"] = mse.item()
            metrics["rmse"] = mse.sqrt().item()
    return metrics


def _ensure_models_registered() -> None:
    """Import every model module so all @registry.register decorators have run.

    Single source of truth — both build_model and load_checkpoint route
    through this. Add new model files here when extending model types.
    """
    import neural_platform.models.mlp          # noqa: F401
    import neural_platform.models.cnn          # noqa: F401
    import neural_platform.models.rnn          # noqa: F401
    import neural_platform.models.transformer  # noqa: F401
    import neural_platform.models.audio        # noqa: F401
    import neural_platform.models.tcn          # noqa: F401
    import neural_platform.models.tabular      # noqa: F401
    import neural_platform.models.video        # noqa: F401
    import neural_platform.models.hf_pipeline  # noqa: F401


class PyTorchAdapter(FrameworkAdapter):
    """Full PyTorch implementation of the NeuralForge training backend."""

    def build_model(self) -> nn.Module:
        _ensure_models_registered()
        model_cfg = self.config.model
        # The HF pipeline wrapper needs to know the resolved task to pick the
        # right `transformers.Auto*` class. We set it on the model_cfg as a
        # private attribute so the model's __init__ can grab it without us
        # changing every model's signature.
        if model_cfg.type.value == "hf_pipeline":
            model_cfg._resolved_task = (
                self.config.training.pipeline_task
                or self.config.training.task.value
            )
        model_cls = registry.get(MODEL, model_cfg.type.value)
        model = model_cls(model_cfg)
        device = self.get_device()
        return model.to(device)

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return _build_optimizer(model, self.config.training)

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        return _build_scheduler(optimizer, self.config.training)

    def build_loss(self) -> nn.Module:
        return _build_loss(self.config)

    def get_device(self) -> torch.device:
        return _resolve_device(self.config.training.device)

    def make_scaler(self) -> Optional[GradScaler]:
        if self.config.training.mixed_precision and torch.cuda.is_available():
            return GradScaler("cuda")
        return None

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scaler: Optional[GradScaler] = None,
    ) -> Tuple[float, Dict[str, float]]:
        device = self.get_device()
        task = self.config.training.task

        # Unpack batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError("Batch must be a (inputs, targets) tuple")

        inputs = _to_device(inputs, device)
        targets = _to_device(targets, device)

        use_amp = scaler is not None
        with autocast("cuda", enabled=use_amp):
            outputs = _forward(model, inputs)
            loss = loss_fn(outputs, targets)

        acc_steps = self.config.training.accumulation_steps
        loss = loss / acc_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = _compute_metrics(outputs.detach(), targets, task)
        metrics["loss"] = (loss * acc_steps).item()
        return loss.item() * acc_steps, metrics

    def optimizer_step(self, model, optimizer, scaler=None):
        """Call after accumulation_steps train_steps."""
        grad_clip = self.config.training.optimizer.grad_clip
        if grad_clip:
            if scaler:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    def eval_step(
        self,
        model: nn.Module,
        batch: Any,
        loss_fn: nn.Module,
    ) -> Tuple[float, Dict[str, float]]:
        device = self.get_device()
        task = self.config.training.task

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError("Batch must be a (inputs, targets) tuple")

        inputs = _to_device(inputs, device)
        targets = _to_device(targets, device)

        with torch.no_grad():
            outputs = _forward(model, inputs)
            loss = loss_fn(outputs, targets)

        metrics = _compute_metrics(outputs, targets, task)
        metrics["loss"] = loss.item()
        return loss.item(), metrics

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str,
        extra: Dict,
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": self.config.model.model_dump(),
                **extra,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> Tuple[nn.Module, Dict]:
        _ensure_models_registered()
        payload = torch.load(path, map_location=self.get_device(), weights_only=False)
        from neural_platform.core.config import ModelConfig
        model_cfg = ModelConfig.model_validate(payload["model_config"])
        model_cls = registry.get(MODEL, model_cfg.type.value)
        cfg_copy = self.config.model_copy()
        cfg_copy.model = model_cfg
        model = model_cls(model_cfg).to(self.get_device())
        model.load_state_dict(payload["state_dict"])
        meta = {k: v for k, v in payload.items() if k not in ("state_dict", "optimizer_state", "model_config")}
        return model, meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(obj, device):
    """Recursively move tensors to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


def _forward(model: nn.Module, inputs):
    """
    Call model.forward with the right signature depending on input type.

    - dict      → model(**inputs) — used by transformer (input_ids, attention_mask)
    - Tensor    → model(inputs)   — the common case
    - list/tuple of Tensors of equal shape → torch.stack and treat as one tensor
    - list/tuple of mixed/non-Tensor items → fail with a clear modality error,
      *not* an opaque "takes 2 positional arguments but N were given" trace
    """
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, torch.Tensor):
        return model(inputs)
    if isinstance(inputs, (list, tuple)):
        # Common cause: a HF text dataset paired with an image/MLP model — the
        # default collate fn yields a tuple of strings. Catch this early and
        # tell the user exactly what's wrong.
        non_tensors = [type(x).__name__ for x in inputs if not isinstance(x, torch.Tensor)]
        if non_tensors:
            sample = non_tensors[:3]
            raise RuntimeError(
                f"Model expects a Tensor input but got a {type(inputs).__name__} of "
                f"non-tensor items (saw {sample}). This usually means the dataset's "
                f"modality doesn't match the model — e.g. a CNN/MLP pointed at a text "
                f"dataset, or a transformer pointed at images. Check `data.source` and "
                f"`data.text_column` against your `model.type`."
            )
        if all(isinstance(x, torch.Tensor) for x in inputs) and len({tuple(x.shape) for x in inputs}) == 1:
            return model(torch.stack(list(inputs), dim=0))
        # Last resort: assume the user really wants multi-positional forward
        return model(*inputs)
    raise RuntimeError(
        f"Unsupported batch input type {type(inputs).__name__} — expected "
        f"Tensor, dict, or a list/tuple of Tensors."
    )
