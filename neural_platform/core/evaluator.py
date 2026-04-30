"""
NeuralForge Evaluator
Accumulates per-batch metrics across an epoch and computes aggregate statistics.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


class MetricAccumulator:
    """Tracks running mean of scalar metrics over batches."""

    def __init__(self):
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            self._sums[k] += v * n
            self._counts[k] += n

    def compute(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def reset(self):
        self._sums.clear()
        self._counts.clear()


class Evaluator:
    """
    Wraps a trained model and evaluates it on a dataset.
    Supports classification, regression, and sequence tasks.
    """

    def __init__(self, adapter, loss_fn):
        self.adapter = adapter
        self.loss_fn = loss_fn

    def evaluate(self, model, dataloader, phase: str = "val") -> Dict[str, float]:
        """
        Run full evaluation over a dataloader.
        Returns aggregated metrics dict.
        """
        model.eval()
        accumulator = MetricAccumulator()

        for batch in dataloader:
            loss_val, metrics = self.adapter.eval_step(model, batch, self.loss_fn)
            n = _batch_size(batch)
            accumulator.update(metrics, n)

        return accumulator.compute()


def _batch_size(batch) -> int:
    """Infer batch size from a batch object."""
    import torch
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        first = batch[0]
        if isinstance(first, torch.Tensor):
            return first.size(0)
        if isinstance(first, dict):
            v = next(iter(first.values()))
            return v.size(0) if isinstance(v, torch.Tensor) else 1
    return 1
