"""Shared optimizer, clipping, and early-stopping utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch


@dataclass(frozen=True)
class OptimizerSettings:
    lr: float = 1e-3
    embedding_lr: float | None = None
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.98)


@dataclass
class OptimizerBundle:
    dense: torch.optim.Optimizer | None
    sparse: torch.optim.Optimizer | None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in (self.dense, self.sparse):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in (self.dense, self.sparse):
            if optimizer is not None:
                optimizer.step()

    def state_dict(self) -> dict[str, object]:
        return {
            "dense": self.dense.state_dict() if self.dense is not None else None,
            "sparse": self.sparse.state_dict() if self.sparse is not None else None,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if self.dense is not None and state.get("dense") is not None:
            self.dense.load_state_dict(state["dense"])
        if self.sparse is not None and state.get("sparse") is not None:
            self.sparse.load_state_dict(state["sparse"])


def build_optimizers(
    model: torch.nn.Module, settings: OptimizerSettings
) -> OptimizerBundle:
    if hasattr(model, "optimizer_parameter_groups"):
        groups = model.optimizer_parameter_groups()
        sparse_parameters = [parameter for parameter in groups["sparse"] if parameter.requires_grad]
        dense_parameters = [parameter for parameter in groups["dense"] if parameter.requires_grad]
    else:
        sparse_parameters = []
        dense_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    dense = (
        torch.optim.AdamW(
            dense_parameters,
            lr=settings.lr,
            betas=settings.betas,
            weight_decay=settings.weight_decay,
        )
        if dense_parameters
        else None
    )
    sparse = (
        torch.optim.SparseAdam(
            sparse_parameters,
            lr=settings.embedding_lr or settings.lr,
            betas=settings.betas,
        )
        if sparse_parameters
        else None
    )
    return OptimizerBundle(dense, sparse)


def clip_global_grad_norm(
    parameters: Iterable[torch.nn.Parameter], max_norm: float, epsilon: float = 1e-12
) -> float:
    parameters = tuple(parameter for parameter in parameters if parameter.grad is not None)
    squared = 0.0
    for parameter in parameters:
        gradient = parameter.grad
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        squared += float(torch.sum(values.float().square()).item())
    norm = math.sqrt(squared)
    if max_norm > 0 and norm > max_norm:
        factor = max_norm / (norm + epsilon)
        for parameter in parameters:
            gradient = parameter.grad
            if gradient.is_sparse:
                gradient = gradient.coalesce()
                gradient._values().mul_(factor)
                parameter.grad = gradient
            else:
                gradient.mul_(factor)
    return norm


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_epoch: int | None = None
    best_metric: float = -math.inf
    bad_epochs: int = 0
    should_stop: bool = False

    def update(self, epoch: int, metric: float) -> bool:
        improved = metric > self.best_metric + self.min_delta
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.should_stop = self.bad_epochs >= self.patience
        return improved


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    steps_per_epoch: int
    patience: int


@dataclass
class TrainingHistory:
    epochs: list[dict[str, float | int]] = field(default_factory=list)
    best_epoch: int | None = None
    best_metric: float = -math.inf


def train_epochs(
    settings: TrainingSettings,
    *,
    train_epoch: Callable[[int, int], dict[str, float]],
    validate: Callable[[int], float],
    on_best: Callable[[int, float], None] | None = None,
    on_epoch: Callable[[int, dict[str, float], float], None] | None = None,
) -> TrainingHistory:
    stopping = EarlyStopping(settings.patience)
    history = TrainingHistory()
    for epoch in range(1, settings.epochs + 1):
        aggregate: dict[str, float] = {}
        for step in range(settings.steps_per_epoch):
            metrics = train_epoch(epoch, step)
            for name, value in metrics.items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value)
        aggregate = {
            name: value / settings.steps_per_epoch for name, value in aggregate.items()
        }
        validation = float(validate(epoch))
        improved = stopping.update(epoch, validation)
        history.epochs.append({"epoch": epoch, **aggregate, "validation_ndcg": validation})
        if improved and on_best is not None:
            on_best(epoch, validation)
        if on_epoch is not None:
            on_epoch(epoch, aggregate, validation)
        if stopping.should_stop:
            break
    history.best_epoch = stopping.best_epoch
    history.best_metric = stopping.best_metric
    return history

