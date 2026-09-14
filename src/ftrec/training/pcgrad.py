"""Sparse-aware PCGrad and gradient-conflict diagnostics."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class TaskGradients:
    names: tuple[str, ...]
    values: tuple[torch.Tensor | None, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError("gradient names and values must have identical lengths")


def collect_task_gradients(
    loss: torch.Tensor,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> TaskGradients:
    parameters = tuple(parameter for _, parameter in named_parameters)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return TaskGradients(
        tuple(name for name, _ in named_parameters),
        tuple(gradient.coalesce() if gradient is not None and gradient.is_sparse else gradient for gradient in gradients),
    )


def _selected(name: str, prefixes: tuple[str, ...] | None) -> bool:
    return prefixes is None or any(name.startswith(prefix) for prefix in prefixes)


def _tensor_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.is_sparse != right.is_sparse:
        raise TypeError("matching parameter gradients must share sparse layout")
    if left.is_sparse:
        return (left.coalesce() * right.coalesce()).sum()
    return torch.sum(left * right)


def gradient_dot(
    left: TaskGradients,
    right: TaskGradients,
    *,
    prefixes: tuple[str, ...] | None = None,
) -> float:
    if left.names != right.names:
        raise ValueError("task gradients must use identical parameter ordering")
    total = 0.0
    for name, left_value, right_value in zip(
        left.names, left.values, right.values, strict=True
    ):
        if not _selected(name, prefixes) or left_value is None or right_value is None:
            continue
        total += float(_tensor_dot(left_value, right_value).item())
    return total


def gradient_norm(
    gradients: TaskGradients, *, prefixes: tuple[str, ...] | None = None
) -> float:
    squared = gradient_dot(gradients, gradients, prefixes=prefixes)
    return math.sqrt(max(squared, 0.0))


def _scale(value: torch.Tensor, factor: float) -> torch.Tensor:
    if value.is_sparse:
        value = value.coalesce()
        return torch.sparse_coo_tensor(
            value.indices(),
            value.values() * factor,
            value.shape,
            dtype=value.dtype,
            device=value.device,
        ).coalesce()
    return value * factor


def _add_scaled(
    left: torch.Tensor | None, right: torch.Tensor | None, factor: float
) -> torch.Tensor | None:
    if right is None:
        return left.clone() if left is not None else None
    if left is None:
        return _scale(right, factor)
    if left.is_sparse != right.is_sparse:
        raise TypeError("matching parameter gradients must share sparse layout")
    if not left.is_sparse:
        return left + factor * right
    left = left.coalesce()
    right = right.coalesce()
    return torch.sparse_coo_tensor(
        torch.cat((left.indices(), right.indices()), dim=1),
        torch.cat((left.values(), right.values() * factor), dim=0),
        left.shape,
        dtype=left.dtype,
        device=left.device,
    ).coalesce()


def _clone(gradients: TaskGradients) -> TaskGradients:
    return TaskGradients(
        gradients.names,
        tuple(value.clone().coalesce() if value is not None and value.is_sparse else value.clone() if value is not None else None for value in gradients.values),
    )


def _combine(
    left: TaskGradients, right: TaskGradients, factor: float
) -> TaskGradients:
    if left.names != right.names:
        raise ValueError("task gradients must use identical parameter ordering")
    return TaskGradients(
        left.names,
        tuple(
            _add_scaled(left_value, right_value, factor)
            for left_value, right_value in zip(left.values, right.values, strict=True)
        ),
    )


def project_pcgrad(
    tasks: Sequence[TaskGradients], *, seed: int, step: int
) -> tuple[TaskGradients, ...]:
    projected, _ = project_pcgrad_with_counts(tasks, seed=seed, step=step)
    return projected


def project_pcgrad_with_counts(
    tasks: Sequence[TaskGradients], *, seed: int, step: int
) -> tuple[tuple[TaskGradients, ...], tuple[int, ...]]:
    if not tasks:
        raise ValueError("PCGrad requires at least one task")
    originals = tuple(_clone(task) for task in tasks)
    projected: list[TaskGradients] = []
    projection_counts: list[int] = []
    for task_index, task in enumerate(originals):
        current = _clone(task)
        count = 0
        peers = [index for index in range(len(tasks)) if index != task_index]
        random.Random(seed * 1_000_003 + step * 101 + task_index).shuffle(peers)
        for peer_index in peers:
            peer = originals[peer_index]
            dot = gradient_dot(current, peer)
            denominator = gradient_dot(peer, peer)
            if dot < 0.0 and denominator > 0.0:
                current = _combine(current, peer, -dot / denominator)
                count += 1
        projected.append(current)
        projection_counts.append(count)
    return tuple(projected), tuple(projection_counts)


def mean_gradients(tasks: Sequence[TaskGradients]) -> TaskGradients:
    if not tasks:
        raise ValueError("cannot average an empty task list")
    names = tasks[0].names
    if any(task.names != names for task in tasks[1:]):
        raise ValueError("task gradients must use identical parameter ordering")
    values: list[torch.Tensor | None] = []
    for parameter_index in range(len(names)):
        total: torch.Tensor | None = None
        for task in tasks:
            total = _add_scaled(total, task.values[parameter_index], 1.0)
        values.append(_scale(total, 1.0 / len(tasks)) if total is not None else None)
    return TaskGradients(names, tuple(values))


def assign_mean_gradients(
    named_parameters: Mapping[str, torch.nn.Parameter],
    tasks: Sequence[TaskGradients],
) -> TaskGradients:
    averaged = mean_gradients(tasks)
    for name, value in zip(averaged.names, averaged.values, strict=True):
        parameter = named_parameters[name]
        parameter.grad = value
    return averaged


def cosine_matrix(
    tasks: Sequence[TaskGradients],
    *,
    prefixes: tuple[str, ...] | None = None,
) -> tuple[tuple[float, ...], ...]:
    norms = [gradient_norm(task, prefixes=prefixes) for task in tasks]
    rows: list[tuple[float, ...]] = []
    for left_index, left in enumerate(tasks):
        row: list[float] = []
        for right_index, right in enumerate(tasks):
            denominator = norms[left_index] * norms[right_index]
            if denominator == 0.0:
                row.append(float("nan"))
            else:
                row.append(gradient_dot(left, right, prefixes=prefixes) / denominator)
        rows.append(tuple(row))
    return tuple(rows)


def negative_pair_ratio(matrix: Sequence[Sequence[float]]) -> float:
    negative = total = 0
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            value = float(matrix[left][right])
            if math.isnan(value):
                continue
            total += 1
            negative += value < 0.0
    return negative / total if total else float("nan")


class GradientConflictLogger:
    def __init__(self, path: str | Path, domain_names: Sequence[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.domain_names = tuple(domain_names)

    def record(
        self,
        *,
        method: str,
        seed: int,
        epoch: int,
        step: int,
        raw: Sequence[TaskGradients],
        projected: Sequence[TaskGradients] | None = None,
        projection_counts: Sequence[int] | None = None,
        groups: Mapping[str, tuple[str, ...] | None] | None = None,
    ) -> dict[str, object]:
        selected_groups = groups or {"full": None}
        raw_groups = {
            name: cosine_matrix(raw, prefixes=prefixes)
            for name, prefixes in selected_groups.items()
        }
        record: dict[str, object] = {
            "domain_names": self.domain_names,
            "epoch": epoch,
            "method": method,
            "negative_gradient_ratio": {
                name: negative_pair_ratio(matrix) for name, matrix in raw_groups.items()
            },
            "raw_cosine": raw_groups,
            "seed": seed,
            "step": step,
        }
        if projected is not None:
            record["projected_cosine"] = {
                name: cosine_matrix(projected, prefixes=prefixes)
                for name, prefixes in selected_groups.items()
            }
        if projection_counts is not None:
            record["projection_counts"] = tuple(int(value) for value in projection_counts)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=True,
                )
                + "\n"
            )
        return record
