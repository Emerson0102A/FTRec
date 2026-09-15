"""Sparse-aware PCGrad and gradient-conflict diagnostics."""

from __future__ import annotations

import json
import math
import random
import csv
import re
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
    value = _gradient_dot_tensor(left, right, prefixes=prefixes)
    return float(value.detach().cpu().item()) if value is not None else 0.0


def _gradient_dot_tensor(
    left: TaskGradients,
    right: TaskGradients,
    *,
    prefixes: tuple[str, ...] | None = None,
) -> torch.Tensor | None:
    if left.names != right.names:
        raise ValueError("task gradients must use identical parameter ordering")
    total: torch.Tensor | None = None
    for name, left_value, right_value in zip(
        left.names, left.values, right.values, strict=True
    ):
        if not _selected(name, prefixes) or left_value is None or right_value is None:
            continue
        value = _tensor_dot(left_value, right_value)
        total = value if total is None else total + value
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
            dot_tensor = _gradient_dot_tensor(current, peer)
            denominator_tensor = _gradient_dot_tensor(peer, peer)
            if dot_tensor is None or denominator_tensor is None:
                continue
            dot, denominator = torch.stack(
                (dot_tensor, denominator_tensor)
            ).detach().cpu().tolist()
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
    if not tasks:
        return ()
    diagonal = [_gradient_dot_tensor(task, task, prefixes=prefixes) for task in tasks]
    template = next((value for value in diagonal if value is not None), None)
    if template is None:
        return tuple(tuple(float("nan") for _ in tasks) for _ in tasks)
    zero = template.new_zeros(())
    norms = [torch.sqrt(torch.clamp(value if value is not None else zero, min=0)) for value in diagonal]
    values: list[torch.Tensor] = []
    for left_index, left in enumerate(tasks):
        for right_index, right in enumerate(tasks):
            dot = _gradient_dot_tensor(left, right, prefixes=prefixes)
            denominator = norms[left_index] * norms[right_index]
            if dot is None:
                dot = zero
            values.append(
                torch.where(
                    denominator > 0,
                    dot / denominator,
                    torch.full_like(denominator, float("nan")),
                )
            )
    host = torch.stack(values).detach().cpu().tolist()
    width = len(tasks)
    return tuple(
        tuple(float(value) for value in host[start : start + width])
        for start in range(0, len(host), width)
    )


def cosine_and_conflict_matrix(
    tasks: Sequence[TaskGradients],
    *,
    prefixes: tuple[str, ...] | None = None,
    epsilon: float = 1e-12,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Return finite cosine/conflict matrices for one parameter group.

    A pair with a missing or near-zero gradient is defined as neutral.  This
    prevents undefined cosines from contaminating the long-running EMA.
    """
    if not tasks:
        return (), ()
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    diagonal = [_gradient_dot_tensor(task, task, prefixes=prefixes) for task in tasks]
    template = next((value for value in diagonal if value is not None), None)
    if template is None:
        zeros = tuple(tuple(0.0 for _ in tasks) for _ in tasks)
        return zeros, zeros
    zero = template.new_zeros(())
    squared_norms = [value if value is not None else zero for value in diagonal]
    values: list[torch.Tensor] = []
    for left_index, left in enumerate(tasks):
        for right_index, right in enumerate(tasks):
            dot = _gradient_dot_tensor(left, right, prefixes=prefixes)
            denominator = torch.sqrt(
                torch.clamp(squared_norms[left_index], min=0)
                * torch.clamp(squared_norms[right_index], min=0)
            )
            safe = denominator > epsilon
            values.append(
                torch.where(safe, (dot if dot is not None else zero) / denominator, zero)
            )
    host = torch.stack(values).detach().cpu().tolist()
    width = len(tasks)
    cosine = tuple(
        tuple(float(value) for value in host[start : start + width])
        for start in range(0, len(host), width)
    )
    conflict = tuple(
        tuple(max(0.0, -value) for value in row) for row in cosine
    )
    return cosine, conflict


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
    def __init__(
        self,
        path: str | Path,
        domain_names: Sequence[str],
        *,
        ema_beta: float = 0.9,
        pairwise_path: str | Path | None = None,
        summary_path: str | Path | None = None,
        layer_table_path: str | Path | None = None,
    ) -> None:
        if not 0 <= ema_beta < 1:
            raise ValueError("ema_beta must be in [0, 1)")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.domain_names = tuple(domain_names)
        self.ema_beta = ema_beta
        self.pairwise_path = Path(pairwise_path) if pairwise_path else self.path.with_name(
            "gradient_conflict_pairs.csv"
        )
        self.summary_path = Path(summary_path) if summary_path else self.path.with_name(
            "gradient_conflict_summary.json"
        )
        self.layer_table_path = (
            Path(layer_table_path)
            if layer_table_path
            else self.path.with_name("gradient_conflict_by_domain_layer.csv")
        )
        self._cosine_ema: dict[tuple[str, int, int], float] = {}
        self._conflict_ema: dict[tuple[str, int, int], float] = {}
        self._latest_method: str | None = None
        self._latest_seed: int | None = None
        self._groups: tuple[str, ...] = ()
        self._steps_logged = 0

    def _update_ema(
        self, store: dict[tuple[str, int, int], float], key: tuple[str, int, int], value: float
    ) -> float:
        previous = store.get(key)
        current = value if previous is None else self.ema_beta * previous + (1 - self.ema_beta) * value
        store[key] = current
        return current

    def _ema_matrix(
        self, store: Mapping[tuple[str, int, int], float], group: str
    ) -> tuple[tuple[float, ...], ...]:
        width = len(self.domain_names)
        return tuple(
            tuple(float(store.get((group, left, right), 0.0)) for right in range(width))
            for left in range(width)
        )

    def _matrix_summary(
        self, matrix: Sequence[Sequence[float]]
    ) -> tuple[dict[str, float], float]:
        width = len(self.domain_names)
        domain = {
            name: (
                sum(float(matrix[index][peer]) for peer in range(width) if peer != index)
                / (width - 1)
                if width > 1
                else 0.0
            )
            for index, name in enumerate(self.domain_names)
        }
        pairs = [
            float(matrix[left][right])
            for left in range(width)
            for right in range(left + 1, width)
        ]
        return domain, sum(pairs) / len(pairs) if pairs else 0.0

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
        del projected  # projected gradient vectors are intentionally never logged
        if len(raw) != len(self.domain_names):
            raise ValueError("domain names and raw task gradients must have identical lengths")
        raw_groups: dict[str, tuple[tuple[float, ...], ...]] = {}
        conflict_groups: dict[str, tuple[tuple[float, ...], ...]] = {}
        cosine_ema_groups: dict[str, tuple[tuple[float, ...], ...]] = {}
        conflict_ema_groups: dict[str, tuple[tuple[float, ...], ...]] = {}
        domain_conflict: dict[str, dict[str, float]] = {}
        aggregate_conflict: dict[str, float] = {}
        domain_conflict_ema: dict[str, dict[str, float]] = {}
        aggregate_conflict_ema: dict[str, float] = {}
        pair_rows: list[dict[str, object]] = []
        for name, prefixes in selected_groups.items():
            cosine, conflict = cosine_and_conflict_matrix(raw, prefixes=prefixes)
            raw_groups[name] = cosine
            conflict_groups[name] = conflict
            for left in range(len(raw)):
                for right in range(len(raw)):
                    key = (name, left, right)
                    cosine_value = self._update_ema(
                        self._cosine_ema, key, float(cosine[left][right])
                    )
                    conflict_value = self._update_ema(
                        self._conflict_ema, key, float(conflict[left][right])
                    )
                    if left < right:
                        pair_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "epoch": epoch,
                                "step": step,
                                "group": name,
                                "domain_i": self.domain_names[left],
                                "domain_j": self.domain_names[right],
                                "cosine": float(cosine[left][right]),
                                "conflict": float(conflict[left][right]),
                                "cosine_ema": cosine_value,
                                "conflict_ema": conflict_value,
                            }
                        )
            cosine_ema_groups[name] = self._ema_matrix(self._cosine_ema, name)
            conflict_ema_groups[name] = self._ema_matrix(self._conflict_ema, name)
            domain_conflict[name], aggregate_conflict[name] = self._matrix_summary(conflict)
            domain_conflict_ema[name], aggregate_conflict_ema[name] = self._matrix_summary(
                conflict_ema_groups[name]
            )
        record: dict[str, object] = {
            "domain_names": self.domain_names,
            "epoch": epoch,
            "method": method,
            "negative_gradient_ratio": {
                name: negative_pair_ratio(matrix) for name, matrix in raw_groups.items()
            },
            "raw_cosine": raw_groups,
            "raw_conflict": conflict_groups,
            "cosine_ema": cosine_ema_groups,
            "conflict_ema": conflict_ema_groups,
            "domain_conflict": domain_conflict,
            "aggregate_conflict": aggregate_conflict,
            "domain_conflict_ema": domain_conflict_ema,
            "aggregate_conflict_ema": aggregate_conflict_ema,
            "seed": seed,
            "step": step,
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
        fieldnames = (
            "seed", "method", "epoch", "step", "group", "domain_i", "domain_j",
            "cosine", "conflict", "cosine_ema", "conflict_ema",
        )
        write_header = not self.pairwise_path.exists() or self.pairwise_path.stat().st_size == 0
        with self.pairwise_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(pair_rows)
        self._latest_method = method
        self._latest_seed = seed
        self._groups = tuple(selected_groups)
        self._steps_logged += 1
        return record

    def _group_summary(self, group: str) -> dict[str, object]:
        matrix = self._ema_matrix(self._conflict_ema, group)
        domain_conflict, aggregate_conflict = self._matrix_summary(matrix)
        return {
            "aggregate_conflict": aggregate_conflict,
            "domain_conflict": domain_conflict,
        }

    def finalize(self) -> dict[str, object]:
        summary: dict[str, object] = {
            "method": self._latest_method,
            "seed": self._latest_seed,
            "ema_beta": self.ema_beta,
            "steps_logged": self._steps_logged,
            "groups": {group: self._group_summary(group) for group in self._groups},
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        layer_groups: dict[int, dict[str, str]] = {}
        for group in self._groups:
            match = re.fullmatch(r"block_(\d+)_(q|v|qv)", group)
            if match:
                layer_groups.setdefault(int(match.group(1)), {})[match.group(2)] = group
        with self.layer_table_path.open("w", encoding="utf-8", newline="") as stream:
            fieldnames = ("domain", "layer", "q_conflict", "v_conflict", "qv_conflict")
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            summaries = summary["groups"]
            assert isinstance(summaries, dict)
            for layer in sorted(layer_groups):
                for domain in self.domain_names:
                    row: dict[str, object] = {"domain": domain, "layer": layer}
                    for suffix in ("q", "v", "qv"):
                        group = layer_groups[layer].get(suffix)
                        value = 0.0
                        if group is not None:
                            value = float(summaries[group]["domain_conflict"][domain])
                        row[f"{suffix}_conflict"] = value
                    writer.writerow(row)
        return summary
