"""Post-training diagnostics for content-only SASRec checkpoints."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from ftrec.data.datasets import SequenceStore, TargetExample
from ftrec.evaluation.ranking import evaluate_model
from ftrec.models.sasrec import SASRec


METRIC_NAMES = ("HR@5", "NDCG@5", "HR@10", "NDCG@10")
DEFAULT_FREQUENCY_LOWER_BOUNDS = (0, 1, 5, 15, 50, 100)


class DualTowerScoringView(nn.Module):
    """Expose one branch or the fixed fusion of a trained dual-tower model."""

    def __init__(self, model: SASRec, component: str) -> None:
        super().__init__()
        if model.config.item_embedding_mode != "content_dual":
            raise ValueError("tower views require a content_dual checkpoint")
        if component not in {"title", "attribute", "fusion"}:
            raise ValueError("component must be title, attribute, or fusion")
        self.model = model
        self.component = component

    def prepare_evaluation_cache(self, *, chunk_size: int = 4096) -> None:
        assert self.model.title_tower is not None
        assert self.model.attribute_tower is not None
        if self.component in {"title", "fusion"}:
            self.model.title_tower.prepare_evaluation_cache(chunk_size=chunk_size)
        if self.component in {"attribute", "fusion"}:
            self.model.attribute_tower.prepare_evaluation_cache(chunk_size=chunk_size)

    def prepare_scoring(
        self, contexts: torch.Tensor, candidate_ids: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del candidate_ids
        assert self.model.title_tower is not None
        assert self.model.attribute_tower is not None
        if self.component == "title":
            return self.model.title_tower.final_state(contexts)
        if self.component == "attribute":
            return self.model.attribute_tower.final_state(contexts)
        return (
            self.model.title_tower.final_state(contexts),
            self.model.attribute_tower.final_state(contexts),
        )

    def score_prepared(
        self,
        prepared: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert self.model.title_tower is not None
        assert self.model.attribute_tower is not None
        if self.component == "title":
            assert isinstance(prepared, torch.Tensor)
            return self.model.title_tower.score_prepared(prepared, candidate_ids)
        if self.component == "attribute":
            assert isinstance(prepared, torch.Tensor)
            return self.model.attribute_tower.score_prepared(prepared, candidate_ids)
        assert isinstance(prepared, tuple)
        title = self.model.title_tower.score_prepared(prepared[0], candidate_ids)
        attribute = self.model.attribute_tower.score_prepared(
            prepared[1], candidate_ids
        )
        return 0.5 * title + 0.5 * attribute


class FrequencyAdaptiveDualTowerScoringView(DualTowerScoringView):
    """Fuse every candidate using only its train-frequency bucket.

    Candidate-specific weights are deployable at inference time. In contrast,
    assigning one weight from the held-out positive item's bucket would leak the
    target identity into ranking and is therefore deliberately unsupported.
    """

    def __init__(
        self,
        model: SASRec,
        *,
        item_bucket_indices: torch.Tensor,
        attribute_weights: Sequence[float],
    ) -> None:
        super().__init__(model, "fusion")
        weights = torch.as_tensor(tuple(attribute_weights), dtype=torch.float32)
        indices = torch.as_tensor(item_bucket_indices, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() != model.config.num_items + 1:
            raise ValueError("item bucket indices must cover padding and every item")
        if weights.ndim != 1 or not weights.numel():
            raise ValueError("attribute weights must be a non-empty vector")
        if torch.any(indices < 0) or torch.any(indices >= weights.numel()):
            raise ValueError("item bucket index is outside the weight vector")
        if torch.any(weights < 0) or torch.any(weights > 1):
            raise ValueError("attribute weights must lie in [0, 1]")
        self.register_buffer("item_bucket_indices", indices, persistent=False)
        self.register_buffer("attribute_weights", weights, persistent=False)

    def score_prepared(
        self,
        prepared: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(prepared, tuple)
        assert self.model.title_tower is not None
        assert self.model.attribute_tower is not None
        title = self.model.title_tower.score_prepared(prepared[0], candidate_ids)
        attribute = self.model.attribute_tower.score_prepared(
            prepared[1], candidate_ids
        )
        bucket_indices = self.item_bucket_indices[candidate_ids]
        weights = self.attribute_weights[bucket_indices].to(title.dtype)
        return torch.lerp(title, attribute, weights)


def training_item_frequencies(store: SequenceStore) -> Counter[int]:
    """Count interactions only in imported training-cohort sequences.

    Evaluation records contain ``valid`` and ``test`` markers. Excluding those
    records prevents validation/test histories from leaking into popularity
    buckets while still counting every interaction in each training sequence.
    """

    counts: Counter[int] = Counter()
    for record in store.records:
        if "train" in record.splits:
            counts.update(record.item_ids)
    return counts


@dataclass(frozen=True)
class FrequencyBucket:
    label: str
    lower: int
    upper: int | None

    def contains(self, value: int) -> bool:
        return value >= self.lower and (self.upper is None or value < self.upper)


def frequency_buckets(
    lower_bounds: Sequence[int] = DEFAULT_FREQUENCY_LOWER_BOUNDS,
) -> tuple[FrequencyBucket, ...]:
    bounds = tuple(int(value) for value in lower_bounds)
    if not bounds or bounds[0] != 0:
        raise ValueError("frequency lower bounds must start at zero")
    if any(left >= right for left, right in zip(bounds, bounds[1:])):
        raise ValueError("frequency lower bounds must be strictly increasing")
    buckets: list[FrequencyBucket] = []
    for index, lower in enumerate(bounds):
        upper = bounds[index + 1] if index + 1 < len(bounds) else None
        if upper is None:
            label = f"{lower}+"
        elif upper == lower + 1:
            label = str(lower)
        else:
            label = f"{lower}-{upper - 1}"
        buckets.append(FrequencyBucket(label, lower, upper))
    return tuple(buckets)


def split_examples_by_frequency(
    examples: Sequence[TargetExample],
    frequencies: Mapping[int, int],
    buckets: Sequence[FrequencyBucket],
) -> dict[str, tuple[TargetExample, ...]]:
    result: dict[str, list[TargetExample]] = {bucket.label: [] for bucket in buckets}
    for example in examples:
        value = int(frequencies.get(example.positive_item, 0))
        for bucket in buckets:
            if bucket.contains(value):
                result[bucket.label].append(example)
                break
        else:  # pragma: no cover - validated buckets start at zero and end open
            raise RuntimeError(f"no frequency bucket for item frequency {value}")
    return {label: tuple(values) for label, values in result.items()}


def item_frequency_bucket_indices(
    *,
    num_items: int,
    frequencies: Mapping[int, int],
    buckets: Sequence[FrequencyBucket],
) -> torch.Tensor:
    if num_items < 1:
        raise ValueError("num_items must be positive")
    result = torch.empty(num_items + 1, dtype=torch.long)
    for item_id in range(num_items + 1):
        frequency = int(frequencies.get(item_id, 0))
        for index, bucket in enumerate(buckets):
            if bucket.contains(frequency):
                result[item_id] = index
                break
        else:  # pragma: no cover - bucket validation guarantees total coverage
            raise RuntimeError(f"no frequency bucket for item frequency {frequency}")
    return result


def aggregate_domain_metrics(
    by_domain: Mapping[int, Mapping[str, float | int | str]],
) -> dict[str, object]:
    """Return macro-domain and user-weighted summaries."""

    nonempty = [
        metrics
        for metrics in by_domain.values()
        if int(metrics.get("num_eval_users", 0)) > 0
    ]
    evaluated = sum(int(metrics["num_eval_users"]) for metrics in nonempty)
    skipped = sum(int(metrics["num_skipped_users"]) for metrics in nonempty)
    macro = {
        name: sum(float(metrics[name]) for metrics in nonempty) / len(nonempty)
        if nonempty
        else math.nan
        for name in METRIC_NAMES
    }
    micro = {
        name: sum(
            float(metrics[name]) * int(metrics["num_eval_users"])
            for metrics in nonempty
        )
        / evaluated
        if evaluated
        else math.nan
        for name in METRIC_NAMES
    }
    return {
        "macro_domain": macro,
        "micro_user": micro,
        "num_eval_users": evaluated,
        "num_skipped_users": skipped,
        "num_nonempty_domains": len(nonempty),
    }


def evaluate_domains(
    model: object,
    examples_by_domain: Mapping[int, Sequence[TargetExample]],
    items_by_domain: Mapping[int, Sequence[int]],
    candidates_by_domain: Mapping[int, object] | None,
    *,
    protocol: str,
    chunk_size: int,
    batch_size: int,
    device: str | torch.device,
    progress: bool,
    description_prefix: str,
) -> dict[int, dict[str, float | int | str]]:
    by_domain: dict[int, dict[str, float | int | str]] = {}
    for domain, examples in sorted(examples_by_domain.items()):
        candidates = (
            candidates_by_domain[domain] if candidates_by_domain is not None else None
        )
        by_domain[domain] = evaluate_model(
            model,
            examples,
            items_by_domain,
            protocol=protocol,
            sampled_candidates=candidates,
            chunk_size=chunk_size,
            batch_size=batch_size,
            device=device,
            progress=progress,
            description=f"{description_prefix} domain-{domain}",
        )
    return by_domain


@dataclass(frozen=True)
class AdaptiveFusionSelection:
    attribute_weights: tuple[float, ...]
    validation_score: float
    global_sweep: tuple[dict[str, object], ...]
    coordinate_steps: tuple[dict[str, object], ...]
    evaluations: int


def select_frequency_adaptive_weights(
    evaluator: Callable[[tuple[float, ...]], float],
    *,
    bucket_count: int,
    alpha_grid: Sequence[float],
    tunable_buckets: Sequence[bool],
    max_coordinate_rounds: int = 2,
) -> AdaptiveFusionSelection:
    """Select candidate-frequency weights using validation metrics only."""

    grid = tuple(float(value) for value in alpha_grid)
    if bucket_count < 1 or len(tunable_buckets) != bucket_count:
        raise ValueError("tunable_buckets must match the positive bucket count")
    if not grid or any(value < 0 or value > 1 for value in grid):
        raise ValueError("alpha grid must contain values in [0, 1]")
    if len(set(grid)) != len(grid):
        raise ValueError("alpha grid values must be unique")
    if max_coordinate_rounds < 1:
        raise ValueError("max_coordinate_rounds must be positive")

    cache: dict[tuple[float, ...], float] = {}

    def score(weights: tuple[float, ...]) -> float:
        if weights not in cache:
            cache[weights] = float(evaluator(weights))
        return cache[weights]

    global_rows: list[dict[str, object]] = []
    best_weights: tuple[float, ...] | None = None
    best_score = -math.inf
    for alpha in grid:
        weights = (alpha,) * bucket_count
        current = score(weights)
        global_rows.append({"attribute_weight": alpha, "validation_score": current})
        if current > best_score:
            best_weights, best_score = weights, current
    assert best_weights is not None

    coordinate_steps: list[dict[str, object]] = []
    tolerance = 1e-12
    for round_index in range(max_coordinate_rounds):
        changed = False
        for bucket_index, tunable in enumerate(tunable_buckets):
            if not tunable:
                continue
            start_weight = best_weights[bucket_index]
            start_score = best_score
            chosen_weights = best_weights
            chosen_score = best_score
            for alpha in grid:
                candidate = list(best_weights)
                candidate[bucket_index] = alpha
                candidate_tuple = tuple(candidate)
                current = score(candidate_tuple)
                if current > chosen_score + tolerance:
                    chosen_weights, chosen_score = candidate_tuple, current
            best_weights, best_score = chosen_weights, chosen_score
            if best_weights[bucket_index] != start_weight:
                changed = True
            coordinate_steps.append(
                {
                    "round": round_index + 1,
                    "bucket_index": bucket_index,
                    "start_attribute_weight": start_weight,
                    "selected_attribute_weight": best_weights[bucket_index],
                    "start_validation_score": start_score,
                    "selected_validation_score": best_score,
                }
            )
        if not changed:
            break
    return AdaptiveFusionSelection(
        attribute_weights=best_weights,
        validation_score=best_score,
        global_sweep=tuple(global_rows),
        coordinate_steps=tuple(coordinate_steps),
        evaluations=len(cache),
    )


def evaluate_component(
    model: object,
    examples_by_domain: Mapping[int, Sequence[TargetExample]],
    items_by_domain: Mapping[int, Sequence[int]],
    candidates_by_domain: Mapping[int, object] | None,
    *,
    protocol: str,
    chunk_size: int,
    batch_size: int,
    device: str | torch.device,
    progress: bool,
    description_prefix: str,
    frequencies: Mapping[int, int],
    buckets: Sequence[FrequencyBucket],
) -> dict[str, object]:
    by_domain = evaluate_domains(
        model,
        examples_by_domain,
        items_by_domain,
        candidates_by_domain,
        protocol=protocol,
        chunk_size=chunk_size,
        batch_size=batch_size,
        device=device,
        progress=progress,
        description_prefix=description_prefix,
    )
    by_bucket: dict[str, dict[int, dict[str, float | int | str]]] = {
        bucket.label: {} for bucket in buckets
    }
    bucket_counts = {bucket.label: 0 for bucket in buckets}
    for domain, examples in sorted(examples_by_domain.items()):
        candidates = (
            candidates_by_domain[domain] if candidates_by_domain is not None else None
        )
        partitions = split_examples_by_frequency(examples, frequencies, buckets)
        for label, selected in partitions.items():
            bucket_counts[label] += len(selected)
            if not selected:
                continue
            by_bucket[label][domain] = evaluate_model(
                model,
                selected,
                items_by_domain,
                protocol=protocol,
                sampled_candidates=candidates,
                chunk_size=chunk_size,
                batch_size=batch_size,
                device=device,
                progress=False,
                description=f"{description_prefix} {label} domain-{domain}",
            )
    frequency = {}
    for bucket in buckets:
        domain_metrics = by_bucket[bucket.label]
        frequency[bucket.label] = {
            "range": {"lower": bucket.lower, "upper_exclusive": bucket.upper},
            "target_count": bucket_counts[bucket.label],
            "by_domain": {str(key): value for key, value in domain_metrics.items()},
            **aggregate_domain_metrics(domain_metrics),
        }
    return {
        "by_domain": {str(key): value for key, value in by_domain.items()},
        **aggregate_domain_metrics(by_domain),
        "by_train_item_frequency": frequency,
    }
