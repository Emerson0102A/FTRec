"""Post-training diagnostics for content-only SASRec checkpoints."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
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
        self, contexts: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    by_domain: dict[int, dict[str, float | int | str]] = {}
    by_bucket: dict[str, dict[int, dict[str, float | int | str]]] = {
        bucket.label: {} for bucket in buckets
    }
    bucket_counts = {bucket.label: 0 for bucket in buckets}
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
