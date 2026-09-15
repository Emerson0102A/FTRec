"""Deterministic negatives and balanced multi-domain batch plans."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .datasets import CandidateMatrix, SequenceStore, TargetExample


class NegativeSamplingError(RuntimeError):
    """Raised when a task has no valid negative candidate."""


class SameDomainNegativeSampler:
    def __init__(self, items_by_domain: Mapping[int, Sequence[int]], seed: int) -> None:
        self.items_by_domain = {
            int(domain): tuple(sorted(int(item) for item in items))
            for domain, items in items_by_domain.items()
        }
        self.random = random.Random(seed)

    def sample(
        self, example: TargetExample, *, rng: random.Random | None = None
    ) -> int:
        catalog = self.items_by_domain.get(example.target_domain, ())
        generator = rng or self.random
        if not catalog:
            raise NegativeSamplingError(
                f"no unseen item in domain {example.target_domain} for user {example.user_id}"
            )
        # Rejection sampling is O(1) for the normal sparse-user case. The
        # bounded exhaustive fallback guarantees termination for dense users.
        for _ in range(64):
            candidate = catalog[generator.randrange(len(catalog))]
            if candidate not in example.seen_items:
                return candidate
        start = generator.randrange(len(catalog))
        for offset in range(len(catalog)):
            candidate = catalog[(start + offset) % len(catalog)]
            if candidate not in example.seen_items:
                return candidate
        raise NegativeSamplingError(
            f"no unseen item in domain {example.target_domain} for user {example.user_id}"
        )

    def sample_many(
        self,
        example: TargetExample,
        count: int,
        *,
        rng: random.Random | None = None,
    ) -> tuple[int, ...]:
        generator = rng or self.random
        catalog = self.items_by_domain.get(example.target_domain, ())
        selected: list[int] = []
        selected_set: set[int] = set()
        attempts = max(64, count * 8)
        for _ in range(attempts):
            if len(selected) >= count or not catalog:
                break
            candidate = catalog[generator.randrange(len(catalog))]
            if candidate not in example.seen_items and candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
        if len(selected) < count and catalog:
            start = generator.randrange(len(catalog))
            for offset in range(len(catalog)):
                candidate = catalog[(start + offset) % len(catalog)]
                if candidate not in example.seen_items and candidate not in selected_set:
                    selected.append(candidate)
                    selected_set.add(candidate)
                    if len(selected) >= count:
                        break
        return tuple(selected)


def build_evaluation_candidates(
    examples: Sequence[TargetExample],
    items_by_domain: Mapping[int, Sequence[int]],
    *,
    count: int,
    evaluation_seed: int,
    split_offset: int,
) -> dict[int, tuple[int, ...]]:
    """Build fixed same-domain candidates keyed by each local example id."""
    if count < 1:
        raise ValueError("evaluation negative count must be positive")
    sampler = SameDomainNegativeSampler(items_by_domain, evaluation_seed)
    candidates: dict[int, tuple[int, ...]] = {}
    for example in examples:
        stable_seed = (
            evaluation_seed * 1_000_003
            + split_offset * 10_007
            + example.target_domain * 1009
            + example.user_id * 101
            + example.positive_item
        )
        negatives = sampler.sample_many(
            example, count, rng=random.Random(stable_seed)
        )
        if len(negatives) != count:
            raise NegativeSamplingError(
                f"requested {count} negatives for user {example.user_id} in domain "
                f"{example.target_domain}, but only {len(negatives)} unseen items exist"
            )
        candidates[example.example_id] = (example.positive_item, *negatives)
    return candidates


def resolve_evaluation_candidates(
    store: SequenceStore,
    examples: Sequence[TargetExample],
    *,
    split: str,
    domain: int,
    count: int,
    evaluation_seed: int,
    split_offset: int,
) -> CandidateMatrix | dict[int, tuple[int, ...]]:
    """Reuse an imported fixed matrix, falling back for synthetic/legacy data."""
    cached = store.evaluation_candidates(
        split=split,
        domain=domain,
        negative_count=count,
        evaluation_seed=evaluation_seed,
    )
    if cached is not None:
        return cached.select(examples)
    return build_evaluation_candidates(
        examples,
        store.items_by_domain,
        count=count,
        evaluation_seed=evaluation_seed,
        split_offset=split_offset,
    )


def evaluation_candidate_manifest(
    *, count: int, evaluation_seed: int
) -> dict[str, object]:
    """Describe the deterministic recipe without duplicating huge candidate files."""
    return {
        "algorithm": "same-domain-user-target-v1",
        "candidates_per_user": count + 1,
        "evaluation_seed": evaluation_seed,
        "negative_count": count,
        "positive_position": 0,
        "scope": "same_domain",
        "split_offsets": {"test": 20_000, "validation": 10_000},
    }


@dataclass(frozen=True)
class BalancedBatchPlan:
    """Compact deterministic batch recipe; identifiers are generated lazily."""

    seed: int
    batch_size: int
    total_steps: int
    example_ids_by_domain: dict[int, tuple[int, ...]]

    @classmethod
    def create(
        cls,
        examples_by_domain: Mapping[int, Sequence[TargetExample]],
        *,
        batch_size: int,
        total_steps: int,
        seed: int,
    ) -> "BalancedBatchPlan":
        if batch_size < 1 or total_steps < 1:
            raise ValueError("batch_size and total_steps must be positive")
        identifiers = {
            int(domain): tuple(example.example_id for example in examples)
            for domain, examples in sorted(examples_by_domain.items())
        }
        empty = [domain for domain, values in identifiers.items() if not values]
        if empty:
            raise ValueError(f"domains have no training examples: {empty}")
        return cls(seed, batch_size, total_steps, identifiers)

    def iter_steps(self, *, limit: int | None = None):
        count = self.total_steps if limit is None else min(limit, self.total_steps)
        states: dict[int, tuple[list[int], int, random.Random]] = {}
        for domain, values in sorted(self.example_ids_by_domain.items()):
            ids = list(values)
            generator = random.Random(self.seed * 1009 + domain)
            generator.shuffle(ids)
            states[domain] = (ids, 0, generator)
        for _ in range(count):
            step: dict[int, tuple[int, ...]] = {}
            for domain in sorted(states):
                ids, cursor, generator = states[domain]
                selected: list[int] = []
                while len(selected) < self.batch_size:
                    if cursor >= len(ids):
                        generator.shuffle(ids)
                        cursor = 0
                    selected.append(ids[cursor])
                    cursor += 1
                states[domain] = (ids, cursor, generator)
                step[domain] = tuple(selected)
            yield step

    def to_dict(self) -> dict[str, object]:
        domains = {}
        for domain, ids in sorted(self.example_ids_by_domain.items()):
            digest = hashlib.sha256()
            for start in range(0, len(ids), 10_000):
                block = ",".join(str(value) for value in ids[start : start + 10_000])
                digest.update(block.encode("ascii"))
                digest.update(b",")
            domains[str(domain)] = {"count": len(ids), "sha256": digest.hexdigest()}
        return {
            "algorithm": "shuffle-cycle-v1",
            "batch_size": self.batch_size,
            "domains": domains,
            "seed": self.seed,
            "total_steps": self.total_steps,
            "version": 2,
        }

    def write(self, path: str | Path) -> None:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        Path(path).write_text(payload + "\n", encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class BalancedBatchManifest:
    seed: int
    batch_size: int
    steps: tuple[dict[int, tuple[int, ...]], ...]

    @classmethod
    def create(
        cls,
        examples_by_domain: Mapping[int, Sequence[TargetExample]],
        *,
        batch_size: int,
        steps: int,
        seed: int,
    ) -> "BalancedBatchManifest":
        if batch_size < 1 or steps < 1:
            raise ValueError("batch_size and steps must be positive")
        domain_ids = sorted(examples_by_domain)
        states: dict[int, dict[str, object]] = {}
        for domain in domain_ids:
            identifiers = [example.example_id for example in examples_by_domain[domain]]
            if not identifiers:
                raise ValueError(f"domain {domain} has no training examples")
            generator = random.Random(seed * 1009 + domain)
            generator.shuffle(identifiers)
            states[domain] = {"ids": identifiers, "cursor": 0, "rng": generator}
        manifest_steps: list[dict[int, tuple[int, ...]]] = []
        for _ in range(steps):
            step: dict[int, tuple[int, ...]] = {}
            for domain in domain_ids:
                state = states[domain]
                selected: list[int] = []
                while len(selected) < batch_size:
                    identifiers = state["ids"]
                    cursor = int(state["cursor"])
                    if cursor >= len(identifiers):
                        state["rng"].shuffle(identifiers)
                        cursor = 0
                    selected.append(identifiers[cursor])
                    state["cursor"] = cursor + 1
                step[domain] = tuple(selected)
            manifest_steps.append(step)
        return cls(seed, batch_size, tuple(manifest_steps))

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "seed": self.seed,
            "steps": [
                {str(domain): list(ids) for domain, ids in sorted(step.items())}
                for step in self.steps
            ],
        }

    def write(self, path: str | Path) -> None:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        Path(path).write_text(payload + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def read(cls, path: str | Path) -> "BalancedBatchManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            seed=int(value["seed"]),
            batch_size=int(value["batch_size"]),
            steps=tuple(
                {int(domain): tuple(ids) for domain, ids in step.items()}
                for step in value["steps"]
            ),
        )
