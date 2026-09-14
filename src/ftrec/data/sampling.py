"""Deterministic negatives and balanced multi-domain batch plans."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .datasets import TargetExample


class NegativeSamplingError(RuntimeError):
    """Raised when a task has no valid negative candidate."""


class SameDomainNegativeSampler:
    def __init__(self, items_by_domain: Mapping[int, Sequence[int]], seed: int) -> None:
        self.items_by_domain = {
            int(domain): tuple(sorted(int(item) for item in items))
            for domain, items in items_by_domain.items()
        }
        self.random = random.Random(seed)

    def sample(self, example: TargetExample) -> int:
        candidates = tuple(
            item
            for item in self.items_by_domain.get(example.target_domain, ())
            if item not in example.seen_items
        )
        if not candidates:
            raise NegativeSamplingError(
                f"no unseen item in domain {example.target_domain} for user {example.user_id}"
            )
        return self.random.choice(candidates)


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

