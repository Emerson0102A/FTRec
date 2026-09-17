from __future__ import annotations

from pathlib import Path

import pytest

from ftrec.data.datasets import TargetExample


def _example(example_id: int, domain: int, seen: frozenset[int]) -> TargetExample:
    return TargetExample(
        example_id=example_id,
        user_id=example_id + 1,
        context_items=(0, 1),
        context_domains=(-1, domain),
        positive_item=domain * 10 + 1,
        target_domain=domain,
        seen_items=seen,
    )


def test_negative_is_unseen_and_from_target_domain() -> None:
    from ftrec.data.sampling import SameDomainNegativeSampler

    example = _example(0, 1, frozenset({11, 12}))
    sampler = SameDomainNegativeSampler({0: (1, 2), 1: (11, 12, 13)}, seed=42)

    assert sampler.sample(example) == 13


def test_negative_sampling_fails_when_domain_catalog_is_exhausted() -> None:
    from ftrec.data.sampling import NegativeSamplingError, SameDomainNegativeSampler

    example = _example(0, 1, frozenset({11, 12}))
    sampler = SameDomainNegativeSampler({1: (11, 12)}, seed=42)

    with pytest.raises(NegativeSamplingError, match="no unseen item"):
        sampler.sample(example)


def test_balanced_batch_manifest_is_deterministic_and_equal_sized(tmp_path: Path) -> None:
    from ftrec.data.sampling import BalancedBatchManifest

    by_domain = {
        domain: tuple(_example(domain * 10 + index, domain, frozenset()) for index in range(domain + 1))
        for domain in range(5)
    }
    first = BalancedBatchManifest.create(by_domain, batch_size=2, steps=3, seed=42)
    second = BalancedBatchManifest.create(by_domain, batch_size=2, steps=3, seed=42)
    first_path = tmp_path / "joint.json"
    second_path = tmp_path / "pcgrad.json"
    first.write(first_path)
    second.write(second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert all(len(step[domain]) == 2 for step in first.steps for domain in range(5))


def test_compact_batch_plan_is_bounded_and_replays_deterministically(tmp_path: Path) -> None:
    from ftrec.data.sampling import BalancedBatchPlan

    by_domain = {
        domain: tuple(_example(domain * 10 + index, domain, frozenset()) for index in range(3))
        for domain in range(5)
    }
    first = BalancedBatchPlan.create(
        by_domain, batch_size=2, total_steps=100_000, seed=42
    )
    second = BalancedBatchPlan.create(
        by_domain, batch_size=2, total_steps=100_000, seed=42
    )
    first_path = tmp_path / "joint.json"
    second_path = tmp_path / "pcgrad.json"
    first.write(first_path)
    second.write(second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.stat().st_size < 2_000
    assert list(first.iter_steps(limit=4)) == list(second.iter_steps(limit=4))


def test_proportional_batch_plan_preserves_total_batch_and_dataset_weights(
    tmp_path: Path,
) -> None:
    from ftrec.data.sampling import ProportionalBatchPlan

    by_domain = {
        0: tuple(_example(index, 0, frozenset()) for index in range(5)),
        1: tuple(_example(10 + index, 1, frozenset()) for index in range(3)),
        2: tuple(_example(20 + index, 2, frozenset()) for index in range(2)),
    }
    first = ProportionalBatchPlan.create(
        by_domain, total_batch_size=10, total_steps=4, seed=42
    )
    second = ProportionalBatchPlan.create(
        by_domain, total_batch_size=10, total_steps=4, seed=42
    )
    path = tmp_path / "proportional.json"
    first.write(path)

    assert first.batch_sizes_by_domain == {0: 5, 1: 3, 2: 2}
    assert list(first.iter_steps()) == list(second.iter_steps())
    assert all(
        {domain: len(ids) for domain, ids in step.items()} == {0: 5, 1: 3, 2: 2}
        for step in first.iter_steps()
    )
    assert '"algorithm":"proportional-shuffle-cycle-v1"' in path.read_text(
        encoding="utf-8"
    )


def test_rejection_sampler_handles_nearly_exhausted_catalog() -> None:
    from ftrec.data.sampling import SameDomainNegativeSampler

    catalog = tuple(range(1, 10_001))
    example = _example(0, 1, frozenset(catalog[:-1]))
    sampler = SameDomainNegativeSampler({1: catalog}, seed=42)

    assert sampler.sample(example) == catalog[-1]


def test_evaluation_candidates_use_stable_user_key_not_example_id() -> None:
    """Catch Single/Joint builders assigning different negatives to one target."""
    from ftrec.data.sampling import build_evaluation_candidates

    first = _example(0, 1, frozenset({11, 12}))
    second = TargetExample(
        example_id=99,
        user_id=first.user_id,
        context_items=first.context_items,
        context_domains=first.context_domains,
        positive_item=first.positive_item,
        target_domain=first.target_domain,
        seen_items=first.seen_items,
    )
    catalog = {1: tuple(range(11, 30))}

    first_map = build_evaluation_candidates(
        (first,), catalog, count=5, evaluation_seed=2026, split_offset=20_000
    )
    second_map = build_evaluation_candidates(
        (second,), catalog, count=5, evaluation_seed=2026, split_offset=20_000
    )

    assert first_map[first.example_id] == second_map[second.example_id]


def test_evaluation_candidates_require_exact_requested_count() -> None:
    """Catch silently evaluating fewer than the paper's 1,000 candidates."""
    from ftrec.data.sampling import (
        NegativeSamplingError,
        build_evaluation_candidates,
    )

    example = _example(0, 1, frozenset({11, 12}))

    with pytest.raises(NegativeSamplingError, match="requested 3 negatives"):
        build_evaluation_candidates(
            (example,),
            {1: (11, 12, 13, 14)},
            count=3,
            evaluation_seed=2026,
            split_offset=20_000,
        )
