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
