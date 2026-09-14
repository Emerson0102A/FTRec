from __future__ import annotations

from ftrec.data.datasets import (
    SequenceRecord,
    SequenceStore,
    build_mixed_examples,
    build_single_domain_examples,
)


def _store() -> SequenceStore:
    return SequenceStore(
        records=(
            SequenceRecord(
                user_id=1,
                item_ids=(10, 20, 11, 21, 22),
                domain_ids=(0, 1, 0, 1, 1),
                timestamps=(1, 2, 3, 4, 5),
                splits=("train", "train", "train", "valid", "test"),
            ),
        ),
        items_by_domain={0: (10, 11), 1: (20, 21, 22, 23)},
    )


def test_mixed_target_uses_cross_domain_context() -> None:
    examples = build_mixed_examples(_store(), split="valid", target_domain=1, maxlen=4)

    assert len(examples) == 1
    assert examples[0].context_items == (0, 10, 20, 11)
    assert examples[0].context_domains == (-1, 0, 1, 0)
    assert examples[0].positive_item == 21


def test_single_domain_filters_context_without_changing_target_cohort() -> None:
    examples = build_single_domain_examples(
        _store(), split="valid", domain=1, maxlen=4
    )

    assert len(examples) == 1
    assert examples[0].context_items == (0, 0, 0, 20)
    assert examples[0].positive_item == 21


def test_single_domain_records_missing_domain_history_as_skipped() -> None:
    store = SequenceStore(
        records=(
            SequenceRecord(2, (10, 20), (0, 1), (1, 2), ("train", "valid")),
        ),
        items_by_domain={0: (10,), 1: (20, 21)},
    )

    examples = build_single_domain_examples(store, split="valid", domain=1, maxlen=3)

    assert examples == []
    assert store.last_build_skipped == 1

