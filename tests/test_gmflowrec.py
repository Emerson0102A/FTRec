from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_source(root: Path, *, corrupt_test_prefix: bool = False) -> Path:
    root.mkdir(parents=True)
    start = datetime(2022, 1, 1)
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array(
                    [0, 0, 0, 1, 1, 1, 1, 1, 1], type=pa.int32()
                ),
                "item_id": pa.array([0, 4, 1, 0, 5, 6, 4, 7, 8], type=pa.int64()),
                "domain_id": pa.array([0, 1, 0, 0, 1, 1, 1, 1, 1], type=pa.int64()),
                "timestamp": pa.array(
                    [start + timedelta(seconds=index) for index in range(9)],
                    type=pa.timestamp("us"),
                ),
            }
        ),
        root / "processed.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "item_id": [[0, 4, 1]],
                "domain_id": [[0, 1, 0]],
                "timestamp": [[start, start + timedelta(seconds=1), start + timedelta(seconds=2)]],
            }
        ),
        root / "train_new.parquet",
    )
    valid_items = [0, 5]
    valid_domains = [0, 1]
    valid_times = [start + timedelta(seconds=3), start + timedelta(seconds=4)]
    pq.write_table(
        pa.table(
            {
                "item_id": [valid_items],
                "domain_id": [valid_domains],
                "timestamp": [valid_times],
                "max_time": [valid_times[-1]],
            }
        ),
        root / "valid_new.parquet",
    )
    test_items = [1 if corrupt_test_prefix else 0, 5, 6]
    test_times = [*valid_times, start + timedelta(seconds=5)]
    pq.write_table(
        pa.table(
            {
                "item_id": [test_items],
                "domain_id": [[0, 1, 1]],
                "timestamp": [test_times],
                "max_time": [valid_times[-1]],
            }
        ),
        root / "test_new.parquet",
    )
    return root


def test_import_offsets_item_zero_and_uses_one_target_per_sequence(tmp_path: Path) -> None:
    """Catch padding/item collisions or turning every history event into a target."""
    from ftrec.data.gmflowrec import GMFlowRecImportSettings, import_gmflowrec
    from ftrec.data.datasets import SequenceStore, build_mixed_examples

    source = _write_source(tmp_path / "source")
    output = tmp_path / "processed"
    import_gmflowrec(
        GMFlowRecImportSettings(
            source_dir=source,
            output_dir=output,
            num_eval_negatives=2,
            verify_paper_statistics=False,
            progress=False,
        )
    )

    store = SequenceStore.from_processed(output)
    assert store.items_by_domain == {0: (1, 2), 1: (5, 6, 7, 8, 9)}
    assert store.records[0].item_ids == (1, 5, 2)
    assert store.records[0].splits == ("context", "context", "train")
    assert store.records[1].item_ids == (1, 6, 7)
    assert store.records[1].splits == ("context", "valid", "test")
    assert sum(
        len(build_mixed_examples(store, split="train", target_domain=domain, maxlen=50))
        for domain in store.items_by_domain
    ) == 1


def test_import_rejects_test_sequence_that_does_not_extend_validation(tmp_path: Path) -> None:
    """Catch silently pairing unrelated validation and test users."""
    from ftrec.data.gmflowrec import (
        GMFlowRecDataError,
        GMFlowRecImportSettings,
        import_gmflowrec,
    )

    source = _write_source(tmp_path / "source", corrupt_test_prefix=True)

    with pytest.raises(GMFlowRecDataError, match="does not extend validation"):
        import_gmflowrec(
            GMFlowRecImportSettings(
                source_dir=source,
                output_dir=tmp_path / "processed",
                num_eval_negatives=2,
                verify_paper_statistics=False,
                progress=False,
            )
        )


def test_import_persists_reusable_same_domain_candidate_matrices(tmp_path: Path) -> None:
    """Catch regenerating candidates per model or including future/seen items."""
    from ftrec.data.gmflowrec import GMFlowRecImportSettings, import_gmflowrec
    from ftrec.data.datasets import SequenceStore

    source = _write_source(tmp_path / "source")
    output = tmp_path / "processed"
    settings = GMFlowRecImportSettings(
        source_dir=source,
        output_dir=output,
        num_eval_negatives=2,
        evaluation_seed=2026,
        verify_paper_statistics=False,
        progress=False,
    )
    import_gmflowrec(settings)
    first_bytes = (output / "evaluation" / "test-domain-1.npy").read_bytes()

    store = SequenceStore.from_processed(output)
    candidates = store.evaluation_candidates(
        split="test", domain=1, negative_count=2, evaluation_seed=2026
    )
    assert candidates is not None
    row = tuple(int(value) for value in candidates[0])
    assert row[0] == 7
    assert len(row) == 3
    assert len(set(row)) == 3
    assert set(row[1:]).issubset(store.items_by_domain[1])
    assert set(row[1:]).isdisjoint({6, 7})
    np.testing.assert_array_equal(candidates.batch((0,)), np.asarray([row], dtype=np.int32))

    import_gmflowrec(settings.__class__(**{**settings.__dict__, "force": True}))
    assert (output / "evaluation" / "test-domain-1.npy").read_bytes() == first_bytes


def test_training_candidate_resolution_prefers_the_shared_cache(tmp_path: Path) -> None:
    """Catch individual model runs rebuilding the released-protocol candidates."""
    from ftrec.data.gmflowrec import GMFlowRecImportSettings, import_gmflowrec
    from ftrec.data.datasets import SequenceStore, build_mixed_examples
    from ftrec.data.sampling import resolve_evaluation_candidates

    output = tmp_path / "processed"
    import_gmflowrec(
        GMFlowRecImportSettings(
            source_dir=_write_source(tmp_path / "source"),
            output_dir=output,
            num_eval_negatives=2,
            verify_paper_statistics=False,
            progress=False,
        )
    )
    store = SequenceStore.from_processed(output)
    examples = tuple(
        build_mixed_examples(store, split="test", target_domain=1, maxlen=50)
    )

    candidates = resolve_evaluation_candidates(
        store,
        examples,
        split="test",
        domain=1,
        count=2,
        evaluation_seed=2026,
        split_offset=20_000,
    )

    assert candidates.__class__.__name__ == "CandidateMatrix"
    assert tuple(int(value) for value in candidates[0])[0] == examples[0].positive_item


def test_cached_candidates_select_stable_user_rows_after_single_domain_skips(
    tmp_path: Path,
) -> None:
    """Catch dense example ids shifting rows after ineligible Single users are skipped."""
    from ftrec.data.datasets import CandidateMatrix, TargetExample

    values = np.asarray([[10, 11, 12], [20, 21, 22]], dtype=np.int32)
    user_ids = np.asarray([100, 200], dtype=np.int64)
    np.save(tmp_path / "candidates.npy", values, allow_pickle=False)
    np.save(tmp_path / "user_ids.npy", user_ids, allow_pickle=False)
    matrix = CandidateMatrix(
        tmp_path / "candidates.npy",
        expected_rows=2,
        expected_width=3,
        key_path=tmp_path / "user_ids.npy",
    )
    remaining = (
        TargetExample(
            example_id=0,
            user_id=200,
            context_items=(0, 1),
            context_domains=(-1, 1),
            positive_item=20,
            target_domain=1,
            seen_items=frozenset({1, 20}),
        ),
    )

    selected = matrix.select(remaining)

    assert len(selected) == 1
    np.testing.assert_array_equal(selected[0], values[1])
    np.testing.assert_array_equal(selected.batch((0,)), values[1:2])
