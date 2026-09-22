import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mdsr_parquet import (
    EvaluationSequenceDataset,
    MDSRParquetData,
    TrainSequenceDataset,
)


def _write_split(path, items, domains):
    pq.write_table(
        pa.table(
            {
                "item_id": pa.array(items, type=pa.list_(pa.int64())),
                "domain_id": pa.array(domains, type=pa.list_(pa.int64())),
            }
        ),
        path,
    )


def _make_data(tmp_path):
    mapping = {
        "user": {"u0": 0, "u1": 1, "u2": 2},
        "item": {f"i{i}": i for i in range(8)},
        "domain_offset": {0: (0, 4), 1: (4, 8)},
    }
    with (tmp_path / "mappings.pkl").open("wb") as handle:
        pickle.dump(mapping, handle)
    _write_split(
        tmp_path / "train_new.parquet",
        [[0, 1, 4, 5], [2, 3, 6]],
        [[0, 0, 1, 1], [0, 0, 1]],
    )
    _write_split(
        tmp_path / "valid_new.parquet",
        [[0, 4, 1], [2, 6]],
        [[0, 1, 0], [0, 1]],
    )
    _write_split(
        tmp_path / "test_new.parquet",
        [[0, 4, 1, 5], [2, 6, 3]],
        [[0, 1, 0, 1], [0, 1, 0]],
    )
    return MDSRParquetData(tmp_path)


def test_loader_preserves_splits_and_only_shifts_item_ids(tmp_path):
    data = _make_data(tmp_path)
    train_items, train_domains = data.train.sequence(0)
    assert train_items.tolist() == [1, 2, 5, 6]
    assert train_domains.tolist() == [0, 0, 1, 1]
    assert data.summary()["train_sequences"] == 2
    assert data.summary()["evaluation_pairing"].startswith("test[row]")


def test_train_examples_are_shifted_and_epoch_deterministic(tmp_path):
    data = _make_data(tmp_path)
    dataset = TrainSequenceDataset(data.train, item_count=8, maxlen=5, seed=7)
    dataset.set_epoch(3)
    sequence, positive, negative, row_id = dataset[0]
    repeated = dataset[0]
    assert row_id == 0
    assert sequence.tolist() == [0, 0, 1, 2, 5]
    assert positive.tolist() == [0, 0, 2, 5, 6]
    assert np.array_equal(negative, repeated[2])
    assert set(negative[negative != 0]).isdisjoint({1, 2, 5, 6})


def test_evaluation_target_and_candidates_follow_each_parquet_row(tmp_path):
    data = _make_data(tmp_path)
    valid = EvaluationSequenceDataset(
        data.valid, item_count=8, maxlen=4, num_negatives=3, seed=11
    )
    test = EvaluationSequenceDataset(
        data.test, item_count=8, maxlen=4, num_negatives=3, seed=11
    )
    valid_sequence, valid_candidates, _ = valid[0]
    test_sequence, test_candidates, _ = test[0]
    assert valid_sequence.tolist() == [0, 0, 1, 5]
    assert valid_candidates[0] == 2
    assert test_sequence.tolist() == [0, 1, 5, 2]
    assert test_candidates[0] == 6
    assert set(valid_candidates[1:]).isdisjoint({1, 5, 2})
    assert set(test_candidates[1:]).isdisjoint({1, 5, 2, 6})
