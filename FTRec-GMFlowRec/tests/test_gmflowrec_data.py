from __future__ import annotations

from gmflowrec_data import GMFlowRecEvaluationDataset, GMFlowRecTrainDataset
from test_mdsr_parquet import _make_data


def test_training_dataset_preserves_item_and_domain_context(tmp_path):
    data = _make_data(tmp_path)
    dataset = GMFlowRecTrainDataset(data.train, maxlen=5)
    items, domains, target, target_domain, row = dataset[0]
    assert items.tolist() == [0, 0, 1, 2, 5]
    assert domains.tolist() == [-1, -1, 0, 0, 1]
    assert (target, target_domain, row) == (6, 1, 0)


def test_evaluation_negatives_are_distinct_unseen_and_same_domain(tmp_path):
    data = _make_data(tmp_path)
    dataset = GMFlowRecEvaluationDataset(
        data.valid, data.metadata, maxlen=4, num_negatives=2, seed=11
    )
    items, domains, target_domain, candidates, _ = dataset[0]
    assert items.tolist() == [0, 0, 1, 5]
    assert domains.tolist() == [-1, -1, 0, 1]
    assert target_domain == 0
    assert candidates[0] == 2
    assert len(set(candidates.tolist())) == 3
    assert set(candidates[1:]).isdisjoint({1, 5, 2})
    assert all(1 <= candidate <= 4 for candidate in candidates[1:])

    repeated = dataset[0][3]
    assert candidates.tolist() == repeated.tolist()
