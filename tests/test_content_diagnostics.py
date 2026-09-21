from __future__ import annotations

import pytest
import torch

from ftrec.analysis.content_diagnostics import (
    DualTowerScoringView,
    aggregate_domain_metrics,
    frequency_buckets,
    split_examples_by_frequency,
    training_item_frequencies,
)
from ftrec.attributes.artifacts import create_embedding_arrays, finish_artifact
from ftrec.data.datasets import SequenceRecord, SequenceStore, TargetExample
from ftrec.models.sasrec import SASRec, SASRecConfig


def _dual_model(tmp_path) -> SASRec:
    title, attributes, present = create_embedding_arrays(
        tmp_path,
        item_count=4,
        attribute_count=3,
        embedding_dim=8,
        dtype="float32",
    )
    for item_id in range(1, 5):
        title[item_id] = item_id
        attributes[item_id] = item_id
        present[item_id] = True
    finish_artifact(
        tmp_path,
        provider="test",
        item_count=4,
        attribute_count=3,
        embedding_dim=8,
        catalog_sha256="catalog",
        title=title,
        attributes=attributes,
        present=present,
        provider_config={"processed_items": 4},
    )
    return SASRec(
        SASRecConfig(
            num_items=4,
            hidden_size=4,
            num_blocks=1,
            num_heads=1,
            dropout=0,
            maxlen=3,
            item_embedding_mode="content_dual",
            attribute_artifact=str(tmp_path),
        )
    ).eval()


def test_dual_tower_views_reproduce_components_and_fusion(tmp_path):
    model = _dual_model(tmp_path)
    contexts = torch.tensor([[0, 1, 2]])
    candidates = torch.tensor([[3, 4]])
    title, attribute = model.score_components(contexts, candidates)

    title_view = DualTowerScoringView(model, "title").eval()
    attribute_view = DualTowerScoringView(model, "attribute").eval()
    fusion_view = DualTowerScoringView(model, "fusion").eval()

    assert torch.allclose(
        title_view.score_prepared(title_view.prepare_scoring(contexts), candidates),
        title,
    )
    assert torch.allclose(
        attribute_view.score_prepared(
            attribute_view.prepare_scoring(contexts), candidates
        ),
        attribute,
    )
    assert torch.allclose(
        fusion_view.score_prepared(fusion_view.prepare_scoring(contexts), candidates),
        0.5 * (title + attribute),
    )


def test_frequency_counts_exclude_evaluation_cohort_and_bucket_targets():
    store = SequenceStore(
        records=(
            SequenceRecord(1, (1, 1, 2), (0, 0, 0), (1, 2, 3), ("context", "context", "train")),
            SequenceRecord(2, (1, 3, 4), (0, 0, 0), (1, 2, 3), ("context", "valid", "test")),
        ),
        items_by_domain={0: (1, 2, 3, 4)},
    )
    counts = training_item_frequencies(store)
    examples = (
        TargetExample(0, 1, (0, 1), (-1, 0), 4, 0, frozenset()),
        TargetExample(1, 2, (0, 1), (-1, 0), 2, 0, frozenset()),
        TargetExample(2, 3, (0, 1), (-1, 0), 1, 0, frozenset()),
    )
    partitions = split_examples_by_frequency(
        examples, counts, frequency_buckets((0, 1, 2, 3))
    )

    assert counts == {1: 2, 2: 1}
    assert [example.positive_item for example in partitions["0"]] == [4]
    assert [example.positive_item for example in partitions["1"]] == [2]
    assert [example.positive_item for example in partitions["2"]] == [1]


def test_domain_metric_aggregation_reports_macro_and_user_weighted_values():
    summary = aggregate_domain_metrics(
        {
            0: {"HR@5": 0.2, "NDCG@5": 0.1, "HR@10": 0.4, "NDCG@10": 0.3, "num_eval_users": 10, "num_skipped_users": 1},
            1: {"HR@5": 0.6, "NDCG@5": 0.5, "HR@10": 0.8, "NDCG@10": 0.7, "num_eval_users": 30, "num_skipped_users": 2},
        }
    )

    assert summary["macro_domain"]["NDCG@10"] == pytest.approx(0.5)
    assert summary["micro_user"]["NDCG@10"] == pytest.approx(0.6)
    assert summary["num_eval_users"] == 40
    assert summary["num_skipped_users"] == 3
