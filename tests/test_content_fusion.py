from __future__ import annotations

import csv
import gzip

import pytest
import torch
from torch import nn

from ftrec.attributes.artifacts import create_embedding_arrays, finish_artifact
from ftrec.data.datasets import TargetExample
from ftrec.models.lora import count_trainable_parameters, inject_lora
from ftrec.models.sasrec import SASRec, SASRecConfig
from ftrec.training.objectives import sampled_bce_loss
from ftrec.training.pretrain import _task_loss


def _artifact(tmp_path):
    title, attributes, present = create_embedding_arrays(
        tmp_path,
        item_count=4,
        attribute_count=3,
        embedding_dim=8,
        dtype="float32",
    )
    for item_id in range(1, 4):
        title[item_id, item_id] = 1
        for attribute_id in range(3):
            attributes[item_id, attribute_id, item_id + attribute_id] = 1
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


def _domain_file(tmp_path):
    path = tmp_path / "items.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("item_id", "domain_id"))
        writer.writeheader()
        for item_id, domain_id in ((1, 10), (2, 10), (3, 20), (4, 20)):
            writer.writerow({"item_id": item_id, "domain_id": domain_id})
    return path


def _model(
    tmp_path,
    mode: str,
    *,
    attribute_pooling: str = "hard_top1",
) -> SASRec:
    _artifact(tmp_path)
    domain_file = (
        str(_domain_file(tmp_path))
        if attribute_pooling == "domain_title_attention"
        else None
    )
    return SASRec(
        SASRecConfig(
            num_items=4,
            hidden_size=4,
            num_blocks=1,
            num_heads=1,
            dropout=0,
            maxlen=3,
            item_embedding_mode=mode,
            attribute_artifact=str(tmp_path),
            attribute_pooling=attribute_pooling,
            item_domain_file=domain_file,
        )
    )


def test_content_modes_are_content_only_and_reject_hybrid_configuration(tmp_path):
    dual = _model(tmp_path / "dual", "content_dual")
    fused = _model(tmp_path / "fused", "content_fused")

    assert dual.item_embedding is None
    assert fused.item_embedding is None
    assert all("item_embedding.weight" not in name for name, _ in dual.named_parameters())
    assert all("item_embedding.weight" not in name for name, _ in fused.named_parameters())

    try:
        SASRecConfig(
            num_items=4,
            item_embedding_mode="id",
            attribute_artifact=str(tmp_path),
        )
    except ValueError as error:
        assert "cannot be combined" in str(error)
    else:  # pragma: no cover - documents the required invariant
        raise AssertionError("hybrid ID+content configuration was accepted")


def test_dual_tower_averages_scores_only_at_inference(tmp_path):
    model = _model(tmp_path, "content_dual").eval()
    contexts = torch.tensor([[0, 1, 2], [0, 2, 3]])
    candidates = torch.tensor([[3, 1], [1, 2]])

    title_scores, attribute_scores = model.score_components(contexts, candidates)

    assert model.title_tower is not model.attribute_tower
    assert torch.allclose(
        model.score(contexts, candidates),
        0.5 * title_scores + 0.5 * attribute_scores,
    )


def test_fused_mode_builds_one_item_embedding_before_one_sasrec(tmp_path):
    model = _model(tmp_path, "content_fused").eval()
    contexts = torch.tensor([[0, 1, 2]])
    candidates = torch.tensor([[3, 1]])

    components = model.score_components(contexts, candidates)

    assert model.fused_tower is not None
    assert model.title_tower is None
    assert model.attribute_tower is None
    assert len(components) == 1
    assert torch.allclose(model.score(contexts, candidates), components[0])


@pytest.mark.parametrize(
    "pooling",
    ("mean_all", "soft_attention", "domain_title_attention"),
)
def test_soft_pooling_uses_all_valid_attributes_and_masks_missing(
    tmp_path, pooling
):
    model = _model(tmp_path, "content_fused", attribute_pooling=pooling)
    assert model.fused_tower is not None
    encoder = model.fused_tower.item_encoder
    ids = torch.tensor([0, 1, 4])
    title = encoder.title_encoder(ids)

    pooled, weights = encoder.pool_attribute_embeddings(ids, title)

    assert weights.shape == (3, 3)
    assert torch.count_nonzero(weights[0]) == 0
    assert torch.count_nonzero(weights[2]) == 0
    assert torch.allclose(weights[1].sum(), torch.tensor(1.0))
    assert torch.all(weights[1] > 0)
    assert torch.count_nonzero(pooled[0]) == 0
    assert torch.count_nonzero(pooled[2]) == 0


def test_attention_pooling_starts_from_mean_and_receives_gradients(tmp_path):
    for pooling in ("soft_attention", "domain_title_attention"):
        model = _model(
            tmp_path / pooling,
            "content_fused",
            attribute_pooling=pooling,
        )
        assert model.fused_tower is not None
        encoder = model.fused_tower.item_encoder
        ids = torch.tensor([1, 2, 3])
        title = encoder.title_encoder(ids)
        pooled, weights = encoder.pool_attribute_embeddings(ids, title)

        assert torch.allclose(weights, torch.full_like(weights, 1 / 3))
        pooled.sum().backward()
        if pooling == "soft_attention":
            assert encoder.attribute_title_query.weight.grad is not None
        else:
            assert encoder.domain_queries.grad is not None
            assert all(
                projection.weight.grad is not None
                for projection in encoder.domain_title_queries
            )


def test_domain_title_attention_can_learn_different_domain_preferences(tmp_path):
    model = _model(
        tmp_path,
        "content_fused",
        attribute_pooling="domain_title_attention",
    )
    assert model.fused_tower is not None
    encoder = model.fused_tower.item_encoder
    assert encoder.num_domains == 2
    with torch.no_grad():
        encoder.domain_queries[1].fill_(1.0)
        encoder.domain_queries[2].fill_(-1.0)
    ids = torch.tensor([1, 3])
    title = encoder.title_encoder(ids)

    _, weights = encoder.pool_attribute_embeddings(ids, title)

    assert not torch.allclose(weights[0], weights[1])


def test_pooling_configuration_rejects_ambiguous_or_missing_domain_inputs(tmp_path):
    with pytest.raises(ValueError, match="content_fused"):
        SASRecConfig(
            num_items=4,
            item_embedding_mode="content_dual",
            attribute_artifact=str(tmp_path),
            attribute_pooling="mean_all",
        )
    with pytest.raises(ValueError, match="item_domain_file"):
        SASRecConfig(
            num_items=4,
            item_embedding_mode="content_fused",
            attribute_artifact=str(tmp_path),
            attribute_pooling="domain_title_attention",
        )


def test_content_encoders_zero_padding_and_missing_items(tmp_path):
    dual = _model(tmp_path / "dual", "content_dual")
    fused = _model(tmp_path / "fused", "content_fused")
    ids = torch.tensor([0, 1, 4])

    assert dual.title_tower is not None and dual.attribute_tower is not None
    assert fused.fused_tower is not None
    for encoder in (
        dual.title_tower.item_encoder,
        dual.attribute_tower.item_encoder,
        fused.fused_tower.item_encoder,
    ):
        values = encoder(ids)
        assert torch.count_nonzero(values[0]) == 0
        assert torch.count_nonzero(values[2]) == 0
        assert torch.isfinite(values[1]).all()


def test_lora_freezes_content_encoders_and_dual_has_two_tower_adapters(tmp_path):
    dual = _model(tmp_path / "dual", "content_dual")
    fused = _model(tmp_path / "fused", "content_fused")
    inject_lora(dual, rank=2, alpha=2, scope="all_linear")
    inject_lora(fused, rank=2, alpha=2, scope="all_linear")

    dual_trainable = tuple(
        name for name, parameter in dual.named_parameters() if parameter.requires_grad
    )
    fused_trainable = tuple(
        name for name, parameter in fused.named_parameters() if parameter.requires_grad
    )
    assert dual_trainable and fused_trainable
    assert all(".lora_" in name for name in dual_trainable)
    assert all(".lora_" in name for name in fused_trainable)
    assert not any("item_encoder" in name for name in dual_trainable)
    assert not any("item_encoder" in name for name in fused_trainable)
    assert count_trainable_parameters(dual) == 2 * count_trainable_parameters(fused)


def test_evaluation_cache_reuses_catalog_item_representations(tmp_path):
    model = _model(tmp_path, "content_fused").eval()
    assert model.fused_tower is not None
    model.prepare_evaluation_cache(chunk_size=2)
    cache = model.fused_tower._evaluation_item_cache
    assert cache is not None
    assert cache.shape == (5, 4)

    contexts = torch.tensor([[0, 1, 2]])
    prepared = model.prepare_scoring(contexts)
    original_forward = model.fused_tower.item_encoder.forward

    def fail_if_reencoded(_item_ids):
        raise AssertionError("candidate content was re-encoded despite the cache")

    model.fused_tower.item_encoder.forward = fail_if_reencoded
    try:
        scores = model.score_prepared(prepared, torch.tensor([[3, 1]]))
    finally:
        model.fused_tower.item_encoder.forward = original_forward

    assert scores.shape == (1, 2)
    model.train()
    assert model.fused_tower._evaluation_item_cache is None


def test_dual_tower_training_sums_independent_bce_losses():
    class TwoTowerScores(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def score_components(self, contexts, candidate_ids):
            del contexts, candidate_ids
            return (
                torch.tensor([[2.0, -1.0]]) + self.anchor,
                torch.tensor([[0.5, 1.5]]) + self.anchor,
            )

    model = TwoTowerScores()
    example = TargetExample(0, 7, (0, 1), (-1, 0), 2, 0, frozenset({1, 2}))
    loss = _task_loss(model, (example,), {0: (1, 2, 3)}, seed=42)
    expected = sampled_bce_loss(
        torch.tensor([[2.0]]), torch.tensor([[-1.0]])
    ) + sampled_bce_loss(torch.tensor([[0.5]]), torch.tensor([[1.5]]))

    assert torch.allclose(loss, expected)
