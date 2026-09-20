from __future__ import annotations

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
        title[item_id] = item_id
        attributes[item_id, 0] = item_id
        attributes[item_id, 1] = item_id + 1
        attributes[item_id, 2] = item_id + 2
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


def _model(tmp_path, mode: str) -> SASRec:
    _artifact(tmp_path)
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
