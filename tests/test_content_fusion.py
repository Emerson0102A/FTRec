from __future__ import annotations

import torch

from ftrec.attributes.artifacts import create_embedding_arrays, finish_artifact
from ftrec.models.content_fusion import unfreeze_content_fusion
from ftrec.models.sasrec import SASRec, SASRecConfig


def _artifact(tmp_path):
    title, attributes, present = create_embedding_arrays(
        tmp_path,
        item_count=3,
        attribute_count=2,
        embedding_dim=6,
        dtype="float32",
    )
    title[1] = 1
    attributes[1] = 1
    present[1] = True
    title[2] = 2
    attributes[2] = 2
    present[2] = True
    finish_artifact(
        tmp_path,
        provider="test",
        item_count=3,
        attribute_count=2,
        embedding_dim=6,
        catalog_sha256="catalog",
        title=title,
        attributes=attributes,
        present=present,
        provider_config={"processed_items": 3},
    )


def _model(tmp_path):
    _artifact(tmp_path)
    return SASRec(
        SASRecConfig(
            num_items=3,
            hidden_size=4,
            num_blocks=1,
            num_heads=1,
            dropout=0,
            maxlen=3,
            attribute_artifact=str(tmp_path),
        )
    )


def test_content_fusion_preserves_padding_and_missing_content(tmp_path):
    model = _model(tmp_path)
    ids = torch.tensor([0, 1, 3])
    fused = model.embed_items(ids)
    assert torch.count_nonzero(fused[0]) == 0
    assert torch.equal(fused[2], model.item_embedding(ids)[2])
    assert not torch.equal(fused[1], model.item_embedding(ids)[1])


def test_scoring_weight_uses_same_fusion_as_candidate_scoring(tmp_path):
    model = _model(tmp_path)
    item_ids = torch.arange(4)
    assert torch.allclose(model.scoring_weight(), model.embed_items(item_ids))


def test_content_fusion_can_be_selected_for_parameter_efficient_training(tmp_path):
    model = _model(tmp_path)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = unfreeze_content_fusion(model)
    assert names
    assert all(name.startswith("content_fusion.") for name in names)
