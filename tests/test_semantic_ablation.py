from __future__ import annotations

import csv
import gzip
import json

import numpy as np
import pytest
import torch

from ftrec.attributes.artifacts import create_embedding_arrays, finish_artifact
from ftrec.models.sasrec import SASRec, SASRecConfig
from ftrec.training.checkpoint import CheckpointMismatchError


def _inputs(tmp_path):
    title, attributes, present = create_embedding_arrays(
        tmp_path, item_count=6, attribute_count=2, embedding_dim=8, dtype="float32"
    )
    for item_id in range(1, 7):
        title[item_id, 0] = item_id
        title[item_id, 1] = 1
        attributes[item_id, 0, item_id] = item_id
        attributes[item_id, 1, 0] = 1
        attributes[item_id, 1, item_id] = item_id
        present[item_id] = True
    finish_artifact(
        tmp_path, provider="test", item_count=6, attribute_count=2,
        embedding_dim=8, catalog_sha256="catalog", title=title,
        attributes=attributes, present=present, provider_config={"processed_items": 6},
    )
    domains = tmp_path / "items.csv.gz"
    with gzip.open(domains, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("item_id", "domain_id"))
        writer.writeheader()
        for item_id in range(1, 7):
            writer.writerow({"item_id": item_id, "domain_id": (item_id - 1) // 3})
    return domains


def _model(tmp_path, control, *, seed=42, domains=None):
    torch.manual_seed(7)
    return SASRec(SASRecConfig(
        num_items=6, hidden_size=4, num_blocks=1, num_heads=1,
        dropout=0, maxlen=3, item_embedding_mode="content_fused",
        attribute_artifact=str(tmp_path), content_ablation=control,
        content_ablation_seed=seed,
        content_ablation_domain_file=str(domains) if domains is not None else None,
    ))


def test_shuffled_content_breaks_item_alignment_within_each_domain(tmp_path):
    domains = _inputs(tmp_path)
    original = _model(tmp_path, "none")
    shuffled = _model(tmp_path, "shuffled", domains=domains)
    again = _model(tmp_path, "shuffled", domains=domains)
    source = original.fused_tower.item_encoder
    changed = shuffled.fused_tower.item_encoder
    repeated = again.fused_tower.item_encoder

    assert torch.equal(changed.title_encoder.title_bank, repeated.title_encoder.title_bank)
    assert not torch.equal(changed.title_encoder.title_bank[1:], source.title_encoder.title_bank[1:])
    for ids in ((1, 2, 3), (4, 5, 6)):
        original_pairs = sorted(
            (float(source.title_encoder.title_bank[i, 0]),
             float(source.attribute_encoder.attribute_bank[i, 0, i])) for i in ids
        )
        shuffled_pairs = sorted(
            (float(changed.title_encoder.title_bank[i, 0]),
             float(changed.attribute_encoder.attribute_bank[i, 0].max())) for i in ids
        )
        assert shuffled_pairs == original_pairs
    assert torch.count_nonzero(changed.title_encoder.title_bank[0]) == 0
    assert tuple(name for name, _ in original.named_parameters()) == tuple(
        name for name, _ in shuffled.named_parameters()
    )


def test_random_content_is_fixed_per_seed_and_preserves_missing_slots(tmp_path):
    _inputs(tmp_path)
    first = _model(tmp_path, "random", seed=42).fused_tower.item_encoder
    second = _model(tmp_path, "random", seed=42).fused_tower.item_encoder
    third = _model(tmp_path, "random", seed=43).fused_tower.item_encoder
    assert torch.equal(first.title_encoder.title_bank, second.title_encoder.title_bank)
    assert not torch.equal(first.title_encoder.title_bank, third.title_encoder.title_bank)
    assert torch.count_nonzero(first.title_encoder.title_bank[0]) == 0
    assert torch.count_nonzero(first.attribute_encoder.attribute_bank[0]) == 0
    assert torch.isfinite(first.attribute_encoder.attribute_bank).all()


def test_attribute_only_ignores_title_values_but_uses_attribute_values(tmp_path):
    _inputs(tmp_path)
    model = _model(tmp_path, "attribute_only")
    encoder = model.fused_tower.item_encoder
    encoder.eval()
    ids = torch.tensor([1, 2])
    before = encoder(ids).detach().clone()
    with torch.no_grad():
        encoder.title_encoder.title_bank[1, 0] += 100
    assert torch.equal(before, encoder(ids))
    with torch.no_grad():
        encoder.attribute_encoder.attribute_bank[1, :, :] = 0
    assert not torch.equal(before, encoder(ids))


def test_shuffled_control_requires_domain_map(tmp_path):
    _inputs(tmp_path)
    with pytest.raises(ValueError, match="content_ablation_domain_file"):
        _model(tmp_path, "shuffled")


def test_adaptation_rejects_base_trained_with_another_content_control(tmp_path):
    from ftrec.training.adapt import validate_base_model_config

    _inputs(tmp_path)
    semantic = _model(tmp_path, "none").config
    shuffled = _model(tmp_path, "shuffled", domains=tmp_path / "items.csv.gz").config
    base = tmp_path / "pretrain" / "best.pt"
    base.parent.mkdir()
    base.touch()
    from ftrec.models.sasrec import model_config_dict

    (base.parent / "resolved_config.json").write_text(
        json.dumps({"model": model_config_dict(semantic)}), encoding="utf-8"
    )
    validate_base_model_config(base, semantic)
    with pytest.raises(CheckpointMismatchError, match="model configuration"):
        validate_base_model_config(base, shuffled)
