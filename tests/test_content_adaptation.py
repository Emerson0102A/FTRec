from __future__ import annotations

import json

import pytest
import torch

from ftrec.attributes.artifacts import create_embedding_arrays, finish_artifact
from ftrec.data.datasets import SequenceRecord, SequenceStore
from ftrec.models.sasrec import SASRec, SASRecConfig
from ftrec.training.adapt import AdaptSettings, train_adaptation
from ftrec.training.checkpoint import save_checkpoint


def _fixture(tmp_path):
    artifact_dir = tmp_path / "artifact"
    title, attributes, present = create_embedding_arrays(
        artifact_dir,
        item_count=5,
        attribute_count=3,
        embedding_dim=8,
        dtype="float32",
    )
    for item_id in range(1, 6):
        title[item_id, item_id % 8] = 1
        for attribute_id in range(3):
            attributes[item_id, attribute_id, (item_id + attribute_id) % 8] = 1
        present[item_id] = True
    finish_artifact(
        artifact_dir,
        provider="test",
        item_count=5,
        attribute_count=3,
        embedding_dim=8,
        catalog_sha256="catalog",
        title=title,
        attributes=attributes,
        present=present,
        provider_config={"processed_items": 5},
    )
    config = SASRecConfig(
        num_items=5,
        hidden_size=4,
        num_blocks=1,
        num_heads=1,
        dropout=0,
        maxlen=3,
        item_embedding_mode="content_fused",
        attribute_artifact=str(artifact_dir),
    )
    torch.manual_seed(42)
    checkpoint = save_checkpoint(
        tmp_path / "base.pt",
        SASRec(config),
        metadata={"method": "joint_proportional", "seed": 42, "data_hash": "data-a"},
    )
    store = SequenceStore(
        (
            SequenceRecord(
                user_id=1,
                item_ids=(1, 2, 3, 4),
                domain_ids=(0, 0, 0, 0),
                timestamps=(1, 2, 3, 4),
                splits=("train", "train", "valid", "test"),
            ),
        ),
        {0: (1, 2, 3, 4, 5)},
    )
    return store, config, checkpoint


@pytest.mark.parametrize(
    ("method", "rank"),
    (("content_adapter", None), ("lora_all_content_adapter", 2)),
)
def test_structured_fused_content_adaptation_runs_and_saves_only_deltas(
    tmp_path, method, rank
):
    store, config, checkpoint = _fixture(tmp_path)
    output = tmp_path / method
    result = train_adaptation(
        store,
        config,
        checkpoint,
        AdaptSettings(
            method=method,
            pretrain_method="joint_proportional",
            domain=0,
            output_dir=output,
            rank=rank,
            alpha=float(rank) if rank is not None else None,
            content_bottleneck_size=2,
            seed=42,
            batch_size=1,
            steps_per_epoch=1,
            epochs=1,
            patience=1,
            lr=1e-2,
            device="cpu",
            evaluation_protocol="sampled",
            num_eval_negatives=1,
            context_mode="target_only",
            min_domain_sequence_length=2,
            data_hash="data-a",
            progress=False,
        ),
    )

    assert any("content_adapter" in name for name in result.trainable_names)
    if method == "lora_all_content_adapter":
        assert any(".lora_" in name for name in result.trainable_names)
    else:
        assert all(".lora_" not in name for name in result.trainable_names)
    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["context_mode"] == "target_only"
    assert payload["content_bottleneck_size"] == 2
    assert payload["target_modules"][-1] == "fused_content_adapter"
    checkpoint_payload = torch.load(
        output / "best.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint_payload["adapter"]
    assert all(
        ".lora_" in name or "content_adapter" in name
        for name in checkpoint_payload["adapter"]
    )


def test_content_adapter_settings_require_a_positive_bottleneck(tmp_path):
    with pytest.raises(ValueError, match="content_bottleneck_size"):
        AdaptSettings(
            method="content_adapter",
            pretrain_method="joint_proportional",
            domain=0,
            output_dir=tmp_path,
        )
