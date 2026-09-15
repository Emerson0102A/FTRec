from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest
import torch

from ftrec.data.datasets import SequenceRecord, SequenceStore
from ftrec.models.sasrec import SASRec, SASRecConfig
from ftrec.training.checkpoint import save_checkpoint


def _fixture(tmp_path: Path):
    config = SASRecConfig(
        num_items=5,
        hidden_size=4,
        num_blocks=1,
        num_heads=1,
        dropout=0,
        maxlen=3,
    )
    torch.manual_seed(42)
    model = SASRec(config)
    checkpoint = save_checkpoint(
        tmp_path / "base.pt",
        model,
        metadata={"method": "joint", "seed": 42, "data_hash": "data-a"},
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


@pytest.mark.parametrize("method", ("lora", "fullft"))
def test_adaptation_run_writes_selected_checkpoint_and_result(
    tmp_path: Path, method: str, capsys
) -> None:
    from ftrec.training.adapt import AdaptSettings, train_adaptation

    store, config, checkpoint = _fixture(tmp_path)
    output = tmp_path / method
    result = train_adaptation(
        store,
        config,
        checkpoint,
        AdaptSettings(
            method=method,
            pretrain_method="joint",
            domain=0,
            output_dir=output,
            rank=2 if method == "lora" else None,
            alpha=2 if method == "lora" else None,
            seed=42,
            batch_size=1,
            steps_per_epoch=1,
            epochs=1,
            patience=1,
            lr=1e-2,
            device="cpu",
            evaluation_protocol="sampled",
            data_hash="data-a",
        ),
    )

    assert result.best_checkpoint.is_file()
    assert (output / "last.pt").is_file()
    assert (output / "result.json").is_file()
    assert (output / "validation_candidates.json").is_file()
    assert (output / "test_candidates.json").is_file()
    assert (output / "resolved_config.json").is_file()
    assert (output / "environment.json").is_file()
    assert result.num_trainable_params > 0
    assert result.num_total_params >= result.num_trainable_params
    assert result.test_metrics["num_eval_users"] == 1
    if method == "lora":
        assert all(
            ("q_proj" in name or "v_proj" in name) and ".lora_" in name
            for name in result.trainable_names
        )
        assert result.num_trainable_params < result.num_total_params
    else:
        assert result.num_trainable_params == result.num_total_params
        assert all(".lora_" not in name for name in result.trainable_names)
    stderr = capsys.readouterr().err
    assert f"train {method} domain-0 seed-42" in stderr
    assert "100%" in stderr


def test_adaptation_rejects_pretrain_method_mismatch(tmp_path: Path) -> None:
    from ftrec.training.adapt import AdaptSettings, train_adaptation
    from ftrec.training.checkpoint import CheckpointMismatchError

    store, config, checkpoint = _fixture(tmp_path)
    with pytest.raises(CheckpointMismatchError, match="method"):
        train_adaptation(
            store,
            config,
            checkpoint,
            AdaptSettings(
                method="lora",
                pretrain_method="pcgrad",
                domain=0,
                output_dir=tmp_path / "wrong",
                rank=1,
                alpha=1,
                epochs=1,
                steps_per_epoch=1,
                patience=1,
                batch_size=1,
                device="cpu",
                data_hash="data-a",
            ),
        )


def test_adapt_config_hash_ignores_output_control_fields(tmp_path: Path) -> None:
    from ftrec.training.adapt import AdaptSettings, adapt_config_hash

    _, config, _ = _fixture(tmp_path)
    settings = AdaptSettings(
        method="lora",
        pretrain_method="joint",
        domain=0,
        output_dir=tmp_path / "first",
        rank=2,
        alpha=2,
    )

    assert adapt_config_hash(config, settings) == adapt_config_hash(
        config,
        replace(
            settings,
            output_dir=tmp_path / "second",
            force=True,
            progress=False,
        ),
    )
