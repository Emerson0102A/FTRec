from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest


def test_auto_adaptation_steps_cover_each_domain_once() -> None:
    """Catch adaptation epochs retaining a fixed step count across domains."""
    from ftrec.training.adapt import resolve_adapt_steps_per_epoch

    assert resolve_adapt_steps_per_epoch(501, batch_size=256, requested=None) == 2
    assert resolve_adapt_steps_per_epoch(501, batch_size=256, requested=9) == 9
import torch

from ftrec.data.datasets import SequenceRecord, SequenceStore
from ftrec.models.sasrec import SASRec, SASRecConfig
from ftrec.training.checkpoint import save_checkpoint


ADAPT_METHODS = (
    "lora",
    "lora_all",
    "lora_all_embedding",
    "embedding",
    "houlsby",
    "pfeiffer",
    "fullft",
)


def _capacity_kwargs(method: str) -> dict[str, int | None]:
    return {
        "rank": 2 if method in {"lora", "lora_all", "lora_all_embedding"} else None,
        "alpha": 2 if method in {"lora", "lora_all", "lora_all_embedding"} else None,
        "bottleneck_size": 2 if method in {"houlsby", "pfeiffer"} else None,
    }


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


@pytest.mark.parametrize("method", ADAPT_METHODS)
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
            **_capacity_kwargs(method),
            seed=42,
            batch_size=1,
            steps_per_epoch=1,
            epochs=1,
            patience=1,
            lr=1e-2,
            device="cpu",
            evaluation_protocol="sampled",
            num_eval_negatives=1,
            data_hash="data-a",
        ),
    )

    assert result.best_checkpoint.is_file()
    assert (output / "last.pt").is_file()
    assert (output / "result.json").is_file()
    assert (output / "evaluation_candidates.json").is_file()
    assert not (output / "validation_candidates.json").exists()
    assert not (output / "test_candidates.json").exists()
    assert (output / "resolved_config.json").is_file()
    assert (output / "environment.json").is_file()
    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["best_epoch"] in (0, 1)
    assert payload["best_validation_ndcg"] == pytest.approx(
        payload["best_validation_metrics"]["NDCG@10"]
    )
    assert "initial_validation_metrics" in payload
    assert payload["context_mode"] == "mixed"
    assert payload["min_domain_sequence_length"] == 1
    assert payload["num_examples"] == {"test": 1, "train": 1, "valid": 1}
    assert result.num_trainable_params > 0
    assert result.num_total_params >= result.num_trainable_params
    assert result.test_metrics["num_eval_users"] == 1
    if method == "lora":
        assert all(
            ("q_proj" in name or "v_proj" in name) and ".lora_" in name
            for name in result.trainable_names
        )
        assert result.num_trainable_params < result.num_total_params
    elif method == "lora_all":
        assert all(".lora_" in name for name in result.trainable_names)
        assert any("ffn.first" in name for name in result.trainable_names)
        assert payload["target_modules"][-1] == "ffn.second"
        assert result.num_trainable_params < result.num_total_params
    elif method == "lora_all_embedding":
        assert any(".lora_" in name for name in result.trainable_names)
        assert "item_embedding_adapter.delta.weight" in result.trainable_names
        assert payload["target_modules"][-1] == "target_item_embedding"
        assert payload["target_embedding_rows"] == 5
        assert result.num_trainable_params < result.num_total_params
    elif method == "embedding":
        assert result.trainable_names == ("item_embedding_adapter.delta.weight",)
        assert payload["target_modules"] == ["target_item_embedding"]
        assert payload["target_embedding_rows"] == 5
        assert result.num_trainable_params < result.num_total_params
    elif method in {"houlsby", "pfeiffer"}:
        assert all("_adapter." in name for name in result.trainable_names)
        assert payload["bottleneck_size"] == 2
        assert result.num_trainable_params < result.num_total_params
    elif method == "fullft":
        assert result.num_trainable_params == result.num_total_params
        assert all(".lora_" not in name for name in result.trainable_names)
    stderr = capsys.readouterr().err
    assert f"train {method} domain-0 seed-42" in stderr
    assert "100%" in stderr


def test_target_only_adaptation_runs_with_filtered_context(
    tmp_path: Path,
) -> None:
    from ftrec.training.adapt import AdaptSettings, train_adaptation

    store, config, checkpoint = _fixture(tmp_path)
    output = tmp_path / "target-only"
    train_adaptation(
        store,
        config,
        checkpoint,
        AdaptSettings(
            method="lora",
            pretrain_method="joint",
            domain=0,
            output_dir=output,
            rank=2,
            alpha=2,
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

    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["context_mode"] == "target_only"
    assert payload["min_domain_sequence_length"] == 2
    assert payload["num_examples"] == {"test": 1, "train": 1, "valid": 1}


@pytest.mark.parametrize("method", ("lora", "fullft"))
def test_adaptation_can_keep_epoch_zero_when_training_does_not_improve_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Catch adaptation being forced to select a checkpoint worse than its base."""
    from ftrec.training.adapt import AdaptSettings, train_adaptation

    store, config, checkpoint = _fixture(tmp_path)
    ndcg_values = iter((0.5, 0.9, 0.1, 0.5))

    def controlled_evaluation(*args, **kwargs):
        ndcg = next(ndcg_values)
        return {
            "HR@5": ndcg,
            "HR@10": ndcg,
            "NDCG@5": ndcg,
            "NDCG@10": ndcg,
            "evaluation_protocol": "sampled",
            "num_eval_users": 1,
            "num_skipped_users": 0,
        }

    monkeypatch.setattr("ftrec.training.adapt._evaluate", controlled_evaluation)
    output = tmp_path / f"epoch-zero-{method}"
    train_adaptation(
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
            num_eval_negatives=1,
            data_hash="data-a",
            progress=False,
        ),
    )

    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["best_epoch"] == 0
    assert payload["best_validation_ndcg"] == pytest.approx(0.9)
    assert payload["initial_validation_metrics"]["NDCG@10"] == pytest.approx(0.9)
    assert payload["best_validation_metrics"] == payload["initial_validation_metrics"]
    metric_rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["epoch"] for row in metric_rows] == [0, 1]
    assert metric_rows[0]["phase"] == "initial_validation"
    assert float(metric_rows[0]["loss"]) == 0.0
    assert float(metric_rows[0]["gradient_norm"]) == 0.0

    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert best["training_state"]["epoch"] == 0


@pytest.mark.parametrize("method", ("lora", "fullft"))
def test_adaptation_replaces_epoch_zero_when_training_improves_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Catch the epoch-zero safeguard preventing a genuinely better update."""
    from ftrec.training.adapt import AdaptSettings, train_adaptation

    store, config, checkpoint = _fixture(tmp_path)
    ndcg_values = iter((0.5, 0.1, 0.9, 0.5))

    def controlled_evaluation(*args, **kwargs):
        ndcg = next(ndcg_values)
        return {
            "HR@5": ndcg,
            "HR@10": ndcg,
            "NDCG@5": ndcg,
            "NDCG@10": ndcg,
            "evaluation_protocol": "sampled",
            "num_eval_users": 1,
            "num_skipped_users": 0,
        }

    monkeypatch.setattr("ftrec.training.adapt._evaluate", controlled_evaluation)
    output = tmp_path / f"trained-best-{method}"
    train_adaptation(
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
            num_eval_negatives=1,
            data_hash="data-a",
            progress=False,
        ),
    )

    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["best_epoch"] == 1
    assert payload["best_validation_ndcg"] == pytest.approx(0.9)
    assert payload["best_validation_metrics"]["NDCG@10"] == pytest.approx(0.9)
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert best["training_state"]["epoch"] == 1


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


def test_adaptation_settings_accept_joint_proportional_backbone(tmp_path: Path) -> None:
    from ftrec.training.adapt import AdaptSettings

    settings = AdaptSettings(
        method="lora_all",
        pretrain_method="joint_proportional",
        domain=0,
        output_dir=tmp_path / "adapt",
        rank=3,
        alpha=3,
    )

    assert settings.pretrain_method == "joint_proportional"


def test_adaptation_settings_validate_context_controls(tmp_path: Path) -> None:
    from ftrec.training.adapt import AdaptSettings

    common = {
        "method": "lora",
        "pretrain_method": "joint_proportional",
        "domain": 0,
        "output_dir": tmp_path / "adapt",
        "rank": 2,
        "alpha": 2,
    }
    with pytest.raises(ValueError, match="context_mode"):
        AdaptSettings(**common, context_mode="unknown")
    with pytest.raises(ValueError, match="min_domain_sequence_length"):
        AdaptSettings(**common, min_domain_sequence_length=0)


def test_context_controls_change_adaptation_fingerprint(tmp_path: Path) -> None:
    from ftrec.training.adapt import AdaptSettings, adapt_config_hash

    _, config, _ = _fixture(tmp_path)
    common = AdaptSettings(
        method="lora",
        pretrain_method="joint",
        domain=0,
        output_dir=tmp_path / "mixed",
        rank=2,
        alpha=2,
    )

    assert adapt_config_hash(config, common) != adapt_config_hash(
        config,
        replace(
            common,
            output_dir=tmp_path / "target-only",
            context_mode="target_only",
            min_domain_sequence_length=5,
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


def test_legacy_lora_hash_omits_inapplicable_bottleneck_field(tmp_path: Path) -> None:
    """Keep existing completed LoRA runs resumable after adding new PEFT methods."""
    from dataclasses import asdict

    from ftrec.config import canonical_hash
    from ftrec.models.sasrec import model_config_dict
    from ftrec.training.adapt import AdaptSettings, adapt_config_hash

    _, config, _ = _fixture(tmp_path)
    settings = AdaptSettings(
        method="lora",
        pretrain_method="joint",
        domain=0,
        output_dir=tmp_path / "legacy",
        rank=2,
        alpha=2,
    )
    legacy = asdict(settings)
    for key in (
        "output_dir",
        "force",
        "progress",
        "bottleneck_size",
        "content_bottleneck_size",
        "num_train_negatives",
        "context_mode",
        "min_domain_sequence_length",
    ):
        legacy.pop(key)

    assert adapt_config_hash(config, settings) == canonical_hash(
        {"model": model_config_dict(config), "training": legacy}
    )
