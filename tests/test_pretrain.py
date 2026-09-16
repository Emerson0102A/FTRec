from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest


def test_auto_pretrain_steps_cover_one_balanced_dataset_pass() -> None:
    """Catch paper epochs retaining the old arbitrary 1,000-step definition."""
    from ftrec.training.pretrain import resolve_pretrain_steps_per_epoch

    assert resolve_pretrain_steps_per_epoch(
        {0: 500, 1: 300, 2: 200}, batch_size=100, requested=None
    ) == 4
    assert resolve_pretrain_steps_per_epoch(
        {0: 500}, batch_size=100, requested=None
    ) == 5
    assert resolve_pretrain_steps_per_epoch(
        {0: 500, 1: 300}, batch_size=100, requested=7
    ) == 7
import torch

from ftrec.data.datasets import TargetExample


def _examples_by_domain() -> dict[int, tuple[TargetExample, ...]]:
    result = {}
    for domain in range(5):
        base = domain * 5 + 1
        result[domain] = (
            TargetExample(
                example_id=0,
                user_id=domain + 1,
                context_items=(0, base, base + 1),
                context_domains=(-1, domain, domain),
                positive_item=base + 2,
                target_domain=domain,
                seen_items=frozenset({base, base + 1, base + 2}),
            ),
        )
    return result


def _model():
    from ftrec.models.sasrec import SASRec, SASRecConfig

    torch.manual_seed(42)
    return SASRec(
        SASRecConfig(num_items=30, hidden_size=4, num_blocks=1, num_heads=1, dropout=0, maxlen=3)
    )


def test_joint_and_pcgrad_use_identical_batch_manifest_bytes(tmp_path: Path) -> None:
    from ftrec.training.pretrain import prepare_batch_manifest

    joint = prepare_batch_manifest(
        tmp_path / "joint.json", _examples_by_domain(), batch_size=1, steps=2, seed=42
    )
    pcgrad = prepare_batch_manifest(
        tmp_path / "pcgrad.json", _examples_by_domain(), batch_size=1, steps=2, seed=42
    )

    assert list(joint.iter_steps()) == list(pcgrad.iter_steps())
    assert (tmp_path / "joint.json").read_bytes() == (tmp_path / "pcgrad.json").read_bytes()


def test_backbone_pcgrad_scope_excludes_only_item_embedding() -> None:
    """Catch the configured backbone scope dropping dense SASRec parameters."""
    from ftrec.training.pretrain import pcgrad_projection_parameter_names

    names = tuple(name for name, _ in _model().named_parameters())

    selected = pcgrad_projection_parameter_names(names, scope="backbone")

    assert "item_embedding.weight" not in selected
    assert "position_embedding.weight" in selected
    assert "blocks.0.attention.q_proj.weight" in selected
    assert "blocks.0.attention.k_proj.weight" in selected
    assert "blocks.0.attention.v_proj.weight" in selected
    assert "blocks.0.attention.out_proj.weight" in selected
    assert "blocks.0.ffn.first.weight" in selected
    assert "blocks.0.attention_norm.weight" in selected
    assert "final_norm.weight" in selected
    assert selected == tuple(name for name in names if name != "item_embedding.weight")


def test_backbone_logging_group_excludes_embedding_even_for_full_projection() -> None:
    """Keep diagnostic group semantics independent from the surgery scope."""
    from ftrec.data.sampling import BalancedBatchManifest
    from ftrec.training.engine import OptimizerSettings, build_optimizers
    from ftrec.training.pretrain import run_multitask_step

    class CapturingLogger:
        groups: dict[str, tuple[str, ...] | None] | None = None

        def record(self, **values) -> None:
            self.groups = values["groups"]

    examples = _examples_by_domain()
    catalogs = {
        domain: tuple(range(domain * 5 + 1, domain * 5 + 6))
        for domain in range(5)
    }
    batches = BalancedBatchManifest.create(
        examples, batch_size=1, steps=1, seed=42
    ).steps[0]
    model = _model()
    logger = CapturingLogger()

    run_multitask_step(
        model,
        examples,
        catalogs,
        batches,
        build_optimizers(model, OptimizerSettings(lr=1e-3)),
        method="pcgrad",
        seed=42,
        global_step=0,
        grad_clip_norm=5.0,
        gradient_logger=logger,  # type: ignore[arg-type]
        pcgrad_projection_scope="full",
    )

    assert logger.groups is not None
    assert "item_embedding.weight" not in logger.groups["backbone"]


def test_checkpoint_gradient_profile_is_deterministic_and_restores_model_mode(
    tmp_path: Path,
) -> None:
    """Catch dropout/RNG history changing the formal best-checkpoint profile."""
    from ftrec.models.sasrec import SASRec, SASRecConfig
    from ftrec.training.pretrain import record_checkpoint_gradient_profile

    examples = _examples_by_domain()
    catalogs = {
        domain: tuple(range(domain * 5 + 1, domain * 5 + 6))
        for domain in range(5)
    }
    torch.manual_seed(7)
    model = SASRec(
        SASRecConfig(
            num_items=30,
            hidden_size=4,
            num_blocks=1,
            num_heads=1,
            dropout=0.5,
            maxlen=3,
        )
    )
    model.eval()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    torch.manual_seed(111)
    expected_next_random = torch.rand(4)
    torch.manual_seed(111)
    cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    record_checkpoint_gradient_profile(
        model,
        examples,
        catalogs,
        output_dir=first,
        method="joint",
        seed=42,
        checkpoint_epoch=3,
        diagnostic_steps=2,
        diagnostic_seed=2026,
        batch_size=1,
        ema_beta=0.9,
        bf16=False,
        progress=False,
        pcgrad_projection_scope="backbone",
    )
    assert not model.training
    torch.testing.assert_close(torch.rand(4), expected_next_random, rtol=0, atol=0)
    if cuda_rng_before:
        for actual, expected in zip(
            torch.cuda.get_rng_state_all(), cuda_rng_before, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    torch.manual_seed(999)
    record_checkpoint_gradient_profile(
        model,
        examples,
        catalogs,
        output_dir=second,
        method="joint",
        seed=42,
        checkpoint_epoch=3,
        diagnostic_steps=2,
        diagnostic_seed=2026,
        batch_size=1,
        ema_beta=0.9,
        bf16=False,
        progress=False,
        pcgrad_projection_scope="backbone",
    )

    assert not model.training
    assert (first / "gradient_conflicts_best_checkpoint.jsonl").read_bytes() == (
        second / "gradient_conflicts_best_checkpoint.jsonl"
    ).read_bytes()


def test_joint_and_pcgrad_observe_same_losses_and_raw_cosines() -> None:
    from ftrec.data.sampling import BalancedBatchManifest
    from ftrec.training.engine import OptimizerSettings, build_optimizers
    from ftrec.training.pretrain import run_multitask_step

    examples = _examples_by_domain()
    catalogs = {domain: tuple(range(domain * 5 + 1, domain * 5 + 6)) for domain in range(5)}
    manifest = BalancedBatchManifest.create(examples, batch_size=1, steps=1, seed=42)
    joint_model = _model()
    pcgrad_model = copy.deepcopy(joint_model)

    joint = run_multitask_step(
        joint_model,
        examples,
        catalogs,
        manifest.steps[0],
        build_optimizers(joint_model, OptimizerSettings(lr=1e-3)),
        method="joint",
        seed=42,
        global_step=0,
        grad_clip_norm=5.0,
        initialization_hash="shared-initialization",
    )
    pcgrad = run_multitask_step(
        pcgrad_model,
        examples,
        catalogs,
        manifest.steps[0],
        build_optimizers(pcgrad_model, OptimizerSettings(lr=1e-3)),
        method="pcgrad",
        seed=42,
        global_step=0,
        grad_clip_norm=5.0,
        initialization_hash="shared-initialization",
    )

    assert pcgrad.domain_losses == pytest.approx(joint.domain_losses)
    for actual, expected in zip(pcgrad.raw_cosine, joint.raw_cosine, strict=True):
        assert actual == pytest.approx(expected)
    assert pcgrad.initialization_hash == joint.initialization_hash
    assert pcgrad.projected_cosine is not None
    torch.testing.assert_close(
        pcgrad_model.item_embedding.weight,
        joint_model.item_embedding.weight,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("method", ["joint", "pcgrad"])
def test_multitask_logging_records_projection_input_raw_gradients(
    method: str, tmp_path: Path
) -> None:
    """Catch logging PCGrad-projected gradients under the raw analysis fields."""
    import json

    from ftrec.data.sampling import BalancedBatchManifest
    from ftrec.training.engine import OptimizerSettings, build_optimizers
    from ftrec.training.pcgrad import GradientConflictLogger
    from ftrec.training.pretrain import run_multitask_step

    examples = _examples_by_domain()
    catalogs = {
        domain: tuple(range(domain * 5 + 1, domain * 5 + 6))
        for domain in range(5)
    }
    manifest = BalancedBatchManifest.create(examples, batch_size=1, steps=1, seed=42)
    model = _model()
    logger = GradientConflictLogger(
        tmp_path / "gradient_conflicts.jsonl", tuple(str(i) for i in range(5))
    )

    result = run_multitask_step(
        model,
        examples,
        catalogs,
        manifest.steps[0],
        build_optimizers(model, OptimizerSettings(lr=1e-3)),
        method=method,
        seed=42,
        global_step=0,
        grad_clip_norm=5.0,
        gradient_logger=logger,
        compute_cosine=True,
    )
    record = json.loads(
        (tmp_path / "gradient_conflicts.jsonl").read_text(encoding="utf-8")
    )

    for actual, expected in zip(record["raw_cosine"]["full"], result.raw_cosine, strict=True):
        assert actual == pytest.approx(expected)
    assert "projected_cosine" not in record


@pytest.mark.parametrize("method", ["joint", "pcgrad"])
def test_logging_toggle_does_not_change_optimizer_step(
    method: str, tmp_path: Path
) -> None:
    """Catch diagnostics mutating gradients, RNG, clipping, or optimizer updates."""
    from ftrec.data.sampling import BalancedBatchManifest
    from ftrec.training.engine import OptimizerSettings, build_optimizers
    from ftrec.training.pcgrad import GradientConflictLogger
    from ftrec.training.pretrain import run_multitask_step

    examples = _examples_by_domain()
    catalogs = {
        domain: tuple(range(domain * 5 + 1, domain * 5 + 6))
        for domain in range(5)
    }
    batches = BalancedBatchManifest.create(
        examples, batch_size=1, steps=1, seed=42
    ).steps[0]
    without_logging = _model()
    with_logging = copy.deepcopy(without_logging)

    run_multitask_step(
        without_logging,
        examples,
        catalogs,
        batches,
        build_optimizers(without_logging, OptimizerSettings(lr=1e-3)),
        method=method,
        seed=42,
        global_step=0,
        grad_clip_norm=5.0,
    )
    run_multitask_step(
        with_logging,
        examples,
        catalogs,
        batches,
        build_optimizers(with_logging, OptimizerSettings(lr=1e-3)),
        method=method,
        seed=42,
        global_step=0,
        grad_clip_norm=5.0,
        gradient_logger=GradientConflictLogger(
            tmp_path / "gradient_conflicts.jsonl", tuple(str(i) for i in range(5))
        ),
    )

    for name, expected in without_logging.state_dict().items():
        torch.testing.assert_close(with_logging.state_dict()[name], expected, rtol=0, atol=0)


def test_single_step_updates_model_with_sparse_and_dense_optimizers() -> None:
    from ftrec.training.engine import OptimizerSettings, build_optimizers
    from ftrec.training.pretrain import run_single_task_step

    model = _model()
    before = model.item_embedding.weight.detach().clone()
    example = _examples_by_domain()[0][0]

    result = run_single_task_step(
        model,
        (example,),
        {0: tuple(range(1, 6))},
        build_optimizers(model, OptimizerSettings(lr=1e-2)),
        seed=42,
        global_step=0,
        grad_clip_norm=5.0,
    )

    assert result.loss > 0
    assert not torch.equal(before, model.item_embedding.weight)


def test_joint_pretraining_run_writes_checkpoints_metrics_and_gradients(
    tmp_path: Path, capsys
) -> None:
    from ftrec.data.datasets import SequenceRecord, SequenceStore
    from ftrec.models.sasrec import SASRecConfig
    from ftrec.training.pretrain import PretrainSettings, train_pretraining

    records = []
    catalogs = {}
    for domain in range(5):
        base = domain * 10 + 1
        catalogs[domain] = tuple(range(base, base + 5))
        records.append(
            SequenceRecord(
                user_id=domain + 1,
                item_ids=(base, base + 1, base + 2, base + 3),
                domain_ids=(domain,) * 4,
                timestamps=(1, 2, 3, 4),
                splits=("train", "train", "valid", "test"),
            )
        )
    store = SequenceStore(tuple(records), catalogs)
    output = tmp_path / "joint"

    result = train_pretraining(
        store,
        SASRecConfig(num_items=45, hidden_size=4, num_blocks=1, num_heads=1, dropout=0, maxlen=3),
        PretrainSettings(
            method="joint",
            output_dir=output,
            seed=42,
            batch_size=1,
            steps_per_epoch=1,
            epochs=1,
            patience=1,
            device="cpu",
            evaluation_protocol="sampled",
            num_eval_negatives=1,
        ),
    )

    assert result.best_checkpoint == output / "best.pt"
    assert result.best_checkpoint.is_file()
    assert (output / "last.pt").is_file()
    assert (output / "result.json").is_file()
    assert (output / "gradient_conflicts.jsonl").is_file()
    assert (output / "gradient_conflict_pairs.csv").is_file()
    assert (output / "gradient_conflict_summary_training.json").is_file()
    assert (output / "gradient_conflicts_best_checkpoint.jsonl").is_file()
    assert (output / "gradient_conflict_pairs_best_checkpoint.csv").is_file()
    assert (output / "gradient_conflict_summary.json").is_file()
    assert (output / "gradient_conflict_by_domain_layer.csv").is_file()
    assert (output / "evaluation_candidates.json").is_file()
    assert not (output / "validation_candidates.json").exists()
    assert not (output / "test_candidates.json").exists()
    assert (output / "resolved_config.json").is_file()
    assert (output / "environment.json").is_file()
    epoch = __import__("json").loads(
        (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert epoch["domain_names"] == {
        "0": "Health",
        "1": "Clothing",
        "2": "Beauty",
        "3": "Grocery",
        "4": "Sports",
    }
    assert set(epoch["domain_losses"]) == {"0", "1", "2", "3", "4"}
    assert set(epoch["domain_losses_by_name"]) == {
        "Health",
        "Clothing",
        "Beauty",
        "Grocery",
        "Sports",
    }
    assert set(epoch["validation_by_name"]) == {
        "Health",
        "Clothing",
        "Beauty",
        "Grocery",
        "Sports",
    }
    for domain_id, domain_name in epoch["domain_names"].items():
        assert epoch["domain_losses_by_name"][domain_name] == epoch["domain_losses"][
            domain_id
        ]
        assert epoch["validation_by_name"][domain_name] == epoch["validation"][
            domain_id
        ]
    result_payload = __import__("json").loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result_payload["domain_names"] == epoch["domain_names"]
    conflict_profile = result_payload["gradient_conflict_profile"]
    assert conflict_profile == {
        "checkpoint_epoch": 1,
        "diagnostic_seed": 2026,
        "diagnostic_steps": 1,
        "pcgrad_projection_scope": "backbone",
        "profile_scope": "best_checkpoint",
    }
    checkpoint_records = [
        __import__("json").loads(line)
        for line in (output / "gradient_conflicts_best_checkpoint.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(checkpoint_records) == 1
    assert checkpoint_records[0]["epoch"] == result_payload["best_epoch"]
    assert checkpoint_records[0]["profile_scope"] == "best_checkpoint"
    checkpoint_summary = __import__("json").loads(
        (output / "gradient_conflict_summary.json").read_text(encoding="utf-8")
    )
    assert checkpoint_summary["checkpoint_epoch"] == result_payload["best_epoch"]
    assert checkpoint_summary["profile_scope"] == "best_checkpoint"
    checkpoint_payload = torch.load(output / "best.pt", weights_only=False)
    assert checkpoint_payload["metadata"]["pcgrad_projection_scope"] == "backbone"
    assert set(result_payload["test_metrics_by_name"]) == {
        "Health",
        "Clothing",
        "Beauty",
        "Grocery",
        "Sports",
    }
    assert set(result_payload["validation_metrics_by_name"]) == {
        "Health",
        "Clothing",
        "Beauty",
        "Grocery",
        "Sports",
    }
    for domain_id, domain_name in result_payload["domain_names"].items():
        assert result_payload["test_metrics_by_name"][domain_name] == result_payload[
            "test_metrics"
        ][domain_id]
        assert result_payload["validation_metrics_by_name"][domain_name] == result_payload[
            "validation_metrics"
        ][domain_id]
    assert len(result.test_metrics) == 5
    stderr = capsys.readouterr().err
    assert "train joint seed-42" in stderr
    assert "100%" in stderr


def test_formal_conflict_profile_uses_best_not_last_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise a real post-best epoch so an epoch label alone cannot pass."""
    from ftrec.data.datasets import SequenceRecord, SequenceStore, build_mixed_examples
    from ftrec.models.sasrec import SASRec, SASRecConfig
    from ftrec.training.checkpoint import load_checkpoint
    from ftrec.training.pretrain import (
        PretrainSettings,
        record_checkpoint_gradient_profile,
        train_pretraining,
    )

    records = []
    catalogs = {}
    for domain in range(5):
        base = domain * 10 + 1
        catalogs[domain] = tuple(range(base, base + 5))
        records.append(
            SequenceRecord(
                user_id=domain + 1,
                item_ids=(base, base + 1, base + 2, base + 3),
                domain_ids=(domain,) * 4,
                timestamps=(1, 2, 3, 4),
                splits=("train", "train", "valid", "test"),
            )
        )
    store = SequenceStore(tuple(records), catalogs)
    validation_values = iter((1.0, 0.0))

    def fake_evaluate(*_args, seed_offset: int, **_kwargs):
        value = 0.5 if seed_offset >= 20_000 else next(validation_values)
        return {domain: {"NDCG@10": value} for domain in range(5)}

    monkeypatch.setattr("ftrec.training.pretrain._evaluate_domains", fake_evaluate)
    model_config = SASRecConfig(
        num_items=45,
        hidden_size=4,
        num_blocks=1,
        num_heads=1,
        dropout=0,
        maxlen=3,
    )
    output = tmp_path / "run"
    train_pretraining(
        store,
        model_config,
        PretrainSettings(
            method="joint",
            output_dir=output,
            seed=42,
            batch_size=1,
            steps_per_epoch=1,
            epochs=3,
            patience=1,
            device="cpu",
            evaluation_protocol="sampled",
            num_eval_negatives=1,
            progress=False,
        ),
    )

    result_payload = __import__("json").loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result_payload["best_epoch"] == 1
    assert result_payload["gradient_conflict_profile"]["checkpoint_epoch"] == 1
    best_model = SASRec(model_config)
    last_model = SASRec(model_config)
    load_checkpoint(output / "best.pt", best_model)
    load_checkpoint(output / "last.pt", last_model)
    examples = {
        domain: tuple(
            build_mixed_examples(
                store, split="train", maxlen=3, target_domain=domain
            )
        )
        for domain in range(5)
    }
    best_profile = tmp_path / "best-profile"
    last_profile = tmp_path / "last-profile"
    best_profile.mkdir()
    last_profile.mkdir()
    common = {
        "examples_by_domain": examples,
        "item_catalogs": catalogs,
        "method": "joint",
        "seed": 42,
        "checkpoint_epoch": 1,
        "diagnostic_steps": 1,
        "diagnostic_seed": 2026,
        "batch_size": 1,
        "ema_beta": 0.9,
        "bf16": False,
        "progress": False,
        "pcgrad_projection_scope": "backbone",
    }
    record_checkpoint_gradient_profile(
        best_model, output_dir=best_profile, **common
    )
    record_checkpoint_gradient_profile(
        last_model, output_dir=last_profile, **common
    )
    formal = (output / "gradient_conflicts_best_checkpoint.jsonl").read_bytes()
    assert formal == (
        best_profile / "gradient_conflicts_best_checkpoint.jsonl"
    ).read_bytes()
    assert formal != (
        last_profile / "gradient_conflicts_best_checkpoint.jsonl"
    ).read_bytes()


def test_evaluation_candidates_are_independent_of_training_seed(tmp_path: Path) -> None:
    """Catch training seeds accidentally changing the paper comparison set."""
    from ftrec.data.datasets import SequenceRecord, SequenceStore
    from ftrec.models.sasrec import SASRecConfig
    from ftrec.training.pretrain import PretrainSettings, train_pretraining

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
        {0: (1, 2, 3, 4, 5, 6)},
    )
    model = SASRecConfig(
        num_items=6,
        hidden_size=4,
        num_blocks=1,
        num_heads=1,
        dropout=0,
        maxlen=3,
    )

    for training_seed in (42, 43):
        train_pretraining(
            store,
            model,
            PretrainSettings(
                method="joint",
                output_dir=tmp_path / f"seed-{training_seed}",
                seed=training_seed,
                evaluation_seed=2026,
                batch_size=1,
                steps_per_epoch=1,
                epochs=1,
                patience=1,
                device="cpu",
                evaluation_protocol="sampled",
                num_eval_negatives=2,
                progress=False,
            ),
        )

    first = (tmp_path / "seed-42" / "evaluation_candidates.json").read_bytes()
    second = (tmp_path / "seed-43" / "evaluation_candidates.json").read_bytes()
    assert first == second
    assert len(first) < 1_000


def test_pretrain_config_hash_ignores_output_control_fields(tmp_path: Path) -> None:
    from ftrec.models.sasrec import SASRecConfig
    from ftrec.training.pretrain import PretrainSettings, pretrain_config_hash

    model = SASRecConfig(num_items=10)
    settings = PretrainSettings(method="joint", output_dir=tmp_path / "first")

    assert pretrain_config_hash(model, settings) == pretrain_config_hash(
        model,
        replace(
            settings,
            output_dir=tmp_path / "second",
            force=True,
            progress=False,
        ),
    )
