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
    assert set(epoch["domain_losses"]) == {"0", "1", "2", "3", "4"}
    assert len(result.test_metrics) == 5
    stderr = capsys.readouterr().err
    assert "train joint seed-42" in stderr
    assert "100%" in stderr


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
