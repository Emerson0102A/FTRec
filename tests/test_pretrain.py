from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
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

    assert joint.steps == pcgrad.steps
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
    )

    assert pcgrad.domain_losses == pytest.approx(joint.domain_losses)
    for actual, expected in zip(pcgrad.raw_cosine, joint.raw_cosine, strict=True):
        assert actual == pytest.approx(expected)
    assert pcgrad.initialization_hash == joint.initialization_hash
    assert pcgrad.projected_cosine is not None


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


def test_joint_pretraining_run_writes_checkpoints_metrics_and_gradients(tmp_path: Path) -> None:
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
        ),
    )

    assert result.best_checkpoint == output / "best.pt"
    assert result.best_checkpoint.is_file()
    assert (output / "last.pt").is_file()
    assert (output / "result.json").is_file()
    assert (output / "gradient_conflicts.jsonl").is_file()
    assert (output / "validation_candidates.json").is_file()
    assert (output / "test_candidates.json").is_file()
    assert (output / "resolved_config.json").is_file()
    assert (output / "environment.json").is_file()
    epoch = __import__("json").loads(
        (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert set(epoch["domain_losses"]) == {"0", "1", "2", "3", "4"}
    assert len(result.test_metrics) == 5


def test_pretrain_config_hash_ignores_output_control_fields(tmp_path: Path) -> None:
    from ftrec.models.sasrec import SASRecConfig
    from ftrec.training.pretrain import PretrainSettings, pretrain_config_hash

    model = SASRecConfig(num_items=10)
    settings = PretrainSettings(method="joint", output_dir=tmp_path / "first")

    assert pretrain_config_hash(model, settings) == pretrain_config_hash(
        model, replace(settings, output_dir=tmp_path / "second", force=True)
    )
