"""Domain LoRA and full-parameter adaptation from pretrained SASRec checkpoints."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ftrec.artifacts import RunDirectory, sha256_file
from ftrec.config import canonical_hash
from ftrec.data.datasets import SequenceStore, TargetExample, build_mixed_examples
from ftrec.data.sampling import BalancedBatchManifest
from ftrec.evaluation.ranking import evaluate_model
from ftrec.models.lora import (
    inject_qv_lora,
    load_adapter_checkpoint,
    lora_parameter_names,
    save_adapter_checkpoint,
)
from ftrec.models.sasrec import SASRec, SASRecConfig
from ftrec.reproducibility import resolve_device, seed_everything
from ftrec.training.checkpoint import load_checkpoint, save_checkpoint
from ftrec.training.engine import EarlyStopping, OptimizerSettings, build_optimizers
from ftrec.training.pretrain import run_single_task_step


@dataclass(frozen=True)
class AdaptSettings:
    method: str
    pretrain_method: str
    domain: int
    output_dir: Path
    rank: int | None = None
    alpha: float | None = None
    seed: int = 42
    batch_size: int = 128
    steps_per_epoch: int = 100
    epochs: int = 100
    patience: int = 10
    lr: float = 1e-4
    embedding_lr: float | None = None
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    device: str = "cpu"
    evaluation_protocol: str = "full"
    num_eval_negatives: int = 100
    evaluation_chunk_size: int = 4096
    data_hash: str = "unknown"
    force: bool = False

    def __post_init__(self) -> None:
        if self.method not in {"lora", "fullft"}:
            raise ValueError("method must be 'lora' or 'fullft'")
        if self.pretrain_method not in {"joint", "pcgrad"}:
            raise ValueError("pretrain_method must be 'joint' or 'pcgrad'")
        if self.domain < 0:
            raise ValueError("domain must be non-negative")
        if self.method == "lora":
            if self.rank is None or self.rank < 1:
                raise ValueError("LoRA adaptation requires a positive rank")
            if self.alpha is None or self.alpha <= 0:
                raise ValueError("LoRA adaptation requires a positive alpha")
        elif self.rank is not None or self.alpha is not None:
            raise ValueError("rank and alpha are only valid for LoRA")
        for name in ("batch_size", "steps_per_epoch", "epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.evaluation_protocol not in {"full", "sampled"}:
            raise ValueError("evaluation_protocol must be 'full' or 'sampled'")


@dataclass(frozen=True)
class AdaptRunResult:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    base_hash: str
    trainable_names: tuple[str, ...]
    num_trainable_params: int
    num_total_params: int
    pretrain_metrics: dict[str, float | int | str]
    validation_metrics: dict[str, float | int | str]
    test_metrics: dict[str, float | int | str]


def _candidate_map(
    examples: tuple[TargetExample, ...],
    store: SequenceStore,
    settings: AdaptSettings,
    *,
    seed_offset: int,
) -> dict[int, tuple[int, ...]] | None:
    if settings.evaluation_protocol == "full":
        return None
    result: dict[int, tuple[int, ...]] = {}
    for example in examples:
        negatives = [
            item
            for item in store.items_by_domain.get(settings.domain, ())
            if item not in example.seen_items
        ]
        random.Random(
            settings.seed * 1_000_003 + seed_offset + example.example_id
        ).shuffle(negatives)
        result[example.example_id] = (
            example.positive_item,
            *negatives[: settings.num_eval_negatives],
        )
    return result


def _evaluate(
    model: SASRec,
    examples: tuple[TargetExample, ...],
    store: SequenceStore,
    settings: AdaptSettings,
    *,
    seed_offset: int,
) -> dict[str, float | int | str]:
    return evaluate_model(
        model,
        examples,
        store.items_by_domain,
        protocol=settings.evaluation_protocol,
        sampled_candidates=_candidate_map(
            examples, store, settings, seed_offset=seed_offset
        ),
        chunk_size=settings.evaluation_chunk_size,
        device=settings.device,
    )


def _metric_value(metrics: dict[str, float | int | str]) -> float:
    value = float(metrics["NDCG@10"])
    return value if math.isfinite(value) else -math.inf


def train_adaptation(
    store: SequenceStore,
    model_config: SASRecConfig,
    base_checkpoint: str | Path,
    settings: AdaptSettings,
) -> AdaptRunResult:
    seed_everything(settings.seed)
    device = resolve_device(settings.device)
    base_checkpoint = Path(base_checkpoint)
    base_hash = sha256_file(base_checkpoint)
    model = SASRec(model_config).to(device)
    expected = {"method": settings.pretrain_method}
    if settings.data_hash != "unknown":
        expected["data_hash"] = settings.data_hash
    load_checkpoint(base_checkpoint, model, expected=expected, map_location=device)

    test_examples = tuple(
        build_mixed_examples(
            store,
            split="test",
            target_domain=settings.domain,
            maxlen=model_config.maxlen,
        )
    )
    pretrain_metrics = _evaluate(
        model, test_examples, store, settings, seed_offset=20_000
    )
    if settings.method == "lora":
        assert settings.rank is not None and settings.alpha is not None
        inject_qv_lora(model, settings.rank, settings.alpha)
        trainable_names = lora_parameter_names(model)
        expected_names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if trainable_names != expected_names:
            raise RuntimeError("non-LoRA parameters are trainable")
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        trainable_names = tuple(name for name, _ in model.named_parameters())
        if any(not parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("FullFT must expose all model parameters")

    num_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    num_total = sum(parameter.numel() for parameter in model.parameters())
    train_examples = tuple(
        build_mixed_examples(
            store,
            split="train",
            target_domain=settings.domain,
            maxlen=model_config.maxlen,
        )
    )
    validation_examples = tuple(
        build_mixed_examples(
            store,
            split="valid",
            target_domain=settings.domain,
            maxlen=model_config.maxlen,
        )
    )
    if not train_examples:
        raise ValueError(f"domain {settings.domain} has no adaptation examples")
    optimizers = build_optimizers(
        model,
        OptimizerSettings(
            lr=settings.lr,
            embedding_lr=settings.embedding_lr,
            weight_decay=settings.weight_decay,
        ),
    )
    config_hash = canonical_hash(
        {"model": asdict(model_config), "training": asdict(settings)}
    )
    metadata = {
        "alpha": settings.alpha,
        "base_hash": base_hash,
        "config_hash": config_hash,
        "data_hash": settings.data_hash,
        "domain": settings.domain,
        "method": settings.method,
        "pretrain_method": settings.pretrain_method,
        "rank": settings.rank,
        "seed": settings.seed,
    }
    stopping = EarlyStopping(settings.patience)
    final_validation: dict[str, float | int | str] = {}

    with RunDirectory(settings.output_dir, force=settings.force) as run:
        assert run.path is not None
        manifest = BalancedBatchManifest.create(
            {settings.domain: train_examples},
            batch_size=settings.batch_size,
            steps=settings.epochs * settings.steps_per_epoch,
            seed=settings.seed,
        )
        manifest.write(run.path / "batch_manifest.json")
        metrics_path = run.path / "metrics.jsonl"
        global_step = 0

        def save_selected(path: Path, epoch: int) -> None:
            state = {"epoch": epoch, "global_step": global_step}
            if settings.method == "lora":
                save_adapter_checkpoint(
                    path, model, metadata=metadata, training_state=state
                )
            else:
                save_checkpoint(
                    path,
                    model,
                    metadata=metadata,
                    training_state=state,
                    optimizer_state=optimizers.state_dict(),
                )

        last_epoch = 0
        for epoch in range(1, settings.epochs + 1):
            last_epoch = epoch
            losses: list[float] = []
            norms: list[float] = []
            lookup = {example.example_id: example for example in train_examples}
            for _ in range(settings.steps_per_epoch):
                identifiers = manifest.steps[global_step][settings.domain]
                batch = tuple(lookup[identifier] for identifier in identifiers)
                step = run_single_task_step(
                    model,
                    batch,
                    store.items_by_domain,
                    optimizers,
                    seed=settings.seed,
                    global_step=global_step,
                    grad_clip_norm=settings.grad_clip_norm,
                )
                losses.append(step.loss)
                norms.append(step.gradient_norm)
                global_step += 1
            final_validation = _evaluate(
                model,
                validation_examples,
                store,
                settings,
                seed_offset=10_000 + epoch,
            )
            selected_metric = _metric_value(final_validation)
            record = {
                "epoch": epoch,
                "gradient_norm": sum(norms) / len(norms),
                "loss": sum(losses) / len(losses),
                "validation": final_validation,
            }
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=True,
                    )
                    + "\n"
                )
            if stopping.update(epoch, selected_metric):
                save_selected(run.path / "best.pt", epoch)
            save_selected(run.path / "last.pt", epoch)
            if stopping.should_stop:
                break

        if not (run.path / "best.pt").exists():
            save_selected(run.path / "best.pt", last_epoch)
        if settings.method == "lora":
            load_adapter_checkpoint(
                run.path / "best.pt", model, expected=metadata, map_location=device
            )
        else:
            load_checkpoint(
                run.path / "best.pt", model, expected=metadata, map_location=device
            )
        test_metrics = _evaluate(
            model, test_examples, store, settings, seed_offset=20_000
        )
        result_payload = {
            **metadata,
            "best_epoch": stopping.best_epoch,
            "checkpoint_path": str(Path(settings.output_dir) / "best.pt"),
            "num_total_params": num_total,
            "num_trainable_params": num_trainable,
            "pretrain_metrics": pretrain_metrics,
            "test_metrics": test_metrics,
            "trainable_names": trainable_names,
            "validation_metrics": final_validation,
        }
        run.write_json("result.json", result_payload)
        run.complete(
            {
                "base_hash": base_hash,
                "config_hash": config_hash,
                "data_hash": settings.data_hash,
                "domain": settings.domain,
                "method": settings.method,
                "pretrain_method": settings.pretrain_method,
                "rank": settings.rank,
                "seed": settings.seed,
            }
        )

    output = Path(settings.output_dir)
    return AdaptRunResult(
        output_dir=output,
        best_checkpoint=output / "best.pt",
        last_checkpoint=output / "last.pt",
        base_hash=base_hash,
        trainable_names=trainable_names,
        num_trainable_params=num_trainable,
        num_total_params=num_total,
        pretrain_metrics=pretrain_metrics,
        validation_metrics=final_validation,
        test_metrics=test_metrics,
    )
