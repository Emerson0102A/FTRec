"""Domain parameter-efficient and full adaptation from SASRec checkpoints."""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

from ftrec.artifacts import RunDirectory, sha256_file
from ftrec.config import canonical_hash, canonical_json
from ftrec.data.datasets import (
    SequenceStore,
    TargetExample,
    build_mixed_examples,
    build_single_domain_examples,
)
from ftrec.data.sampling import (
    BalancedBatchPlan,
    SameDomainNegativeSampler,
    evaluation_candidate_manifest,
    resolve_evaluation_candidates,
)
from ftrec.evaluation.ranking import evaluate_model
from ftrec.models.adapters import adapter_parameter_names, inject_adapters
from ftrec.models.content_adapter import (
    content_adapter_parameter_names,
    inject_fused_content_adapter,
)
from ftrec.models.embedding_adapter import (
    inject_target_embedding_adapter,
    target_embedding_parameter_names,
)
from ftrec.models.lora import (
    LORA_SCOPES,
    inject_lora,
    load_adapter_checkpoint,
    lora_parameter_names,
    save_adapter_checkpoint,
)
from ftrec.models.sasrec import SASRec, SASRecConfig, model_config_dict
from ftrec.reproducibility import resolve_device, runtime_metadata, seed_everything
from ftrec.training.checkpoint import CheckpointMismatchError, load_checkpoint, save_checkpoint
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
    bottleneck_size: int | None = None
    content_bottleneck_size: int | None = None
    seed: int = 42
    batch_size: int = 128
    steps_per_epoch: int | None = 100
    epochs: int = 100
    patience: int = 10
    lr: float = 1e-4
    embedding_lr: float | None = None
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    device: str = "cpu"
    evaluation_protocol: str = "full"
    num_train_negatives: int = 1
    num_eval_negatives: int = 100
    evaluation_seed: int = 2026
    evaluation_chunk_size: int = 4096
    evaluation_batch_size: int = 128
    context_mode: str = "mixed"
    min_domain_sequence_length: int = 1
    bf16: bool = False
    data_hash: str = "unknown"
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.method not in {
            "lora",
            "lora_all",
            "lora_all_content_adapter",
            "lora_all_embedding",
            "content_adapter",
            "embedding",
            "houlsby",
            "pfeiffer",
            "fullft",
        }:
            raise ValueError(
                "unsupported adaptation method"
            )
        if self.pretrain_method not in {"joint", "pcgrad", "joint_proportional"}:
            raise ValueError(
                "pretrain_method must be 'joint', 'pcgrad', or "
                "'joint_proportional'"
            )
        if self.domain < 0:
            raise ValueError("domain must be non-negative")
        if self.method in {
            "lora",
            "lora_all",
            "lora_all_content_adapter",
            "lora_all_embedding",
        }:
            if self.rank is None or self.rank < 1:
                raise ValueError("LoRA adaptation requires a positive rank")
            if self.alpha is None or self.alpha <= 0:
                raise ValueError("LoRA adaptation requires a positive alpha")
            if self.bottleneck_size is not None:
                raise ValueError("bottleneck_size is only valid for bottleneck adapters")
        elif self.method in {"houlsby", "pfeiffer"}:
            if self.bottleneck_size is None or self.bottleneck_size < 1:
                raise ValueError("adapter adaptation requires a positive bottleneck_size")
            if self.rank is not None or self.alpha is not None:
                raise ValueError("rank and alpha are only valid for LoRA")
        elif self.rank is not None or self.alpha is not None:
            raise ValueError("rank and alpha are only valid for LoRA")
        elif self.bottleneck_size is not None:
            raise ValueError("bottleneck_size is only valid for bottleneck adapters")
        if self.method in {"content_adapter", "lora_all_content_adapter"}:
            if self.content_bottleneck_size is None or self.content_bottleneck_size < 1:
                raise ValueError(
                    "content adaptation requires a positive content_bottleneck_size"
                )
        elif self.content_bottleneck_size is not None:
            raise ValueError(
                "content_bottleneck_size is only valid for content adaptation"
            )
        for name in ("batch_size", "epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.steps_per_epoch is not None and self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive or automatic")
        if self.evaluation_protocol not in {"full", "sampled"}:
            raise ValueError("evaluation_protocol must be 'full' or 'sampled'")
        if self.num_train_negatives < 1:
            raise ValueError("num_train_negatives must be positive")
        if self.context_mode not in {"mixed", "target_only"}:
            raise ValueError("context_mode must be 'mixed' or 'target_only'")
        if self.min_domain_sequence_length < 1:
            raise ValueError("min_domain_sequence_length must be positive")


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


def is_lora_method(method: str) -> bool:
    return method in {
        "lora",
        "lora_all",
        "lora_all_content_adapter",
        "lora_all_embedding",
    }


def is_parameter_efficient_method(method: str) -> bool:
    return method in {
        "lora",
        "lora_all",
        "lora_all_content_adapter",
        "lora_all_embedding",
        "content_adapter",
        "embedding",
        "houlsby",
        "pfeiffer",
    }


def target_modules_for_method(method: str) -> tuple[str, ...]:
    if method == "lora":
        return LORA_SCOPES["qv"]
    if method == "lora_all":
        return LORA_SCOPES["all_linear"]
    if method == "lora_all_content_adapter":
        return (*LORA_SCOPES["all_linear"], "fused_content_adapter")
    if method == "lora_all_embedding":
        return (*LORA_SCOPES["all_linear"], "target_item_embedding")
    if method == "embedding":
        return ("target_item_embedding",)
    if method == "content_adapter":
        return ("fused_content_adapter",)
    if method == "houlsby":
        return ("attention_adapter", "ffn_adapter")
    if method == "pfeiffer":
        return ("ffn_adapter",)
    return ("all",)


def resolve_adapt_steps_per_epoch(
    example_count: int, *, batch_size: int, requested: int | None
) -> int:
    """Cover each target-domain adaptation example once per automatic epoch."""
    if requested is not None:
        return requested
    if example_count < 1:
        raise ValueError("cannot resolve steps without training examples")
    return max(1, math.ceil(example_count / batch_size))


def adapt_config_hash(model_config: SASRecConfig, settings: AdaptSettings) -> str:
    training = asdict(settings)
    training.pop("output_dir")
    training.pop("force")
    training.pop("progress")
    # Preserve the lineage hash of legacy LoRA/FullFT runs created before
    # bottleneck adapters were added. An inapplicable null must not make an
    # otherwise identical completed run look conflicting.
    if training.get("bottleneck_size") is None:
        training.pop("bottleneck_size")
    if training.get("content_bottleneck_size") is None:
        training.pop("content_bottleneck_size")
    if training.get("num_train_negatives") == 1:
        training.pop("num_train_negatives")
    if training.get("context_mode") == "mixed":
        training.pop("context_mode")
    if training.get("min_domain_sequence_length") == 1:
        training.pop("min_domain_sequence_length")
    return canonical_hash(
        {"model": model_config_dict(model_config), "training": training}
    )


def validate_base_model_config(
    base_checkpoint: str | Path, model_config: SASRecConfig
) -> None:
    """Reject a checkpoint whose external frozen content bank belongs to another arm."""

    resolved = Path(base_checkpoint).parent / "resolved_config.json"
    if not resolved.is_file():
        return
    previous = json.loads(resolved.read_text(encoding="utf-8"))
    if previous.get("model") != model_config_dict(model_config):
        raise CheckpointMismatchError(
            f"base checkpoint model configuration differs from adaptation: {resolved}"
        )


def _build_adaptation_examples(
    store: SequenceStore,
    model_config: SASRecConfig,
    settings: AdaptSettings,
    *,
    split: str,
) -> tuple[tuple[TargetExample, ...], int]:
    if settings.context_mode == "target_only":
        examples = build_single_domain_examples(
            store,
            split=split,
            domain=settings.domain,
            maxlen=model_config.maxlen,
            min_domain_sequence_length=settings.min_domain_sequence_length,
        )
    else:
        examples = build_mixed_examples(
            store,
            split=split,
            target_domain=settings.domain,
            maxlen=model_config.maxlen,
            min_domain_sequence_length=settings.min_domain_sequence_length,
        )
    return tuple(examples), store.last_build_skipped


def _candidate_map(
    examples: tuple[TargetExample, ...],
    store: SequenceStore,
    settings: AdaptSettings,
    *,
    split: str,
    seed_offset: int,
) -> object | None:
    if settings.evaluation_protocol == "full":
        return None
    return resolve_evaluation_candidates(
        store,
        examples,
        split=split,
        domain=settings.domain,
        count=settings.num_eval_negatives,
        evaluation_seed=settings.evaluation_seed,
        split_offset=seed_offset,
    )


def _evaluate(
    model: SASRec,
    examples: tuple[TargetExample, ...],
    store: SequenceStore,
    settings: AdaptSettings,
    *,
    seed_offset: int,
    sampled_candidates: object | None = None,
) -> dict[str, float | int | str]:
    return evaluate_model(
        model,
        examples,
        store.items_by_domain,
        protocol=settings.evaluation_protocol,
        sampled_candidates=(
            sampled_candidates
            if settings.evaluation_protocol == "sampled"
            else None
        ),
        chunk_size=settings.evaluation_chunk_size,
        batch_size=settings.evaluation_batch_size,
        device=settings.device,
        progress=settings.progress,
        description=f"evaluate domain-{settings.domain}",
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
    validate_base_model_config(base_checkpoint, model_config)
    base_hash = sha256_file(base_checkpoint)
    model = SASRec(model_config).to(device)
    expected = {"method": settings.pretrain_method}
    if settings.data_hash != "unknown":
        expected["data_hash"] = settings.data_hash
    load_checkpoint(base_checkpoint, model, expected=expected, map_location=device)

    test_examples, test_skipped = _build_adaptation_examples(
        store, model_config, settings, split="test"
    )
    test_candidates = _candidate_map(
        test_examples, store, settings, split="test", seed_offset=20_000
    )
    pretrain_metrics = _evaluate(
        model,
        test_examples,
        store,
        settings,
        seed_offset=20_000,
        sampled_candidates=test_candidates,
    )
    target_embedding_rows: int | None = None
    if is_lora_method(settings.method):
        assert settings.rank is not None and settings.alpha is not None
        scope = "qv" if settings.method == "lora" else "all_linear"
        inject_lora(model, settings.rank, settings.alpha, scope=scope)
        if settings.method == "lora_all_content_adapter":
            assert settings.content_bottleneck_size is not None
            inject_fused_content_adapter(
                model,
                bottleneck_size=settings.content_bottleneck_size,
                freeze_existing=False,
            )
            trainable_names = tuple(
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
            content_names = content_adapter_parameter_names(model)
            if not content_names:
                raise RuntimeError("fused content adapter is not trainable")
            if any(
                ".lora_" not in name and name not in content_names
                for name in trainable_names
            ):
                raise RuntimeError("unexpected parameters are trainable")
        elif settings.method == "lora_all_embedding":
            embedding_adapter = inject_target_embedding_adapter(
                model,
                store.items_by_domain[settings.domain],
                freeze_existing=False,
            )
            target_embedding_rows = embedding_adapter.num_target_items
            trainable_names = tuple(
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
            if not target_embedding_parameter_names(model):
                raise RuntimeError("target embedding adapter is not trainable")
            if any(
                ".lora_" not in name
                and not name.startswith("item_embedding_adapter.")
                for name in trainable_names
            ):
                raise RuntimeError("unexpected parameters are trainable")
        else:
            trainable_names = lora_parameter_names(model)
        expected_names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if trainable_names != expected_names:
            raise RuntimeError("non-LoRA parameters are trainable")
    elif settings.method == "content_adapter":
        assert settings.content_bottleneck_size is not None
        inject_fused_content_adapter(
            model,
            bottleneck_size=settings.content_bottleneck_size,
            freeze_existing=True,
        )
        trainable_names = content_adapter_parameter_names(model)
        expected_names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if not trainable_names or trainable_names != expected_names:
            raise RuntimeError("non-content-adapter parameters are trainable")
    elif settings.method == "embedding":
        embedding_adapter = inject_target_embedding_adapter(
            model,
            store.items_by_domain[settings.domain],
            freeze_existing=True,
        )
        target_embedding_rows = embedding_adapter.num_target_items
        trainable_names = target_embedding_parameter_names(model)
        expected_names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if trainable_names != expected_names:
            raise RuntimeError("non-embedding parameters are trainable")
    elif settings.method in {"houlsby", "pfeiffer"}:
        assert settings.bottleneck_size is not None
        inject_adapters(
            model,
            method=settings.method,
            bottleneck_size=settings.bottleneck_size,
        )
        trainable_names = adapter_parameter_names(model)
        expected_names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if trainable_names != expected_names:
            raise RuntimeError("non-adapter parameters are trainable")
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
    train_examples, train_skipped = _build_adaptation_examples(
        store, model_config, settings, split="train"
    )
    validation_examples, validation_skipped = _build_adaptation_examples(
        store, model_config, settings, split="valid"
    )
    if not train_examples:
        raise ValueError(f"domain {settings.domain} has no adaptation examples")
    steps_per_epoch = resolve_adapt_steps_per_epoch(
        len(train_examples),
        batch_size=settings.batch_size,
        requested=settings.steps_per_epoch,
    )
    validation_candidates = _candidate_map(
        validation_examples, store, settings, split="valid", seed_offset=10_000
    )
    optimizers = build_optimizers(
        model,
        OptimizerSettings(
            lr=settings.lr,
            embedding_lr=settings.embedding_lr,
            weight_decay=settings.weight_decay,
        ),
    )
    config_hash = adapt_config_hash(model_config, settings)
    metadata = {
        "alpha": settings.alpha,
        "bottleneck_size": settings.bottleneck_size,
        "content_bottleneck_size": settings.content_bottleneck_size,
        "base_hash": base_hash,
        "config_hash": config_hash,
        "context_mode": settings.context_mode,
        "data_hash": settings.data_hash,
        "domain": settings.domain,
        "embedding_lr": settings.embedding_lr,
        "lr": settings.lr,
        "method": settings.method,
        "min_domain_sequence_length": settings.min_domain_sequence_length,
        "num_train_negatives": settings.num_train_negatives,
        "pretrain_method": settings.pretrain_method,
        "rank": settings.rank,
        "seed": settings.seed,
        "target_embedding_rows": target_embedding_rows,
        "target_embedding_params": (
            target_embedding_rows * model_config.hidden_size
            if target_embedding_rows is not None
            else None
        ),
        "target_modules": target_modules_for_method(settings.method),
    }
    stopping = EarlyStopping(settings.patience)
    final_validation: dict[str, float | int | str] = {}

    with RunDirectory(settings.output_dir, force=settings.force) as run:
        assert run.path is not None
        run.write_json(
            "resolved_config.json",
            json.loads(
                canonical_json(
                    {
                        "base_checkpoint": base_checkpoint,
                        "base_hash": base_hash,
                        "model": model_config_dict(model_config),
                        "training": asdict(settings),
                    }
                )
            ),
        )
        run.write_json("environment.json", runtime_metadata(device))
        manifest = BalancedBatchPlan.create(
            {settings.domain: train_examples},
            batch_size=settings.batch_size,
            total_steps=settings.epochs * steps_per_epoch,
            seed=settings.seed,
        )
        manifest.write(run.path / "batch_manifest.json")
        batch_steps = iter(manifest.iter_steps())
        if validation_candidates is not None and test_candidates is not None:
            run.write_json(
                "evaluation_candidates.json",
                evaluation_candidate_manifest(
                    count=settings.num_eval_negatives,
                    evaluation_seed=settings.evaluation_seed,
                ),
            )
        metrics_path = run.path / "metrics.jsonl"
        global_step = 0
        lookup = {example.example_id: example for example in train_examples}
        samplers = {
            settings.domain: SameDomainNegativeSampler(
                {settings.domain: store.items_by_domain[settings.domain]}, settings.seed
            )
        }

        def save_selected(path: Path, epoch: int) -> None:
            state = {"epoch": epoch, "global_step": global_step}
            if is_parameter_efficient_method(settings.method):
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

        initial_validation = _evaluate(
            model,
            validation_examples,
            store,
            settings,
            seed_offset=10_000,
            sampled_candidates=validation_candidates,
        )
        initial_metric = _metric_value(initial_validation)
        stopping.update(0, initial_metric)
        best_validation = dict(initial_validation)
        save_selected(run.path / "best.pt", 0)
        with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    {
                        "elapsed_seconds": 0.0,
                        "epoch": 0,
                        "eta_seconds": None,
                        "gradient_norm": 0.0,
                        "loss": 0.0,
                        "phase": "initial_validation",
                        "validation": initial_validation,
                        "validation_ndcg": initial_metric,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=True,
                )
                + "\n"
            )
        print(
            json.dumps(
                {
                    "domain": settings.domain,
                    "epoch": 0,
                    "event": "adapt_initial_validation_complete",
                    "method": settings.method,
                    "validation_ndcg": initial_metric,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )

        last_epoch = 0
        training_started = time.perf_counter()
        training_progress = tqdm(
            total=settings.epochs * steps_per_epoch,
            desc=(
                f"train {settings.method} domain-{settings.domain} "
                f"seed-{settings.seed}"
            ),
            unit="step",
            mininterval=1.0,
            dynamic_ncols=True,
            disable=not settings.progress,
        )
        for epoch in range(1, settings.epochs + 1):
            last_epoch = epoch
            losses: list[float] = []
            norms: list[float] = []
            for _ in range(steps_per_epoch):
                identifiers = next(batch_steps)[settings.domain]
                batch = tuple(lookup[identifier] for identifier in identifiers)
                step = run_single_task_step(
                    model,
                    batch,
                    store.items_by_domain,
                    optimizers,
                    seed=settings.seed,
                    global_step=global_step,
                    grad_clip_norm=settings.grad_clip_norm,
                    bf16=settings.bf16,
                    samplers=samplers,
                    initialization_hash=base_hash,
                    num_negatives=settings.num_train_negatives,
                )
                losses.append(step.loss)
                norms.append(step.gradient_norm)
                global_step += 1
                training_progress.update(1)
            final_validation = _evaluate(
                model,
                validation_examples,
                store,
                settings,
                seed_offset=10_000 + epoch,
                sampled_candidates=validation_candidates,
            )
            selected_metric = _metric_value(final_validation)
            elapsed_seconds = time.perf_counter() - training_started
            eta_seconds = elapsed_seconds / epoch * (settings.epochs - epoch)
            record = {
                "elapsed_seconds": elapsed_seconds,
                "epoch": epoch,
                "eta_seconds": eta_seconds,
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
            print(
                json.dumps(
                    {
                        "domain": settings.domain,
                        "elapsed_seconds": round(elapsed_seconds, 1),
                        "epoch": epoch,
                        "eta_seconds": round(eta_seconds, 1),
                        "event": "adapt_epoch_complete",
                        "method": settings.method,
                        "validation_ndcg": selected_metric,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
            if stopping.update(epoch, selected_metric):
                best_validation = dict(final_validation)
                save_selected(run.path / "best.pt", epoch)
            save_selected(run.path / "last.pt", epoch)
            training_progress.set_postfix(
                epoch=epoch,
                loss=f"{record['loss']:.4f}",
                val_ndcg=f"{selected_metric:.4f}",
            )
            if stopping.should_stop:
                training_progress.total = global_step
                training_progress.refresh()
                break

        training_progress.close()

        if not (run.path / "best.pt").exists():
            save_selected(run.path / "best.pt", last_epoch)
        if is_parameter_efficient_method(settings.method):
            load_adapter_checkpoint(
                run.path / "best.pt", model, expected=metadata, map_location=device
            )
        else:
            load_checkpoint(
                run.path / "best.pt", model, expected=metadata, map_location=device
            )
        test_metrics = _evaluate(
            model,
            test_examples,
            store,
            settings,
            seed_offset=20_000,
            sampled_candidates=test_candidates,
        )
        result_payload = {
            **metadata,
            "best_epoch": stopping.best_epoch,
            "best_validation_metrics": best_validation,
            "best_validation_ndcg": stopping.best_metric,
            "checkpoint_path": str(Path(settings.output_dir) / "best.pt"),
            "initial_validation_metrics": initial_validation,
            "num_examples": {
                "test": len(test_examples),
                "train": len(train_examples),
                "valid": len(validation_examples),
            },
            "num_filtered_examples": {
                "test": test_skipped,
                "train": train_skipped,
                "valid": validation_skipped,
            },
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
                "context_mode": settings.context_mode,
                "method": settings.method,
                "min_domain_sequence_length": settings.min_domain_sequence_length,
                "pretrain_method": settings.pretrain_method,
                "rank": settings.rank,
                "bottleneck_size": settings.bottleneck_size,
                "content_bottleneck_size": settings.content_bottleneck_size,
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
