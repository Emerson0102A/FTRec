"""Single-task and balanced multi-domain SASRec pretraining."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from ftrec.artifacts import RunDirectory
from ftrec.config import canonical_hash
from ftrec.data.datasets import (
    SequenceStore,
    TargetExample,
    build_mixed_examples,
    build_single_domain_examples,
)
from ftrec.data.sampling import (
    BalancedBatchManifest,
    SameDomainNegativeSampler,
)
from ftrec.evaluation.ranking import evaluate_model
from ftrec.models.sasrec import SASRec, SASRecConfig
from ftrec.reproducibility import resolve_device, seed_everything
from ftrec.training.checkpoint import load_checkpoint, model_state_hash, save_checkpoint
from ftrec.training.engine import (
    EarlyStopping,
    OptimizerBundle,
    OptimizerSettings,
    build_optimizers,
    clip_global_grad_norm,
)
from ftrec.training.objectives import sampled_bce_loss
from ftrec.training.pcgrad import (
    GradientConflictLogger,
    assign_mean_gradients,
    collect_task_gradients,
    cosine_matrix,
    project_pcgrad,
)


@dataclass(frozen=True)
class PretrainSettings:
    method: str
    output_dir: Path
    seed: int = 42
    domain: int | None = None
    batch_size: int = 128
    steps_per_epoch: int = 100
    epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    embedding_lr: float | None = None
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    device: str = "cpu"
    evaluation_protocol: str = "full"
    num_eval_negatives: int = 100
    evaluation_chunk_size: int = 4096
    gradient_log_interval: int = 1
    data_hash: str = "unknown"
    force: bool = False

    def __post_init__(self) -> None:
        if self.method not in {"single", "joint", "pcgrad"}:
            raise ValueError("method must be 'single', 'joint', or 'pcgrad'")
        if self.method == "single" and self.domain is None:
            raise ValueError("single-domain pretraining requires domain")
        if self.method != "single" and self.domain is not None:
            raise ValueError("domain is only valid for single-domain pretraining")
        if self.domain is not None and self.domain < 0:
            raise ValueError("domain must be non-negative")
        for name in ("batch_size", "steps_per_epoch", "epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.evaluation_protocol not in {"full", "sampled"}:
            raise ValueError("evaluation_protocol must be 'full' or 'sampled'")
        if self.num_eval_negatives < 1 or self.evaluation_chunk_size < 1:
            raise ValueError("evaluation sizes must be positive")
        if self.gradient_log_interval < 1:
            raise ValueError("gradient_log_interval must be positive")


@dataclass(frozen=True)
class PretrainRunResult:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    initialization_hash: str
    validation_metrics: dict[int, dict[str, float | int | str]]
    test_metrics: dict[int, dict[str, float | int | str]]


@dataclass(frozen=True)
class MultiTaskStepResult:
    domain_losses: dict[int, float]
    raw_cosine: tuple[tuple[float, ...], ...]
    projected_cosine: tuple[tuple[float, ...], ...] | None
    gradient_norm: float
    initialization_hash: str


@dataclass(frozen=True)
class SingleTaskStepResult:
    loss: float
    gradient_norm: float
    initialization_hash: str


def prepare_batch_manifest(
    path: str | Path,
    examples_by_domain: Mapping[int, Sequence[TargetExample]],
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> BalancedBatchManifest:
    manifest = BalancedBatchManifest.create(
        examples_by_domain, batch_size=batch_size, steps=steps, seed=seed
    )
    manifest.write(path)
    return manifest


def _task_loss(
    model: torch.nn.Module,
    examples: Sequence[TargetExample],
    item_catalogs: Mapping[int, Sequence[int]],
    *,
    seed: int,
) -> torch.Tensor:
    if not examples:
        raise ValueError("a task micro-batch cannot be empty")
    device = next(model.parameters()).device
    contexts = torch.tensor(
        [example.context_items for example in examples],
        dtype=torch.long,
        device=device,
    )
    negatives = []
    samplers: dict[int, SameDomainNegativeSampler] = {}
    for example in examples:
        sampler = samplers.setdefault(
            example.target_domain,
            SameDomainNegativeSampler(item_catalogs, seed + example.target_domain),
        )
        negatives.append(sampler.sample(example))
    candidate_ids = torch.tensor(
        [
            [example.positive_item, negative]
            for example, negative in zip(examples, negatives, strict=True)
        ],
        dtype=torch.long,
        device=device,
    )
    logits = model.score(contexts, candidate_ids)
    return sampled_bce_loss(logits[:, 0], logits[:, 1])


def run_multitask_step(
    model: torch.nn.Module,
    examples_by_domain: Mapping[int, Sequence[TargetExample]],
    item_catalogs: Mapping[int, Sequence[int]],
    step_batches: Mapping[int, Sequence[int]],
    optimizers: OptimizerBundle,
    *,
    method: str,
    seed: int,
    global_step: int,
    grad_clip_norm: float,
    gradient_logger: GradientConflictLogger | None = None,
    epoch: int = 0,
) -> MultiTaskStepResult:
    if method not in {"joint", "pcgrad"}:
        raise ValueError("method must be 'joint' or 'pcgrad'")
    initialization_hash = model_state_hash(model)
    model.train()
    optimizers.zero_grad()
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    task_gradients = []
    domain_losses: dict[int, float] = {}
    for domain in sorted(step_batches):
        lookup = {example.example_id: example for example in examples_by_domain[domain]}
        batch = tuple(lookup[identifier] for identifier in step_batches[domain])
        loss = _task_loss(
            model,
            batch,
            item_catalogs,
            seed=seed * 100_003 + global_step * 101 + domain,
        )
        domain_losses[domain] = float(loss.detach().cpu())
        task_gradients.append(collect_task_gradients(loss, named_parameters))
    raw = tuple(task_gradients)
    raw_cosine = cosine_matrix(raw)
    if method == "pcgrad":
        gradients_for_step = project_pcgrad(raw, seed=seed, step=global_step)
        projected_cosine = cosine_matrix(gradients_for_step)
    else:
        gradients_for_step = raw
        projected_cosine = None
    if gradient_logger is not None:
        groups: dict[str, tuple[str, ...] | None] = {"full": None}
        if hasattr(model, "logging_parameter_groups"):
            groups.update(model.logging_parameter_groups())
        gradient_logger.record(
            method=method,
            seed=seed,
            epoch=epoch,
            step=global_step,
            raw=raw,
            projected=gradients_for_step if method == "pcgrad" else None,
            groups=groups,
        )
    assign_mean_gradients(dict(named_parameters), gradients_for_step)
    norm = clip_global_grad_norm(model.parameters(), grad_clip_norm)
    optimizers.step()
    return MultiTaskStepResult(
        domain_losses,
        raw_cosine,
        projected_cosine,
        norm,
        initialization_hash,
    )


def run_single_task_step(
    model: torch.nn.Module,
    examples: Sequence[TargetExample],
    item_catalogs: Mapping[int, Sequence[int]],
    optimizers: OptimizerBundle,
    *,
    seed: int,
    global_step: int,
    grad_clip_norm: float,
) -> SingleTaskStepResult:
    if not examples:
        raise ValueError("single-task batch cannot be empty")
    initialization_hash = model_state_hash(model)
    model.train()
    optimizers.zero_grad()
    loss = _task_loss(
        model,
        examples,
        item_catalogs,
        seed=seed * 100_003 + global_step * 101,
    )
    loss.backward()
    norm = clip_global_grad_norm(model.parameters(), grad_clip_norm)
    optimizers.step()
    return SingleTaskStepResult(float(loss.detach().cpu()), norm, initialization_hash)


def _examples_for_domains(
    store: SequenceStore,
    *,
    split: str,
    domains: Sequence[int],
    maxlen: int,
    single_domain: bool,
) -> dict[int, tuple[TargetExample, ...]]:
    builder = build_single_domain_examples if single_domain else build_mixed_examples
    result: dict[int, tuple[TargetExample, ...]] = {}
    for domain in domains:
        keyword = "domain" if single_domain else "target_domain"
        examples = builder(store, split=split, maxlen=maxlen, **{keyword: domain})
        result[domain] = tuple(examples)
    return result


def _sampled_candidates(
    examples: Sequence[TargetExample],
    items_by_domain: Mapping[int, Sequence[int]],
    *,
    count: int,
    seed: int,
) -> dict[int, tuple[int, ...]]:
    candidates: dict[int, tuple[int, ...]] = {}
    for example in examples:
        negatives = [
            item
            for item in items_by_domain.get(example.target_domain, ())
            if item not in example.seen_items
        ]
        generator = random.Random(
            seed * 1_000_003 + example.target_domain * 1009 + example.example_id
        )
        generator.shuffle(negatives)
        candidates[example.example_id] = (
            example.positive_item,
            *negatives[:count],
        )
    return candidates


def _evaluate_domains(
    model: SASRec,
    examples_by_domain: Mapping[int, Sequence[TargetExample]],
    items_by_domain: Mapping[int, Sequence[int]],
    settings: PretrainSettings,
    *,
    seed_offset: int,
) -> dict[int, dict[str, float | int | str]]:
    metrics: dict[int, dict[str, float | int | str]] = {}
    for domain, examples in sorted(examples_by_domain.items()):
        sampled = None
        if settings.evaluation_protocol == "sampled":
            sampled = _sampled_candidates(
                examples,
                items_by_domain,
                count=settings.num_eval_negatives,
                seed=settings.seed + seed_offset,
            )
        metrics[domain] = evaluate_model(
            model,
            examples,
            items_by_domain,
            protocol=settings.evaluation_protocol,
            sampled_candidates=sampled,
            chunk_size=settings.evaluation_chunk_size,
            device=settings.device,
        )
    return metrics


def _macro_ndcg(metrics: Mapping[int, Mapping[str, float | int | str]]) -> float:
    values = [float(domain_metrics["NDCG@10"]) for domain_metrics in metrics.values()]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else -math.inf


def train_pretraining(
    store: SequenceStore,
    model_config: SASRecConfig,
    settings: PretrainSettings,
) -> PretrainRunResult:
    """Train one reproducible Single, Joint, or PCGrad pretraining run."""
    seed_everything(settings.seed)
    device = resolve_device(settings.device)
    domains = (
        (int(settings.domain),)
        if settings.method == "single"
        else tuple(sorted(store.items_by_domain))
    )
    if not domains:
        raise ValueError("the processed dataset has no domains")
    model = SASRec(model_config).to(device)
    initialization_hash = model_state_hash(model)
    single_domain = settings.method == "single"
    train_examples = _examples_for_domains(
        store,
        split="train",
        domains=domains,
        maxlen=model_config.maxlen,
        single_domain=single_domain,
    )
    validation_examples = _examples_for_domains(
        store,
        split="valid",
        domains=domains,
        maxlen=model_config.maxlen,
        single_domain=single_domain,
    )
    test_examples = _examples_for_domains(
        store,
        split="test",
        domains=domains,
        maxlen=model_config.maxlen,
        single_domain=single_domain,
    )
    empty = [domain for domain, examples in train_examples.items() if not examples]
    if empty:
        raise ValueError(f"no training examples for domains: {empty}")

    optimizer_settings = OptimizerSettings(
        lr=settings.lr,
        embedding_lr=settings.embedding_lr,
        weight_decay=settings.weight_decay,
    )
    optimizers = build_optimizers(model, optimizer_settings)
    stopping = EarlyStopping(settings.patience)
    config_hash = canonical_hash(
        {"model": asdict(model_config), "training": asdict(settings)}
    )
    metadata = {
        "config_hash": config_hash,
        "data_hash": settings.data_hash,
        "domain": settings.domain,
        "initialization_hash": initialization_hash,
        "method": settings.method,
        "seed": settings.seed,
    }
    num_total = sum(parameter.numel() for parameter in model.parameters())
    num_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    final_validation: dict[int, dict[str, float | int | str]] = {}

    with RunDirectory(settings.output_dir, force=settings.force) as run:
        assert run.path is not None
        total_steps = settings.epochs * settings.steps_per_epoch
        manifest = prepare_batch_manifest(
            run.path / "batch_manifest.json",
            train_examples,
            batch_size=settings.batch_size,
            steps=total_steps,
            seed=settings.seed,
        )
        gradient_logger = None
        if settings.method in {"joint", "pcgrad"}:
            gradient_logger = GradientConflictLogger(
                run.path / "gradient_conflicts.jsonl",
                [str(domain) for domain in domains],
            )
        metrics_path = run.path / "metrics.jsonl"
        global_step = 0
        last_epoch = 0
        for epoch in range(1, settings.epochs + 1):
            last_epoch = epoch
            losses: list[float] = []
            norms: list[float] = []
            for _ in range(settings.steps_per_epoch):
                batches = manifest.steps[global_step]
                if settings.method == "single":
                    domain = domains[0]
                    lookup = {
                        example.example_id: example for example in train_examples[domain]
                    }
                    batch = tuple(lookup[index] for index in batches[domain])
                    step_result = run_single_task_step(
                        model,
                        batch,
                        store.items_by_domain,
                        optimizers,
                        seed=settings.seed,
                        global_step=global_step,
                        grad_clip_norm=settings.grad_clip_norm,
                    )
                    losses.append(step_result.loss)
                    norms.append(step_result.gradient_norm)
                else:
                    step_result = run_multitask_step(
                        model,
                        train_examples,
                        store.items_by_domain,
                        batches,
                        optimizers,
                        method=settings.method,
                        seed=settings.seed,
                        global_step=global_step,
                        grad_clip_norm=settings.grad_clip_norm,
                        gradient_logger=(
                            gradient_logger
                            if global_step % settings.gradient_log_interval == 0
                            else None
                        ),
                        epoch=epoch,
                    )
                    losses.extend(step_result.domain_losses.values())
                    norms.append(step_result.gradient_norm)
                global_step += 1

            final_validation = _evaluate_domains(
                model,
                validation_examples,
                store.items_by_domain,
                settings,
                seed_offset=10_000 + epoch,
            )
            validation_ndcg = _macro_ndcg(final_validation)
            epoch_record = {
                "epoch": epoch,
                "gradient_norm": sum(norms) / len(norms),
                "loss": sum(losses) / len(losses),
                "validation": final_validation,
                "validation_macro_ndcg": validation_ndcg,
            }
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(
                        epoch_record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=True,
                    )
                    + "\n"
                )
            if stopping.update(epoch, validation_ndcg):
                save_checkpoint(
                    run.path / "best.pt",
                    model,
                    metadata=metadata,
                    training_state={
                        "epoch": epoch,
                        "global_step": global_step,
                        "validation_macro_ndcg": validation_ndcg,
                    },
                    optimizer_state=optimizers.state_dict(),
                )
            save_checkpoint(
                run.path / "last.pt",
                model,
                metadata=metadata,
                training_state={"epoch": epoch, "global_step": global_step},
                optimizer_state=optimizers.state_dict(),
            )
            if stopping.should_stop:
                break

        if not (run.path / "best.pt").exists():
            save_checkpoint(
                run.path / "best.pt",
                model,
                metadata=metadata,
                training_state={"epoch": last_epoch, "global_step": global_step},
                optimizer_state=optimizers.state_dict(),
            )
        load_checkpoint(run.path / "best.pt", model, expected=metadata, map_location=device)
        final_test = _evaluate_domains(
            model,
            test_examples,
            store.items_by_domain,
            settings,
            seed_offset=20_000,
        )
        result_payload = {
            "best_epoch": stopping.best_epoch,
            "best_validation_macro_ndcg": stopping.best_metric,
            "checkpoint_path": str(Path(settings.output_dir) / "best.pt"),
            "config_hash": config_hash,
            "data_hash": settings.data_hash,
            "initialization_hash": initialization_hash,
            "method": settings.method,
            "num_total_params": num_total,
            "num_trainable_params": num_trainable,
            "seed": settings.seed,
            "test_metrics": final_test,
            "validation_metrics": final_validation,
        }
        run.write_json("result.json", result_payload)
        run.complete(
            {
                "config_hash": config_hash,
                "data_hash": settings.data_hash,
                "method": settings.method,
                "seed": settings.seed,
            }
        )

    output = Path(settings.output_dir)
    return PretrainRunResult(
        output_dir=output,
        best_checkpoint=output / "best.pt",
        last_checkpoint=output / "last.pt",
        initialization_hash=initialization_hash,
        validation_metrics=final_validation,
        test_metrics=final_test,
    )
