"""Single-task and balanced multi-domain SASRec pretraining."""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from tqdm.auto import tqdm

from ftrec.artifacts import RunDirectory
from ftrec.config import canonical_hash, canonical_json
from ftrec.data.amazon import DOMAIN_BY_ID
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
from ftrec.models.sasrec import SASRec, SASRecConfig
from ftrec.reproducibility import resolve_device, runtime_metadata, seed_everything
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
    project_pcgrad_with_counts,
)


@dataclass(frozen=True)
class PretrainSettings:
    method: str
    output_dir: Path
    seed: int = 42
    domain: int | None = None
    batch_size: int = 128
    steps_per_epoch: int | None = 100
    epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    embedding_lr: float | None = None
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    device: str = "cpu"
    evaluation_protocol: str = "full"
    num_eval_negatives: int = 100
    evaluation_seed: int = 2026
    evaluation_chunk_size: int = 4096
    evaluation_batch_size: int = 128
    gradient_log_interval: int = 1
    gradient_conflict_enabled: bool = True
    gradient_conflict_ema_beta: float = 0.9
    bf16: bool = False
    data_hash: str = "unknown"
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"single", "joint", "pcgrad"}:
            raise ValueError("method must be 'single', 'joint', or 'pcgrad'")
        if self.method == "single" and self.domain is None:
            raise ValueError("single-domain pretraining requires domain")
        if self.method != "single" and self.domain is not None:
            raise ValueError("domain is only valid for single-domain pretraining")
        if self.domain is not None and self.domain < 0:
            raise ValueError("domain must be non-negative")
        for name in ("batch_size", "epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.steps_per_epoch is not None and self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive or automatic")
        if self.evaluation_protocol not in {"full", "sampled"}:
            raise ValueError("evaluation_protocol must be 'full' or 'sampled'")
        if (
            self.num_eval_negatives < 1
            or self.evaluation_chunk_size < 1
            or self.evaluation_batch_size < 1
        ):
            raise ValueError("evaluation sizes must be positive")
        if self.gradient_log_interval < 1:
            raise ValueError("gradient_log_interval must be positive")
        if not 0 <= self.gradient_conflict_ema_beta < 1:
            raise ValueError("gradient_conflict_ema_beta must be in [0, 1)")


@dataclass(frozen=True)
class PretrainRunResult:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    initialization_hash: str
    validation_metrics: dict[int, dict[str, float | int | str]]
    test_metrics: dict[int, dict[str, float | int | str]]


def _domain_name(domain: int) -> str:
    specification = DOMAIN_BY_ID.get(domain)
    return specification.name if specification is not None else str(domain)


def _domain_names(domains: Sequence[int]) -> dict[int, str]:
    return {domain: _domain_name(domain) for domain in domains}


def _metrics_by_domain_name(
    metrics: Mapping[int, dict[str, float | int | str]],
) -> dict[str, dict[str, float | int | str]]:
    return {_domain_name(domain): values for domain, values in metrics.items()}


def pretrain_config_hash(
    model_config: SASRecConfig, settings: PretrainSettings
) -> str:
    training = asdict(settings)
    training.pop("output_dir")
    training.pop("force")
    training.pop("progress")
    return canonical_hash({"model": asdict(model_config), "training": training})


@dataclass(frozen=True)
class MultiTaskStepResult:
    domain_losses: dict[int, float]
    raw_cosine: tuple[tuple[float, ...], ...]
    projected_cosine: tuple[tuple[float, ...], ...] | None
    gradient_norm: float
    initialization_hash: str
    projection_counts: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SingleTaskStepResult:
    loss: float
    gradient_norm: float
    initialization_hash: str


def resolve_pretrain_steps_per_epoch(
    example_counts: Mapping[int, int], *, batch_size: int, requested: int | None
) -> int:
    """Define one balanced epoch as one dataset-sized amount of target work."""
    if requested is not None:
        return requested
    if not example_counts:
        raise ValueError("cannot resolve steps without training examples")
    return max(1, math.ceil(sum(example_counts.values()) / (len(example_counts) * batch_size)))


def prepare_batch_manifest(
    path: str | Path,
    examples_by_domain: Mapping[int, Sequence[TargetExample]],
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> BalancedBatchPlan:
    manifest = BalancedBatchPlan.create(
        examples_by_domain, batch_size=batch_size, total_steps=steps, seed=seed
    )
    manifest.write(path)
    return manifest


def _task_loss(
    model: torch.nn.Module,
    examples: Sequence[TargetExample],
    item_catalogs: Mapping[int, Sequence[int]],
    *,
    seed: int,
    bf16: bool = False,
    samplers: Mapping[int, SameDomainNegativeSampler] | None = None,
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
    sampler_cache = dict(samplers or {})
    generators: dict[int, random.Random] = {}
    for example in examples:
        domain = example.target_domain
        sampler = sampler_cache.get(domain)
        if sampler is None:
            sampler = SameDomainNegativeSampler(item_catalogs, seed + domain)
            sampler_cache[domain] = sampler
        generator = generators.get(domain)
        if generator is None:
            generator = random.Random(seed + domain)
            generators[domain] = generator
        negatives.append(sampler.sample(example, rng=generator))
    candidate_ids = torch.tensor(
        [
            [example.positive_item, negative]
            for example, negative in zip(examples, negatives, strict=True)
        ],
        dtype=torch.long,
        device=device,
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=bf16,
    ):
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
    bf16: bool = False,
    example_lookup_by_domain: Mapping[int, Mapping[int, TargetExample]] | None = None,
    samplers: Mapping[int, SameDomainNegativeSampler] | None = None,
    initialization_hash: str = "not-computed",
    compute_cosine: bool = True,
) -> MultiTaskStepResult:
    if method not in {"joint", "pcgrad"}:
        raise ValueError("method must be 'joint' or 'pcgrad'")
    model.train()
    optimizers.zero_grad()
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    task_gradients = []
    domain_loss_tensors: dict[int, torch.Tensor] = {}
    for domain in sorted(step_batches):
        lookup = (
            example_lookup_by_domain[domain]
            if example_lookup_by_domain is not None
            else {example.example_id: example for example in examples_by_domain[domain]}
        )
        batch = tuple(lookup[identifier] for identifier in step_batches[domain])
        loss = _task_loss(
            model,
            batch,
            item_catalogs,
            seed=seed * 100_003 + global_step * 101 + domain,
            bf16=bf16,
            samplers=samplers,
        )
        domain_loss_tensors[domain] = loss.detach()
        task_gradients.append(collect_task_gradients(loss, named_parameters))
    loss_values = torch.stack(
        [domain_loss_tensors[domain] for domain in sorted(domain_loss_tensors)]
    ).cpu().tolist()
    domain_losses = {
        domain: float(value)
        for domain, value in zip(sorted(domain_loss_tensors), loss_values, strict=True)
    }
    raw = tuple(task_gradients)
    raw_cosine = cosine_matrix(raw) if compute_cosine else ()
    if method == "pcgrad":
        gradients_for_step, projection_counts = project_pcgrad_with_counts(
            raw, seed=seed, step=global_step
        )
        projected_cosine = cosine_matrix(gradients_for_step) if compute_cosine else ()
    else:
        gradients_for_step = raw
        projected_cosine = None
        projection_counts = None
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
            projection_counts=projection_counts,
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
        projection_counts,
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
    bf16: bool = False,
    samplers: Mapping[int, SameDomainNegativeSampler] | None = None,
    initialization_hash: str = "not-computed",
) -> SingleTaskStepResult:
    if not examples:
        raise ValueError("single-task batch cannot be empty")
    model.train()
    optimizers.zero_grad()
    loss = _task_loss(
        model,
        examples,
        item_catalogs,
        seed=seed * 100_003 + global_step * 101,
        bf16=bf16,
        samplers=samplers,
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
    store: SequenceStore,
    examples: Sequence[TargetExample],
    *,
    split: str,
    domain: int,
    count: int,
    evaluation_seed: int,
    split_offset: int,
) -> object:
    return resolve_evaluation_candidates(
        store,
        examples,
        split=split,
        domain=domain,
        count=count,
        evaluation_seed=evaluation_seed,
        split_offset=split_offset,
    )


def _evaluate_domains(
    model: SASRec,
    examples_by_domain: Mapping[int, Sequence[TargetExample]],
    items_by_domain: Mapping[int, Sequence[int]],
    settings: PretrainSettings,
    *,
    seed_offset: int,
    sampled_candidates_by_domain: Mapping[
        int, Mapping[int, Sequence[int]]
    ]
    | None = None,
) -> dict[int, dict[str, float | int | str]]:
    metrics: dict[int, dict[str, float | int | str]] = {}
    for domain, examples in sorted(examples_by_domain.items()):
        sampled = None
        if settings.evaluation_protocol == "sampled":
            sampled = (
                sampled_candidates_by_domain[domain]
                if sampled_candidates_by_domain is not None
                else _sampled_candidates(
                    # This fallback is retained for callers that do not prepare
                    # candidates up front; production passes the resolved map.
                    # Synthetic stores have no processed directory and use the
                    # deterministic in-memory recipe.
                    SequenceStore((), dict(items_by_domain)),
                    examples,
                    split="valid" if seed_offset == 10_000 else "test",
                    domain=domain,
                    count=settings.num_eval_negatives,
                    evaluation_seed=settings.evaluation_seed,
                    split_offset=seed_offset,
                )
            )
        metrics[domain] = evaluate_model(
            model,
            examples,
            items_by_domain,
            protocol=settings.evaluation_protocol,
            sampled_candidates=sampled,
            chunk_size=settings.evaluation_chunk_size,
            batch_size=settings.evaluation_batch_size,
            device=settings.device,
            progress=settings.progress,
            description=f"evaluate domain-{domain}",
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
    steps_per_epoch = resolve_pretrain_steps_per_epoch(
        {domain: len(examples) for domain, examples in train_examples.items()},
        batch_size=settings.batch_size,
        requested=settings.steps_per_epoch,
    )

    optimizer_settings = OptimizerSettings(
        lr=settings.lr,
        embedding_lr=settings.embedding_lr,
        weight_decay=settings.weight_decay,
    )
    optimizers = build_optimizers(model, optimizer_settings)
    stopping = EarlyStopping(settings.patience)
    config_hash = pretrain_config_hash(model_config, settings)
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
    validation_candidates = None
    test_candidates = None
    if settings.evaluation_protocol == "sampled":
        validation_candidates = {
            domain: _sampled_candidates(
                store,
                examples,
                split="valid",
                domain=domain,
                count=settings.num_eval_negatives,
                evaluation_seed=settings.evaluation_seed,
                split_offset=10_000,
            )
            for domain, examples in validation_examples.items()
        }
        test_candidates = {
            domain: _sampled_candidates(
                store,
                examples,
                split="test",
                domain=domain,
                count=settings.num_eval_negatives,
                evaluation_seed=settings.evaluation_seed,
                split_offset=20_000,
            )
            for domain, examples in test_examples.items()
        }

    with RunDirectory(settings.output_dir, force=settings.force) as run:
        assert run.path is not None
        run.write_json(
            "resolved_config.json",
            json.loads(
                canonical_json(
                    {"model": asdict(model_config), "training": asdict(settings)}
                )
            ),
        )
        run.write_json("environment.json", runtime_metadata(device))
        total_steps = settings.epochs * steps_per_epoch
        manifest = prepare_batch_manifest(
            run.path / "batch_manifest.json",
            train_examples,
            batch_size=settings.batch_size,
            steps=total_steps,
            seed=settings.seed,
        )
        batch_steps = iter(manifest.iter_steps())
        example_lookup_by_domain = {
            domain: {example.example_id: example for example in examples}
            for domain, examples in train_examples.items()
        }
        shared_sampler = SameDomainNegativeSampler(store.items_by_domain, settings.seed)
        samplers = {domain: shared_sampler for domain in domains}
        if validation_candidates is not None and test_candidates is not None:
            run.write_json(
                "evaluation_candidates.json",
                evaluation_candidate_manifest(
                    count=settings.num_eval_negatives,
                    evaluation_seed=settings.evaluation_seed,
                ),
            )
        gradient_logger = None
        if settings.gradient_conflict_enabled and settings.method in {"joint", "pcgrad"}:
            gradient_logger = GradientConflictLogger(
                run.path / "gradient_conflicts.jsonl",
                [DOMAIN_BY_ID[domain].name if domain in DOMAIN_BY_ID else str(domain) for domain in domains],
                ema_beta=settings.gradient_conflict_ema_beta,
            )
        metrics_path = run.path / "metrics.jsonl"
        global_step = 0
        last_epoch = 0
        training_started = time.perf_counter()
        training_progress = tqdm(
            total=total_steps,
            desc=f"train {settings.method} seed-{settings.seed}",
            unit="step",
            mininterval=1.0,
            dynamic_ncols=True,
            disable=not settings.progress,
        )
        for epoch in range(1, settings.epochs + 1):
            last_epoch = epoch
            losses: list[float] = []
            norms: list[float] = []
            domain_loss_values: dict[int, list[float]] = {
                domain: [] for domain in domains
            }
            for _ in range(steps_per_epoch):
                batches = next(batch_steps)
                if settings.method == "single":
                    domain = domains[0]
                    lookup = example_lookup_by_domain[domain]
                    batch = tuple(lookup[index] for index in batches[domain])
                    step_result = run_single_task_step(
                        model,
                        batch,
                        store.items_by_domain,
                        optimizers,
                        seed=settings.seed,
                        global_step=global_step,
                        grad_clip_norm=settings.grad_clip_norm,
                        bf16=settings.bf16,
                        samplers=samplers,
                        initialization_hash=initialization_hash,
                    )
                    losses.append(step_result.loss)
                    domain_loss_values[domain].append(step_result.loss)
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
                        bf16=settings.bf16,
                        example_lookup_by_domain=example_lookup_by_domain,
                        samplers=samplers,
                        initialization_hash=initialization_hash,
                        compute_cosine=False,
                    )
                    losses.extend(step_result.domain_losses.values())
                    for domain, value in step_result.domain_losses.items():
                        domain_loss_values[domain].append(value)
                    norms.append(step_result.gradient_norm)
                global_step += 1
                training_progress.update(1)

            final_validation = _evaluate_domains(
                model,
                validation_examples,
                store.items_by_domain,
                settings,
                seed_offset=10_000 + epoch,
                sampled_candidates_by_domain=validation_candidates,
            )
            validation_ndcg = _macro_ndcg(final_validation)
            elapsed_seconds = time.perf_counter() - training_started
            eta_seconds = elapsed_seconds / epoch * (settings.epochs - epoch)
            epoch_record = {
                "domain_names": _domain_names(domains),
                "elapsed_seconds": elapsed_seconds,
                "epoch": epoch,
                "eta_seconds": eta_seconds,
                "gradient_norm": sum(norms) / len(norms),
                "loss": sum(losses) / len(losses),
                "domain_losses": {
                    domain: sum(values) / len(values)
                    for domain, values in domain_loss_values.items()
                },
                "domain_losses_by_name": {
                    _domain_name(domain): sum(values) / len(values)
                    for domain, values in domain_loss_values.items()
                },
                "validation": final_validation,
                "validation_by_name": _metrics_by_domain_name(final_validation),
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
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(elapsed_seconds, 1),
                        "epoch": epoch,
                        "eta_seconds": round(eta_seconds, 1),
                        "event": "pretrain_epoch_complete",
                        "method": settings.method,
                        "validation_macro_ndcg": validation_ndcg,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
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
            training_progress.set_postfix(
                epoch=epoch,
                loss=f"{epoch_record['loss']:.4f}",
                val_ndcg=f"{validation_ndcg:.4f}",
            )
            if stopping.should_stop:
                training_progress.total = global_step
                training_progress.refresh()
                break

        training_progress.close()

        if gradient_logger is not None:
            gradient_logger.finalize()

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
            sampled_candidates_by_domain=test_candidates,
        )
        result_payload = {
            "best_epoch": stopping.best_epoch,
            "best_validation_macro_ndcg": stopping.best_metric,
            "checkpoint_path": str(Path(settings.output_dir) / "best.pt"),
            "config_hash": config_hash,
            "data_hash": settings.data_hash,
            "domain_names": _domain_names(domains),
            "initialization_hash": initialization_hash,
            "method": settings.method,
            "num_total_params": num_total,
            "num_trainable_params": num_trainable,
            "seed": settings.seed,
            "test_metrics": final_test,
            "test_metrics_by_name": _metrics_by_domain_name(final_test),
            "validation_metrics": final_validation,
            "validation_metrics_by_name": _metrics_by_domain_name(final_validation),
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
