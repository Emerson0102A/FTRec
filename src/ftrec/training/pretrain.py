"""Single-task and balanced multi-domain SASRec pretraining."""

from __future__ import annotations

import json
import math
import random
import shutil
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
    ProportionalBatchPlan,
    SameDomainNegativeSampler,
    evaluation_candidate_manifest,
    resolve_evaluation_candidates,
)
from ftrec.evaluation.ranking import evaluate_model
from ftrec.models.sasrec import SASRec, SASRecConfig, model_config_dict
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
    TaskGradients,
    collect_task_gradients,
    cosine_matrix,
    mean_gradients,
    project_pcgrad_with_counts,
    weighted_gradients,
)


@dataclass(frozen=True)
class PretrainMethodSpec:
    single_task: bool
    single_domain_context: bool
    matched_domain_history_cohort: bool
    optimizer_method: str


PRETRAIN_METHODS = (
    "single",
    "single_mixed",
    "joint_domain",
    "joint_mixed_matched",
    "joint",
    "joint_proportional",
    "pcgrad",
)

_METHOD_SPECS = {
    "single": PretrainMethodSpec(True, True, True, "single"),
    "single_mixed": PretrainMethodSpec(True, False, True, "single"),
    "joint_domain": PretrainMethodSpec(False, True, True, "joint"),
    "joint_mixed_matched": PretrainMethodSpec(False, False, True, "joint"),
    "joint": PretrainMethodSpec(False, False, False, "joint"),
    "joint_proportional": PretrainMethodSpec(False, False, False, "joint"),
    "pcgrad": PretrainMethodSpec(False, False, False, "pcgrad"),
}


def method_spec(method: str) -> PretrainMethodSpec:
    try:
        return _METHOD_SPECS[method]
    except KeyError as error:
        raise ValueError(
            f"method must be one of: {', '.join(PRETRAIN_METHODS)}"
        ) from error


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
    gradient_conflict_checkpoint_steps: int = 1
    gradient_conflict_checkpoint_seed: int = 2026
    pcgrad_projection_scope: str = "backbone"
    bf16: bool = False
    data_hash: str = "unknown"
    force: bool = False
    progress: bool = True
    resume: bool = False
    evaluate_test_each_epoch: bool = False
    snapshot_epochs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        specification = method_spec(self.method)
        if specification.single_task and self.domain is None:
            raise ValueError("single-task pretraining requires domain")
        if not specification.single_task and self.domain is not None:
            raise ValueError("domain is only valid for single-task pretraining")
        if self.domain is not None and self.domain < 0:
            raise ValueError("domain must be non-negative")
        for name in ("batch_size", "epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.steps_per_epoch is not None and self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive or automatic")
        if (
            tuple(sorted(set(self.snapshot_epochs))) != self.snapshot_epochs
            or any(epoch < 1 or epoch > self.epochs for epoch in self.snapshot_epochs)
        ):
            raise ValueError("snapshot_epochs must be unique, sorted, and within epochs")
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
        if self.gradient_conflict_checkpoint_steps < 1:
            raise ValueError("gradient_conflict_checkpoint_steps must be positive")
        if self.pcgrad_projection_scope not in {"backbone", "full"}:
            raise ValueError("pcgrad_projection_scope must be 'backbone' or 'full'")
        if self.force and self.resume:
            raise ValueError("force and resume cannot be enabled together")
        if self.resume and self.gradient_conflict_enabled:
            raise ValueError(
                "resuming training-trajectory gradient logging is not supported; "
                "disable gradient_conflict before resuming"
            )


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


def pretrain_settings_dict(settings: PretrainSettings) -> dict[str, object]:
    """Serialize settings without changing legacy hashes when monitoring is off."""

    values = asdict(settings)
    if not settings.evaluate_test_each_epoch:
        values.pop("evaluate_test_each_epoch")
    if not settings.snapshot_epochs:
        values.pop("snapshot_epochs")
    return values


def pretrain_config_hash(
    model_config: SASRecConfig, settings: PretrainSettings
) -> str:
    training = pretrain_settings_dict(settings)
    training.pop("output_dir")
    training.pop("force")
    training.pop("progress")
    training.pop("resume")
    return canonical_hash(
        {"model": model_config_dict(model_config), "training": training}
    )


_RESUME_MUTABLE_SETTINGS = {
    "epochs",
    "force",
    "output_dir",
    "patience",
    "progress",
    "resume",
}


def _validate_resume_configuration(
    source: Path,
    model_config: SASRecConfig,
    settings: PretrainSettings,
) -> None:
    """Allow only the stopping horizon to change across an in-place resume."""
    resolved_path = source / "resolved_config.json"
    if not resolved_path.is_file():
        raise FileNotFoundError(f"resume metadata is missing: {resolved_path}")
    previous = json.loads(resolved_path.read_text(encoding="utf-8"))
    current = json.loads(
        canonical_json(
            {
                "model": model_config_dict(model_config),
                "training": pretrain_settings_dict(settings),
            }
        )
    )
    if previous.get("model") != current["model"]:
        raise ValueError("resume model configuration does not match the checkpoint")
    previous_training = dict(previous.get("training", {}))
    current_training = dict(current["training"])
    for key in _RESUME_MUTABLE_SETTINGS:
        previous_training.pop(key, None)
        current_training.pop(key, None)
    if previous_training != current_training:
        changed = sorted(
            key
            for key in set(previous_training) | set(current_training)
            if previous_training.get(key) != current_training.get(key)
        )
        raise ValueError(
            "resume training configuration changed unsupported fields: "
            + ", ".join(changed)
        )


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


def combine_multitask_gradients(
    tasks: Sequence[TaskGradients],
    *,
    method: str,
    domain_batch_sizes: Sequence[int],
    task_domains: Sequence[int] | None = None,
    private_parameter_owners: Mapping[str, int] | None = None,
) -> TaskGradients:
    """Reduce gradients, retaining full owner gradients for private parameters."""
    if method == "joint_proportional":
        combined = weighted_gradients(tasks, domain_batch_sizes)
    else:
        combined = mean_gradients(tasks)
    owners = dict(private_parameter_owners or {})
    if not owners:
        return combined
    if task_domains is None or len(task_domains) != len(tasks):
        raise ValueError("task_domains must align with tasks when private parameters are used")
    task_by_domain = dict(zip(task_domains, tasks, strict=True))
    unknown = set(owners) - set(combined.names)
    if unknown:
        raise ValueError(f"private parameter owners contain unknown names: {sorted(unknown)}")
    values = list(combined.values)
    for index, name in enumerate(combined.names):
        owner = owners.get(name)
        if owner is None:
            continue
        if owner not in task_by_domain:
            raise ValueError(f"private parameter {name!r} has absent owner domain {owner}")
        owner_task = task_by_domain[owner]
        owner_value = owner_task.values[index]
        values[index] = owner_value.clone() if owner_value is not None else None
    return TaskGradients(combined.names, tuple(values))


def combine_domain_losses(
    losses: Mapping[int, float],
    *,
    method: str,
    domain_batch_sizes: Mapping[int, int],
) -> float:
    if not losses:
        raise ValueError("cannot combine empty domain losses")
    if method != "joint_proportional":
        return sum(losses.values()) / len(losses)
    total = sum(domain_batch_sizes[domain] for domain in losses)
    return sum(
        loss * domain_batch_sizes[domain] for domain, loss in losses.items()
    ) / total


def pcgrad_projection_parameter_names(
    names: Sequence[str], *, scope: str
) -> tuple[str, ...]:
    """Resolve the parameters PCGrad may project without changing task gradients elsewhere."""
    names = tuple(str(name) for name in names)
    if scope == "full":
        return names
    if scope == "backbone":
        return backbone_parameter_names(names)
    raise ValueError("pcgrad projection scope must be 'backbone' or 'full'")


def backbone_parameter_names(names: Sequence[str]) -> tuple[str, ...]:
    """Return the stable non-item-embedding diagnostic parameter group."""
    return tuple(str(name) for name in names if name != "item_embedding.weight")


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
    proportional: bool = False,
) -> BalancedBatchPlan | ProportionalBatchPlan:
    manifest = (
        ProportionalBatchPlan.create(
            examples_by_domain,
            total_batch_size=batch_size * len(examples_by_domain),
            total_steps=steps,
            seed=seed,
        )
        if proportional
        else BalancedBatchPlan.create(
            examples_by_domain, batch_size=batch_size, total_steps=steps, seed=seed
        )
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
    num_negatives: int = 1,
) -> torch.Tensor:
    if not examples:
        raise ValueError("a task micro-batch cannot be empty")
    if num_negatives < 1:
        raise ValueError("num_negatives must be positive")
    device = next(model.parameters()).device
    contexts = torch.tensor(
        [example.context_items for example in examples],
        dtype=torch.long,
        device=device,
    )
    negatives: list[tuple[int, ...]] = []
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
        sampled = (
            (sampler.sample(example, rng=generator),)
            if num_negatives == 1
            else sampler.sample_many(example, num_negatives, rng=generator)
        )
        if len(sampled) != num_negatives:
            raise ValueError(
                f"domain {domain} cannot provide {num_negatives} distinct negatives"
            )
        negatives.append(sampled)
    candidate_ids = torch.tensor(
        [
            [example.positive_item, *negative]
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
        component_scorer = getattr(model, "score_components", None)
        logits_by_tower = (
            component_scorer(contexts, candidate_ids)
            if callable(component_scorer)
            else (model.score(contexts, candidate_ids),)
        )
        losses: list[torch.Tensor] = []
        for logits in logits_by_tower:
            negative_logits = logits[:, 1:]
            positive_logits = logits[:, :1].expand_as(negative_logits)
            losses.append(sampled_bce_loss(positive_logits, negative_logits))
        # MyRec supervises title and attribute towers independently and sums
        # their BCE losses. Single-tower models naturally contribute one term.
        return torch.stack(losses).sum()


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
    pcgrad_projection_scope: str = "backbone",
) -> MultiTaskStepResult:
    specification = method_spec(method)
    if specification.single_task:
        raise ValueError("multitask step requires a shared pretraining method")
    optimizer_method = specification.optimizer_method
    model.train()
    optimizers.zero_grad()
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    parameters_by_name = dict(named_parameters)
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
    projection_names = pcgrad_projection_parameter_names(
        tuple(name for name, _ in named_parameters), scope=pcgrad_projection_scope
    )
    backbone_names = backbone_parameter_names(
        tuple(name for name, _ in named_parameters)
    )
    if optimizer_method == "pcgrad":
        gradients_for_step, projection_counts = project_pcgrad_with_counts(
            raw,
            seed=seed,
            step=global_step,
            projection_names=projection_names,
        )
        projected_cosine = cosine_matrix(gradients_for_step) if compute_cosine else ()
    else:
        gradients_for_step = raw
        projected_cosine = None
        projection_counts = None
    if gradient_logger is not None:
        groups: dict[str, tuple[str, ...] | None] = {
            "full": None,
            "backbone": backbone_names,
        }
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
    task_domains = tuple(sorted(step_batches))
    private_owner_getter = getattr(model, "private_parameter_owners", None)
    private_parameter_owners = (
        private_owner_getter() if callable(private_owner_getter) else {}
    )
    combined = combine_multitask_gradients(
        gradients_for_step,
        method=method,
        domain_batch_sizes=tuple(
            len(step_batches[domain]) for domain in task_domains
        ),
        task_domains=task_domains,
        private_parameter_owners=private_parameter_owners,
    )
    for name, value in zip(combined.names, combined.values, strict=True):
        parameters_by_name[name].grad = value
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
    num_negatives: int = 1,
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
        num_negatives=num_negatives,
    )
    loss.backward()
    norm = clip_global_grad_norm(model.parameters(), grad_clip_norm)
    optimizers.step()
    return SingleTaskStepResult(float(loss.detach().cpu()), norm, initialization_hash)


def record_checkpoint_gradient_profile(
    model: torch.nn.Module,
    examples_by_domain: Mapping[int, Sequence[TargetExample]],
    item_catalogs: Mapping[int, Sequence[int]],
    *,
    output_dir: Path,
    method: str,
    seed: int,
    checkpoint_epoch: int,
    diagnostic_steps: int,
    diagnostic_seed: int,
    batch_size: int,
    ema_beta: float,
    bf16: bool,
    progress: bool,
    pcgrad_projection_scope: str,
) -> dict[str, object]:
    """Measure raw domain conflicts at one fixed checkpoint without updating it."""
    before_hash = model_state_hash(model)
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    names = tuple(name for name, _ in named_parameters)
    projection_names = pcgrad_projection_parameter_names(
        names, scope=pcgrad_projection_scope
    )
    backbone_names = backbone_parameter_names(names)
    groups: dict[str, tuple[str, ...] | None] = {
        "full": None,
        "backbone": backbone_names,
    }
    if hasattr(model, "logging_parameter_groups"):
        groups.update(model.logging_parameter_groups())
    plan = BalancedBatchPlan.create(
        examples_by_domain,
        batch_size=batch_size,
        total_steps=diagnostic_steps,
        seed=diagnostic_seed,
    )
    plan.write(output_dir / "gradient_profile_batch_manifest.json")
    lookup_by_domain = {
        domain: {example.example_id: example for example in examples}
        for domain, examples in examples_by_domain.items()
    }
    shared_sampler = SameDomainNegativeSampler(item_catalogs, diagnostic_seed)
    samplers = {domain: shared_sampler for domain in examples_by_domain}
    logger = GradientConflictLogger(
        output_dir / "gradient_conflicts_best_checkpoint.jsonl",
        [
            DOMAIN_BY_ID[domain].name if domain in DOMAIN_BY_ID else str(domain)
            for domain in sorted(examples_by_domain)
        ],
        ema_beta=ema_beta,
        pairwise_path=output_dir / "gradient_conflict_pairs_best_checkpoint.csv",
        summary_path=output_dir / "gradient_conflict_summary.json",
        layer_table_path=output_dir / "gradient_conflict_by_domain_layer.csv",
        profile_scope="best_checkpoint",
    )
    profile_progress = tqdm(
        total=diagnostic_steps,
        desc=f"profile {method} best checkpoint",
        unit="step",
        mininterval=1.0,
        dynamic_ncols=True,
        disable=not progress,
    )
    was_training = model.training
    device = next(model.parameters()).device
    rng_devices: list[int] = []
    if device.type == "cuda":
        rng_devices.append(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    try:
        # Profiling must depend only on the checkpoint and diagnostic seed, not on
        # how many post-best epochs happened to consume dropout RNG state.
        with torch.random.fork_rng(devices=rng_devices):
            model.train()
            for diagnostic_step, batches in enumerate(plan.iter_steps()):
                raw: list[TaskGradients] = []
                for domain in sorted(batches):
                    task_seed = (
                        diagnostic_seed * 100_003
                        + diagnostic_step * 101
                        + domain
                    )
                    torch.random.default_generator.manual_seed(task_seed)
                    if device.type == "cuda":
                        torch.cuda.default_generators[rng_devices[0]].manual_seed(
                            task_seed
                        )
                    batch = tuple(
                        lookup_by_domain[domain][identifier]
                        for identifier in batches[domain]
                    )
                    loss = _task_loss(
                        model,
                        batch,
                        item_catalogs,
                        seed=task_seed,
                        bf16=bf16,
                        samplers=samplers,
                    )
                    raw.append(collect_task_gradients(loss, named_parameters))
                logger.record(
                    method=method,
                    seed=seed,
                    epoch=checkpoint_epoch,
                    step=diagnostic_step,
                    raw=tuple(raw),
                    groups=groups,
                )
                profile_progress.update(1)
    finally:
        profile_progress.close()
        model.train(was_training)
    after_hash = model_state_hash(model)
    if after_hash != before_hash:
        raise RuntimeError("checkpoint conflict profiling modified model parameters")
    metadata = {
        "checkpoint_epoch": checkpoint_epoch,
        "diagnostic_seed": diagnostic_seed,
        "diagnostic_steps": diagnostic_steps,
        "pcgrad_projection_scope": pcgrad_projection_scope,
        "profile_scope": "best_checkpoint",
    }
    logger.finalize(metadata=metadata)
    return metadata


def build_pretraining_examples(
    store: SequenceStore,
    *,
    split: str,
    domains: Sequence[int],
    maxlen: int,
    method: str,
) -> dict[int, tuple[TargetExample, ...]]:
    specification = method_spec(method)
    builder = (
        build_single_domain_examples
        if specification.single_domain_context
        else build_mixed_examples
    )
    result: dict[int, tuple[TargetExample, ...]] = {}
    for domain in domains:
        if specification.single_domain_context:
            examples = builder(store, split=split, maxlen=maxlen, domain=domain)
        else:
            examples = builder(
                store,
                split=split,
                maxlen=maxlen,
                target_domain=domain,
                require_target_domain_history=(
                    specification.matched_domain_history_cohort
                ),
            )
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
    was_training = model.training
    model.eval()
    try:
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
    finally:
        if was_training:
            model.train()
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
    """Train one reproducible pretraining or context-ablation run."""
    seed_everything(settings.seed)
    device = resolve_device(settings.device)
    specification = method_spec(settings.method)
    domains = (
        (int(settings.domain),)
        if specification.single_task
        else tuple(sorted(store.items_by_domain))
    )
    if not domains:
        raise ValueError("the processed dataset has no domains")
    model = SASRec(model_config).to(device)
    initialization_hash = model_state_hash(model)
    train_examples = build_pretraining_examples(
        store,
        split="train",
        domains=domains,
        maxlen=model_config.maxlen,
        method=settings.method,
    )
    validation_examples = build_pretraining_examples(
        store,
        split="valid",
        domains=domains,
        maxlen=model_config.maxlen,
        method=settings.method,
    )
    test_examples = build_pretraining_examples(
        store,
        split="test",
        domains=domains,
        maxlen=model_config.maxlen,
        method=settings.method,
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

    resume_source: Path | None = None
    resume_from_epoch: int | None = None
    start_epoch = 1
    global_step = 0
    if settings.resume:
        resume_source = Path(settings.output_dir).resolve()
        _validate_resume_configuration(resume_source, model_config, settings)
        expected = {
            "data_hash": settings.data_hash,
            "domain": settings.domain,
            "method": settings.method,
            "pcgrad_projection_scope": settings.pcgrad_projection_scope,
            "seed": settings.seed,
        }
        loaded_best = load_checkpoint(
            resume_source / "best.pt", model, expected=expected, map_location=device
        )
        best_epoch = int(loaded_best.training_state["epoch"])
        best_metric = float(loaded_best.training_state["validation_macro_ndcg"])
        loaded_last = load_checkpoint(
            resume_source / "last.pt",
            model,
            expected=expected,
            map_location=device,
            restore_rng=True,
        )
        resume_from_epoch = int(loaded_last.training_state["epoch"])
        global_step = int(loaded_last.training_state["global_step"])
        if settings.epochs <= resume_from_epoch:
            raise ValueError(
                f"target epochs ({settings.epochs}) must exceed the resume epoch "
                f"({resume_from_epoch})"
            )
        if best_epoch > resume_from_epoch:
            raise ValueError("best checkpoint is newer than the last checkpoint")
        bad_epochs = resume_from_epoch - best_epoch
        if bad_epochs >= settings.patience:
            raise ValueError(
                "the previous run already satisfies the requested early-stopping "
                "patience"
            )
        optimizers.load_state_dict(loaded_last.optimizer_state)
        initialization_hash = str(loaded_last.metadata["initialization_hash"])
        stopping = EarlyStopping(
            settings.patience,
            best_epoch=best_epoch,
            best_metric=best_metric,
            bad_epochs=bad_epochs,
        )
        start_epoch = resume_from_epoch + 1

    metadata = {
        "config_hash": config_hash,
        "data_hash": settings.data_hash,
        "domain": settings.domain,
        "initialization_hash": initialization_hash,
        "method": settings.method,
        "pcgrad_projection_scope": settings.pcgrad_projection_scope,
        "seed": settings.seed,
    }

    with RunDirectory(
        settings.output_dir, force=settings.force or settings.resume
    ) as run:
        assert run.path is not None
        if resume_source is not None:
            shutil.copytree(resume_source, run.path, dirs_exist_ok=True)
        run.write_json(
            "resolved_config.json",
            json.loads(
                canonical_json(
                    {
                        "model": model_config_dict(model_config),
                        "training": pretrain_settings_dict(settings),
                    }
                )
            ),
        )
        run.write_json("environment.json", runtime_metadata(device))
        total_steps = settings.epochs * steps_per_epoch
        remaining_steps = total_steps - global_step
        if remaining_steps < 1:
            raise ValueError("resume checkpoint has no remaining training steps")
        manifest = prepare_batch_manifest(
            run.path / "batch_manifest.json",
            train_examples,
            batch_size=settings.batch_size,
            steps=total_steps,
            seed=settings.seed,
            proportional=settings.method == "joint_proportional",
        )
        batch_steps = iter(manifest.iter_steps())
        for _ in range(global_step):
            next(batch_steps)
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
        if settings.gradient_conflict_enabled and not specification.single_task:
            gradient_logger = GradientConflictLogger(
                run.path / "gradient_conflicts.jsonl",
                [DOMAIN_BY_ID[domain].name if domain in DOMAIN_BY_ID else str(domain) for domain in domains],
                ema_beta=settings.gradient_conflict_ema_beta,
                summary_path=run.path / "gradient_conflict_summary_training.json",
                layer_table_path=run.path / "gradient_conflict_by_domain_layer_training.csv",
                profile_scope="training_trajectory",
            )
        metrics_path = run.path / "metrics.jsonl"
        last_epoch = start_epoch - 1
        training_started = time.perf_counter()
        training_progress = tqdm(
            total=remaining_steps,
            desc=f"train {settings.method} seed-{settings.seed}",
            unit="step",
            mininterval=1.0,
            dynamic_ncols=True,
            disable=not settings.progress,
        )
        for epoch in range(start_epoch, settings.epochs + 1):
            last_epoch = epoch
            losses: list[float] = []
            norms: list[float] = []
            domain_loss_values: dict[int, list[float]] = {
                domain: [] for domain in domains
            }
            for _ in range(steps_per_epoch):
                batches = next(batch_steps)
                if specification.single_task:
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
                        pcgrad_projection_scope=settings.pcgrad_projection_scope,
                    )
                    losses.append(
                        combine_domain_losses(
                            step_result.domain_losses,
                            method=settings.method,
                            domain_batch_sizes={
                                domain: len(identifiers)
                                for domain, identifiers in batches.items()
                            },
                        )
                    )
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
            test_monitor: dict[int, dict[str, float | int | str]] | None = None
            test_monitor_ndcg: float | None = None
            if settings.evaluate_test_each_epoch:
                test_monitor = _evaluate_domains(
                    model,
                    test_examples,
                    store.items_by_domain,
                    settings,
                    seed_offset=20_000,
                    sampled_candidates_by_domain=test_candidates,
                )
                test_monitor_ndcg = _macro_ndcg(test_monitor)
            elapsed_seconds = time.perf_counter() - training_started
            completed_epochs = epoch - start_epoch + 1
            eta_seconds = (
                elapsed_seconds
                / completed_epochs
                * (settings.epochs - epoch)
            )
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
            if test_monitor is not None:
                epoch_record.update(
                    {
                        "test_monitor": test_monitor,
                        "test_monitor_by_name": _metrics_by_domain_name(
                            test_monitor
                        ),
                        "test_monitor_macro_ndcg": test_monitor_ndcg,
                    }
                )
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
            progress_record = {
                "elapsed_seconds": round(elapsed_seconds, 1),
                "epoch": epoch,
                "eta_seconds": round(eta_seconds, 1),
                "event": "pretrain_epoch_complete",
                "method": settings.method,
                "validation_macro_ndcg": validation_ndcg,
            }
            if test_monitor_ndcg is not None:
                progress_record["test_monitor_macro_ndcg"] = test_monitor_ndcg
            print(
                json.dumps(
                    progress_record,
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
            if epoch in settings.snapshot_epochs:
                save_checkpoint(
                    run.path / "snapshots" / f"epoch-{epoch:04d}.pt",
                    model,
                    metadata=metadata,
                    training_state={
                        "epoch": epoch,
                        "global_step": global_step,
                        "validation_macro_ndcg": validation_ndcg,
                    },
                )
            training_progress.set_postfix(
                epoch=epoch,
                loss=f"{epoch_record['loss']:.4f}",
                val_ndcg=f"{validation_ndcg:.4f}",
            )
            if stopping.should_stop:
                training_progress.total = training_progress.n
                training_progress.refresh()
                break

        training_progress.close()

        if gradient_logger is not None:
            gradient_logger.finalize(
                metadata={
                    "last_trained_epoch": last_epoch,
                    "profile_scope": "training_trajectory",
                }
            )

        if not (run.path / "best.pt").exists():
            save_checkpoint(
                run.path / "best.pt",
                model,
                metadata=metadata,
                training_state={"epoch": last_epoch, "global_step": global_step},
                optimizer_state=optimizers.state_dict(),
            )
        checkpoint_expected = (
            metadata
            if resume_source is None
            else {
                "data_hash": settings.data_hash,
                "domain": settings.domain,
                "method": settings.method,
                "pcgrad_projection_scope": settings.pcgrad_projection_scope,
                "seed": settings.seed,
            }
        )
        loaded_best = load_checkpoint(
            run.path / "best.pt",
            model,
            expected=checkpoint_expected,
            map_location=device,
        )
        if resume_source is not None:
            save_checkpoint(
                run.path / "best.pt",
                model,
                metadata=metadata,
                training_state=loaded_best.training_state,
                optimizer_state=loaded_best.optimizer_state,
            )
        conflict_profile: dict[str, object] | None = None
        if gradient_logger is not None:
            checkpoint_epoch = int(
                loaded_best.training_state.get("epoch", stopping.best_epoch or last_epoch)
            )
            conflict_profile = record_checkpoint_gradient_profile(
                model,
                train_examples,
                store.items_by_domain,
                output_dir=run.path,
                method=settings.method,
                seed=settings.seed,
                checkpoint_epoch=checkpoint_epoch,
                diagnostic_steps=settings.gradient_conflict_checkpoint_steps,
                diagnostic_seed=settings.gradient_conflict_checkpoint_seed,
                batch_size=settings.batch_size,
                ema_beta=settings.gradient_conflict_ema_beta,
                bf16=settings.bf16,
                progress=settings.progress,
                pcgrad_projection_scope=settings.pcgrad_projection_scope,
            )
        final_validation = _evaluate_domains(
            model,
            validation_examples,
            store.items_by_domain,
            settings,
            seed_offset=10_000,
            sampled_candidates_by_domain=validation_candidates,
        )
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
            "gradient_conflict_profile": conflict_profile,
            "initialization_hash": initialization_hash,
            "method": settings.method,
            "pcgrad_projection_scope": settings.pcgrad_projection_scope,
            "num_total_params": num_total,
            "num_trainable_params": num_trainable,
            "last_epoch": last_epoch,
            "resume_from_epoch": resume_from_epoch,
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
                "last_epoch": last_epoch,
                "resume_from_epoch": resume_from_epoch,
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
