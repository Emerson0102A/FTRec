"""Train Single, Joint, or PCGrad SASRec backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ftrec.artifacts import sha256_file
from ftrec.config import load_config
from ftrec.data.datasets import (
    SequenceStore,
    build_mixed_examples,
    build_single_domain_examples,
)
from ftrec.models.sasrec import SASRecConfig
from ftrec.training.pretrain import (
    PretrainSettings,
    pretrain_config_hash,
    train_pretraining,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/joint.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/sasrec.yaml")
    )
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=("single", "joint", "pcgrad"))
    parser.add_argument("--domain", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _pick(argument: object | None, config: dict[str, object], key: str) -> object:
    return argument if argument is not None else config[key]


def _steps_per_epoch(value: object) -> int | None:
    if isinstance(value, str) and value.lower() == "auto":
        return None
    return int(value)


def _data_hash(processed_dir: Path) -> str:
    completion = processed_dir / "COMPLETE.json"
    if completion.is_file():
        value = json.loads(completion.read_text(encoding="utf-8"))
        if value.get("data_hash"):
            return str(value["data_hash"])
    return sha256_file(processed_dir / "manifest.json")


def _resolved_output(
    config: dict[str, object], method: str, domain: int | None, seed: int
) -> Path:
    if config.get("output_dir"):
        return Path(str(config["output_dir"]))
    root = Path(str(config.get("output_root", "runs")))
    scope = f"domain-{domain}" if domain is not None else "all-domains"
    return root / "pretrain" / method / scope / f"seed-{seed}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    gradient_config = config.get("gradient_conflict", {})
    if not isinstance(gradient_config, dict):
        raise TypeError("gradient_conflict configuration must be a mapping")
    model_values = load_config(args.model_config)
    processed_dir = Path(_pick(args.processed_dir, config, "processed_dir"))
    method = str(_pick(args.method, config, "method"))
    seed = int(_pick(args.seed, config, "seed"))
    domain_value = args.domain if args.domain is not None else config.get("domain")
    domain = int(domain_value) if domain_value is not None else None
    output_dir = args.output_dir or _resolved_output(config, method, domain, seed)
    store = SequenceStore.from_processed(processed_dir)
    num_items = max(
        (item for catalog in store.items_by_domain.values() for item in catalog),
        default=0,
    )
    model_values = {**model_values, "num_items": num_items}
    model_config = SASRecConfig(**model_values)
    settings = PretrainSettings(
        method=method,
        output_dir=output_dir,
        seed=seed,
        domain=domain,
        batch_size=int(config["batch_size"]),
        steps_per_epoch=_steps_per_epoch(
            _pick(args.steps_per_epoch, config, "steps_per_epoch")
        ),
        epochs=int(_pick(args.epochs, config, "epochs")),
        patience=int(config["patience"]),
        lr=float(config["lr"]),
        embedding_lr=(
            float(config["embedding_lr"])
            if config.get("embedding_lr") is not None
            else None
        ),
        weight_decay=float(config.get("weight_decay", 0.0)),
        grad_clip_norm=float(config.get("grad_clip_norm", 5.0)),
        device=str(_pick(args.device, config, "device")),
        evaluation_protocol=str(config.get("evaluation_protocol", "full")),
        num_eval_negatives=int(config.get("num_eval_negatives", 100)),
        evaluation_seed=int(config.get("evaluation_seed", 2026)),
        evaluation_chunk_size=int(config.get("evaluation_chunk_size", 4096)),
        evaluation_batch_size=int(config.get("evaluation_batch_size", 128)),
        gradient_log_interval=int(
            gradient_config.get("log_interval", config.get("gradient_log_interval", 10))
        ),
        gradient_conflict_enabled=bool(gradient_config.get("enabled", True)),
        gradient_conflict_ema_beta=float(gradient_config.get("ema_beta", 0.9)),
        gradient_conflict_checkpoint_steps=int(
            gradient_config.get("checkpoint_steps", 1)
        ),
        gradient_conflict_checkpoint_seed=int(
            gradient_config.get("checkpoint_seed", 2026)
        ),
        pcgrad_projection_scope=str(config.get("pcgrad_projection_scope", "backbone")),
        bf16=bool(config.get("bf16", False)),
        data_hash=_data_hash(processed_dir),
        force=args.force,
        progress=bool(config.get("progress", True)) and not args.no_progress,
    )
    domains = (domain,) if method == "single" else tuple(sorted(store.items_by_domain))
    counts: dict[int, int] = {}
    for domain_id in domains:
        if domain_id is None:
            continue
        if method == "single":
            examples = build_single_domain_examples(
                store, split="train", domain=domain_id, maxlen=model_config.maxlen
            )
        else:
            examples = build_mixed_examples(
                store,
                split="train",
                target_domain=domain_id,
                maxlen=model_config.maxlen,
            )
        counts[domain_id] = len(examples)
    resolved_hash = pretrain_config_hash(model_config, settings)
    completion_path = output_dir / "COMPLETE.json"
    decision = "create"
    if completion_path.is_file() and not args.force:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        exact = (
            completion.get("config_hash") == resolved_hash
            and completion.get("data_hash") == settings.data_hash
            and completion.get("method") == settings.method
            and completion.get("seed") == settings.seed
        )
        decision = "skip" if exact else "conflict"
    preview = {
        "config_hash": resolved_hash,
        "data_hash": settings.data_hash,
        "decision": decision,
        "method": method,
        "output_dir": str(output_dir),
        "seed": seed,
        "train_examples": counts,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, sort_keys=True))
        return 0 if decision != "conflict" else 2
    if decision == "skip":
        print(json.dumps(preview, ensure_ascii=False, sort_keys=True))
        return 0
    if decision == "conflict":
        raise FileExistsError(
            f"completed output has another configuration: {output_dir}; use --force"
        )
    result = train_pretraining(store, model_config, settings)
    print(
        json.dumps(
            {
                **preview,
                "best_checkpoint": str(result.best_checkpoint),
                "decision": "completed",
                "test_metrics": result.test_metrics,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
