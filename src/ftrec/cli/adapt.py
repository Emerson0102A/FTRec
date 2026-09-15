"""Run the LoRA or FullFT adaptation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from ftrec.artifacts import sha256_file
from ftrec.config import load_config
from ftrec.data.datasets import SequenceStore
from ftrec.models.sasrec import SASRecConfig
from ftrec.training.adapt import AdaptSettings, adapt_config_hash, train_adaptation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/lora.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/sasrec.yaml")
    )
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=("lora", "fullft"))
    parser.add_argument("--pretrain-method", choices=("joint", "pcgrad"))
    parser.add_argument("--domain", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--ranks", type=int, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _list_or_selected(value: object, selected: object | None) -> tuple[object, ...]:
    if selected is not None:
        return (selected,)
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _data_hash(processed_dir: Path) -> str:
    completion = processed_dir / "COMPLETE.json"
    if completion.is_file():
        value = json.loads(completion.read_text(encoding="utf-8"))
        if value.get("data_hash"):
            return str(value["data_hash"])
    return sha256_file(processed_dir / "manifest.json")


def _base_path(root: Path, pretrain_method: str, seed: int) -> Path:
    return root / "pretrain" / pretrain_method / "all-domains" / f"seed-{seed}" / "best.pt"


def _output_path(
    root: Path,
    method: str,
    pretrain_method: str,
    domain: int,
    rank: int | None,
    seed: int,
) -> Path:
    path = root / "adapt" / method / pretrain_method / f"domain-{domain}"
    if rank is not None:
        path = path / f"rank-{rank}"
    return path / f"seed-{seed}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    method = args.method or str(config["method"])
    processed_dir = args.processed_dir or Path(str(config["processed_dir"]))
    output_root = Path(str(config.get("output_root", "runs")))
    base_root = Path(str(config.get("base_root", output_root)))
    pretrain_methods = tuple(
        str(value)
        for value in _list_or_selected(
            config.get("pretrain_methods", ("joint", "pcgrad")),
            args.pretrain_method,
        )
    )
    domains = tuple(
        int(value)
        for value in _list_or_selected(config.get("domains", range(5)), args.domain)
    )
    seeds = tuple(
        int(value) for value in _list_or_selected(config.get("seeds", (42,)), args.seed)
    )
    ranks: tuple[int | None, ...]
    if method == "lora":
        if args.rank is not None and args.ranks is not None:
            raise ValueError("--rank and --ranks cannot be used together")
        if args.ranks is not None:
            ranks = tuple(args.ranks)
        else:
            ranks = tuple(
                int(value)
                for value in _list_or_selected(
                    config.get("ranks", (1, 2, 4, 8, 16)), args.rank
                )
            )
    else:
        if args.rank is not None or args.ranks is not None:
            raise ValueError("--rank and --ranks are only valid with LoRA")
        ranks = (None,)
    combinations = len(pretrain_methods) * len(domains) * len(ranks) * len(seeds)
    progress_enabled = bool(config.get("progress", True)) and not args.no_progress
    if args.output_dir is not None and combinations != 1:
        raise ValueError("--output-dir requires selecting one exact combination")
    if args.base_checkpoint is not None and combinations != 1:
        raise ValueError("--base-checkpoint requires selecting one exact combination")

    store = SequenceStore.from_processed(processed_dir)
    model_values = load_config(args.model_config)
    model_values["num_items"] = max(
        item for catalog in store.items_by_domain.values() for item in catalog
    )
    model_config = SASRecConfig(**model_values)
    data_hash = _data_hash(processed_dir)
    reports: list[dict[str, object]] = []
    failures = 0
    matrix_progress = tqdm(
        total=combinations,
        desc=f"{method} matrix",
        unit="run",
        mininterval=1.0,
        dynamic_ncols=True,
        disable=not progress_enabled,
    )
    for pretrain_method in pretrain_methods:
        for domain in domains:
            for rank in ranks:
                for seed in seeds:
                    base = args.base_checkpoint or _base_path(
                        base_root, pretrain_method, seed
                    )
                    output = args.output_dir or _output_path(
                        output_root,
                        method,
                        pretrain_method,
                        domain,
                        rank,
                        seed,
                    )
                    alpha_config = config.get("alpha")
                    alpha = (
                        float(alpha_config)
                        if alpha_config is not None
                        else float(rank) if rank is not None else None
                    )
                    settings = AdaptSettings(
                        method=method,
                        pretrain_method=pretrain_method,
                        domain=domain,
                        output_dir=output,
                        rank=rank,
                        alpha=alpha,
                        seed=seed,
                        batch_size=int(config["batch_size"]),
                        steps_per_epoch=int(config["steps_per_epoch"]),
                        epochs=int(config["epochs"]),
                        patience=int(config["patience"]),
                        lr=float(config["lr"]),
                        embedding_lr=(
                            float(config["embedding_lr"])
                            if config.get("embedding_lr") is not None
                            else None
                        ),
                        weight_decay=float(config.get("weight_decay", 0.0)),
                        grad_clip_norm=float(config.get("grad_clip_norm", 5.0)),
                        device=args.device or str(config["device"]),
                        evaluation_protocol=str(
                            config.get("evaluation_protocol", "full")
                        ),
                        num_eval_negatives=int(config.get("num_eval_negatives", 100)),
                        evaluation_seed=int(config.get("evaluation_seed", 2026)),
                        evaluation_chunk_size=int(
                            config.get("evaluation_chunk_size", 4096)
                        ),
                        evaluation_batch_size=int(
                            config.get("evaluation_batch_size", 128)
                        ),
                        bf16=bool(config.get("bf16", False)),
                        data_hash=data_hash,
                        force=args.force,
                        progress=(
                            progress_enabled
                        ),
                    )
                    config_hash = adapt_config_hash(model_config, settings)
                    decision = "create"
                    completion_path = output / "COMPLETE.json"
                    if not base.is_file():
                        decision = "missing-base"
                    elif completion_path.is_file() and not args.force:
                        completion = json.loads(
                            completion_path.read_text(encoding="utf-8")
                        )
                        exact = (
                            completion.get("config_hash") == config_hash
                            and completion.get("data_hash") == data_hash
                            and completion.get("base_hash") == sha256_file(base)
                        )
                        decision = "skip" if exact else "conflict"
                    report: dict[str, object] = {
                        "base_checkpoint": str(base),
                        "config_hash": config_hash,
                        "decision": decision,
                        "domain": domain,
                        "method": method,
                        "output_dir": str(output),
                        "pretrain_method": pretrain_method,
                        "rank": rank,
                        "seed": seed,
                    }
                    reports.append(report)
                    if args.dry_run or decision == "skip":
                        failures += decision in {"missing-base", "conflict"}
                        matrix_progress.set_postfix(
                            decision=decision, domain=domain, seed=seed
                        )
                        matrix_progress.update(1)
                        continue
                    if decision != "create":
                        matrix_progress.close()
                        raise FileExistsError(
                            f"cannot run {output}: {decision}; use --force for conflicts"
                        )
                    result = train_adaptation(store, model_config, base, settings)
                    report["best_checkpoint"] = str(result.best_checkpoint)
                    report["decision"] = "completed"
                    matrix_progress.set_postfix(
                        decision="completed", domain=domain, seed=seed
                    )
                    matrix_progress.update(1)
    matrix_progress.close()
    print(
        json.dumps(
            {"combinations": combinations, "runs": reports},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
