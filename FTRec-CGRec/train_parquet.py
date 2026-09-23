"""Train released-code or metadata-hierarchy CGRec on GMFlowRec Parquet splits."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from parquet_data import (
    CGRecCategoryMapping, CGRecDomainCategoryMapping,
    CGRecEvaluationDataset, CGRecTrainDataset,
    domain_remap, load_parquet_data,
)
from parquet_eval import evaluate_model
from parquet_model import CGRecParquetModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet_dir", type=Path, required=True)
    category_options = parser.add_mutually_exclusive_group(required=True)
    category_options.add_argument("--category_catalog", type=Path)
    category_options.add_argument("--official_domain_categories", action="store_true")
    category_options.add_argument("--item_only", action="store_true")
    parser.add_argument("--run_dir", type=Path, default=Path("runs/cgrec-parquet"))
    parser.add_argument("--target_domain", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_eval_negatives", type=int, default=999)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--disable_shapley", action="store_true")
    parser.add_argument("--no_precompute_eval", action="store_true")
    parser.add_argument("--cpu_threads", type=int, default=2)
    args = parser.parse_args()
    positive = (
        "epochs", "patience", "batch_size", "eval_batch_size", "maxlen",
        "hidden_size", "num_layers", "num_heads", "num_eval_negatives",
        "lr", "cpu_threads",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("training, model, and evaluation sizes must be positive")
    if args.num_workers < 0 or args.weight_decay < 0:
        parser.error("num_workers and weight_decay cannot be negative")
    if args.hidden_size % args.num_heads:
        parser.error("hidden_size must be divisible by num_heads")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0, 1)")
    for name in ("max_train_examples", "max_eval_examples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int, pin: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        # Workers must pick up train_data.set_epoch() before each epoch so
        # sampled training negatives change reproducibly.
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),
    )


def train_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    examples = 0
    for batch in loader:
        tensors = [value.to(device, non_blocking=True) for value in batch[:-1]]
        loss = model.train_loss(*tensors)
        if not torch.isfinite(loss):
            raise ValueError("CGRec produced a non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        size = len(tensors[0])
        total_loss += float(loss.detach()) * size
        examples += size
    if not examples:
        raise ValueError("training split is empty")
    return total_loss / examples


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke test")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    started = time.time()
    data = load_parquet_data(args.parquet_dir)
    if args.category_catalog is not None:
        categories = CGRecCategoryMapping(
            args.category_catalog, args.parquet_dir / "mappings.pkl", data.metadata
        )
    elif args.official_domain_categories:
        categories = CGRecDomainCategoryMapping(data.metadata, args.target_domain)
    else:
        categories = None
    train_data = CGRecTrainDataset(
        data.train, data.metadata, args.target_domain, args.maxlen, args.seed,
        max_examples=args.max_train_examples, categories=categories,
    )
    valid_data = CGRecEvaluationDataset(
        data.valid, data.metadata, args.target_domain, args.maxlen,
        args.num_eval_negatives, args.eval_seed, args.max_eval_examples,
        categories=categories,
    )
    test_data = CGRecEvaluationDataset(
        data.test, data.metadata, args.target_domain, args.maxlen,
        args.num_eval_negatives, args.eval_seed, args.max_eval_examples,
        categories=categories,
    )
    if not valid_data or not test_data:
        raise ValueError("selected target domain has no validation or test cases")
    if not args.no_precompute_eval:
        valid_data.precompute()
        test_data.precompute()
    pin = device.type == "cuda"
    train_loader = make_loader(
        train_data, args.batch_size, True, args.num_workers, args.seed, pin
    )
    valid_loader = make_loader(
        valid_data, args.eval_batch_size, False, args.num_workers, args.eval_seed, pin
    )
    test_loader = make_loader(
        test_data, args.eval_batch_size, False, args.num_workers, args.eval_seed, pin
    )
    model = CGRecParquetModel(
        item_count=data.metadata.item_count,
        maxlen=args.maxlen,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device=device,
        shapley=not args.disable_shapley,
        cat1_size=categories.cat1_size if categories else 1,
        cat2_size=categories.cat2_size if categories else 1,
        hierarchical=categories is not None,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output = args.run_dir / f"domain-{args.target_domain}" / f"seed-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration["parquet_dir"] = str(args.parquet_dir.resolve())
    if args.category_catalog is not None:
        configuration["category_catalog"] = str(args.category_catalog.resolve())
    configuration["run_dir"] = str(output.resolve())
    (output / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True), encoding="utf-8"
    )

    best_ndcg = -1.0
    best_epoch = 0
    best_validation = None
    history = []
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_data.set_epoch(epoch)
        training_loss = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate_model(model, valid_loader, device)
        history.append({"epoch": epoch, "training_loss": training_loss, "validation": validation})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if validation["ndcg@10"] > best_ndcg + 1e-12:
            best_ndcg = float(validation["ndcg@10"])
            best_epoch = epoch
            best_validation = validation
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, output / "best.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate_model(model, test_loader, device)
    report = {
        "target_domain": args.target_domain,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "validation": best_validation,
        "test": test_metrics,
        "history": history,
        "elapsed_seconds": round(time.time() - started, 2),
        "protocol": {
            "source": "cpark88/CGRec at 9108dd04637decd616448e30d1d604eddb2a3543",
            "dataset": "GMFlowRec MDSR-Amazon train_new/valid_new/test_new.parquet",
            "category_features": categories.source if categories else "unavailable; item-level CGRec",
            "category_catalog": (
                str(categories.path.resolve()) if categories and categories.path else None
            ),
            "category_catalog_sha256": categories.sha256 if categories else None,
            "cat1_size": categories.cat1_size if categories else None,
            "cat2_size": categories.cat2_size if categories else None,
            "category_missing_coarse": categories.missing_coarse if categories else None,
            "category_missing_fine": categories.missing_fine if categories else None,
            "category_levels": categories.level_description if categories else None,
            "shapley": not args.disable_shapley,
            "item_id": "parquet zero-based + 5; IDs 0..4 reserved",
            "domain_mapping": domain_remap(args.target_domain),
            "eval_negatives": args.num_eval_negatives,
            "eval_negative_seed": args.eval_seed,
            "eval_negative_sampling": "GMFlowRecEvaluationDataset, same domain, distinct, unseen",
            "checkpoint_selection": "best validation ndcg@10",
            "train_sequences": len(train_data),
            "validation_sequences": len(valid_data),
            "test_sequences": len(test_data),
        },
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
