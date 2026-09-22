"""Train and evaluate the paper-driven GMFlowRec reproduction."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gmflowrec import GMFlowRec, GMFlowRecConfig
from gmflowrec_data import (
    GMFlowRecEvaluationDataset,
    GMFlowRecTrainDataset,
    evaluate_gmflowrec,
)
from mdsr_parquet import MDSRParquetData


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return lowered == "true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet_dir", default="data/MDSR-Amazon")
    parser.add_argument("--run_dir", default="runs/gmflowrec")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval_seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_mixtures", type=int, default=8)
    parser.add_argument("--fusion_weight", type=float, default=0.9)
    parser.add_argument("--prior_weight", type=float, default=0.1)
    parser.add_argument("--gmm_weight", type=float, default=1e-4)
    parser.add_argument("--min_scale", type=float, default=1e-3)
    parser.add_argument("--ode_steps", type=int, default=8)
    parser.add_argument("--num_eval_negatives", type=int, default=999)
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="optional prefix limit for smoke tests",
    )
    parser.add_argument(
        "--max_eval_examples",
        type=int,
        default=None,
        help="optional prefix limit per evaluation split for smoke tests",
    )
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--inference_only", type=str2bool, default=False)
    args = parser.parse_args()
    positive = (
        "batch_size", "eval_batch_size", "epochs", "patience", "eval_every",
        "maxlen", "hidden_units", "num_blocks", "num_heads", "num_mixtures",
        "ode_steps", "num_eval_negatives",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("batch, model, epoch, and evaluation counts must be positive")
    if args.max_train_examples is not None and args.max_train_examples <= 0:
        parser.error("--max_train_examples must be positive")
    if args.max_eval_examples is not None and args.max_eval_examples <= 0:
        parser.error("--max_eval_examples must be positive")
    if args.inference_only and not args.checkpoint:
        parser.error("--inference_only true requires --checkpoint")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, shuffle: bool, args, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


def load_checkpoint(model: GMFlowRec, path: str, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise ValueError("GMFlowRec checkpoint is missing its model state")
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = MDSRParquetData(args.parquet_dir)
    config = GMFlowRecConfig(
        item_count=data.metadata.item_count,
        domain_offsets=data.metadata.domain_offsets,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        num_mixtures=args.num_mixtures,
        fusion_weight=args.fusion_weight,
        prior_weight=args.prior_weight,
        gmm_weight=args.gmm_weight,
        min_scale=args.min_scale,
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"arguments": vars(args), "model": config.to_dict(), "data": data.summary()},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    model = GMFlowRec(config).to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)

    valid_dataset = GMFlowRecEvaluationDataset(
        data.valid, data.metadata, args.maxlen, args.num_eval_negatives, args.eval_seed
    )
    test_dataset = GMFlowRecEvaluationDataset(
        data.test, data.metadata, args.maxlen, args.num_eval_negatives, args.eval_seed
    )
    if args.max_eval_examples is not None:
        valid_dataset = Subset(
            valid_dataset, range(min(args.max_eval_examples, len(valid_dataset)))
        )
        test_dataset = Subset(
            test_dataset, range(min(args.max_eval_examples, len(test_dataset)))
        )
    valid_loader = make_loader(
        valid_dataset, args.eval_batch_size, False, args, args.eval_seed
    )
    test_loader = make_loader(
        test_dataset, args.eval_batch_size, False, args, args.eval_seed
    )

    if args.inference_only:
        metrics = evaluate_gmflowrec(model, test_loader, device, args.ode_steps)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    train_dataset = GMFlowRecTrainDataset(data.train, args.maxlen)
    if args.max_train_examples is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.max_train_examples, len(train_dataset)))
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    best_score = float("-inf")
    best_epoch = 0
    stale_evaluations = 0
    best_path = run_dir / "best.pt"
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(
            train_dataset, args.batch_size, True, args, args.seed + epoch
        )
        model.train()
        running = {"loss": 0.0, "recommendation_loss": 0.0,
                   "prior_loss": 0.0, "gmm_loss": 0.0}
        for step, (items, domains, targets, target_domains, _) in enumerate(
            train_loader, 1
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if use_amp else torch.float32,
                enabled=use_amp,
            ):
                losses = model.training_objective(
                    items.to(device, non_blocking=True),
                    domains.to(device, non_blocking=True),
                    targets.to(device, non_blocking=True),
                    target_domains.to(device, non_blocking=True),
                )
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for name in running:
                running[name] += float(losses[name].detach())
            if step % 100 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    + " ".join(f"{key}={value / step:.6f}" for key, value in running.items())
                )

        record = {
            "epoch": epoch,
            "train": {key: value / len(train_loader) for key, value in running.items()},
        }
        if epoch % args.eval_every == 0:
            validation = evaluate_gmflowrec(
                model, valid_loader, device, steps=args.ode_steps
            )
            record["validation"] = validation
            score = validation["overall"]["ndcg@10"]
            print(
                f"validation epoch={epoch} NDCG@10={score:.6f} "
                f"HR@10={validation['overall']['hr@10']:.6f}"
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale_evaluations = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": config.to_dict(),
                        "epoch": epoch,
                        "validation": validation,
                    },
                    best_path,
                )
            else:
                stale_evaluations += 1
        history.append(record)
        if stale_evaluations >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    checkpoint = load_checkpoint(model, str(best_path), device)
    test_metrics = evaluate_gmflowrec(model, test_loader, device, args.ode_steps)
    result = {
        "best_epoch": best_epoch,
        "best_validation": checkpoint["validation"],
        "test_at_selected_epoch": test_metrics,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "implementation": "paper-driven reproduction; see docs/gmflowrec-reproduction.md",
    }
    with (run_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(test_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
