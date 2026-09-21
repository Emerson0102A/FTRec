"""Evaluate content towers and target-item train-frequency buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ftrec.analysis.content_diagnostics import (
    DEFAULT_FREQUENCY_LOWER_BOUNDS,
    DualTowerScoringView,
    evaluate_component,
    frequency_buckets,
    training_item_frequencies,
)
from ftrec.config import load_config
from ftrec.data.amazon import DOMAIN_BY_ID
from ftrec.data.datasets import SequenceStore
from ftrec.data.sampling import resolve_evaluation_candidates
from ftrec.models.sasrec import SASRec, SASRecConfig
from ftrec.reproducibility import resolve_device
from ftrec.training.checkpoint import load_checkpoint
from ftrec.training.pretrain import build_pretraining_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--frequency-lower-bounds",
        default=",".join(str(value) for value in DEFAULT_FREQUENCY_LOWER_BOUNDS),
        help="comma-separated inclusive lower bounds; must start at 0",
    )
    return parser


def _parse_bounds(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("frequency lower bounds must be integers") from error


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment = load_config(args.config)
    model_values = load_config(args.model_config)
    processed_dir = args.processed_dir or Path(str(experiment["processed_dir"]))
    store = SequenceStore.from_processed(processed_dir)
    num_items = max(
        (item for values in store.items_by_domain.values() for item in values),
        default=0,
    )
    model_config = SASRecConfig(**{**model_values, "num_items": num_items})
    if model_config.item_embedding_mode == "id":
        raise ValueError("content diagnostics require a content-only model")
    device = resolve_device(str(args.device or experiment.get("device", "cpu")))
    model = SASRec(model_config).to(device)
    loaded = load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    method = str(experiment["method"])
    domains = tuple(sorted(store.items_by_domain))
    examples_by_domain = build_pretraining_examples(
        store,
        split=args.split,
        domains=domains,
        maxlen=model_config.maxlen,
        method=method,
    )
    protocol = str(experiment.get("evaluation_protocol", "full"))
    negative_count = int(experiment.get("num_eval_negatives", 100))
    evaluation_seed = int(experiment.get("evaluation_seed", 2026))
    split_offset = 10_000 if args.split == "valid" else 20_000
    candidates_by_domain = None
    if protocol == "sampled":
        candidates_by_domain = {
            domain: resolve_evaluation_candidates(
                store,
                examples,
                split=args.split,
                domain=domain,
                count=negative_count,
                evaluation_seed=evaluation_seed,
                split_offset=split_offset,
            )
            for domain, examples in examples_by_domain.items()
        }

    buckets = frequency_buckets(_parse_bounds(args.frequency_lower_bounds))
    frequencies = training_item_frequencies(store)
    if model_config.item_embedding_mode == "content_dual":
        components: dict[str, object] = {
            name: DualTowerScoringView(model, name).eval()
            for name in ("fusion", "title", "attribute")
        }
    else:
        components = {"fused": model}

    results = {}
    for name, scoring_model in components.items():
        results[name] = evaluate_component(
            scoring_model,
            examples_by_domain,
            store.items_by_domain,
            candidates_by_domain,
            protocol=protocol,
            chunk_size=int(experiment.get("evaluation_chunk_size", 4096)),
            batch_size=int(experiment.get("evaluation_batch_size", 128)),
            device=device,
            progress=bool(experiment.get("progress", True)) and not args.no_progress,
            description_prefix=name,
            frequencies=frequencies,
            buckets=buckets,
        )

    output = args.output or args.checkpoint.parent / "content-diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": loaded.metadata.get("epoch"),
        "model_config": str(args.model_config),
        "experiment_config": str(args.config),
        "split": args.split,
        "evaluation_protocol": protocol,
        "evaluation_seed": evaluation_seed,
        "num_eval_negatives": negative_count if protocol == "sampled" else None,
        "frequency_definition": (
            "number of item occurrences in imported training-cohort sequences only"
        ),
        "frequency_lower_bounds": [bucket.lower for bucket in buckets],
        "domain_names": {
            str(domain): (
                DOMAIN_BY_ID[domain].name if domain in DOMAIN_BY_ID else str(domain)
            )
            for domain in domains
        },
        "components": results,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "components": list(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
