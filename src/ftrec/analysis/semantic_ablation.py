"""Paired adaptation-gain summary for the semantic alignment ablation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_ARMS = ("id", "random", "shuffled", "attribute", "semantic")


def summarize_semantic_ablation(
    root: str | Path,
    *,
    arms: tuple[str, ...] = DEFAULT_ARMS,
    domains: tuple[int, ...] = (0, 1, 2, 3, 4),
    seed: int = 42,
) -> dict[str, object]:
    root = Path(root)
    if not arms or not domains:
        raise ValueError("arms and domains must be nonempty")
    rows: list[dict[str, object]] = []
    matched: dict[int, tuple[object, ...]] = {}
    for arm in arms:
        for domain in domains:
            path = (
                root / arm / "adapt" / "lora_all" / "joint_proportional"
                / f"domain-{domain}" / "rank-5" / f"seed-{seed}" / "result.json"
            )
            result = json.loads(path.read_text(encoding="utf-8"))
            cohort = (
                result["data_hash"], result["num_examples"],
                result["context_mode"], result["min_domain_sequence_length"],
                result["num_train_negatives"], result["method"],
                result["rank"], result["seed"], result["domain"],
            )
            if domain in matched and cohort != matched[domain]:
                raise ValueError(f"target cohort or training protocol differs for domain {domain}: {arm}")
            matched[domain] = cohort
            before = float(result["pretrain_metrics"]["NDCG@10"])
            after = float(result["test_metrics"]["NDCG@10"])
            if not math.isfinite(before) or not math.isfinite(after):
                raise ValueError(f"non-finite NDCG@10: {path}")
            rows.append({
                "arm": arm, "domain": domain, "seed": seed,
                "pretrain_ndcg10": before, "adapted_ndcg10": after,
                "gain": after - before, "best_epoch": result["best_epoch"],
                "num_trainable_params": result["num_trainable_params"],
            })
    macro = {
        arm: sum(float(row["gain"]) for row in rows if row["arm"] == arm)
        / len(domains)
        for arm in arms
    }
    return {
        "seed": seed,
        "domains": list(domains),
        "rows": rows,
        "macro_gain": macro,
        "aligned_minus_shuffled_gain": (
            macro["semantic"] - macro["shuffled"]
            if "semantic" in macro and "shuffled" in macro else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs-attributes/semantic-ft-ablation"))
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--domains", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = summarize_semantic_ablation(
        args.root, arms=tuple(args.arms), domains=tuple(args.domains), seed=args.seed
    )
    output = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
