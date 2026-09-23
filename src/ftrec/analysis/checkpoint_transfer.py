"""Compare joint validation quality with fixed-budget target-domain adaptation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    for index, value in enumerate(values):
        equals = [position for position, candidate in enumerate(values) if candidate == value]
        result[index] = sum(sorted(values).index(value) + 1 + offset for offset in range(len(equals))) / len(equals)
    return result


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x, y = _ranks(left), _ranks(right)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def _ndcg(result: dict[str, object], key: str, path: Path) -> float:
    metric = float(result[key]["NDCG@10"])  # type: ignore[index]
    if not math.isfinite(metric):
        raise ValueError(f"non-finite {key} NDCG@10: {path}")
    return metric


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_checkpoint_transfer(
    root: str | Path,
    *,
    arms: tuple[str, ...] = ("id", "semantic"),
    epochs: tuple[int, ...] = (5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300),
    domains: tuple[int, ...] = (0, 1, 2, 3, 4),
    seed: int = 42,
    ft_epochs: int = 10,
) -> dict[str, object]:
    root = Path(root)
    if not arms or not epochs or not domains or ft_epochs < 1:
        raise ValueError("arms, epochs, and domains must be nonempty; ft_epochs must be positive")
    if len(set(arms)) != len(arms) or len(set(epochs)) != len(epochs) or len(set(domains)) != len(domains):
        raise ValueError("arms, epochs, and domains must be unique")
    cohort_by_domain: dict[int, tuple[object, ...]] = {}
    summaries: dict[str, object] = {}
    for arm in arms:
        pretrain_path = root / arm / "pretrain" / "joint_proportional" / "all-domains" / f"seed-{seed}" / "metrics.jsonl"
        pretrain = {int(record["epoch"]): record for record in _records(pretrain_path)}
        rows: list[dict[str, object]] = []
        for epoch in epochs:
            if epoch not in pretrain:
                raise ValueError(f"missing pretrain epoch {epoch}: {pretrain_path}")
            joint_macro = float(pretrain[epoch]["validation_macro_ndcg"])
            if not math.isfinite(joint_macro):
                raise ValueError(f"non-finite joint NDCG@10: {pretrain_path} epoch {epoch}")
            ft_by_domain: dict[int, float] = {}
            selected_ft_by_domain: dict[int, float] = {}
            initial_by_domain: dict[int, float] = {}
            joint_by_domain: dict[int, float] = {}
            for domain in domains:
                run = root / arm / "adapt" / f"epoch-{epoch:04d}" / f"domain-{domain}" / f"seed-{seed}"
                result_path = run / "result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                ft_records = _records(run / "metrics.jsonl")
                observed_epochs = [int(record["epoch"]) for record in ft_records]
                if observed_epochs != list(range(ft_epochs + 1)):
                    raise ValueError(f"fixed budget of {ft_epochs} epochs was not completed: {run}")
                if int(result["best_epoch"]) > ft_epochs:
                    raise ValueError(f"best_epoch exceeds fixed budget: {run}")
                validation_values = [
                    _ndcg(record, "validation", run / "metrics.jsonl")
                    for record in ft_records
                ]
                selected_epoch = max(range(len(validation_values)), key=validation_values.__getitem__)
                selected_value = _ndcg(result, "best_validation_metrics", result_path)
                if (int(result["best_epoch"]) != selected_epoch
                        or not math.isclose(selected_value, validation_values[selected_epoch], abs_tol=1e-10)
                        or not math.isclose(
                            _ndcg(result, "initial_validation_metrics", result_path),
                            validation_values[0], abs_tol=1e-10,
                        )):
                    raise ValueError(f"selected validation does not match epoch metrics: {run}")
                cohort = tuple(result[key] for key in (
                    "data_hash", "num_examples", "context_mode", "min_domain_sequence_length",
                    "num_train_negatives", "method", "lr", "seed", "domain",
                ))
                if domain in cohort_by_domain and cohort != cohort_by_domain[domain]:
                    raise ValueError(f"target cohort or fine-tuning protocol differs: {run}")
                cohort_by_domain[domain] = cohort
                ft_by_domain[domain] = validation_values[-1]
                selected_ft_by_domain[domain] = selected_value
                initial_by_domain[domain] = validation_values[0]
                joint_by_domain[domain] = float(pretrain[epoch]["validation"][str(domain)]["NDCG@10"])  # type: ignore[index]
            rows.append({
                "epoch": epoch,
                "joint_macro_ndcg10": joint_macro,
                "ft_macro_ndcg10": sum(ft_by_domain.values()) / len(domains),
                "selected_ft_macro_ndcg10": sum(selected_ft_by_domain.values()) / len(domains),
                "initial_target_macro_ndcg10": sum(initial_by_domain.values()) / len(domains),
                "ft_gain_macro_ndcg10": sum(ft_by_domain[d] - initial_by_domain[d] for d in domains) / len(domains),
                "joint_by_domain": joint_by_domain,
                "ft_by_domain": ft_by_domain,
                "selected_ft_by_domain": selected_ft_by_domain,
                "initial_target_by_domain": initial_by_domain,
            })
        summaries[arm] = {
            "rows": rows,
            "spearman_macro": _spearman(
                [float(row["joint_macro_ndcg10"]) for row in rows],
                [float(row["ft_macro_ndcg10"]) for row in rows],
            ),
            "spearman_by_domain": {
                domain: _spearman(
                    [float(row["joint_by_domain"][domain]) for row in rows],  # type: ignore[index]
                    [float(row["ft_by_domain"][domain]) for row in rows],  # type: ignore[index]
                ) for domain in domains
            },
            "best_joint_epoch": max(rows, key=lambda row: float(row["joint_macro_ndcg10"]))["epoch"],
            "best_ft_epoch": max(rows, key=lambda row: float(row["ft_macro_ndcg10"]))["epoch"],
            "best_ft_by_domain": {
                domain: max(rows, key=lambda row: float(row["ft_by_domain"][domain]))["epoch"]  # type: ignore[index]
                for domain in domains
            },
        }
    return {"seed": seed, "epochs": list(epochs), "domains": list(domains),
            "ft_budget_epochs": ft_epochs, "arms": summaries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs-attributes/checkpoint-transfer"))
    parser.add_argument("--arms", nargs="+", default=("id", "semantic"))
    parser.add_argument("--epochs", type=int, nargs="+", default=(5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300))
    parser.add_argument("--domains", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = summarize_checkpoint_transfer(
        args.root, arms=tuple(args.arms), epochs=tuple(args.epochs),
        domains=tuple(args.domains), seed=args.seed, ft_epochs=args.ft_epochs,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
