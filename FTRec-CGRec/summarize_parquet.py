"""Compare completed CGRec Parquet runs with GMFlowRec Table 1."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DOMAIN_NAMES = (
    "Health", "Clothing", "Beauty", "Grocery", "Sports",
)
METRICS = ("hr@5", "hr@10", "ndcg@5", "ndcg@10")
# Percent values reported for CGRec in GMFlowRec Table 1.
PAPER = (
    (16.44, 23.63, 11.20, 13.51),
    (16.40, 23.48, 11.47, 13.75),
    (16.47, 24.17, 11.49, 13.97),
    (14.77, 20.72, 10.15, 12.05),
    (11.66, 17.15, 7.71, 9.48),
)


def summarize(run_dir: Path) -> list[str]:
    runs: dict[int, dict[int, dict]] = {domain: {} for domain in range(5)}
    for path in sorted(run_dir.glob("domain-*/seed-*/results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        domain, seed = result["target_domain"], result["seed"]
        if domain not in runs:
            raise ValueError(f"unexpected target domain {domain} in {path}")
        if seed in runs[domain]:
            raise ValueError(f"duplicate domain {domain}, seed {seed}")
        runs[domain][seed] = result["test"]

    lines = [
        "CGRec on GMFlowRec MDSR-Amazon Parquet; values in percent.",
        "Each metric cell: local mean ± sample standard deviation / GMFlowRec Table 1 CGRec.",
        "| Domain | Runs | HR@5 | HR@10 | NDCG@5 | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, name in enumerate(DOMAIN_NAMES):
        cells = []
        for metric, paper_value in zip(METRICS, PAPER[domain]):
            values = [100 * run[metric] for run in runs[domain].values()]
            if not values:
                local = "—"
            elif len(values) == 1:
                local = f"{values[0]:.2f}"
            else:
                local = f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"
            cells.append(f"{local} / {paper_value:.2f}")
        lines.append(
            f"| {name} | {len(runs[domain])}/5 | " + " | ".join(cells) + " |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=Path("runs/cgrec-parquet"))
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error(f"run directory does not exist: {args.run_dir}")
    print("\n".join(summarize(args.run_dir)))


if __name__ == "__main__":
    main()
