"""Run the complete synthetic CPU experiment smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ftrec.config import load_config
from ftrec.smoke import run_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = args.output_dir or Path(str(config["output_dir"]))
    seed = args.seed if args.seed is not None else int(config["seed"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "decision": (
                        "replace"
                        if output_dir.exists() and args.force
                        else "conflict" if output_dir.exists() else "create"
                    ),
                    "model_runs": 67,
                    "output_dir": str(output_dir),
                    "seed": seed,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    report = run_smoke(
        output_dir,
        seed=seed,
        force=args.force,
    )
    print(
        json.dumps(
            {
                **report.__dict__,
                "output_dir": str(report.output_dir),
                "evaluation_protocols": sorted(report.evaluation_protocols),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
