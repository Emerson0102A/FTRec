"""Import the official GMFlowRec Amazon Parquet release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ftrec.data.gmflowrec import GMFlowRecImportSettings, import_gmflowrec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/MDSR-Amazon"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/gmflowrec-amazon")
    )
    parser.add_argument("--num-eval-negatives", type=int, default=999)
    parser.add_argument("--evaluation-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--skip-paper-statistics-check", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = GMFlowRecImportSettings(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_eval_negatives=args.num_eval_negatives,
        evaluation_seed=args.evaluation_seed,
        verify_paper_statistics=not args.skip_paper_statistics_check,
        batch_size=args.batch_size,
        progress=not args.no_progress,
        force=args.force,
    )
    preview = {
        "evaluation_seed": settings.evaluation_seed,
        "num_eval_negatives": settings.num_eval_negatives,
        "output_dir": str(settings.output_dir),
        "source_dir": str(settings.source_dir),
        "verify_paper_statistics": settings.verify_paper_statistics,
    }
    if args.dry_run:
        print(json.dumps({**preview, "decision": "dry-run"}, sort_keys=True))
        return 0
    result = import_gmflowrec(settings)
    print(
        json.dumps(
            {
                **preview,
                "data_hash": result.data_hash,
                "decision": "completed",
                "evaluation_sequences": result.evaluation_sequences,
                "interactions": result.interactions,
                "items": result.items,
                "train_sequences": result.train_sequences,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

