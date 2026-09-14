"""Preprocess Amazon five-domain reviews."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ftrec.config import load_config
from ftrec.data.preprocessing import PreprocessSettings, preprocess_amazon5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/amazon5.yaml"))
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-user-interactions", type=int)
    parser.add_argument("--min-item-interactions", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sqlite-path", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    values = {
        "input_dir": args.input_dir or Path(config["input_dir"]),
        "output_dir": args.output_dir or Path(config["output_dir"]),
        "min_user_interactions": args.min_user_interactions
        or int(config["min_user_interactions"]),
        "min_item_interactions": args.min_item_interactions
        or int(config["min_item_interactions"]),
        "batch_size": args.batch_size or int(config["batch_size"]),
        "sqlite_path": args.sqlite_path,
        "force": args.force,
    }
    result = preprocess_amazon5(PreprocessSettings(**values))
    payload = {
        "output_dir": str(result.output_dir),
        "users": result.export.users,
        "items": result.export.items,
        "interactions": result.export.interactions,
        "validation": asdict(result.validation),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

