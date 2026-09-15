"""Run the approved one-seed hypothesis pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PilotStage:
    stage: str
    model_runs: int
    argv: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "model_runs": self.model_runs,
            "stage": self.stage,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranks", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--with-fullft", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _base_command(module: str, config: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        module,
        "--config",
        config,
        "--seed",
        str(seed),
    ]


def build_pilot_stages(
    *,
    seed: int,
    ranks: tuple[int, ...],
    with_fullft: bool,
    processed_dir: Path | None = None,
    device: str | None = None,
    no_progress: bool = False,
    force: bool = False,
) -> tuple[PilotStage, ...]:
    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("pilot LoRA ranks must be positive")
    if len(set(ranks)) != len(ranks):
        raise ValueError("pilot LoRA ranks must be unique")

    commands = [
        PilotStage(
            "joint",
            1,
            tuple(
                _base_command(
                    "ftrec.cli.pretrain", "configs/experiment/joint.yaml", seed
                )
            ),
        ),
        PilotStage(
            "pcgrad",
            1,
            tuple(
                _base_command(
                    "ftrec.cli.pretrain", "configs/experiment/pcgrad.yaml", seed
                )
            ),
        ),
        PilotStage(
            "lora",
            2 * 5 * len(ranks),
            tuple(
                _base_command("ftrec.cli.adapt", "configs/experiment/lora.yaml", seed)
                + ["--ranks", *(str(rank) for rank in ranks)]
            ),
        ),
    ]
    if with_fullft:
        commands.append(
            PilotStage(
                "fullft",
                2 * 5,
                tuple(
                    _base_command(
                        "ftrec.cli.adapt", "configs/experiment/fullft.yaml", seed
                    )
                ),
            )
        )

    common: list[str] = []
    if processed_dir is not None:
        common += ["--processed-dir", str(processed_dir)]
    if device is not None:
        common += ["--device", device]
    if no_progress:
        common.append("--no-progress")
    if force:
        common.append("--force")
    return tuple(
        PilotStage(stage.stage, stage.model_runs, (*stage.argv, *common))
        for stage in commands
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ranks = tuple(args.ranks)
    stages = build_pilot_stages(
        seed=args.seed,
        ranks=ranks,
        with_fullft=args.with_fullft,
        processed_dir=args.processed_dir,
        device=args.device,
        no_progress=args.no_progress,
        force=args.force,
    )
    payload = {
        "model_runs": sum(stage.model_runs for stage in stages),
        "ranks": list(ranks),
        "seed": args.seed,
        "stages": [stage.to_dict() for stage in stages],
        "with_fullft": args.with_fullft,
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    for stage in stages:
        result = subprocess.run(stage.argv, check=False)
        if result.returncode != 0:
            return result.returncode
    print(json.dumps({**payload, "decision": "completed"}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
