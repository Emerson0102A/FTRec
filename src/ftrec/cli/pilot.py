"""Run the approved one-seed hypothesis pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
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


@dataclass(frozen=True)
class PilotJob:
    stage: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class PilotPhase:
    name: str
    jobs: tuple[PilotJob, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranks", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--pretrain-workers", type=int, default=1)
    parser.add_argument("--adapt-workers", type=int, default=1)
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
    epochs: int = 3,
    steps_per_epoch: int = 100,
) -> tuple[PilotStage, ...]:
    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("pilot LoRA ranks must be positive")
    if len(set(ranks)) != len(ranks):
        raise ValueError("pilot LoRA ranks must be unique")
    if epochs < 1 or steps_per_epoch < 1:
        raise ValueError("pilot epochs and steps_per_epoch must be positive")

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
    common += ["--epochs", str(epochs), "--steps-per-epoch", str(steps_per_epoch)]
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


def _remove_rank_matrix(argv: tuple[str, ...]) -> tuple[str, ...]:
    start = argv.index("--ranks")
    end = start + 1
    while end < len(argv) and not argv[end].startswith("--"):
        end += 1
    return (*argv[:start], *argv[end:])


def expand_pilot_jobs(
    stages: tuple[PilotStage, ...], *, ranks: tuple[int, ...]
) -> tuple[PilotPhase, ...]:
    """Expand aggregate matrix commands into independently schedulable jobs."""
    by_name = {stage.stage: stage for stage in stages}
    pretrain = tuple(
        PilotJob(name, by_name[name].argv)
        for name in ("joint", "pcgrad")
        if name in by_name
    )
    phases: list[PilotPhase] = [PilotPhase("pretrain", pretrain)]
    if "lora" in by_name:
        base = _remove_rank_matrix(by_name["lora"].argv)
        jobs = tuple(
            PilotJob(
                "lora",
                (
                    *base,
                    "--pretrain-method",
                    pretrain_method,
                    "--domain",
                    str(domain),
                    "--rank",
                    str(rank),
                ),
            )
            for pretrain_method in ("joint", "pcgrad")
            for domain in range(5)
            for rank in ranks
        )
        phases.append(PilotPhase("lora", jobs))
    if "fullft" in by_name:
        base = by_name["fullft"].argv
        jobs = tuple(
            PilotJob(
                "fullft",
                (*base, "--pretrain-method", pretrain_method, "--domain", str(domain)),
            )
            for pretrain_method in ("joint", "pcgrad")
            for domain in range(5)
        )
        phases.append(PilotPhase("fullft", jobs))
    return tuple(phases)


def _run_phase(phase: PilotPhase, *, workers: int) -> int:
    active: set[subprocess.Popen] = set()
    lock = threading.Lock()

    def run(job: PilotJob) -> int:
        process = subprocess.Popen(job.argv)
        with lock:
            active.add(process)
        try:
            return process.wait()
        finally:
            with lock:
                active.discard(process)

    def stop(futures: list[concurrent.futures.Future[int]]) -> None:
        for future in futures:
            future.cancel()
        with lock:
            processes = tuple(active)
        for process in processes:
            try:
                process.terminate()
            except OSError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except OSError:
                pass

    if workers == 1:
        for job in phase.jobs:
            return_code = run(job)
            if return_code:
                return return_code
        return 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures: list[concurrent.futures.Future[int]] = []
    try:
        futures = [executor.submit(run, job) for job in phase.jobs]
        for future in concurrent.futures.as_completed(futures):
            return_code = future.result()
            if return_code:
                stop(futures)
                return return_code
        return 0
    except BaseException:
        stop(futures)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pretrain_workers < 1 or args.adapt_workers < 1:
        raise ValueError("pilot worker counts must be positive")
    ranks = tuple(args.ranks)
    stages = build_pilot_stages(
        seed=args.seed,
        ranks=ranks,
        with_fullft=args.with_fullft,
        processed_dir=args.processed_dir,
        device=args.device,
        no_progress=args.no_progress,
        force=args.force,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
    )
    phases = expand_pilot_jobs(stages, ranks=ranks)
    payload = {
        "adapt_workers": args.adapt_workers,
        "model_runs": sum(stage.model_runs for stage in stages),
        "phases": [{"name": phase.name, "jobs": len(phase.jobs)} for phase in phases],
        "pretrain_workers": args.pretrain_workers,
        "ranks": list(ranks),
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "stages": [stage.to_dict() for stage in stages],
        "with_fullft": args.with_fullft,
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    for phase in phases:
        workers = args.pretrain_workers if phase.name == "pretrain" else args.adapt_workers
        return_code = _run_phase(phase, workers=min(workers, len(phase.jobs)))
        if return_code:
            return return_code
    print(json.dumps({**payload, "decision": "completed"}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
