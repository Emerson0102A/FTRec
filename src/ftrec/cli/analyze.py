"""Aggregate completed runs, compute Recovery, and render experiment figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ftrec.analysis.plotting import generate_figures
from ftrec.analysis.recovery import (
    RecoveryRow,
    analysis_warnings,
    compute_recovery_rows,
)
from ftrec.analysis.results import (
    add_macro_rows,
    aggregate_results,
    collect_result_rows,
    write_result_rows,
    write_summary_rows,
)
from ftrec.artifacts import RunDirectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--recovery-epsilon", type=float, default=1e-12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _read_gradient_records(root: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("gradient_conflicts.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                records.append(json.loads(line))
    return tuple(records)


def _write_recovery(path: Path, rows: tuple[RecoveryRow, ...]) -> None:
    fields = tuple(
        RecoveryRow(0, "d", "p", 1, "m", "full", 0, 0, 0, 0, False, "").to_dict()
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = collect_result_rows(args.runs_root)
    with_macros = add_macro_rows(rows)
    recovery_rows = compute_recovery_rows(rows, epsilon=args.recovery_epsilon)
    gradients = _read_gradient_records(args.runs_root)
    warnings = analysis_warnings(rows, recovery_rows)
    preview = {
        "gradient_records": len(gradients),
        "recovery_rows": len(recovery_rows),
        "result_rows": len(rows),
        "warnings": warnings,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, sort_keys=True))
        return 0
    with RunDirectory(args.output_dir, force=args.force) as run:
        assert run.path is not None
        write_result_rows(run.path / "results.csv", with_macros)
        write_summary_rows(run.path / "summary.csv", aggregate_results(with_macros))
        _write_recovery(run.path / "recovery.csv", recovery_rows)
        (run.path / "warnings.json").write_text(
            json.dumps(
                {"warnings": warnings},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        figures = generate_figures(
            with_macros, recovery_rows, gradients, run.path / "figures"
        )
        run.complete(
            {
                "figure_count": len(figures),
                "recovery_rows": len(recovery_rows),
                "result_rows": len(rows),
            }
        )
    print(
        json.dumps(
            {**preview, "figure_count": 8, "output_dir": str(args.output_dir)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
