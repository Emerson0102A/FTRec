from __future__ import annotations

from pathlib import Path


def test_complete_smoke_runs_every_experiment_branch(tmp_path: Path) -> None:
    from ftrec.smoke import run_smoke

    report = run_smoke(tmp_path / "smoke", seed=42)

    assert report.ok
    assert report.completed == {
        "preprocess": 1,
        "single": 5,
        "joint": 1,
        "pcgrad": 1,
        "lora": 50,
        "fullft": 10,
        "analysis": 1,
    }
    assert report.model_runs == 67
    assert report.lora_ranks == (1, 2, 4, 8, 16)
    assert report.figure_count >= 8
    assert report.evaluation_protocols == frozenset({"sampled"})
    assert report.recovery_rows == 100
    assert len(report.semantic_hash) == 64
    assert len(tuple((tmp_path / "smoke" / "runs").rglob("evaluation_candidates.json"))) == 67
    assert not tuple((tmp_path / "smoke" / "runs").rglob("validation_candidates.json"))
    assert not tuple((tmp_path / "smoke" / "runs").rglob("test_candidates.json"))
    pcgrad_log = next(
        (tmp_path / "smoke" / "runs" / "pretrain" / "pcgrad").rglob(
            "gradient_conflicts.jsonl"
        )
    )
    assert '"projection_counts"' in pcgrad_log.read_text(encoding="utf-8")


def test_complete_smoke_is_semantically_deterministic(tmp_path: Path) -> None:
    from ftrec.smoke import run_smoke

    first = run_smoke(tmp_path / "first", seed=42)
    second = run_smoke(tmp_path / "second", seed=42)

    assert first.semantic_hash == second.semantic_hash
