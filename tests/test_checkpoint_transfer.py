from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _write_run(root: Path, arm: str, epoch: int, domain: int, *, joint: float, ft: float, last_epoch: int = 3) -> None:
    base = root / arm / "pretrain" / "joint_proportional" / "all-domains" / "seed-42"
    base.mkdir(parents=True, exist_ok=True)
    with (base / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        if domain == 0:
            stream.write(json.dumps({"epoch": epoch, "validation_macro_ndcg": joint,
                                     "validation": {"0": {"NDCG@10": joint}, "1": {"NDCG@10": joint}}}) + "\n")
    run = root / arm / "adapt" / f"epoch-{epoch:04d}" / f"domain-{domain}" / "seed-42"
    run.mkdir(parents=True, exist_ok=True)
    (run / "result.json").write_text(json.dumps({
        "best_validation_metrics": {"NDCG@10": ft + 0.1},
        "initial_validation_metrics": {"NDCG@10": 0.1},
        "data_hash": "data-a", "num_examples": {"valid": 10},
        "context_mode": "target_only", "min_domain_sequence_length": 2,
        "num_train_negatives": 31, "method": "fullft", "lr": 0.0001,
        "seed": 42, "domain": domain, "best_epoch": 1,
    }), encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps({"epoch": step, "validation": {
            "NDCG@10": 0.1 if step == 0 else ft + 0.1 if step == 1 else ft,
        }})
                  for step in range(last_epoch + 1)) + "\n",
        encoding="utf-8",
    )


def test_checkpoint_transfer_reports_rank_reversal_and_matched_budget(tmp_path: Path) -> None:
    from ftrec.analysis.checkpoint_transfer import summarize_checkpoint_transfer

    for arm in ("id", "semantic"):
        for epoch, joint, ft in ((5, 0.2, 0.4), (10, 0.3, 0.3), (20, 0.4, 0.2)):
            for domain in (0, 1):
                _write_run(tmp_path, arm, epoch, domain, joint=joint, ft=ft)
    report = summarize_checkpoint_transfer(
        tmp_path, arms=("id", "semantic"), epochs=(5, 10, 20),
        domains=(0, 1), seed=42, ft_epochs=3,
    )
    for arm in ("id", "semantic"):
        summary = report["arms"][arm]
        assert summary["spearman_macro"] == pytest.approx(-1.0)
        assert summary["best_joint_epoch"] == 20
        assert summary["best_ft_epoch"] == 5
        assert len(summary["rows"]) == 3
        assert summary["rows"][0]["ft_macro_ndcg10"] == pytest.approx(0.4)


def test_checkpoint_transfer_rejects_short_finetuning(tmp_path: Path) -> None:
    from ftrec.analysis.checkpoint_transfer import summarize_checkpoint_transfer

    _write_run(tmp_path, "id", 5, 0, joint=0.2, ft=0.3, last_epoch=2)
    with pytest.raises(ValueError, match="fixed budget"):
        summarize_checkpoint_transfer(
            tmp_path, arms=("id",), epochs=(5,), domains=(0,), seed=42, ft_epochs=3,
        )


def test_server_plan_uses_matched_models_snapshots_and_budget() -> None:
    root = Path(__file__).parents[1]
    bash = (
        "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    )
    if bash is None:
        pytest.skip("bash is unavailable")
    result = subprocess.run(
        [bash, "scripts/run_checkpoint_transfer.sh"], cwd=root,
        env={**os.environ, "ACTION": "plan", "SNAPSHOT_EPOCHS": "5 10",
             "DOMAINS": "0", "FT_EPOCHS": "3"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("ftrec.cli.pretrain") == 2
    assert result.stdout.count("ftrec.cli.adapt") == 4
    assert "--snapshot-epochs 5 10" in result.stdout
    assert "--epochs 3 --patience 4" in result.stdout


def test_snapshot_adaptation_checks_parent_run_model_config(tmp_path: Path) -> None:
    from ftrec.models.sasrec import SASRecConfig, model_config_dict
    from ftrec.training.adapt import validate_base_model_config
    from ftrec.training.checkpoint import CheckpointMismatchError

    original = SASRecConfig(num_items=10, hidden_size=4)
    other = SASRecConfig(num_items=10, hidden_size=8)
    (tmp_path / "resolved_config.json").write_text(
        json.dumps({"model": model_config_dict(original)}), encoding="utf-8",
    )
    snapshot = tmp_path / "snapshots" / "epoch-0005.pt"
    snapshot.parent.mkdir()
    snapshot.touch()
    validate_base_model_config(snapshot, original)
    with pytest.raises(CheckpointMismatchError, match="model configuration"):
        validate_base_model_config(snapshot, other)


def test_checkpoint_transfer_rejects_inconsistent_selected_result(tmp_path: Path) -> None:
    from ftrec.analysis.checkpoint_transfer import summarize_checkpoint_transfer

    _write_run(tmp_path, "id", 5, 0, joint=0.2, ft=0.3)
    result_path = tmp_path / "id" / "adapt" / "epoch-0005" / "domain-0" / "seed-42" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["best_validation_metrics"]["NDCG@10"] = 0.9
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="selected validation"):
        summarize_checkpoint_transfer(
            tmp_path, arms=("id",), epochs=(5,), domains=(0,), seed=42, ft_epochs=3,
        )
