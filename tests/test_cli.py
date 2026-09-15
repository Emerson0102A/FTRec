from __future__ import annotations

import gzip
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


CLI_MODULES = (
    "ftrec.cli.preprocess",
    "ftrec.cli.pretrain",
    "ftrec.cli.adapt",
    "ftrec.cli.analyze",
    "ftrec.cli.smoke",
)


@pytest.mark.parametrize("module", CLI_MODULES)
def test_all_commands_expose_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_preprocess_and_smoke_dry_run_do_not_create_outputs(tmp_path: Path) -> None:
    preprocess_output = tmp_path / "processed"
    preprocess = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftrec.cli.preprocess",
            "--input-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(preprocess_output),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert preprocess.returncode == 0, preprocess.stderr
    assert '"decision": "create"' in preprocess.stdout
    assert not preprocess_output.exists()

    smoke_output = tmp_path / "smoke"
    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftrec.cli.smoke",
            "--output-dir",
            str(smoke_output),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert '"model_runs": 67' in smoke.stdout
    assert not smoke_output.exists()


def test_server_scripts_are_fail_fast_and_stage_scoped() -> None:
    root = Path(__file__).parents[1]
    names = (
        "run_preprocess.sh",
        "run_single.sh",
        "run_joint.sh",
        "run_pcgrad.sh",
        "run_lora.sh",
        "run_fullft.sh",
        "run_analysis.sh",
        "check_environment.sh",
    )
    for name in names:
        text = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert 'BASH_SOURCE[0]' in text


def test_production_training_configs_enable_bf16() -> None:
    root = Path(__file__).parents[1]
    for name in ("single.yaml", "joint.yaml", "pcgrad.yaml", "lora.yaml", "fullft.yaml"):
        config = yaml.safe_load(
            (root / "configs" / "experiment" / name).read_text(encoding="utf-8")
        )
        assert config["bf16"] is True


def test_adaptation_dry_run_reports_matrix_progress(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    with gzip.open(processed / "sequences.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(
            '{"domain_ids":[0,0,0],"item_ids":[1,2,3],"splits":["train","valid","test"],"timestamps":[1,2,3],"user_id":1}\n'
        )
    with gzip.open(processed / "items.csv.gz", "wt", encoding="utf-8") as stream:
        stream.write("item_id,domain_id,domain,parent_asin\n1,0,Health,item-1\n")
    (processed / "manifest.json").write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "fullft.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "processed_dir": str(processed),
                "base_root": str(tmp_path / "runs"),
                "output_root": str(tmp_path / "runs"),
                "method": "fullft",
                "pretrain_methods": ["joint"],
                "domains": [0],
                "seeds": [42],
                "device": "cpu",
                "bf16": False,
                "progress": True,
                "batch_size": 1,
                "steps_per_epoch": 1,
                "epochs": 1,
                "patience": 1,
                "lr": 0.001,
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "hidden_size": 4,
                "num_blocks": 1,
                "num_heads": 1,
                "dropout": 0.0,
                "maxlen": 3,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftrec.cli.adapt",
            "--config",
            str(config_path),
            "--model-config",
            str(model_path),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "fullft matrix" in result.stderr
    assert "100%" in result.stderr
