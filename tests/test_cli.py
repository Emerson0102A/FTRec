from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


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
