"""Exercise the unmodified CGRec model on its published sample."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cgrec_cpu_probe_trains_and_scores_sample() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "FTRec-CGRec" / "verify_cpu.py"),
            "--train-users",
            "1",
            "--eval-users",
            "1",
            "--max-seq-length",
            "8",
            "--hidden-size",
            "8",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["shaply_value"] == "y"
    assert metrics["train_users"] == 1
    assert metrics["eval_users"] == 1
    assert math.isfinite(metrics["train_loss"])
    assert 0 <= metrics["hit_at_5"] <= 1
    assert 0 <= metrics["ndcg_at_5"] <= 1
