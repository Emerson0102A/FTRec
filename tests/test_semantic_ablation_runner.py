from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _bash() -> str:
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    return str(git_bash) if git_bash.exists() else "bash"


def test_plan_reuses_existing_baselines_and_only_schedules_missing_arms(tmp_path):
    root = Path(__file__).parents[1]
    existing = tmp_path / "semantic-adapt"
    for domain in range(5):
        run = (
            existing / "adapt" / "lora_all" / "joint_proportional"
            / f"domain-{domain}" / "rank-5" / "seed-42"
        )
        run.mkdir(parents=True)
        (run / "result.json").write_text("{}", encoding="utf-8")
        (run / "COMPLETE.json").write_text("{}", encoding="utf-8")
    environment = dict(os.environ)
    environment.update({
        "ACTION": "plan", "SEED": "42",
        "SEMANTIC_ADAPT_ROOT": existing.as_posix(),
    })
    process = subprocess.run(
        [_bash(), "scripts/run_semantic_ft_ablation.sh"],
        cwd=root, env=environment, text=True, capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    lines = process.stdout.splitlines()
    assert sum("ftrec.cli.pretrain" in line for line in lines) == 3
    assert sum("ftrec.cli.adapt" in line for line in lines) == 20
    assert sum("reuse result" in line for line in lines) == 5
    assert not any("ftrec.cli.pretrain" in line and "sasrec.yaml" in line for line in lines)
    assert not any(
        "ftrec.cli.pretrain" in line and "sasrec_structured_title_fused.yaml" in line
        for line in lines
    )
