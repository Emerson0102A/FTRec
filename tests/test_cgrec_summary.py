"""The sweep report keeps the paper's metric scale and sample variability."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "FTRec-CGRec" / "summarize_parquet.py"


def test_summary_reads_runs_and_reports_percent_mean_and_sample_std(tmp_path):
    metrics = (
        {"hr@5": 0.1, "hr@10": 0.2, "ndcg@5": 0.05, "ndcg@10": 0.08},
        {"hr@5": 0.2, "hr@10": 0.3, "ndcg@5": 0.1, "ndcg@10": 0.12},
    )
    for seed, test in enumerate(metrics):
        result = tmp_path / "domain-0" / f"seed-{seed}" / "results.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({"target_domain": 0, "seed": seed, "test": test}))

    completed = subprocess.run(
        [sys.executable, str(SUMMARY), "--run_dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Health" in completed.stdout
    assert "2/5" in completed.stdout
    assert "15.00 ± 7.07" in completed.stdout
    assert "16.44" in completed.stdout
