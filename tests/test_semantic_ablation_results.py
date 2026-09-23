from __future__ import annotations

import json

import pytest

from ftrec.analysis.semantic_ablation import summarize_semantic_ablation


def _write(root, arm, domain, *, before, after, examples=4):
    path = (
        root / arm / "adapt" / "lora_all" / "joint_proportional"
        / f"domain-{domain}" / "rank-5" / "seed-42" / "result.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "data_hash": "same-data", "method": "lora_all", "domain": domain,
        "seed": 42, "rank": 5, "context_mode": "target_only",
        "min_domain_sequence_length": 2, "num_train_negatives": 31,
        "num_examples": {"train": examples, "valid": 2, "test": 2},
        "num_trainable_params": 32,
        "pretrain_metrics": {"NDCG@10": before},
        "test_metrics": {"NDCG@10": after},
        "best_epoch": 1,
    }), encoding="utf-8")


def test_semantic_report_uses_paired_gains_and_equal_domain_macro(tmp_path):
    _write(tmp_path, "semantic", 0, before=0.1, after=0.3)
    _write(tmp_path, "semantic", 1, before=0.2, after=0.3)
    _write(tmp_path, "shuffled", 0, before=0.1, after=0.2)
    _write(tmp_path, "shuffled", 1, before=0.2, after=0.2)
    report = summarize_semantic_ablation(
        tmp_path, arms=("semantic", "shuffled"), domains=(0, 1), seed=42
    )
    assert report["macro_gain"]["semantic"] == pytest.approx(0.15)
    assert report["macro_gain"]["shuffled"] == pytest.approx(0.05)
    assert report["aligned_minus_shuffled_gain"] == pytest.approx(0.1)


def test_semantic_report_rejects_different_target_cohorts(tmp_path):
    _write(tmp_path, "semantic", 0, before=0.1, after=0.2)
    _write(tmp_path, "shuffled", 0, before=0.1, after=0.2, examples=5)
    with pytest.raises(ValueError, match="cohort"):
        summarize_semantic_ablation(
            tmp_path, arms=("semantic", "shuffled"), domains=(0,), seed=42
        )


def test_semantic_report_reads_previously_completed_semantic_run(tmp_path):
    prior = tmp_path / "prior-semantic"
    current = tmp_path / "new"
    _write(prior, "semantic", 0, before=0.1, after=0.3)
    _write(current, "shuffled", 0, before=0.1, after=0.2)
    report = summarize_semantic_ablation(
        current, arms=("semantic", "shuffled"), domains=(0,), seed=42,
        fallback_roots={"semantic": prior / "semantic"},
    )
    assert report["aligned_minus_shuffled_gain"] == pytest.approx(0.1)
