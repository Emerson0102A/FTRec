import math
import csv
import json
from pathlib import Path

import pytest
import torch


def test_cosine_matrix_is_symmetric_and_negative_ratio_uses_unique_pairs() -> None:
    from ftrec.training.pcgrad import TaskGradients, cosine_matrix, negative_pair_ratio

    tasks = (
        TaskGradients(("w",), (torch.tensor([1.0, 0.0]),)),
        TaskGradients(("w",), (torch.tensor([-1.0, 0.0]),)),
        TaskGradients(("w",), (torch.tensor([0.0, 1.0]),)),
    )

    matrix = cosine_matrix(tasks)

    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(-1.0)
    assert matrix[0][2] == pytest.approx(0.0)
    assert matrix[1][0] == pytest.approx(matrix[0][1])
    assert negative_pair_ratio(matrix) == pytest.approx(1 / 3)


def test_zero_gradient_cosine_is_nan_not_a_fake_zero() -> None:
    from ftrec.training.pcgrad import TaskGradients, cosine_matrix

    tasks = (
        TaskGradients(("w",), (torch.zeros(2),)),
        TaskGradients(("w",), (torch.ones(2),)),
    )

    matrix = cosine_matrix(tasks)

    assert math.isnan(matrix[0][0])
    assert math.isnan(matrix[0][1])


def test_parameter_prefix_selects_layer_group() -> None:
    from ftrec.training.pcgrad import TaskGradients, cosine_matrix

    tasks = (
        TaskGradients(
            ("item_embedding.weight", "blocks.0.attention.q_proj.weight"),
            (torch.tensor([1.0]), torch.tensor([1.0])),
        ),
        TaskGradients(
            ("item_embedding.weight", "blocks.0.attention.q_proj.weight"),
            (torch.tensor([-1.0]), torch.tensor([1.0])),
        ),
    )

    full = cosine_matrix(tasks)
    attention = cosine_matrix(tasks, prefixes=("blocks.0.attention",))

    assert full[0][1] == pytest.approx(0.0)
    assert attention[0][1] == pytest.approx(1.0)


def test_dynamic_q_v_qv_groups_match_lora_projection_parameters() -> None:
    """Catch Q/V diagnostics including K/out projections or assuming two blocks."""
    from ftrec.models.sasrec import SASRec, SASRecConfig

    model = SASRec(
        SASRecConfig(
            num_items=10,
            hidden_size=6,
            num_blocks=3,
            num_heads=2,
            dropout=0.0,
            maxlen=4,
        )
    )
    groups = model.logging_parameter_groups()

    for layer in range(3):
        q = f"blocks.{layer}.attention.q_proj.weight"
        v = f"blocks.{layer}.attention.v_proj.weight"
        assert groups[f"block_{layer}_q"] == (q,)
        assert groups[f"block_{layer}_v"] == (v,)
        assert groups[f"block_{layer}_qv"] == (q, v)
    assert "block_3_q" not in groups


@pytest.mark.parametrize(
    ("right", "expected_cosine", "expected_conflict"),
    [
        (torch.tensor([1.0, 2.0]), 1.0, 0.0),
        (torch.tensor([-1.0, -2.0]), -1.0, 1.0),
    ],
)
def test_cosine_and_conflict_match_hand_calculation(
    right: torch.Tensor, expected_cosine: float, expected_conflict: float
) -> None:
    """Catch the conflict sign or normalization being implemented backwards."""
    from ftrec.training.pcgrad import (
        TaskGradients,
        cosine_and_conflict_matrix,
    )

    tasks = (
        TaskGradients(("w",), (torch.tensor([1.0, 2.0]),)),
        TaskGradients(("w",), (right,)),
    )

    cosine, conflict = cosine_and_conflict_matrix(tasks)

    assert cosine[0][1] == pytest.approx(expected_cosine)
    assert conflict[0][1] == pytest.approx(expected_conflict)


def test_near_zero_group_is_safe_and_nonconflicting() -> None:
    """Catch zero-norm groups leaking NaN into EMA and summary files."""
    from ftrec.training.pcgrad import TaskGradients, cosine_and_conflict_matrix

    tasks = (
        TaskGradients(("w",), (torch.zeros(2),)),
        TaskGradients(("w",), (torch.ones(2),)),
    )

    cosine, conflict = cosine_and_conflict_matrix(tasks)

    assert cosine[0][1] == 0.0
    assert conflict[0][1] == 0.0


def test_logger_writes_pairwise_ema_and_domain_layer_summaries(tmp_path: Path) -> None:
    """Catch missing join-ready Q/V/QV artifacts or an incorrect EMA update."""
    from ftrec.training.pcgrad import GradientConflictLogger, TaskGradients

    names = (
        "blocks.0.attention.q_proj.weight",
        "blocks.0.attention.v_proj.weight",
    )
    groups = {
        "block_0_q": ("blocks.0.attention.q_proj",),
        "block_0_v": ("blocks.0.attention.v_proj",),
        "block_0_qv": (
            "blocks.0.attention.q_proj",
            "blocks.0.attention.v_proj",
        ),
    }
    opposite = (
        TaskGradients(names, (torch.tensor([1.0]), torch.tensor([2.0]))),
        TaskGradients(names, (torch.tensor([-1.0]), torch.tensor([-2.0]))),
    )
    same = (
        TaskGradients(names, (torch.tensor([1.0]), torch.tensor([2.0]))),
        TaskGradients(names, (torch.tensor([1.0]), torch.tensor([2.0]))),
    )
    logger = GradientConflictLogger(
        tmp_path / "gradient_conflicts.jsonl",
        ("Health", "Beauty"),
        ema_beta=0.5,
    )
    logger.record(method="joint", seed=42, epoch=1, step=0, raw=opposite, groups=groups)
    logger.record(method="joint", seed=42, epoch=1, step=1, raw=same, groups=groups)
    logger.finalize()

    records = [
        json.loads(line)
        for line in (tmp_path / "gradient_conflicts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["raw_conflict"]["block_0_qv"][0][1] == pytest.approx(1.0)
    assert records[0]["domain_conflict"]["block_0_qv"] == pytest.approx(
        {"Health": 1.0, "Beauty": 1.0}
    )
    assert records[0]["aggregate_conflict"]["block_0_qv"] == pytest.approx(1.0)
    assert records[1]["conflict_ema"]["block_0_qv"][0][1] == pytest.approx(0.5)
    assert records[1]["domain_conflict_ema"]["block_0_qv"] == pytest.approx(
        {"Health": 0.5, "Beauty": 0.5}
    )
    assert records[1]["aggregate_conflict_ema"]["block_0_qv"] == pytest.approx(0.5)

    with (tmp_path / "gradient_conflict_pairs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    qv = [row for row in rows if row["group"] == "block_0_qv"]
    assert [float(row["conflict"]) for row in qv] == pytest.approx([1.0, 0.0])
    assert float(qv[-1]["conflict_ema"]) == pytest.approx(0.5)

    summary = json.loads(
        (tmp_path / "gradient_conflict_summary.json").read_text(encoding="utf-8")
    )
    assert summary["groups"]["block_0_qv"]["aggregate_conflict"] == pytest.approx(0.5)
    assert summary["groups"]["block_0_qv"]["domain_conflict"] == pytest.approx(
        {"Health": 0.5, "Beauty": 0.5}
    )

    with (tmp_path / "gradient_conflict_by_domain_layer.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        layer_rows = list(csv.DictReader(stream))
    assert layer_rows == [
        {
            "domain": "Health",
            "layer": "0",
            "q_conflict": "0.5",
            "v_conflict": "0.5",
            "qv_conflict": "0.5",
        },
        {
            "domain": "Beauty",
            "layer": "0",
            "q_conflict": "0.5",
            "v_conflict": "0.5",
            "qv_conflict": "0.5",
        },
    ]


def test_logger_marks_best_checkpoint_profile_scope(tmp_path: Path) -> None:
    """Catch a checkpoint-local conflict summary being mistaken for training history."""
    from ftrec.training.pcgrad import GradientConflictLogger, TaskGradients

    tasks = (
        TaskGradients(("w",), (torch.tensor([1.0]),)),
        TaskGradients(("w",), (torch.tensor([-1.0]),)),
    )
    logger = GradientConflictLogger(
        tmp_path / "gradient_conflicts_best_checkpoint.jsonl",
        ("Health", "Beauty"),
        profile_scope="best_checkpoint",
    )
    record = logger.record(
        method="joint", seed=42, epoch=3, step=0, raw=tasks
    )
    summary = logger.finalize(
        metadata={"checkpoint_epoch": 3, "diagnostic_steps": 1, "diagnostic_seed": 2026}
    )

    assert record["profile_scope"] == "best_checkpoint"
    assert summary["profile_scope"] == "best_checkpoint"
    assert summary["checkpoint_epoch"] == 3
    assert summary["diagnostic_steps"] == 1
    assert summary["diagnostic_seed"] == 2026
