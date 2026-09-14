from __future__ import annotations

from dataclasses import replace

import pytest


def _row(seed: int = 42, domain: str = "Health", ndcg: float = 0.2, users: int = 3):
    from ftrec.analysis.results import ResultRow

    return ResultRow(
        seed=seed,
        domain=domain,
        pretrain_method="joint",
        adapt_method="none",
        lora_rank=None,
        split="test",
        evaluation_protocol="full",
        hr_at_10=0.5,
        ndcg_at_10=ndcg,
        num_eval_users=users,
        num_skipped_users=0,
        num_trainable_params=100,
        num_total_params=100,
        checkpoint_path="best.pt",
        config_hash="config",
        data_hash="data",
    )


def test_result_row_requires_protocol_and_parameter_counts() -> None:
    from ftrec.analysis.results import ResultRow, ResultSchemaError

    with pytest.raises(ResultSchemaError):
        ResultRow.from_dict({"seed": 42, "domain": "Health", "NDCG@10": 0.2})


def test_aggregation_reports_sample_std_over_seeds() -> None:
    from ftrec.analysis.results import aggregate_results

    rows = tuple(_row(seed=seed, ndcg=value) for seed, value in enumerate((0.2, 0.3, 0.4)))
    summaries = aggregate_results(rows)
    ndcg = next(row for row in summaries if row.metric == "NDCG@10")

    assert ndcg.mean == pytest.approx(0.3)
    assert ndcg.std == pytest.approx(0.1)
    assert ndcg.seeds == 3


def test_duplicate_keys_are_rejected() -> None:
    from ftrec.analysis.results import ResultSchemaError, validate_result_rows

    row = _row()
    with pytest.raises(ResultSchemaError, match="duplicate"):
        validate_result_rows((row, row))


def test_macro_rows_exclude_domains_without_evaluable_users() -> None:
    from ftrec.analysis.results import add_macro_rows

    rows = (
        _row(domain="Health", ndcg=0.2, users=2),
        replace(_row(domain="Beauty", ndcg=float("nan"), users=0), num_skipped_users=2),
        _row(domain="Sports", ndcg=0.4, users=1),
    )
    macro = next(row for row in add_macro_rows(rows) if row.domain == "Macro")

    assert macro.ndcg_at_10 == pytest.approx(0.3)
    assert macro.contributing_domains == 2
