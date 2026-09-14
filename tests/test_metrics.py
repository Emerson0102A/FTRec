import math

import pytest


@pytest.mark.parametrize(
    "rank, expected_hr, expected_ndcg",
    [(0, 1.0, 1.0), (9, 1.0, 1 / math.log2(11)), (10, 0.0, 0.0)],
)
def test_top10_metric_boundaries(rank, expected_hr, expected_ndcg) -> None:
    from ftrec.evaluation.metrics import metrics_for_rank

    assert metrics_for_rank(rank, 10) == pytest.approx((expected_hr, expected_ndcg))


def test_accumulator_tracks_skips_without_changing_denominator() -> None:
    from ftrec.evaluation.metrics import RankingMetrics

    metrics = RankingMetrics(k=10)
    metrics.add_rank(0)
    metrics.add_rank(20)
    metrics.skip()

    result = metrics.compute()
    assert result == pytest.approx({"HR@10": 0.5, "NDCG@10": 0.5, "num_eval_users": 2, "num_skipped_users": 1})

