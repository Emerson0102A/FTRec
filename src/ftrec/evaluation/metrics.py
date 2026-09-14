"""HR and NDCG metric accumulation."""

from __future__ import annotations

import math
from dataclasses import dataclass


def metrics_for_rank(rank: int, k: int) -> tuple[float, float]:
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if k < 1:
        raise ValueError("k must be positive")
    if rank >= k:
        return 0.0, 0.0
    return 1.0, 1.0 / math.log2(rank + 2)


@dataclass
class RankingMetrics:
    k: int = 10
    hit_sum: float = 0.0
    ndcg_sum: float = 0.0
    num_eval_users: int = 0
    num_skipped_users: int = 0

    def add_rank(self, rank: int) -> None:
        hit, ndcg = metrics_for_rank(rank, self.k)
        self.hit_sum += hit
        self.ndcg_sum += ndcg
        self.num_eval_users += 1

    def skip(self) -> None:
        self.num_skipped_users += 1

    def compute(self) -> dict[str, float | int]:
        denominator = self.num_eval_users
        return {
            f"HR@{self.k}": self.hit_sum / denominator if denominator else float("nan"),
            f"NDCG@{self.k}": self.ndcg_sum / denominator if denominator else float("nan"),
            "num_eval_users": self.num_eval_users,
            "num_skipped_users": self.num_skipped_users,
        }

