"""Sampled ranking metrics shared with GMFlowRec's evaluation protocol."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor


def rank_of_positive(scores: Tensor, candidates: Tensor) -> Tensor:
    """Zero-based rank of candidate zero; lower item ID wins a score tie."""
    if scores.ndim != 2 or scores.shape != candidates.shape:
        raise ValueError("scores and candidates must have the same [batch, items] shape")
    target_scores = scores[:, :1]
    target_ids = candidates[:, :1]
    outranks = scores[:, 1:] > target_scores
    outranks |= (scores[:, 1:] == target_scores) & (candidates[:, 1:] < target_ids)
    return outranks.sum(dim=1)


def summarize_ranks(ranks: Iterable[int]) -> dict[str, float | int]:
    values = list(ranks)
    if not values:
        raise ValueError("cannot report metrics for an empty target domain")
    result: dict[str, float | int] = {"count": len(values)}
    for cutoff in (5, 10):
        result[f"hr@{cutoff}"] = sum(rank < cutoff for rank in values) / len(values)
        result[f"ndcg@{cutoff}"] = sum(
            1.0 / math.log2(rank + 2) for rank in values if rank < cutoff
        ) / len(values)
    return result


@torch.inference_mode()
def evaluate_model(model, loader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    ranks: list[int] = []
    for items, domains, candidates, _ in loader:
        items = items.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)
        candidates = candidates.to(device, non_blocking=True)
        scores = model.score(items, domains, candidates)
        ranks.extend(rank_of_positive(scores, candidates).cpu().tolist())
    return summarize_ranks(ranks)
