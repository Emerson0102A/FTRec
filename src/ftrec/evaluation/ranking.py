"""Deterministic sampled and chunked full-catalog ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ftrec.data.datasets import TargetExample

from .metrics import RankingMetrics


def rank_ground_truth_chunked(
    model: object,
    *,
    context: torch.Tensor,
    target_item: int,
    candidates: Sequence[int],
    seen_items: frozenset[int],
    chunk_size: int,
) -> int:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if target_item not in candidates:
        raise ValueError("target item is absent from candidates")
    device = context.device
    context_batch = context.unsqueeze(0)
    target_ids = torch.tensor([[target_item]], dtype=torch.long, device=device)
    with torch.no_grad():
        target_score = model.score(context_batch, target_ids)[0, 0]
        rank = 0
        for start in range(0, len(candidates), chunk_size):
            chunk = tuple(candidates[start : start + chunk_size])
            eligible = [
                item for item in chunk if item == target_item or item not in seen_items
            ]
            if not eligible:
                continue
            item_ids = torch.tensor([eligible], dtype=torch.long, device=device)
            scores = model.score(context_batch, item_ids)[0]
            identifiers = torch.tensor(eligible, dtype=torch.long, device=device)
            ahead = scores > target_score
            tied_ahead = (scores == target_score) & (identifiers < target_item)
            rank += int((ahead | tied_ahead).sum().item())
    return rank


def evaluate_model(
    model: object,
    examples: Sequence[TargetExample],
    items_by_domain: Mapping[int, Sequence[int]],
    *,
    protocol: str = "full",
    sampled_candidates: Mapping[int, Sequence[int]] | None = None,
    chunk_size: int = 4096,
    k: int = 10,
    device: str | torch.device = "cpu",
) -> dict[str, float | int | str]:
    if protocol not in {"full", "sampled"}:
        raise ValueError(f"unsupported evaluation protocol: {protocol}")
    if protocol == "sampled" and sampled_candidates is None:
        raise ValueError("sampled evaluation requires persisted candidates")
    metrics = RankingMetrics(k=k)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        for example in examples:
            if protocol == "sampled":
                candidates = tuple(sampled_candidates[example.example_id])
            else:
                candidates = tuple(items_by_domain.get(example.target_domain, ()))
            if not candidates or example.positive_item not in candidates:
                metrics.skip()
                continue
            context = torch.tensor(
                example.context_items, dtype=torch.long, device=torch.device(device)
            )
            rank = rank_ground_truth_chunked(
                model,
                context=context,
                target_item=example.positive_item,
                candidates=candidates,
                seen_items=example.seen_items,
                chunk_size=chunk_size,
            )
            metrics.add_rank(rank)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    result = metrics.compute()
    result["evaluation_protocol"] = protocol
    return result

