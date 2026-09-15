"""Deterministic sampled and chunked full-catalog ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from tqdm.auto import tqdm

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
    batch_size: int = 128,
    k: int = 10,
    device: str | torch.device = "cpu",
    progress: bool = False,
    description: str = "evaluate",
) -> dict[str, float | int | str]:
    if protocol not in {"full", "sampled"}:
        raise ValueError(f"unsupported evaluation protocol: {protocol}")
    if protocol == "sampled" and sampled_candidates is None:
        raise ValueError("sampled evaluation requires persisted candidates")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    metrics = RankingMetrics(k=k)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    progress_bar = tqdm(
        total=len(examples),
        desc=description,
        unit="users",
        mininterval=1.0,
        dynamic_ncols=True,
        disable=not progress,
    )
    try:
        supports_batched_scoring = hasattr(model, "final_state") and hasattr(
            model, "item_embedding"
        )
        if protocol == "sampled" and supports_batched_scoring:
            assert sampled_candidates is not None
            _evaluate_sampled_batched(
                model,
                examples,
                sampled_candidates,
                metrics,
                batch_size=batch_size,
                device=torch.device(device),
                progress_bar=progress_bar,
            )
            examples = ()
        elif protocol == "full" and supports_batched_scoring:
            _evaluate_full_batched(
                model,
                examples,
                items_by_domain,
                metrics,
                chunk_size=chunk_size,
                batch_size=batch_size,
                device=torch.device(device),
                progress_bar=progress_bar,
            )
            examples = ()
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
            progress_bar.update(1)
    finally:
        progress_bar.close()
        if was_training and hasattr(model, "train"):
            model.train()
    result = metrics.compute()
    result["evaluation_protocol"] = protocol
    return result


def _evaluate_sampled_batched(
    model: object,
    examples: Sequence[TargetExample],
    sampled_candidates: Mapping[int, Sequence[int]],
    metrics: RankingMetrics,
    *,
    batch_size: int,
    device: torch.device,
    progress_bar: tqdm,
) -> None:
    """Score fixed candidate sets in user batches instead of one user at a time."""
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            source_batch = examples[offset : offset + batch_size]
            batch: list[TargetExample] = []
            rows: list[tuple[int, ...]] = []
            for example in source_batch:
                candidates = tuple(sampled_candidates.get(example.example_id, ()))
                eligible = tuple(
                    item
                    for item in candidates
                    if item == example.positive_item or item not in example.seen_items
                )
                if not eligible or example.positive_item not in eligible:
                    metrics.skip()
                    progress_bar.update(1)
                    continue
                batch.append(example)
                rows.append(eligible)
            if not batch:
                continue

            width = max(len(row) for row in rows)
            candidate_ids = torch.zeros(
                (len(batch), width), dtype=torch.long, device=device
            )
            eligible_mask = torch.zeros(
                (len(batch), width), dtype=torch.bool, device=device
            )
            for row_index, row in enumerate(rows):
                candidate_ids[row_index, : len(row)] = torch.tensor(
                    row, dtype=torch.long, device=device
                )
                eligible_mask[row_index, : len(row)] = True

            contexts = torch.tensor(
                [example.context_items for example in batch],
                dtype=torch.long,
                device=device,
            )
            states = model.final_state(contexts)
            scores = torch.einsum(
                "bd,bcd->bc", states, model.item_embedding(candidate_ids)
            )
            target_ids = torch.tensor(
                [example.positive_item for example in batch],
                dtype=torch.long,
                device=device,
            )
            target_scores = torch.einsum(
                "bd,bd->b", states, model.item_embedding(target_ids)
            )
            ahead = scores > target_scores.unsqueeze(1)
            tied_ahead = (scores == target_scores.unsqueeze(1)) & (
                candidate_ids < target_ids.unsqueeze(1)
            )
            not_target = candidate_ids != target_ids.unsqueeze(1)
            ranks = (
                (ahead | tied_ahead) & eligible_mask & not_target
            ).sum(dim=1)
            for rank in ranks.cpu().tolist():
                metrics.add_rank(int(rank))
            progress_bar.update(len(batch))


def _evaluate_full_batched(
    model: object,
    examples: Sequence[TargetExample],
    items_by_domain: Mapping[int, Sequence[int]],
    metrics: RankingMetrics,
    *,
    chunk_size: int,
    batch_size: int,
    device: torch.device,
    progress_bar: tqdm,
) -> None:
    by_domain: dict[int, list[TargetExample]] = {}
    for example in examples:
        catalog = items_by_domain.get(example.target_domain, ())
        if not catalog or example.positive_item not in catalog:
            metrics.skip()
            progress_bar.update(1)
        else:
            by_domain.setdefault(example.target_domain, []).append(example)
    with torch.no_grad():
        for domain, domain_examples in sorted(by_domain.items()):
            catalog = tuple(items_by_domain[domain])
            for offset in range(0, len(domain_examples), batch_size):
                batch = domain_examples[offset : offset + batch_size]
                contexts = torch.tensor(
                    [example.context_items for example in batch],
                    dtype=torch.long,
                    device=device,
                )
                states = model.final_state(contexts)
                target_ids = torch.tensor(
                    [example.positive_item for example in batch],
                    dtype=torch.long,
                    device=device,
                )
                target_scores = torch.einsum(
                    "bd,bd->b", states, model.item_embedding(target_ids)
                )
                ranks = torch.zeros(len(batch), dtype=torch.long, device=device)
                for start in range(0, len(catalog), chunk_size):
                    chunk = catalog[start : start + chunk_size]
                    identifiers = torch.tensor(chunk, dtype=torch.long, device=device)
                    scores = torch.einsum(
                        "bd,cd->bc", states, model.item_embedding(identifiers)
                    )
                    eligible = torch.ones(
                        (len(batch), len(chunk)), dtype=torch.bool, device=device
                    )
                    positions = {item: index for index, item in enumerate(chunk)}
                    for row, example in enumerate(batch):
                        excluded = [
                            positions[item]
                            for item in example.seen_items
                            if item != example.positive_item and item in positions
                        ]
                        if excluded:
                            eligible[row, excluded] = False
                    ahead = scores > target_scores.unsqueeze(1)
                    tied = scores == target_scores.unsqueeze(1)
                    tied_ahead = tied & (
                        identifiers.unsqueeze(0) < target_ids.unsqueeze(1)
                    )
                    not_target = identifiers.unsqueeze(0) != target_ids.unsqueeze(1)
                    ranks += ((ahead | tied_ahead) & eligible & not_target).sum(dim=1)
                for rank in ranks.cpu().tolist():
                    metrics.add_rank(int(rank))
                progress_bar.update(len(batch))
