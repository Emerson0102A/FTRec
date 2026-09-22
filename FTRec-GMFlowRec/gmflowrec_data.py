"""Domain-aware training and evaluation data for the GMFlowRec reproduction."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from mdsr_parquet import MDSRMetadata, ParquetSequenceStore


def _padded_context(
    items: np.ndarray, domains: np.ndarray, maxlen: int
) -> tuple[np.ndarray, np.ndarray]:
    clipped_items = items[-maxlen:]
    clipped_domains = domains[-maxlen:]
    padded_items = np.zeros(maxlen, dtype=np.int64)
    padded_domains = np.full(maxlen, -1, dtype=np.int64)
    padded_items[-len(clipped_items) :] = clipped_items
    padded_domains[-len(clipped_domains) :] = clipped_domains
    return padded_items, padded_domains


def _sample_domain_negatives(
    rng: np.random.Generator,
    domain_range: tuple[int, int],
    excluded: set[int],
    size: int,
) -> np.ndarray:
    """Sample distinct one-based item IDs from a zero-based domain range."""

    start, end = domain_range
    first_item, last_item_exclusive = start + 1, end + 1
    available = (end - start) - sum(
        first_item <= item < last_item_exclusive for item in excluded
    )
    if size > available:
        raise ValueError(
            f"requested {size} negatives but target domain only has {available} unseen items"
        )

    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < size:
        draw_count = max(16, 2 * (size - len(selected)))
        draws = rng.integers(first_item, last_item_exclusive, size=draw_count)
        for raw_item in draws:
            item = int(raw_item)
            if item in excluded or item in selected_set:
                continue
            selected.append(item)
            selected_set.add(item)
            if len(selected) == size:
                break
    return np.asarray(selected, dtype=np.int64)


class GMFlowRecTrainDataset(Dataset):
    """Use the final interaction of every training row as its next-item target."""

    def __init__(self, store: ParquetSequenceStore, maxlen: int):
        self.store = store
        self.maxlen = int(maxlen)
        if self.maxlen <= 0:
            raise ValueError("maxlen must be positive")

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int):
        items, domains = self.store.sequence(index)
        if len(items) < 2:
            raise ValueError(f"training row {index} has fewer than two interactions")
        context_items, context_domains = _padded_context(
            items[:-1], domains[:-1], self.maxlen
        )
        return (
            context_items,
            context_domains,
            np.int64(items[-1]),
            np.int64(domains[-1]),
            np.int64(index),
        )


class GMFlowRecEvaluationDataset(Dataset):
    """Target plus deterministic, distinct same-domain negatives."""

    def __init__(
        self,
        store: ParquetSequenceStore,
        metadata: MDSRMetadata,
        maxlen: int,
        num_negatives: int = 999,
        seed: int = 3407,
    ):
        self.store = store
        self.metadata = metadata
        self.maxlen = int(maxlen)
        self.num_negatives = int(num_negatives)
        self.seed = int(seed)
        if self.maxlen <= 0 or self.num_negatives <= 0:
            raise ValueError("maxlen and num_negatives must be positive")

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int):
        items, domains = self.store.sequence(index)
        if len(items) < 2:
            raise ValueError(f"evaluation row {index} has fewer than two interactions")
        target = int(items[-1])
        target_domain = int(domains[-1])
        context_items, context_domains = _padded_context(
            items[:-1], domains[:-1], self.maxlen
        )
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        candidates = np.empty(self.num_negatives + 1, dtype=np.int64)
        candidates[0] = target
        candidates[1:] = _sample_domain_negatives(
            rng,
            self.metadata.domain_offsets[target_domain],
            set(int(item) for item in items),
            self.num_negatives,
        )
        return (
            context_items,
            context_domains,
            np.int64(target_domain),
            candidates,
            np.int64(index),
        )


@torch.inference_mode()
def evaluate_gmflowrec(model, loader, device: str | torch.device, steps: int = 8) -> dict:
    """Evaluate HR/NDCG at 5 and 10 globally and per target domain."""

    model.eval()
    totals = defaultdict(lambda: {"count": 0, "hr@5": 0.0, "hr@10": 0.0,
                                 "ndcg@5": 0.0, "ndcg@10": 0.0})
    device = torch.device(device)
    for items, domains, target_domains, candidates, _ in loader:
        logits = model.predict(
            items.to(device, non_blocking=True),
            domains.to(device, non_blocking=True),
            target_domains.to(device, non_blocking=True),
            candidates.to(device, non_blocking=True),
            steps=steps,
        )
        # Candidate 0 is the positive. On score ties, rank the lower global item
        # ID first so results do not depend on candidate-array order.
        negative_scores = logits[:, 1:]
        target_scores = logits[:, :1]
        negative_ids = candidates[:, 1:].to(logits.device)
        target_ids = candidates[:, :1].to(logits.device)
        outranks = negative_scores.gt(target_scores)
        outranks |= negative_scores.eq(target_scores) & negative_ids.lt(target_ids)
        ranks = outranks.sum(dim=1).cpu().numpy()
        domain_values = target_domains.numpy()
        for rank, domain in zip(ranks.tolist(), domain_values.tolist()):
            for key in ("overall", str(domain)):
                bucket = totals[key]
                bucket["count"] += 1
                for cutoff in (5, 10):
                    if rank < cutoff:
                        bucket[f"hr@{cutoff}"] += 1.0
                        bucket[f"ndcg@{cutoff}"] += 1.0 / np.log2(rank + 2.0)

    if not totals["overall"]["count"]:
        raise ValueError("evaluation loader produced no examples")
    result = {}
    for key, bucket in totals.items():
        count = bucket["count"]
        result[key] = {"count": count}
        for metric in ("hr@5", "hr@10", "ndcg@5", "ndcg@10"):
            result[key][metric] = bucket[metric] / count
    return result
