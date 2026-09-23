"""Adapt GMFlowRec's published Parquet splits to CGRec's ID conventions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


GMFLOW_DIR = Path(__file__).resolve().parents[1] / "FTRec-GMFlowRec"
if str(GMFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(GMFLOW_DIR))

from gmflowrec_data import GMFlowRecEvaluationDataset  # noqa: E402
from mdsr_parquet import MDSRMetadata, MDSRParquetData, ParquetSequenceStore  # noqa: E402


def load_parquet_data(root: str | Path) -> MDSRParquetData:
    """Read the supplied splits and verify paired validation/test histories."""
    data = MDSRParquetData(root)
    if sorted(data.metadata.domain_offsets) != list(range(5)):
        raise ValueError("CGRec reproduction expects the five Amazon domains 0..4")
    return data


def domain_remap(target_domain: int) -> dict[int, int]:
    """CGRec's fixed target is 5; the other four domains are 6..9."""
    if target_domain not in range(5):
        raise ValueError("target_domain must be 0..4")
    sources = [domain for domain in range(5) if domain != target_domain]
    return {target_domain: 5, **{domain: 6 + i for i, domain in enumerate(sources)}}


def _map_domains(domains: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    result = np.zeros(domains.shape, dtype=np.int64)
    for original, mapped in mapping.items():
        result[domains == original] = mapped
    return result


def _pad(values: np.ndarray, maxlen: int) -> np.ndarray:
    clipped = values[-maxlen:]
    result = np.zeros(maxlen, dtype=np.int64)
    if len(clipped):
        result[-len(clipped) :] = clipped
    return result


class CGRecTrainDataset(Dataset):
    """Next-item training pairs from every supplied training interaction."""

    def __init__(
        self,
        store: ParquetSequenceStore,
        metadata: MDSRMetadata,
        target_domain: int,
        maxlen: int,
        seed: int,
        max_examples: int | None = None,
    ):
        if maxlen <= 0 or (max_examples is not None and max_examples <= 0):
            raise ValueError("maxlen and max_examples must be positive")
        self.store = store
        self.item_count = metadata.item_count
        self.mapping = domain_remap(target_domain)
        self.maxlen = maxlen
        self.seed = seed
        self.epoch = 0
        self.size = min(len(store), max_examples or len(store))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        items, domains = self.store.sequence(index)
        if len(items) < 2:
            raise ValueError(f"training row {index} has fewer than two interactions")
        # The GMFlowRec loader uses +1; CGRec reserves IDs 0..4.
        window_items = items[-(self.maxlen + 1) :].astype(np.int64) + 4
        window_domains = domains[-(self.maxlen + 1) :]
        item_input = _pad(window_items[:-1], self.maxlen)
        item_pos = _pad(window_items[1:], self.maxlen)
        type_input = _pad(_map_domains(window_domains[:-1], self.mapping), self.maxlen)

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        seen = set((items.astype(np.int64) + 4).tolist())
        if len(seen) >= self.item_count:
            raise ValueError(f"training row {index} exhausts the item catalogue")
        negatives = np.zeros(self.maxlen, dtype=np.int64)
        for position in np.flatnonzero(item_input):
            candidate = int(rng.integers(5, self.item_count + 5))
            while candidate in seen:
                candidate = int(rng.integers(5, self.item_count + 5))
            negatives[position] = candidate
        return item_input, item_pos, negatives, type_input, index


class CGRecEvaluationDataset(Dataset):
    """Use exactly GMFlowRec's fixed same-domain negative sampling."""

    def __init__(
        self,
        store: ParquetSequenceStore,
        metadata: MDSRMetadata,
        target_domain: int,
        maxlen: int,
        num_negatives: int,
        eval_seed: int,
        max_examples: int | None = None,
    ):
        if max_examples is not None and max_examples <= 0:
            raise ValueError("max_examples must be positive")
        self.mapping = domain_remap(target_domain)
        self.base = GMFlowRecEvaluationDataset(
            store, metadata, maxlen=maxlen,
            num_negatives=num_negatives, seed=eval_seed,
        )
        last_positions = store.offsets[1:] - 1
        rows = np.flatnonzero(store.domains[last_positions] == target_domain)
        self.rows = rows[:max_examples] if max_examples is not None else rows
        self._cache: list[tuple[np.ndarray, ...]] | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        if index < 0 or index >= len(self.rows):
            raise IndexError(index)
        if self._cache is not None:
            return self._cache[index]
        row = int(self.rows[index])
        items, domains, _, candidates, _ = self.base[row]
        shifted_items = np.where(items > 0, items + 4, 0).astype(np.int64)
        shifted_candidates = (candidates + 4).astype(np.int64)
        return shifted_items, _map_domains(domains, self.mapping), shifted_candidates, row

    def precompute(self) -> None:
        """Freeze sampled candidates once, avoiding resampling every epoch."""
        if self._cache is None:
            self._cache = [self[index] for index in range(len(self.rows))]
