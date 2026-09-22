"""Native loader for the official MDSR Amazon parquet splits.

The parquet files store zero-based item IDs and one sequence per row.  FTRec
reserves item 0 for padding, so this module performs the only permitted ID
conversion: ``ftrec_item_id = mdsr_item_id + 1``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MDSRMetadata:
    user_count: int
    item_count: int
    domain_offsets: dict[int, tuple[int, int]]


class ParquetSequenceStore:
    """Compact, read-only access to list-valued item/domain parquet columns."""

    def __init__(self, path: str | Path, metadata: MDSRMetadata):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"MDSR split not found: {self.path}")

        table = pq.read_table(self.path, columns=["item_id", "domain_id"], memory_map=True)
        item_lists = table["item_id"].combine_chunks()
        domain_lists = table["domain_id"].combine_chunks()
        self.offsets = np.asarray(item_lists.offsets, dtype=np.int64)
        domain_offsets = np.asarray(domain_lists.offsets, dtype=np.int64)
        if not np.array_equal(self.offsets, domain_offsets):
            raise ValueError(f"Item/domain list boundaries differ in {self.path}")

        # Keep the flattened arrays compact. IDs are shifted once here and are
        # thereafter in FTRec's 1-based item space.
        self.items = np.asarray(item_lists.values, dtype=np.int32) + 1
        self.domains = np.asarray(domain_lists.values, dtype=np.int8)
        self._validate(metadata)

    def _validate(self, metadata: MDSRMetadata) -> None:
        if self.items.size == 0:
            raise ValueError(f"Empty MDSR split: {self.path}")
        if int(self.items.min()) < 1 or int(self.items.max()) > metadata.item_count:
            raise ValueError(f"Item ID outside mappings.pkl range in {self.path}")
        if set(np.unique(self.domains).tolist()) != set(metadata.domain_offsets):
            raise ValueError(f"Unexpected domain IDs in {self.path}")

        zero_based_items = self.items.astype(np.int64) - 1
        for domain_id, (start, end) in metadata.domain_offsets.items():
            mask = self.domains == domain_id
            if mask.any() and not np.logical_and(
                zero_based_items[mask] >= start,
                zero_based_items[mask] < end,
            ).all():
                raise ValueError(
                    f"Item/domain mapping mismatch for domain {domain_id} in {self.path}"
                )

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def sequence(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.items[start:end], self.domains[start:end]

    @property
    def min_length(self) -> int:
        return int(np.diff(self.offsets).min())

    @property
    def max_length(self) -> int:
        return int(np.diff(self.offsets).max())


class MDSRParquetData:
    """The official train/validation/test splits plus their shared mapping."""

    def __init__(self, root: str | Path, validate_pairing: bool = True):
        self.root = Path(root)
        mapping_path = self.root / "mappings.pkl"
        if not mapping_path.is_file():
            raise FileNotFoundError(f"MDSR mapping not found: {mapping_path}")
        with mapping_path.open("rb") as handle:
            mapping = pickle.load(handle)

        offsets = {
            int(domain_id): (int(bounds[0]), int(bounds[1]))
            for domain_id, bounds in mapping["domain_offset"].items()
        }
        self.metadata = MDSRMetadata(
            user_count=len(mapping["user"]),
            item_count=len(mapping["item"]),
            domain_offsets=offsets,
        )
        self.train = ParquetSequenceStore(self.root / "train_new.parquet", self.metadata)
        self.valid = ParquetSequenceStore(self.root / "valid_new.parquet", self.metadata)
        self.test = ParquetSequenceStore(self.root / "test_new.parquet", self.metadata)
        if validate_pairing:
            self._validate_evaluation_pairing()

    def _validate_evaluation_pairing(self) -> None:
        if len(self.valid) != len(self.test):
            raise ValueError("valid_new and test_new must contain the same number of rows")
        valid_lengths = np.diff(self.valid.offsets)
        test_lengths = np.diff(self.test.offsets)
        if not np.array_equal(test_lengths, valid_lengths + 1):
            raise ValueError("Each test sequence must be one item longer than validation")
        for index in range(len(self.valid)):
            valid_items, valid_domains = self.valid.sequence(index)
            test_items, test_domains = self.test.sequence(index)
            if not np.array_equal(valid_items, test_items[:-1]):
                raise ValueError(f"Validation/test item prefix mismatch at row {index}")
            if not np.array_equal(valid_domains, test_domains[:-1]):
                raise ValueError(f"Validation/test domain prefix mismatch at row {index}")

    def summary(self) -> dict:
        return {
            "root": str(self.root.resolve()),
            "user_count": self.metadata.user_count,
            "item_count": self.metadata.item_count,
            "domain_offsets_zero_based": self.metadata.domain_offsets,
            "train_sequences": len(self.train),
            "validation_sequences": len(self.valid),
            "test_sequences": len(self.test),
            "train_length_range": [self.train.min_length, self.train.max_length],
            "validation_length_range": [self.valid.min_length, self.valid.max_length],
            "test_length_range": [self.test.min_length, self.test.max_length],
            "id_conversion": "ftrec_item_id = mdsr_item_id + 1",
            "evaluation_pairing": "test[row] == valid[row] + one final target",
        }


def _sample_unseen(
    rng: np.random.Generator,
    item_count: int,
    excluded: set[int],
    size: int,
) -> np.ndarray:
    result = np.empty(size, dtype=np.int32)
    for index in range(size):
        candidate = int(rng.integers(1, item_count + 1))
        while candidate in excluded:
            candidate = int(rng.integers(1, item_count + 1))
        result[index] = candidate
    return result


class TrainSequenceDataset(Dataset):
    """Create SASRec next-item training tuples directly from train_new.parquet."""

    def __init__(
        self,
        store: ParquetSequenceStore,
        item_count: int,
        maxlen: int,
        seed: int,
    ):
        self.store = store
        self.item_count = item_count
        self.maxlen = maxlen
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        items, _ = self.store.sequence(index)
        if len(items) < 2:
            raise ValueError(f"Training row {index} has fewer than two interactions")
        window = items[-(self.maxlen + 1) :]
        sequence = np.zeros(self.maxlen, dtype=np.int32)
        positive = np.zeros(self.maxlen, dtype=np.int32)
        length = len(window) - 1
        sequence[-length:] = window[:-1]
        positive[-length:] = window[1:]

        # The epoch participates in the seed so each epoch gets new negatives,
        # while repeated experimental runs remain exactly reproducible.
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(index)])
        )
        seen = set(int(item) for item in items)
        negative = np.zeros(self.maxlen, dtype=np.int32)
        negative[-length:] = _sample_unseen(rng, self.item_count, seen, length)
        return sequence, positive, negative, int(index)


class EvaluationSequenceDataset(Dataset):
    """Deterministic target-plus-negative candidates for validation or test."""

    def __init__(
        self,
        store: ParquetSequenceStore,
        item_count: int,
        maxlen: int,
        num_negatives: int,
        seed: int,
    ):
        self.store = store
        self.item_count = item_count
        self.maxlen = maxlen
        self.num_negatives = num_negatives
        self.seed = seed

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, int]:
        items, _ = self.store.sequence(index)
        if len(items) < 2:
            raise ValueError(f"Evaluation row {index} has fewer than two interactions")
        context, target = items[:-1], int(items[-1])
        sequence = np.zeros(self.maxlen, dtype=np.int32)
        clipped = context[-self.maxlen :]
        sequence[-len(clipped) :] = clipped

        candidates = np.empty(self.num_negatives + 1, dtype=np.int32)
        candidates[0] = target
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        candidates[1:] = _sample_unseen(
            rng,
            self.item_count,
            set(int(item) for item in items),
            self.num_negatives,
        )
        return sequence, candidates, int(index)


@torch.no_grad()
def evaluate_loader(model, loader, device: str) -> tuple[float, float, float]:
    """Return NDCG@10, HR@10 and MRR@10 for a deterministic loader."""

    del device  # The SASRec instance already owns its target device.
    model.eval()
    ndcg = 0.0
    hits = 0.0
    mrr = 0.0
    count = 0
    for sequences, candidates, row_ids in loader:
        sequence_np = sequences.numpy()
        candidate_np = candidates.numpy()
        logits = model.predict(row_ids.numpy(), sequence_np, candidate_np)
        target_scores = logits[:, :1]
        ranks = (logits[:, 1:] > target_scores).sum(dim=1).cpu().numpy()
        matched = ranks < 10
        ndcg += np.where(matched, 1.0 / np.log2(ranks + 2), 0.0).sum()
        hits += matched.sum()
        mrr += np.where(matched, 1.0 / (ranks + 1), 0.0).sum()
        count += len(ranks)
    if count == 0:
        raise ValueError("Evaluation loader produced no rows")
    return float(ndcg / count), float(hits / count), float(mrr / count)
