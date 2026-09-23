"""Adapt GMFlowRec's published Parquet splits to CGRec's ID conventions."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
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


class CGRecCategoryMapping:
    """Two category levels from Amazon metadata, aligned by original ASIN."""

    def __init__(self, catalog_path: str | Path, mappings_path: str | Path, metadata: MDSRMetadata):
        with Path(mappings_path).open("rb") as stream:
            mapping = pickle.load(stream)
        asin_by_id = [""] * metadata.item_count
        for asin, zero_id in mapping["item"].items():
            asin_by_id[int(zero_id)] = str(asin)
        if not all(asin_by_id):
            raise ValueError("mappings.pkl does not cover every item ID")

        cat1_keys: list[tuple[int, str, str] | None] = [None] * metadata.item_count
        cat2_keys: list[tuple[int, str] | None] = [None] * metadata.item_count
        self.missing_coarse = 0
        self.missing_fine = 0
        path = Path(catalog_path)
        opener = gzip.open if path.suffix == ".gz" else open
        digest = hashlib.sha256()
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                digest.update(line.encode("utf-8"))
                if not line.strip():
                    continue
                row = json.loads(line)
                zero_id = int(row["item_id"]) - 1
                if not 0 <= zero_id < metadata.item_count or cat1_keys[zero_id] is not None:
                    raise ValueError(f"catalog has an invalid or duplicate item ID: {row['item_id']}")
                domain = int(row["domain_id"])
                bounds = metadata.domain_offsets.get(domain)
                if bounds is None or not bounds[0] <= zero_id < bounds[1]:
                    raise ValueError(f"catalog domain mismatch for item ID {row['item_id']}")
                if str(row["parent_asin"]) != asin_by_id[zero_id]:
                    raise ValueError(f"catalog ASIN mismatch for item ID {row['item_id']}")
                raw_categories = row.get("categories")
                categories = raw_categories if isinstance(raw_categories, list) else []
                coarse = str(categories[1]).strip() if len(categories) >= 2 else "\x00missing"
                fine = str(categories[2]).strip() if len(categories) >= 3 else (
                    "\x00terminal" if len(categories) >= 2 else "\x00missing"
                )
                if len(categories) < 2:
                    self.missing_coarse += 1
                if len(categories) < 3:
                    self.missing_fine += 1
                cat2_keys[zero_id] = (domain, coarse)
                cat1_keys[zero_id] = (domain, coarse, fine)
        if any(key is None for key in cat1_keys):
            raise ValueError("catalog does not cover every item in mappings.pkl")

        # Match the released vocabularies: IDs 0..4 are special tokens.
        cat1_vocab = {key: value for value, key in enumerate(sorted(set(cat1_keys)), start=5)}
        cat2_vocab = {key: value for value, key in enumerate(sorted(set(cat2_keys)), start=5)}
        self.cat1_by_item = np.zeros(metadata.item_count + 5, dtype=np.int64)
        self.cat2_by_item = np.zeros(metadata.item_count + 5, dtype=np.int64)
        for zero_id, (cat1, cat2) in enumerate(zip(cat1_keys, cat2_keys)):
            self.cat1_by_item[zero_id + 5] = cat1_vocab[cat1]
            self.cat2_by_item[zero_id + 5] = cat2_vocab[cat2]
        self.cat1_size = len(cat1_vocab) + 5
        self.cat2_size = len(cat2_vocab) + 5
        self.negative_min = 5
        self.source = "Amazon metadata categories; hierarchical CGRec"
        self.level_description = (
            "cat2=categories[1], cat1=categories[2]; domain/path-aware; "
            "missing and terminal values use explicit tokens"
        )
        self.path = path
        self.sha256 = digest.hexdigest()


class CGRecDomainCategoryMapping:
    """Mirror the released dataset's cat1=cat2=domain sequence behavior."""

    def __init__(self, metadata: MDSRMetadata, target_domain: int):
        mapping = domain_remap(target_domain)
        self.cat1_by_item = np.zeros(metadata.item_count + 5, dtype=np.int64)
        for domain, (start, end) in metadata.domain_offsets.items():
            self.cat1_by_item[start + 5 : end + 5] = mapping[domain]
        self.cat2_by_item = self.cat1_by_item
        # The released run_pretrain.py sizes category embeddings as len(vocab)
        # +2 and +1; its tiny category vocabs contain five special tokens and
        # five domain names. Negatives exclude reserved IDs 0..4.
        self.cat1_size = 12
        self.cat2_size = 11
        self.negative_min = 5
        self.source = "released code: cat1=cat2=remapped domain IDs"
        self.level_description = "both category streams copy the domain sequence"
        self.missing_coarse = 0
        self.missing_fine = 0
        self.path = None
        self.sha256 = None


def _category_negatives(
    rng: np.random.Generator, items: np.ndarray, all_categories: np.ndarray,
    size: int, minimum: int,
) -> np.ndarray:
    result = np.zeros(len(items), dtype=np.int64)
    seen = set(all_categories.tolist())
    seen.discard(0)
    available = (size - minimum) - sum(minimum <= value < size for value in seen)
    if available <= 0:
        raise ValueError("user sequence exhausts the category vocabulary")
    for position in np.flatnonzero(items):
        candidate = int(rng.integers(minimum, size))
        while candidate in seen:
            candidate = int(rng.integers(minimum, size))
        result[position] = candidate
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
        categories: CGRecCategoryMapping | CGRecDomainCategoryMapping | None = None,
    ):
        if maxlen <= 0 or (max_examples is not None and max_examples <= 0):
            raise ValueError("maxlen and max_examples must be positive")
        self.store = store
        self.item_count = metadata.item_count
        self.mapping = domain_remap(target_domain)
        self.maxlen = maxlen
        self.seed = seed
        self.categories = categories
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
        zeros = np.zeros(self.maxlen, dtype=np.int64)
        if self.categories is None:
            cat1_input = cat1_pos = cat1_neg = zeros
            cat2_input = cat2_pos = cat2_neg = zeros
        else:
            cat1_input = self.categories.cat1_by_item[item_input]
            cat1_pos = self.categories.cat1_by_item[item_pos]
            cat2_input = self.categories.cat2_by_item[item_input]
            cat2_pos = self.categories.cat2_by_item[item_pos]
            all_items = items.astype(np.int64) + 4
            cat1_neg = _category_negatives(
                rng, item_input, self.categories.cat1_by_item[all_items],
                self.categories.cat1_size, self.categories.negative_min,
            )
            cat2_neg = _category_negatives(
                rng, item_input, self.categories.cat2_by_item[all_items],
                self.categories.cat2_size, self.categories.negative_min,
            )
        return (
            item_input, item_pos, negatives,
            cat1_input, cat1_pos, cat1_neg,
            cat2_input, cat2_pos, cat2_neg,
            type_input, index,
        )


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
        categories: CGRecCategoryMapping | CGRecDomainCategoryMapping | None = None,
    ):
        if max_examples is not None and max_examples <= 0:
            raise ValueError("max_examples must be positive")
        self.mapping = domain_remap(target_domain)
        self.categories = categories
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
        if self.categories is None:
            cat1 = cat2 = np.zeros(len(shifted_items), dtype=np.int64)
        else:
            cat1 = self.categories.cat1_by_item[shifted_items]
            cat2 = self.categories.cat2_by_item[shifted_items]
        return (
            shifted_items, cat1, cat2, _map_domains(domains, self.mapping),
            shifted_candidates, row,
        )

    def precompute(self) -> None:
        """Freeze sampled candidates once, avoiding resampling every epoch."""
        if self._cache is None:
            self._cache = [self[index] for index in range(len(self.rows))]
