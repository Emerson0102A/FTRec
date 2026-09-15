"""Processed sequence loading and domain-task example construction."""

from __future__ import annotations

import csv
import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SequenceRecord:
    user_id: int
    item_ids: tuple[int, ...]
    domain_ids: tuple[int, ...]
    timestamps: tuple[int, ...]
    splits: tuple[str, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.item_ids),
            len(self.domain_ids),
            len(self.timestamps),
            len(self.splits),
        }
        if len(lengths) != 1:
            raise ValueError(f"sequence arrays differ in length for user {self.user_id}")


@dataclass
class SequenceStore:
    records: tuple[SequenceRecord, ...]
    items_by_domain: dict[int, tuple[int, ...]]
    processed_dir: Path | None = None
    last_build_skipped: int = field(default=0, init=False)

    @classmethod
    def from_processed(cls, directory: str | Path) -> "SequenceStore":
        root = Path(directory)
        records: list[SequenceRecord] = []
        with gzip.open(root / "sequences.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                records.append(
                    SequenceRecord(
                        int(value["user_id"]),
                        tuple(int(item) for item in value["item_ids"]),
                        tuple(int(domain) for domain in value["domain_ids"]),
                        tuple(int(timestamp) for timestamp in value["timestamps"]),
                        tuple(str(split) for split in value["splits"]),
                    )
                )
        items: dict[int, list[int]] = {}
        with gzip.open(root / "items.csv.gz", "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                items.setdefault(int(row["domain_id"]), []).append(int(row["item_id"]))
        return cls(
            tuple(records),
            {domain: tuple(sorted(values)) for domain, values in items.items()},
            root,
        )

    def evaluation_candidates(
        self,
        *,
        split: str,
        domain: int,
        negative_count: int,
        evaluation_seed: int,
    ) -> "CandidateMatrix | None":
        if self.processed_dir is None:
            return None
        manifest_path = self.processed_dir / "evaluation" / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["negative_count"]) != negative_count:
            raise ValueError("cached evaluation negative count differs from configuration")
        if int(manifest["evaluation_seed"]) != evaluation_seed:
            raise ValueError("cached evaluation seed differs from configuration")
        entry = manifest.get("splits", {}).get(split, {}).get(str(domain))
        if entry is None:
            return None
        key_file = entry.get("key_file")
        return CandidateMatrix(
            self.processed_dir / "evaluation" / str(entry["file"]),
            expected_rows=int(entry["rows"]),
            expected_width=negative_count + 1,
            key_path=(
                self.processed_dir / "evaluation" / str(key_file)
                if key_file is not None
                else None
            ),
        )


class CandidateMatrix:
    """Read-only candidates keyed by local example id."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_rows: int,
        expected_width: int,
        key_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        # Linux can replace an open mapped file atomically. Windows cannot, so
        # tests and local force-imports use a normal read-only array there.
        values = np.load(
            self.path,
            mmap_mode="r" if os.name != "nt" else None,
            allow_pickle=False,
        )
        values.flags.writeable = False
        if values.dtype != np.int32 or values.shape != (expected_rows, expected_width):
            raise ValueError(
                f"invalid candidate matrix {self.path}: {values.dtype} {values.shape}"
            )
        self.values = values
        self._row_indices = np.arange(expected_rows, dtype=np.int64)
        self.row_keys: np.ndarray | None = None
        if key_path is not None:
            keys = np.load(key_path, mmap_mode="r" if os.name != "nt" else None, allow_pickle=False)
            if keys.dtype != np.int64 or keys.shape != (expected_rows,):
                raise ValueError(f"invalid candidate row keys {key_path}: {keys.dtype} {keys.shape}")
            if np.unique(keys).size != expected_rows:
                raise ValueError(f"candidate row keys are not unique: {key_path}")
            keys.flags.writeable = False
            self.row_keys = keys

    def __len__(self) -> int:
        return int(self._row_indices.size)

    def __getitem__(self, example_id: int) -> np.ndarray:
        return self.values[int(self._row_indices[example_id])]

    def get(self, example_id: int, default: object = None) -> np.ndarray | object:
        if 0 <= example_id < len(self):
            return self[example_id]
        return default

    def batch(self, example_ids: Sequence[int]) -> np.ndarray:
        local = np.asarray(example_ids, dtype=np.int64)
        return np.asarray(self.values[self._row_indices[local]])

    def select(self, examples: Sequence["TargetExample"]) -> "CandidateMatrix":
        """Return a dense example-id view selected by stable imported user ids."""
        if self.row_keys is None:
            if len(self) != len(examples):
                raise ValueError(
                    f"cached candidates have {len(self)} rows; expected {len(examples)}"
                )
            selected = self
        else:
            key_to_row = {int(key): index for index, key in enumerate(self.row_keys)}
            try:
                rows = np.asarray(
                    [key_to_row[example.user_id] for example in examples], dtype=np.int64
                )
            except KeyError as error:
                raise ValueError(f"cached candidates are missing user {error.args[0]}") from error
            selected = object.__new__(type(self))
            selected.path = self.path
            selected.values = self.values
            selected.row_keys = self.row_keys
            selected._row_indices = rows
        for example in examples:
            if int(selected[example.example_id][0]) != example.positive_item:
                raise ValueError(
                    f"cached candidates do not match example {example.example_id} "
                    f"for user {example.user_id}"
                )
        return selected


@dataclass(frozen=True)
class TargetExample:
    example_id: int
    user_id: int
    context_items: tuple[int, ...]
    context_domains: tuple[int, ...]
    positive_item: int
    target_domain: int
    seen_items: frozenset[int]


def _left_pad(values: Iterable[int], length: int, padding: int) -> tuple[int, ...]:
    trimmed = tuple(values)[-length:]
    return (padding,) * (length - len(trimmed)) + trimmed


def _build_examples(
    store: SequenceStore,
    *,
    split: str,
    target_domain: int,
    maxlen: int,
    single_domain: bool,
) -> list[TargetExample]:
    if maxlen < 1:
        raise ValueError("maxlen must be positive")
    examples: list[TargetExample] = []
    skipped = 0
    next_id = 0
    for record in store.records:
        seen = frozenset(record.item_ids)
        for index, (domain, row_split) in enumerate(
            zip(record.domain_ids, record.splits, strict=True)
        ):
            if domain != target_domain or row_split != split:
                continue
            if single_domain:
                prior = [
                    (item, prior_domain)
                    for item, prior_domain in zip(
                        record.item_ids[:index], record.domain_ids[:index], strict=True
                    )
                    if prior_domain == target_domain
                ]
            else:
                prior = list(
                    zip(record.item_ids[:index], record.domain_ids[:index], strict=True)
                )
            if not prior:
                skipped += 1
                continue
            context_items = _left_pad((item for item, _ in prior), maxlen, 0)
            context_domains = _left_pad(
                (prior_domain for _, prior_domain in prior), maxlen, -1
            )
            examples.append(
                TargetExample(
                    example_id=next_id,
                    user_id=record.user_id,
                    context_items=context_items,
                    context_domains=context_domains,
                    positive_item=record.item_ids[index],
                    target_domain=target_domain,
                    seen_items=seen,
                )
            )
            next_id += 1
    store.last_build_skipped = skipped
    return examples


def build_mixed_examples(
    store: SequenceStore, *, split: str, target_domain: int, maxlen: int
) -> list[TargetExample]:
    return _build_examples(
        store,
        split=split,
        target_domain=target_domain,
        maxlen=maxlen,
        single_domain=False,
    )


def build_single_domain_examples(
    store: SequenceStore, *, split: str, domain: int, maxlen: int
) -> list[TargetExample]:
    return _build_examples(
        store,
        split=split,
        target_domain=domain,
        maxlen=maxlen,
        single_domain=True,
    )
