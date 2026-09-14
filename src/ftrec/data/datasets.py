"""Processed sequence loading and domain-task example construction."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


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
        )


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

