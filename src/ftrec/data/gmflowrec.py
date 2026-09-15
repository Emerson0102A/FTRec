"""Import the official GMFlowRec Amazon release into FTRec artifacts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from ftrec.artifacts import RunDirectory, sha256_file, write_json


DOMAIN_NAMES = (
    "Health",
    "Clothing",
    "Beauty",
    "Grocery",
    "Sports",
)
PAPER_DOMAIN_STATISTICS = {
    0: (45_662, 3_616_188),
    1: (81_229, 5_083_612),
    2: (41_091, 2_780_942),
    3: (30_786, 1_899_959),
    4: (20_078, 928_236),
}
PAPER_USERS = 580_329
PAPER_INTERACTIONS = 14_308_937
PAPER_SPLIT_ROWS = {"train": 491_445, "valid": 88_885, "test": 88_885}
SOURCE_FILES = (
    "processed.parquet",
    "train_new.parquet",
    "valid_new.parquet",
    "test_new.parquet",
)


class GMFlowRecDataError(ValueError):
    """Raised when the released data violates its documented contract."""


@dataclass(frozen=True)
class GMFlowRecImportSettings:
    source_dir: Path
    output_dir: Path
    num_eval_negatives: int = 999
    evaluation_seed: int = 2026
    verify_paper_statistics: bool = True
    batch_size: int = 4096
    progress: bool = True
    force: bool = False

    def __post_init__(self) -> None:
        if self.num_eval_negatives < 1:
            raise ValueError("num_eval_negatives must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class GMFlowRecImportResult:
    output_dir: Path
    train_sequences: int
    evaluation_sequences: int
    items: int
    interactions: int
    data_hash: str


def _require_schema(path: Path, expected: dict[str, pa.DataType]) -> None:
    if not path.is_file():
        raise GMFlowRecDataError(f"missing source file: {path.name}")
    schema = pq.ParquetFile(path).schema_arrow
    actual = {field.name: field.type for field in schema}
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise GMFlowRecDataError(f"{path.name} is missing columns: {missing}")
    wrong = [
        name
        for name, expected_type in expected.items()
        if actual[name] != expected_type
    ]
    if wrong:
        details = ", ".join(
            f"{name}={actual[name]} (expected {expected[name]})" for name in wrong
        )
        raise GMFlowRecDataError(f"{path.name} has incompatible schema: {details}")


def _validate_schemas(source: Path) -> None:
    _require_schema(
        source / "processed.parquet",
        {
            "user_id": pa.int32(),
            "item_id": pa.int64(),
            "domain_id": pa.int64(),
            "timestamp": pa.timestamp("us"),
        },
    )
    sequence_schema = {
        "item_id": pa.list_(pa.int64()),
        "domain_id": pa.list_(pa.int64()),
        "timestamp": pa.list_(pa.timestamp("us")),
    }
    _require_schema(source / "train_new.parquet", sequence_schema)
    _require_schema(source / "valid_new.parquet", sequence_schema)
    _require_schema(source / "test_new.parquet", sequence_schema)


def _catalogs_and_statistics(
    source: Path, *, verify_paper_statistics: bool
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    table = pq.read_table(
        source / "processed.parquet", columns=["user_id", "item_id", "domain_id"]
    )
    users = table["user_id"].combine_chunks().to_numpy()
    items = table["item_id"].combine_chunks().to_numpy()
    domains = table["domain_id"].combine_chunks().to_numpy()
    domain_stats: dict[int, dict[str, int]] = {}
    catalogs: dict[int, np.ndarray] = {}
    for domain in sorted(int(value) for value in np.unique(domains)):
        mask = domains == domain
        source_items = np.unique(items[mask])
        catalogs[domain] = (source_items + 1).astype(np.int32, copy=False)
        domain_stats[domain] = {
            "items": int(source_items.size),
            "interactions": int(mask.sum()),
            "users": int(np.unique(users[mask]).size),
        }
    statistics = {
        "users": int(np.unique(users).size),
        "interactions": int(table.num_rows),
        "domains": {str(key): value for key, value in domain_stats.items()},
    }
    if verify_paper_statistics:
        if statistics["users"] != PAPER_USERS:
            raise GMFlowRecDataError(
                f"processed.parquet has {statistics['users']} users; expected {PAPER_USERS}"
            )
        if statistics["interactions"] != PAPER_INTERACTIONS:
            raise GMFlowRecDataError(
                "processed.parquet interaction count does not match the paper"
            )
        for domain, (expected_items, expected_interactions) in PAPER_DOMAIN_STATISTICS.items():
            actual = domain_stats.get(domain)
            if actual is None or (
                actual["items"], actual["interactions"]
            ) != (expected_items, expected_interactions):
                raise GMFlowRecDataError(
                    f"domain {domain} statistics do not match the paper: {actual}"
                )
    return catalogs, statistics


def _iter_rows(path: Path, batch_size: int) -> Iterator[dict[str, object]]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _timestamp_us(value: datetime) -> int:
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return int(aware.timestamp() * 1_000_000)


def _sequence_payload(
    row: dict[str, object], *, user_id: int, splits: tuple[str, ...]
) -> dict[str, object]:
    item_ids = tuple(int(item) + 1 for item in row["item_id"])
    domain_ids = tuple(int(domain) for domain in row["domain_id"])
    timestamps = tuple(_timestamp_us(value) for value in row["timestamp"])
    if not (len(item_ids) == len(domain_ids) == len(timestamps) == len(splits)):
        raise GMFlowRecDataError(f"sequence arrays differ in length for imported user {user_id}")
    return {
        "domain_ids": domain_ids,
        "item_ids": item_ids,
        "splits": splits,
        "timestamps": timestamps,
        "user_id": user_id,
    }


def _write_line(stream: io.TextIOBase, payload: dict[str, object]) -> None:
    stream.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _open_gzip_text(path: Path) -> tuple[io.BufferedWriter, gzip.GzipFile, io.TextIOWrapper]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    return raw, compressed, text


def _close_gzip_text(handles: tuple[io.BufferedWriter, gzip.GzipFile, io.TextIOWrapper]) -> None:
    raw, compressed, text = handles
    text.flush()
    text.detach()
    compressed.close()
    raw.close()


def _count_target_domains(path: Path, batch_size: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in _iter_rows(path, batch_size):
        domains = row["domain_id"]
        if not domains:
            raise GMFlowRecDataError(f"{path.name} contains an empty sequence")
        domain = int(domains[-1])
        counts[domain] = counts.get(domain, 0) + 1
    return counts


def _stable_rng(
    *, evaluation_seed: int, split: str, user_id: int, domain: int, positive: int
) -> np.random.Generator:
    payload = f"{evaluation_seed}|{split}|{user_id}|{domain}|{positive}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def _candidate_row(
    catalog: np.ndarray,
    *,
    seen_items: tuple[int, ...],
    positive: int,
    negative_count: int,
    evaluation_seed: int,
    split: str,
    user_id: int,
    domain: int,
) -> np.ndarray:
    seen = np.asarray(sorted(set(seen_items)), dtype=np.int32)
    available = int(catalog.size - seen.size)
    if available < negative_count:
        raise GMFlowRecDataError(
            f"domain {domain} has only {available} unseen items for user {user_id}; "
            f"need {negative_count}"
        )
    rng = _stable_rng(
        evaluation_seed=evaluation_seed,
        split=split,
        user_id=user_id,
        domain=domain,
        positive=positive,
    )
    draw_size = min(int(catalog.size), negative_count + int(seen.size))
    draw = rng.choice(catalog, size=draw_size, replace=False)
    negatives = draw[~np.isin(draw, seen, assume_unique=False)][:negative_count]
    if negatives.size != negative_count:
        eligible = catalog[~np.isin(catalog, seen, assume_unique=False)]
        negatives = rng.choice(eligible, size=negative_count, replace=False)
    return np.concatenate(
        (np.asarray([positive], dtype=np.int32), negatives.astype(np.int32, copy=False))
    )


def _write_items(path: Path, catalogs: dict[int, np.ndarray]) -> int:
    handles = _open_gzip_text(path)
    try:
        writer = csv.writer(handles[2], lineterminator="\n")
        writer.writerow(("item_id", "domain_id", "source_item_id"))
        for domain, catalog in sorted(catalogs.items()):
            for item in catalog:
                writer.writerow((int(item), domain, int(item) - 1))
    finally:
        _close_gzip_text(handles)
    return sum(int(catalog.size) for catalog in catalogs.values())


def _allocate_candidate_matrices(
    root: Path,
    counts: dict[str, dict[int, int]],
    negative_count: int,
) -> tuple[
    dict[tuple[str, int], np.memmap],
    dict[tuple[str, int], np.memmap],
    dict[str, dict[str, object]],
]:
    evaluation_dir = root / "evaluation"
    evaluation_dir.mkdir(parents=True)
    matrices: dict[tuple[str, int], np.memmap] = {}
    row_keys: dict[tuple[str, int], np.memmap] = {}
    manifest: dict[str, dict[str, object]] = {"valid": {}, "test": {}}
    for split in ("valid", "test"):
        for domain, rows in sorted(counts[split].items()):
            relative = f"{split}-domain-{domain}.npy"
            matrix = np.lib.format.open_memmap(
                evaluation_dir / relative,
                mode="w+",
                dtype=np.int32,
                shape=(rows, negative_count + 1),
            )
            key_relative = f"{split}-domain-{domain}-user-ids.npy"
            keys = np.lib.format.open_memmap(
                evaluation_dir / key_relative,
                mode="w+",
                dtype=np.int64,
                shape=(rows,),
            )
            matrices[(split, domain)] = matrix
            row_keys[(split, domain)] = keys
            manifest[split][str(domain)] = {
                "file": relative,
                "key_file": key_relative,
                "rows": rows,
            }
    return matrices, row_keys, manifest


def import_gmflowrec(settings: GMFlowRecImportSettings) -> GMFlowRecImportResult:
    """Validate, convert, and atomically publish one official release."""
    source = Path(settings.source_dir).resolve()
    _validate_schemas(source)
    catalogs, source_statistics = _catalogs_and_statistics(
        source, verify_paper_statistics=settings.verify_paper_statistics
    )
    split_rows = {
        split: pq.ParquetFile(source / f"{split}_new.parquet").metadata.num_rows
        for split in ("train", "valid", "test")
    }
    if split_rows["valid"] != split_rows["test"]:
        raise GMFlowRecDataError("valid_new.parquet and test_new.parquet row counts differ")
    if settings.verify_paper_statistics and split_rows != PAPER_SPLIT_ROWS:
        raise GMFlowRecDataError(
            f"released split row counts changed: {split_rows}; expected {PAPER_SPLIT_ROWS}"
        )

    counts = {
        "valid": _count_target_domains(source / "valid_new.parquet", settings.batch_size),
        "test": _count_target_domains(source / "test_new.parquet", settings.batch_size),
    }
    result: GMFlowRecImportResult | None = None
    with RunDirectory(settings.output_dir, force=settings.force) as run:
        assert run.path is not None
        items = _write_items(run.path / "items.csv.gz", catalogs)
        matrices, candidate_row_keys, candidate_files = _allocate_candidate_matrices(
            run.path, counts, settings.num_eval_negatives
        )
        cursors = {(split, domain): 0 for split, domain in matrices}
        sequence_handles = _open_gzip_text(run.path / "sequences.jsonl.gz")
        try:
            train_progress = tqdm(
                total=split_rows["train"],
                desc="import GMFlowRec train",
                unit="seq",
                dynamic_ncols=True,
                disable=not settings.progress,
            )
            next_user_id = 1
            for row in _iter_rows(source / "train_new.parquet", settings.batch_size):
                length = len(row["item_id"])
                if length < 2:
                    raise GMFlowRecDataError("train_new.parquet contains a sequence shorter than 2")
                _write_line(
                    sequence_handles[2],
                    _sequence_payload(
                        row,
                        user_id=next_user_id,
                        splits=("context",) * (length - 1) + ("train",),
                    ),
                )
                next_user_id += 1
                train_progress.update(1)
            train_progress.close()

            eval_progress = tqdm(
                total=split_rows["test"],
                desc="import GMFlowRec eval",
                unit="seq",
                dynamic_ncols=True,
                disable=not settings.progress,
            )
            valid_rows = _iter_rows(source / "valid_new.parquet", settings.batch_size)
            test_rows = _iter_rows(source / "test_new.parquet", settings.batch_size)
            for valid, test in zip(valid_rows, test_rows, strict=True):
                if (
                    test["item_id"][:-1] != valid["item_id"]
                    or test["domain_id"][:-1] != valid["domain_id"]
                    or test["timestamp"][:-1] != valid["timestamp"]
                ):
                    raise GMFlowRecDataError(
                        f"test sequence for imported user {next_user_id} does not extend validation"
                    )
                length = len(test["item_id"])
                if length < 3:
                    raise GMFlowRecDataError("test_new.parquet contains a sequence shorter than 3")
                payload = _sequence_payload(
                    test,
                    user_id=next_user_id,
                    splits=("context",) * (length - 2) + ("valid", "test"),
                )
                _write_line(sequence_handles[2], payload)
                for split, position in (("valid", -2), ("test", -1)):
                    domain = int(payload["domain_ids"][position])
                    positive = int(payload["item_ids"][position])
                    seen_items = tuple(
                        int(item)
                        for item, seen_domain in zip(
                            payload["item_ids"], payload["domain_ids"], strict=True
                        )
                        if int(seen_domain) == domain
                    )
                    cursor = cursors[(split, domain)]
                    matrices[(split, domain)][cursor] = _candidate_row(
                        catalogs[domain],
                        seen_items=seen_items,
                        positive=positive,
                        negative_count=settings.num_eval_negatives,
                        evaluation_seed=settings.evaluation_seed,
                        split=split,
                        user_id=next_user_id,
                        domain=domain,
                    )
                    candidate_row_keys[(split, domain)][cursor] = next_user_id
                    cursors[(split, domain)] = cursor + 1
                next_user_id += 1
                eval_progress.update(1)
            eval_progress.close()
        finally:
            _close_gzip_text(sequence_handles)
            for matrix in matrices.values():
                matrix.flush()
                matrix._mmap.close()
            for keys in candidate_row_keys.values():
                keys.flush()
                keys._mmap.close()

        evaluation_manifest = {
            "algorithm": "gmflowrec-fixed-same-domain-v1",
            "candidate_count": settings.num_eval_negatives + 1,
            "evaluation_seed": settings.evaluation_seed,
            "negative_count": settings.num_eval_negatives,
            "positive_position": 0,
            "splits": candidate_files,
        }
        write_json(run.path / "evaluation" / "manifest.json", evaluation_manifest)
        summary = {
            "format": "gmflowrec-amazon",
            "item_id_offset": 1,
            "items": items,
            "source": source_statistics,
            "split_rows": split_rows,
            "sequence_records": split_rows["train"] + split_rows["test"],
        }
        write_json(run.path / "summary.json", summary)
        artifact_hashes = {
            "items.csv.gz": sha256_file(run.path / "items.csv.gz"),
            "sequences.jsonl.gz": sha256_file(run.path / "sequences.jsonl.gz"),
            "summary.json": sha256_file(run.path / "summary.json"),
            "evaluation/manifest.json": sha256_file(
                run.path / "evaluation" / "manifest.json"
            ),
        }
        for split_files in candidate_files.values():
            for value in split_files.values():
                for file_key in ("file", "key_file"):
                    relative = Path("evaluation") / str(value[file_key])
                    artifact_hashes[relative.as_posix()] = sha256_file(run.path / relative)
        source_hashes = {name: sha256_file(source / name) for name in SOURCE_FILES}
        manifest = {
            "artifacts": artifact_hashes,
            "config": {
                "evaluation_seed": settings.evaluation_seed,
                "num_eval_negatives": settings.num_eval_negatives,
            },
            "schema_version": 1,
            "source_commit": "1d725aafcdd6c21933c186318096331ad885c571",
            "source_files": source_hashes,
        }
        write_json(run.path / "manifest.json", manifest)
        data_hash = sha256_file(run.path / "manifest.json")
        run.complete({"data_hash": data_hash, "schema_version": 1})
        result = GMFlowRecImportResult(
            Path(settings.output_dir).resolve(),
            split_rows["train"],
            split_rows["test"],
            items,
            int(source_statistics["interactions"]),
            data_hash,
        )
    assert result is not None
    return result
