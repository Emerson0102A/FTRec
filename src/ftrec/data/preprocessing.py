"""Memory-bounded Amazon ingestion and exact joint k-core filtering."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from ftrec.artifacts import RunDirectory

from .amazon import AMAZON5_DOMAINS, DOMAIN_BY_ID, DomainSpec


REQUIRED_COLUMNS = frozenset({"user_id", "parent_asin", "rating", "timestamp"})


def _progress(event: str, **values: object) -> None:
    print(
        json.dumps({"event": event, **values}, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


class InputSchemaError(ValueError):
    """Raised when an input file does not satisfy the raw schema."""


class DataInvariantError(RuntimeError):
    """Raised when processed interactions violate a required invariant."""


@dataclass(frozen=True)
class PreprocessSettings:
    input_dir: Path
    output_dir: Path
    min_user_interactions: int = 10
    min_item_interactions: int = 15
    batch_size: int = 100_000
    sqlite_path: Path | None = None
    force: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dir", Path(self.input_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.sqlite_path is not None:
            object.__setattr__(self, "sqlite_path", Path(self.sqlite_path))
        if self.min_user_interactions < 1 or self.min_item_interactions < 1:
            raise ValueError("k-core thresholds must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class IngestReport:
    raw_rows: int
    valid_rows: int
    retained_rows: int
    duplicate_rows: int
    invalid_missing_identity: int
    invalid_timestamp: int
    invalid_structure: int
    per_domain: dict[str, dict[str, int]]


@dataclass(frozen=True)
class KCoreIteration:
    index: int
    low_users: int
    low_items: int
    deleted_edges: int


@dataclass(frozen=True)
class KCoreReport:
    iterations: tuple[KCoreIteration, ...]
    initial_edges: int
    retained_edges: int

    @property
    def rounds(self) -> int:
        return len(self.iterations)


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    artifact_hashes: dict[str, str]
    users: int
    items: int
    interactions: int


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: tuple[str, ...]
    users: int = 0
    items: int = 0
    interactions: int = 0


@dataclass(frozen=True)
class PreprocessResult:
    output_dir: Path
    ingest: IngestReport
    kcore: KCoreReport
    export: ExportResult
    validation: ValidationReport


def open_database(path: str | Path) -> sqlite3.Connection:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("PRAGMA mmap_size=268435456")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            user_raw TEXT NOT NULL,
            domain_id INTEGER NOT NULL,
            item_raw TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            source_ordinal INTEGER NOT NULL,
            PRIMARY KEY (user_raw, domain_id, item_raw)
        ) WITHOUT ROWID
        """
    )
    return connection


def _input_path(settings: PreprocessSettings, domain: DomainSpec) -> Path:
    path = settings.input_dir / domain.filename
    if not path.is_file():
        raise InputSchemaError(f"{domain.name}: input file not found: {path}")
    return path


def ingest_amazon5(
    connection: sqlite3.Connection,
    settings: PreprocessSettings,
    domains: tuple[DomainSpec, ...] = AMAZON5_DOMAINS,
) -> IngestReport:
    counters = {
        domain.name: {
            "raw_rows": 0,
            "valid_rows": 0,
            "invalid_missing_identity": 0,
            "invalid_timestamp": 0,
            "invalid_structure": 0,
        }
        for domain in domains
    }
    raw_rows = valid_rows = missing = bad_timestamp = bad_structure = 0
    ordinal = 0
    upsert = """
        INSERT INTO interactions
            (user_raw, domain_id, item_raw, timestamp, source_ordinal)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_raw, domain_id, item_raw) DO UPDATE SET
            timestamp=excluded.timestamp,
            source_ordinal=excluded.source_ordinal
        WHERE excluded.timestamp < interactions.timestamp
           OR (excluded.timestamp = interactions.timestamp
               AND excluded.source_ordinal < interactions.source_ordinal)
    """
    started = time.perf_counter()
    input_sizes = {domain.name: _input_path(settings, domain).stat().st_size for domain in domains}
    total_input_bytes = sum(input_sizes.values())
    completed_bytes = 0
    for domain in domains:
        path = _input_path(settings, domain)
        _progress("preprocess_ingest_domain_start", domain=domain.name, path=str(path))
        batch: list[tuple[str, int, str, int, int]] = []
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            absent = sorted(REQUIRED_COLUMNS - fields)
            if absent:
                raise InputSchemaError(
                    f"{domain.name}: missing required columns: {', '.join(absent)}"
                )
            for row in reader:
                raw_rows += 1
                counters[domain.name]["raw_rows"] += 1
                ordinal += 1
                if None in row or any(value is None for value in row.values()):
                    bad_structure += 1
                    counters[domain.name]["invalid_structure"] += 1
                    continue
                user = row["user_id"].strip()
                item = row["parent_asin"].strip()
                raw_time = row["timestamp"].strip()
                if not user or not item or not raw_time:
                    missing += 1
                    counters[domain.name]["invalid_missing_identity"] += 1
                    continue
                try:
                    timestamp = int(raw_time)
                except ValueError:
                    bad_timestamp += 1
                    counters[domain.name]["invalid_timestamp"] += 1
                    continue
                valid_rows += 1
                counters[domain.name]["valid_rows"] += 1
                batch.append((user, domain.domain_id, item, timestamp, ordinal))
                if len(batch) >= settings.batch_size:
                    connection.executemany(upsert, batch)
                    connection.commit()
                    batch.clear()
                    if counters[domain.name]["raw_rows"] % 1_000_000 < settings.batch_size:
                        elapsed = time.perf_counter() - started
                        try:
                            current_bytes = int(stream.buffer.fileobj.tell())
                        except (AttributeError, OSError):
                            current_bytes = 0
                        fraction = (
                            min(completed_bytes + current_bytes, total_input_bytes)
                            / total_input_bytes
                            if total_input_bytes
                            else 0.0
                        )
                        eta = elapsed * (1.0 - fraction) / fraction if fraction else None
                        _progress(
                            "preprocess_ingest_progress",
                            domain=domain.name,
                            elapsed_seconds=round(elapsed, 1),
                            eta_seconds=round(eta, 1) if eta is not None else None,
                            raw_rows=raw_rows,
                            rows_per_second=round(raw_rows / elapsed, 1),
                        )
        if batch:
            connection.executemany(upsert, batch)
            connection.commit()
        completed_bytes += input_sizes[domain.name]
        elapsed = time.perf_counter() - started
        fraction = completed_bytes / total_input_bytes if total_input_bytes else 1.0
        eta = elapsed * (1.0 - fraction) / fraction if fraction else 0.0
        _progress(
            "preprocess_ingest_domain_complete",
            domain=domain.name,
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(eta, 1),
            raw_rows=raw_rows,
            rows_per_second=round(raw_rows / elapsed, 1),
        )
    _progress("preprocess_build_indexes_start", elapsed_seconds=round(time.perf_counter() - started, 1))
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_raw)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_interactions_item ON interactions(domain_id, item_raw)"
    )
    connection.commit()
    retained = connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    return IngestReport(
        raw_rows=raw_rows,
        valid_rows=valid_rows,
        retained_rows=retained,
        duplicate_rows=valid_rows - retained,
        invalid_missing_identity=missing,
        invalid_timestamp=bad_timestamp,
        invalid_structure=bad_structure,
        per_domain=counters,
    )


def run_joint_k_core(
    connection: sqlite3.Connection, user_min: int, item_min: int
) -> KCoreReport:
    if user_min < 1 or item_min < 1:
        raise ValueError("k-core thresholds must be positive")
    initial = connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    started = time.perf_counter()
    iterations: list[KCoreIteration] = []
    index = 1
    while True:
        connection.execute("DROP TABLE IF EXISTS temp.low_users")
        connection.execute("DROP TABLE IF EXISTS temp.low_items")
        connection.execute(
            "CREATE TEMP TABLE low_users AS "
            "SELECT user_raw FROM interactions GROUP BY user_raw HAVING COUNT(*) < ?",
            (user_min,),
        )
        connection.execute(
            "CREATE TEMP TABLE low_items AS "
            "SELECT domain_id, item_raw FROM interactions "
            "GROUP BY domain_id, item_raw HAVING COUNT(*) < ?",
            (item_min,),
        )
        connection.execute(
            "CREATE INDEX temp.idx_low_users ON low_users(user_raw)"
        )
        connection.execute(
            "CREATE INDEX temp.idx_low_items ON low_items(domain_id, item_raw)"
        )
        low_users = connection.execute("SELECT COUNT(*) FROM low_users").fetchone()[0]
        low_items = connection.execute("SELECT COUNT(*) FROM low_items").fetchone()[0]
        deleted = connection.execute(
            """
            SELECT COUNT(*) FROM interactions AS edge
            WHERE edge.user_raw IN (SELECT user_raw FROM low_users)
               OR EXISTS (
                    SELECT 1 FROM low_items AS item
                    WHERE item.domain_id=edge.domain_id AND item.item_raw=edge.item_raw
               )
            """
        ).fetchone()[0]
        if deleted:
            connection.execute(
                """
                DELETE FROM interactions
                WHERE user_raw IN (SELECT user_raw FROM low_users)
                   OR EXISTS (
                        SELECT 1 FROM low_items AS item
                        WHERE item.domain_id=interactions.domain_id
                          AND item.item_raw=interactions.item_raw
                   )
                """
            )
            connection.commit()
        iterations.append(KCoreIteration(index, low_users, low_items, deleted))
        _progress(
            "preprocess_kcore_iteration",
            deleted_edges=deleted,
            elapsed_seconds=round(time.perf_counter() - started, 1),
            iteration=index,
            low_items=low_items,
            low_users=low_users,
        )
        if deleted == 0:
            break
        index += 1
    assert_k_core_invariants(connection, user_min, item_min)
    retained = connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    return KCoreReport(tuple(iterations), initial, retained)


def assert_k_core_invariants(
    connection: sqlite3.Connection, user_min: int, item_min: int
) -> None:
    bad_user = connection.execute(
        "SELECT user_raw, COUNT(*) FROM interactions GROUP BY user_raw "
        "HAVING COUNT(*) < ? LIMIT 1",
        (user_min,),
    ).fetchone()
    if bad_user is not None:
        raise DataInvariantError(f"user below k-core threshold: {bad_user}")
    bad_item = connection.execute(
        "SELECT domain_id, item_raw, COUNT(*) FROM interactions "
        "GROUP BY domain_id, item_raw HAVING COUNT(*) < ? LIMIT 1",
        (item_min,),
    ).fetchone()
    if bad_item is not None:
        raise DataInvariantError(f"item below k-core threshold: {bad_item}")


def _create_stable_mappings(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS user_map")
    connection.execute("DROP TABLE IF EXISTS item_map")
    connection.execute(
        "CREATE TABLE user_map (user_raw TEXT PRIMARY KEY, user_id INTEGER UNIQUE NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE item_map (domain_id INTEGER NOT NULL, item_raw TEXT NOT NULL, "
        "item_id INTEGER UNIQUE NOT NULL, PRIMARY KEY(domain_id, item_raw))"
    )
    users = connection.execute(
        "SELECT DISTINCT user_raw FROM interactions ORDER BY user_raw"
    )
    connection.executemany(
        "INSERT INTO user_map VALUES (?, ?)",
        ((row[0], index) for index, row in enumerate(users, start=1)),
    )
    items = connection.execute(
        "SELECT DISTINCT domain_id, item_raw FROM interactions ORDER BY domain_id, item_raw"
    )
    connection.executemany(
        "INSERT INTO item_map VALUES (?, ?, ?)",
        ((row[0], row[1], index) for index, row in enumerate(items, start=1)),
    )
    connection.commit()


def _iter_user_groups(connection: sqlite3.Connection) -> Iterator[list[dict[str, object]]]:
    cursor = connection.execute(
        """
        SELECT users.user_id, items.item_id, edge.domain_id, edge.item_raw,
               edge.timestamp
        FROM interactions AS edge
        JOIN user_map AS users ON users.user_raw=edge.user_raw
        JOIN item_map AS items
          ON items.domain_id=edge.domain_id AND items.item_raw=edge.item_raw
        ORDER BY users.user_id, edge.timestamp, edge.domain_id, edge.item_raw
        """
    )
    current_user: int | None = None
    group: list[dict[str, object]] = []
    for user_id, item_id, domain_id, item_raw, timestamp in cursor:
        if current_user is not None and user_id != current_user:
            yield _assign_positions_and_splits(group)
            group = []
        current_user = user_id
        group.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "domain_id": domain_id,
                "domain": DOMAIN_BY_ID[domain_id].name,
                "item_raw": item_raw,
                "timestamp": timestamp,
            }
        )
    if group:
        yield _assign_positions_and_splits(group)


def _assign_positions_and_splits(group: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(group) < 3:
        raise DataInvariantError(
            f"user {group[0]['user_id']} has fewer than three retained interactions"
        )
    for position, row in enumerate(group):
        row["position"] = position
        if position == len(group) - 2:
            row["split"] = "valid"
        elif position == len(group) - 1:
            row["split"] = "test"
        else:
            row["split"] = "train"
    return group


def _gzip_text_writer(path: Path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    return raw, compressed, text


def _write_mapping_files(connection: sqlite3.Connection, output_dir: Path) -> None:
    raw, compressed, text = _gzip_text_writer(output_dir / "users.csv.gz")
    try:
        writer = csv.writer(text, lineterminator="\n")
        writer.writerow(["user_id", "raw_user_id"])
        writer.writerows(
            connection.execute("SELECT user_id, user_raw FROM user_map ORDER BY user_id")
        )
    finally:
        text.close()
        compressed.close()
        raw.close()
    raw, compressed, text = _gzip_text_writer(output_dir / "items.csv.gz")
    try:
        writer = csv.writer(text, lineterminator="\n")
        writer.writerow(["item_id", "domain_id", "domain", "parent_asin"])
        for item_id, domain_id, item_raw in connection.execute(
            "SELECT item_id, domain_id, item_raw FROM item_map ORDER BY item_id"
        ):
            writer.writerow([item_id, domain_id, DOMAIN_BY_ID[domain_id].name, item_raw])
    finally:
        text.close()
        compressed.close()
        raw.close()


INTERACTION_SCHEMA = pa.schema(
    [
        ("user_id", pa.int64()),
        ("item_id", pa.int64()),
        ("domain_id", pa.int8()),
        ("domain", pa.string()),
        ("timestamp", pa.int64()),
        ("position", pa.int32()),
        ("split", pa.string()),
    ]
)


def _write_interactions_and_sequences(
    connection: sqlite3.Connection, output_dir: Path, *, row_group_size: int
) -> tuple[int, int]:
    parquet_path = output_dir / "interactions.parquet"
    raw, compressed, sequence_stream = _gzip_text_writer(
        output_dir / "sequences.jsonl.gz"
    )
    writer = pq.ParquetWriter(
        parquet_path,
        INTERACTION_SCHEMA,
        compression="zstd",
        use_dictionary=False,
        version="2.6",
    )
    users = interactions = 0
    buffered: list[dict[str, object]] = []
    try:
        for group in _iter_user_groups(connection):
            serializable = [
                {key: row[key] for key in INTERACTION_SCHEMA.names} for row in group
            ]
            buffered.extend(serializable)
            if len(buffered) >= row_group_size:
                writer.write_table(
                    pa.Table.from_pylist(buffered, schema=INTERACTION_SCHEMA),
                    row_group_size=row_group_size,
                )
                buffered.clear()
            sequence = {
                "domain_ids": [row["domain_id"] for row in group],
                "item_ids": [row["item_id"] for row in group],
                "splits": [row["split"] for row in group],
                "timestamps": [row["timestamp"] for row in group],
                "user_id": group[0]["user_id"],
            }
            sequence_stream.write(
                json.dumps(
                    sequence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            users += 1
            interactions += len(group)
        if buffered:
            writer.write_table(
                pa.Table.from_pylist(buffered, schema=INTERACTION_SCHEMA),
                row_group_size=row_group_size,
            )
    finally:
        writer.close()
        sequence_stream.close()
        compressed.close()
        raw.close()
    return users, interactions


def _write_statistics(connection: sqlite3.Connection, output_dir: Path) -> None:
    domain_rows: list[list[object]] = []
    for domain in AMAZON5_DOMAINS:
        users, items, interactions = connection.execute(
            """
            SELECT COUNT(DISTINCT user_raw), COUNT(DISTINCT item_raw), COUNT(*)
            FROM interactions WHERE domain_id=?
            """,
            (domain.domain_id,),
        ).fetchone()
        average = interactions / users if users else 0.0
        domain_rows.append(
            [domain.domain_id, domain.name, users, items, interactions, average]
        )
    with (output_dir / "domain_stats.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "domain_id",
                "domain",
                "users",
                "items",
                "interactions",
                "avg_sequence_length",
            ]
        )
        writer.writerows(domain_rows)
    with (output_dir / "user_overlap.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["domain_a_id", "domain_a", "domain_b_id", "domain_b", "shared_users", "union_users", "jaccard"]
        )
        for left in AMAZON5_DOMAINS:
            for right in AMAZON5_DOMAINS:
                shared = connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT user_raw FROM interactions WHERE domain_id=?
                        INTERSECT
                        SELECT user_raw FROM interactions WHERE domain_id=?
                    )
                    """,
                    (left.domain_id, right.domain_id),
                ).fetchone()[0]
                union = connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT user_raw FROM interactions WHERE domain_id=?
                        UNION
                        SELECT user_raw FROM interactions WHERE domain_id=?
                    )
                    """,
                    (left.domain_id, right.domain_id),
                ).fetchone()[0]
                writer.writerow(
                    [
                        left.domain_id,
                        left.name,
                        right.domain_id,
                        right.name,
                        shared,
                        union,
                        shared / union if union else 0.0,
                    ]
                )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def export_processed_dataset(
    connection: sqlite3.Connection,
    output_dir: str | Path,
    ingest_report: IngestReport,
    kcore_report: KCoreReport,
    settings: PreprocessSettings,
) -> ExportResult:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _progress("preprocess_export_start", output_dir=str(target))
    _create_stable_mappings(connection)
    _write_mapping_files(connection, target)
    users, interactions = _write_interactions_and_sequences(
        connection, target, row_group_size=settings.batch_size
    )
    _write_statistics(connection, target)
    items = connection.execute("SELECT COUNT(*) FROM item_map").fetchone()[0]
    sequence_stats = connection.execute(
        "SELECT MIN(degree), MAX(degree), AVG(degree) FROM "
        "(SELECT COUNT(*) AS degree FROM interactions GROUP BY user_raw)"
    ).fetchone()
    summary = {
        "final": {
            "interactions": interactions,
            "items": items,
            "split_counts": {
                "test": users,
                "train": interactions - 2 * users,
                "valid": users,
            },
            "users": users,
        },
        "ingest": asdict(ingest_report),
        "kcore": {
            "initial_edges": kcore_report.initial_edges,
            "iterations": [asdict(item) for item in kcore_report.iterations],
            "retained_edges": kcore_report.retained_edges,
        },
        "sequence_length": {
            "avg": sequence_stats[2] or 0.0,
            "max": sequence_stats[1] or 0,
            "min": sequence_stats[0] or 0,
        },
    }
    _write_json(target / "summary.json", summary)
    artifact_names = [
        "domain_stats.csv",
        "interactions.parquet",
        "items.csv.gz",
        "sequences.jsonl.gz",
        "summary.json",
        "user_overlap.csv",
        "users.csv.gz",
    ]
    hashes = {name: _file_hash(target / name) for name in artifact_names}
    inputs = {}
    for domain in AMAZON5_DOMAINS:
        path = settings.input_dir / domain.filename
        if path.exists():
            stat = path.stat()
            inputs[domain.name] = {
                "filename": domain.filename,
                "sha256": _file_hash(path),
                "size": stat.st_size,
            }
    manifest = {
        "config": {
            "batch_size": settings.batch_size,
            "min_item_interactions": settings.min_item_interactions,
            "min_user_interactions": settings.min_user_interactions,
        },
        "inputs": inputs,
        "schema_version": 1,
        "sha256": hashes,
    }
    _write_json(target / "manifest.json", manifest)
    _progress(
        "preprocess_export_complete",
        elapsed_seconds=round(time.perf_counter() - started, 1),
        interactions=interactions,
        items=items,
        users=users,
    )
    return ExportResult(target, hashes, users, items, interactions)


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_processed_dataset(output_dir: str | Path) -> ValidationReport:
    root = Path(output_dir)
    errors: list[str] = []
    required = {
        "domain_stats.csv",
        "interactions.parquet",
        "items.csv.gz",
        "manifest.json",
        "sequences.jsonl.gz",
        "summary.json",
        "user_overlap.csv",
        "users.csv.gz",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        return ValidationReport(False, tuple(f"missing artifact: {name}" for name in missing))
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return ValidationReport(False, (f"invalid JSON artifact: {error}",))
    for name, expected in manifest.get("sha256", {}).items():
        actual = _file_hash(root / name)
        if actual != expected:
            errors.append(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    try:
        users_map = _read_gzip_csv(root / "users.csv.gz")
        items_map = _read_gzip_csv(root / "items.csv.gz")
        expected_users = list(range(1, len(users_map) + 1))
        actual_users = [int(row["user_id"]) for row in users_map]
        if actual_users != expected_users:
            errors.append("user IDs are not contiguous from 1")
        expected_items = list(range(1, len(items_map) + 1))
        actual_items = [int(row["item_id"]) for row in items_map]
        if actual_items != expected_items:
            errors.append("item IDs are not contiguous from 1")
        item_domain = {
            int(row["item_id"]): int(row["domain_id"]) for row in items_map
        }
        item_raw = {int(row["item_id"]): row["parent_asin"] for row in items_map}
        user_counts: dict[int, int] = {}
        item_counts: dict[int, int] = {}
        groups: list[dict[str, object]] = []
        last_user = 0
        last_key: tuple[int, int, str] | None = None
        with gzip.open(
            root / "sequences.jsonl.gz", "rt", encoding="utf-8"
        ) as sequence_stream:
            sequences = iter(sequence_stream)
            for batch in pq.ParquetFile(root / "interactions.parquet").iter_batches():
                for row in batch.to_pylist():
                    user_id = row["user_id"]
                    if user_id != last_user:
                        if groups:
                            _validate_user_group(groups, next(sequences), errors)
                            groups = []
                        if user_id <= last_user:
                            errors.append("Parquet rows are not ordered by user_id")
                        last_user = user_id
                        last_key = None
                    key = (row["timestamp"], row["domain_id"], item_raw[row["item_id"]])
                    if last_key is not None and key < last_key:
                        errors.append(f"user {user_id} sequence is not chronologically sorted")
                    last_key = key
                    if item_domain.get(row["item_id"]) != row["domain_id"]:
                        errors.append(f"item/domain mismatch for item {row['item_id']}")
                    groups.append(row)
                    user_counts[user_id] = user_counts.get(user_id, 0) + 1
                    item_counts[row["item_id"]] = item_counts.get(row["item_id"], 0) + 1
            if groups:
                _validate_user_group(groups, next(sequences), errors)
            try:
                next(sequences)
                errors.append("sequences.jsonl.gz contains extra users")
            except StopIteration:
                pass
        user_min = int(manifest["config"]["min_user_interactions"])
        item_min = int(manifest["config"]["min_item_interactions"])
        if user_counts and min(user_counts.values()) < user_min:
            errors.append("a user is below the configured k-core threshold")
        if item_counts and min(item_counts.values()) < item_min:
            errors.append("an item is below the configured k-core threshold")
        interactions = sum(user_counts.values())
        if interactions != summary["final"]["interactions"]:
            errors.append("summary interaction count does not match Parquet")
        if len(user_counts) != summary["final"]["users"]:
            errors.append("summary user count does not match Parquet")
        if len(item_counts) != summary["final"]["items"]:
            errors.append("summary item count does not match Parquet")
        overlap_rows = (root / "user_overlap.csv").read_text(encoding="utf-8").splitlines()
        if len(overlap_rows) != 26:
            errors.append("user_overlap.csv must contain 25 domain pairs")
    except (KeyError, OSError, ValueError, StopIteration, pa.ArrowException) as error:
        errors.append(f"artifact validation failed: {error}")
        return ValidationReport(False, tuple(errors))
    return ValidationReport(
        not errors,
        tuple(errors),
        users=len(user_counts),
        items=len(item_counts),
        interactions=sum(user_counts.values()),
    )


def _validate_user_group(
    group: list[dict[str, object]], sequence_line: str, errors: list[str]
) -> None:
    user_id = group[0]["user_id"]
    positions = [row["position"] for row in group]
    if positions != list(range(len(group))):
        errors.append(f"user {user_id} positions are not contiguous")
    splits = [row["split"] for row in group]
    expected = ["train"] * (len(group) - 2) + ["valid", "test"]
    if splits != expected:
        errors.append(f"user {user_id} must have train* then exactly one valid and test")
    try:
        sequence = json.loads(sequence_line)
    except json.JSONDecodeError as error:
        errors.append(f"invalid sequence JSON for user {user_id}: {error}")
        return
    expected_sequence = {
        "domain_ids": [row["domain_id"] for row in group],
        "item_ids": [row["item_id"] for row in group],
        "splits": splits,
        "timestamps": [row["timestamp"] for row in group],
        "user_id": user_id,
    }
    if sequence != expected_sequence:
        errors.append(f"sequence JSON does not match Parquet for user {user_id}")


def preprocess_amazon5(settings: PreprocessSettings) -> PreprocessResult:
    staged_result: tuple[IngestReport, KCoreReport, ExportResult, ValidationReport] | None = None
    with RunDirectory(settings.output_dir, force=settings.force) as run:
        if run.path is None:
            raise RuntimeError("preprocessing staging directory was not created")
        database_path = settings.sqlite_path or (run.path / "preprocessing.sqlite")
        connection = open_database(database_path)
        try:
            ingest = ingest_amazon5(connection, settings)
            kcore = run_joint_k_core(
                connection,
                settings.min_user_interactions,
                settings.min_item_interactions,
            )
            export = export_processed_dataset(connection, run.path, ingest, kcore, settings)
        finally:
            connection.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{database_path}{suffix}").unlink(missing_ok=True)
        validation = validate_processed_dataset(run.path)
        if not validation.ok:
            raise DataInvariantError(
                "processed dataset validation failed: " + "; ".join(validation.errors)
            )
        manifest_hash = _file_hash(run.path / "manifest.json")
        run.complete({"data_hash": manifest_hash, "schema_version": 1})
        staged_result = (ingest, kcore, export, validation)
    if staged_result is None:
        raise RuntimeError("preprocessing did not produce a result")
    ingest, kcore, export, validation = staged_result
    published_export = ExportResult(
        Path(settings.output_dir).resolve(),
        export.artifact_hashes,
        export.users,
        export.items,
        export.interactions,
    )
    return PreprocessResult(
        Path(settings.output_dir).resolve(), ingest, kcore, published_export, validation
    )
