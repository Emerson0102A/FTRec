"""Memory-bounded Amazon ingestion and exact joint k-core filtering."""

from __future__ import annotations

import csv
import gzip
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .amazon import AMAZON5_DOMAINS, DomainSpec


REQUIRED_COLUMNS = frozenset({"user_id", "parent_asin", "rating", "timestamp"})


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


def open_database(path: str | Path) -> sqlite3.Connection:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
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
    for domain in domains:
        path = _input_path(settings, domain)
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
        if batch:
            connection.executemany(upsert, batch)
            connection.commit()
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

