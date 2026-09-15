from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


def _reports():
    from ftrec.data.preprocessing import IngestReport, KCoreIteration, KCoreReport

    ingest = IngestReport(8, 8, 8, 0, 0, 0, 0, {})
    core = KCoreReport((KCoreIteration(1, 0, 0, 0),), 8, 8)
    return ingest, core


def _filtered_database(path: Path):
    from ftrec.data.preprocessing import open_database

    connection = open_database(path)
    connection.executemany(
        "INSERT INTO interactions VALUES (?, ?, ?, ?, ?)",
        [
            ("u-b", 2, "z", 100, 1),
            ("u-b", 0, "b", 100, 2),
            ("u-b", 0, "a", 100, 3),
            ("u-b", 4, "s", 200, 4),
            ("u-a", 0, "a", 50, 5),
            ("u-a", 1, "c", 60, 6),
            ("u-a", 2, "d", 70, 7),
            ("u-a", 3, "g", 80, 8),
        ],
    )
    connection.commit()
    return connection


def test_export_maps_sorts_and_splits_deterministically(tmp_path: Path) -> None:
    from ftrec.data.preprocessing import PreprocessSettings, export_processed_dataset

    connection = _filtered_database(tmp_path / "stage.sqlite")
    ingest, core = _reports()
    settings = PreprocessSettings(tmp_path, tmp_path / "unused", 1, 1)

    first = export_processed_dataset(connection, tmp_path / "first", ingest, core, settings)
    second = export_processed_dataset(connection, tmp_path / "second", ingest, core, settings)
    connection.close()

    assert first.artifact_hashes == second.artifact_hashes
    rows = pq.read_table(tmp_path / "first" / "interactions.parquet").to_pylist()
    user_one = [row for row in rows if row["user_id"] == 1]
    user_two = [row for row in rows if row["user_id"] == 2]
    assert [row["position"] for row in user_one] == [0, 1, 2, 3]
    assert [row["split"] for row in user_one] == ["train", "train", "valid", "test"]
    assert [row["domain_id"] for row in user_two[:3]] == [0, 0, 2]
    assert [row["item_id"] for row in user_two[:2]] == sorted(
        row["item_id"] for row in user_two[:2]
    )


def test_export_batches_many_users_into_bounded_parquet_row_groups(tmp_path: Path) -> None:
    from ftrec.data.preprocessing import PreprocessSettings, export_processed_dataset

    connection = _filtered_database(tmp_path / "stage.sqlite")
    ingest, core = _reports()
    output = tmp_path / "processed"
    export_processed_dataset(
        connection,
        output,
        ingest,
        core,
        PreprocessSettings(tmp_path, output, 1, 1, batch_size=100),
    )
    connection.close()

    assert pq.ParquetFile(output / "interactions.parquet").num_row_groups == 1


def test_validator_accepts_export_and_detects_changed_hash(tmp_path: Path) -> None:
    from ftrec.data.preprocessing import (
        PreprocessSettings,
        export_processed_dataset,
        validate_processed_dataset,
    )

    connection = _filtered_database(tmp_path / "stage.sqlite")
    ingest, core = _reports()
    output = tmp_path / "processed"
    export_processed_dataset(
        connection,
        output,
        ingest,
        core,
        PreprocessSettings(tmp_path, output, 1, 1),
    )
    connection.close()

    assert validate_processed_dataset(output).ok
    with (output / "domain_stats.csv").open("ab") as stream:
        stream.write(b"corrupt")
    report = validate_processed_dataset(output)

    assert not report.ok
    assert any("SHA-256" in error and "domain_stats.csv" in error for error in report.errors)


def test_domain_statistics_and_overlap_cover_all_domains(tmp_path: Path) -> None:
    from ftrec.data.preprocessing import PreprocessSettings, export_processed_dataset

    connection = _filtered_database(tmp_path / "stage.sqlite")
    ingest, core = _reports()
    output = tmp_path / "processed"
    export_processed_dataset(
        connection,
        output,
        ingest,
        core,
        PreprocessSettings(tmp_path, output, 1, 1),
    )
    connection.close()

    stats = (output / "domain_stats.csv").read_text(encoding="utf-8").splitlines()
    overlap = (output / "user_overlap.csv").read_text(encoding="utf-8").splitlines()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert len(stats) == 6
    assert len(overlap) == 26
    assert summary["final"]["users"] == 2
    assert summary["final"]["interactions"] == 8


def test_preprocess_amazon5_publishes_a_valid_dataset_atomically(tmp_path: Path) -> None:
    from data_helpers import write_amazon5_fixture
    from ftrec.data.preprocessing import (
        PreprocessSettings,
        preprocess_amazon5,
        validate_processed_dataset,
    )

    rows = {
        domain: [(f"u{user}", f"{domain}-item", "5", str(1000 + index)) for user in range(3)]
        for index, domain in enumerate(["Health", "Clothing", "Beauty", "Grocery", "Sports"])
    }
    input_dir = write_amazon5_fixture(tmp_path / "raw", rows)
    output = tmp_path / "processed"

    result = preprocess_amazon5(
        PreprocessSettings(
            input_dir=input_dir,
            output_dir=output,
            min_user_interactions=5,
            min_item_interactions=3,
            batch_size=2,
        )
    )

    assert result.validation.ok
    assert validate_processed_dataset(output).ok
    assert (output / "COMPLETE.json").is_file()
    assert not list(tmp_path.glob("*.sqlite*"))
    assert not list(tmp_path.glob(".processed.staging-*"))


def test_preprocess_reports_ingest_kcore_and_export_progress(
    tmp_path: Path, capsys
) -> None:
    from data_helpers import write_amazon5_fixture
    from ftrec.data.preprocessing import PreprocessSettings, preprocess_amazon5

    rows = {
        domain: [(f"u{user}", f"{domain}-item", "5", str(1000 + index)) for user in range(3)]
        for index, domain in enumerate(["Health", "Clothing", "Beauty", "Grocery", "Sports"])
    }
    input_dir = write_amazon5_fixture(tmp_path / "raw", rows)

    preprocess_amazon5(
        PreprocessSettings(
            input_dir=input_dir,
            output_dir=tmp_path / "processed",
            min_user_interactions=5,
            min_item_interactions=3,
            batch_size=2,
        )
    )

    stderr = capsys.readouterr().err
    assert "ingest Health" in stderr
    assert "joint k-core" in stderr
    assert "export interactions" in stderr
    assert "100%" in stderr
