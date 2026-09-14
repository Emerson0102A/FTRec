from pathlib import Path

import pytest

from data_helpers import write_amazon5_fixture


def test_ingest_keeps_earliest_namespaced_interaction(tmp_path: Path) -> None:
    from ftrec.data.preprocessing import PreprocessSettings, ingest_amazon5, open_database

    input_dir = write_amazon5_fixture(
        tmp_path / "raw",
        {
            "Health": [
                ("u1", "item-a", "5.0", "3000"),
                ("u1", "item-a", "1.0", "1000"),
            ]
        },
    )
    settings = PreprocessSettings(input_dir=input_dir, output_dir=tmp_path / "out", batch_size=2)

    with open_database(tmp_path / "stage.sqlite") as connection:
        report = ingest_amazon5(connection, settings)
        row = connection.execute(
            "SELECT timestamp FROM interactions "
            "WHERE user_raw='u1' AND domain_id=0 AND item_raw='item-a'"
        ).fetchone()

    assert row == (1000,)
    assert report.valid_rows == 6
    assert report.retained_rows == 5
    assert report.duplicate_rows == 1


def test_ingest_counts_invalid_rows_and_keeps_domain_namespaces(tmp_path: Path) -> None:
    from ftrec.data.preprocessing import PreprocessSettings, ingest_amazon5, open_database

    input_dir = write_amazon5_fixture(
        tmp_path / "raw",
        {
            "Health": [
                ("", "missing-user", "5", "10"),
                ("u1", "same", "5", "bad"),
                ("u1", "same", "5", "20"),
            ],
            "Beauty": [("u1", "same", "5", "30")],
        },
    )

    with open_database(tmp_path / "stage.sqlite") as connection:
        report = ingest_amazon5(
            connection,
            PreprocessSettings(input_dir=input_dir, output_dir=tmp_path / "out"),
        )
        namespaced = connection.execute(
            "SELECT domain_id FROM interactions WHERE item_raw='same' ORDER BY domain_id"
        ).fetchall()

    assert report.invalid_missing_identity == 1
    assert report.invalid_timestamp == 1
    assert namespaced == [(0,), (2,)]


def test_missing_required_column_names_domain_and_column(tmp_path: Path) -> None:
    from ftrec.data.preprocessing import InputSchemaError, PreprocessSettings, ingest_amazon5, open_database

    input_dir = write_amazon5_fixture(
        tmp_path / "raw",
        header_by_domain={"Beauty": ["user_id", "parent_asin", "rating"]},
    )
    with open_database(tmp_path / "stage.sqlite") as connection:
        with pytest.raises(InputSchemaError, match=r"Beauty.*timestamp"):
            ingest_amazon5(
                connection,
                PreprocessSettings(input_dir=input_dir, output_dir=tmp_path / "out"),
            )


def test_joint_k_core_recomputes_both_sides_until_stable(tmp_path: Path) -> None:
    from ftrec.data.preprocessing import open_database, run_joint_k_core

    edges = [
        ("u1", 0, "a", 1),
        ("u1", 0, "b", 2),
        ("u2", 0, "a", 1),
        ("u2", 0, "b", 2),
        ("u3", 0, "b", 1),
        ("u3", 0, "c", 2),
    ]
    with open_database(tmp_path / "graph.sqlite") as connection:
        connection.executemany(
            "INSERT INTO interactions VALUES (?, ?, ?, ?, 0)", edges
        )
        report = run_joint_k_core(connection, user_min=2, item_min=2)
        retained = connection.execute(
            "SELECT user_raw, item_raw FROM interactions ORDER BY user_raw, item_raw"
        ).fetchall()

    assert report.iterations[-1].deleted_edges == 0
    assert len(report.iterations) == 3
    assert retained == [("u1", "a"), ("u1", "b"), ("u2", "a"), ("u2", "b")]


def test_user_degree_is_joint_across_domains(tmp_path: Path) -> None:
    from ftrec.data.preprocessing import open_database, run_joint_k_core

    with open_database(tmp_path / "graph.sqlite") as connection:
        connection.executemany(
            "INSERT INTO interactions VALUES (?, ?, ?, ?, 0)",
            [("u", 0, "a", 1), ("u", 1, "b", 2)],
        )
        run_joint_k_core(connection, user_min=2, item_min=1)
        count = connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]

    assert count == 2
