import json
from pathlib import Path

import pytest


def test_run_directory_publishes_only_after_complete(tmp_path: Path) -> None:
    from ftrec.artifacts import RunDirectory

    final = tmp_path / "run"
    with RunDirectory(final) as run:
        run.write_json("metrics.json", {"value": 1})
        assert not final.exists()
        run.complete({"config_hash": "abc"})

    assert json.loads((final / "COMPLETE.json").read_text(encoding="utf-8")) == {
        "config_hash": "abc"
    }


def test_run_directory_does_not_publish_failed_stage(tmp_path: Path) -> None:
    from ftrec.artifacts import RunDirectory

    final = tmp_path / "run"
    with pytest.raises(RuntimeError, match="boom"):
        with RunDirectory(final) as run:
            run.write_json("partial.json", {"value": 1})
            raise RuntimeError("boom")

    assert not final.exists()


def test_deterministic_json_has_stable_bytes(tmp_path: Path) -> None:
    from ftrec.artifacts import write_json

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json(first, {"b": 2, "a": 1})
    write_json(second, {"a": 1, "b": 2})

    assert first.read_bytes() == second.read_bytes() == b'{"a":1,"b":2}\n'
