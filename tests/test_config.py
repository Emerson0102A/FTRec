from pathlib import Path

import pytest
import yaml


def test_load_config_applies_typed_nested_overrides(tmp_path: Path) -> None:
    from ftrec.config import load_config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"model": {"hidden_size": 64, "dropout": 0.2}, "seed": 1}),
        encoding="utf-8",
    )

    config = load_config(path, ["model.dropout=0.0", "seed=42"])

    assert config == {"model": {"hidden_size": 64, "dropout": 0.0}, "seed": 42}


def test_load_config_rejects_unknown_override(tmp_path: Path) -> None:
    from ftrec.config import ConfigError, load_config

    path = tmp_path / "config.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown override"):
        load_config(path, ["missing.value=2"])


def test_canonical_hash_is_order_independent() -> None:
    from ftrec.config import canonical_hash

    assert canonical_hash({"b": 2, "a": {"x": 1}}) == canonical_hash(
        {"a": {"x": 1}, "b": 2}
    )

