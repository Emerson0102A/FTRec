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


def test_production_configs_follow_gmflowrec_training_protocol() -> None:
    """Catch a run silently reverting to the old data, batch, or learning rate."""
    from ftrec.config import load_config

    root = Path(__file__).parents[1]
    for name in (
        "single",
        "joint",
        "pcgrad",
        "lora",
        "fullft",
    ):
        config = load_config(root / "configs" / "experiment" / f"{name}.yaml")
        assert config["processed_dir"] == "data/processed/gmflowrec-amazon"
        assert config["batch_size"] == 256
        assert config["steps_per_epoch"] == "auto"
        assert config["epochs"] == 100
        assert config["lr"] == 0.001
        assert config["evaluation_protocol"] == "sampled"
        assert config["num_eval_negatives"] == 999

    for name in ("lora_all", "houlsby", "pfeiffer"):
        config = load_config(root / "configs" / "experiment" / f"{name}.yaml")
        assert config["processed_dir"] == "data/processed/gmflowrec-amazon"
        assert config["base_root"] == "runs-lr1e-4"
        assert config["output_root"] == "runs-lr1e-4"
        assert config["pretrain_methods"] == ["joint_proportional"]
        assert config["batch_size"] == 256
        assert config["steps_per_epoch"] == "auto"
        assert config["epochs"] == 100
        assert config["lr"] == pytest.approx(0.0001)
        assert config["evaluation_protocol"] == "sampled"
        assert config["num_eval_negatives"] == 999

    assert load_config(root / "configs" / "experiment" / "fullft.yaml")[
        "embedding_lr"
    ] == 0.001
    assert load_config(root / "configs" / "experiment" / "lora_all.yaml")[
        "ranks"
    ] == [3, 5]
    assert load_config(root / "configs" / "experiment" / "houlsby.yaml")[
        "bottleneck_sizes"
    ] == [8, 16]
    assert load_config(root / "configs" / "experiment" / "pfeiffer.yaml")[
        "bottleneck_sizes"
    ] == [16, 32]
    for name in ("joint", "pcgrad"):
        conflict = load_config(root / "configs" / "experiment" / f"{name}.yaml")[
            "gradient_conflict"
        ]
        assert conflict == {
            "enabled": True,
            "ema_beta": 0.9,
            "log_interval": 100,
            "checkpoint_steps": 100,
            "checkpoint_seed": 2026,
        }
    assert load_config(root / "configs" / "experiment" / "pcgrad.yaml")[
        "pcgrad_projection_scope"
    ] == "backbone"
