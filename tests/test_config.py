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
        assert config["pretrain_methods"] == ["joint"]
        assert config["batch_size"] == 256
        assert config["steps_per_epoch"] == "auto"
        assert config["epochs"] == 100
        assert config["lr"] == pytest.approx(0.0001)
        assert config["evaluation_protocol"] == "sampled"
        assert config["num_eval_negatives"] == 999

    for name in ("embedding", "lora_all_embedding"):
        config = load_config(root / "configs" / "experiment" / f"{name}.yaml")
        assert config["base_root"] == "runs-lr1e-4"
        assert config["output_root"] == "runs-lr1e-4"
        assert config["pretrain_methods"] == ["joint_proportional"]
        assert config["seeds"] == [42]
        assert config["lr"] == pytest.approx(0.0001)
        assert config["embedding_lr"] == pytest.approx(0.0001)
    assert load_config(
        root / "configs" / "experiment" / "lora_all_embedding.yaml"
    )["ranks"] == [5]

    fullft_lr1e4 = load_config(
        root / "configs" / "experiment" / "fullft_lr1e4.yaml"
    )
    assert fullft_lr1e4["method"] == "fullft"
    assert fullft_lr1e4["base_root"] == "runs-lr1e-4"
    assert fullft_lr1e4["output_root"] == "runs-lr1e-4"
    assert fullft_lr1e4["pretrain_methods"] == ["joint_proportional"]
    assert fullft_lr1e4["seeds"] == [42]
    assert fullft_lr1e4["lr"] == pytest.approx(0.0001)
    assert fullft_lr1e4["embedding_lr"] == pytest.approx(0.0001)

    for name, method in (
        ("lora_all_multineg31", "lora_all"),
        ("fullft_multineg31", "fullft"),
        ("embedding_multineg31", "embedding"),
        ("lora_all_embedding_multineg31", "lora_all_embedding"),
    ):
        config = load_config(root / "configs" / "experiment" / f"{name}.yaml")
        assert config["method"] == method
        assert config["base_root"] == "runs-lr1e-4"
        assert config["output_root"] == "runs-multineg31"
        assert config["pretrain_methods"] == ["joint_proportional"]
        assert config["seeds"] == [42]
        assert config["num_train_negatives"] == 31
        assert config["lr"] == pytest.approx(0.0001)
        if method in {"embedding", "lora_all_embedding", "fullft"}:
            assert config["embedding_lr"] == pytest.approx(0.0001)

    assert load_config(
        root
        / "configs"
        / "experiment"
        / "lora_all_embedding_multineg31.yaml"
    )["ranks"] == [5]
    for name, context_mode, output_root in (
        ("context_mixed_min5", "mixed", "runs-context-ablation/mixed-min5"),
        (
            "context_target_only_min5",
            "target_only",
            "runs-context-ablation/target-only-min5",
        ),
    ):
        config = load_config(root / "configs" / "experiment" / f"{name}.yaml")
        assert config["method"] == "lora_all_embedding"
        assert config["base_root"] == "runs-lr1e-4"
        assert config["output_root"] == output_root
        assert config["pretrain_methods"] == ["joint_proportional"]
        assert config["ranks"] == [5]
        assert config["seeds"] == [42]
        assert config["num_train_negatives"] == 31
        assert config["lr"] == pytest.approx(0.0001)
        assert config["embedding_lr"] == pytest.approx(0.0001)
        assert config["context_mode"] == context_mode
        assert config["min_domain_sequence_length"] == 5

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


def test_attribute_pretraining_configs_allow_late_convergence() -> None:
    from ftrec.config import load_config

    root = Path(__file__).parents[1] / "configs" / "experiment"
    for name in (
        "attribute_llm2attr_pretrain",
        "attribute_llm2attr_fused_pretrain",
        "attribute_structured_title_pretrain",
        "attribute_structured_title_fused_pretrain",
    ):
        config = load_config(root / f"{name}.yaml")
        assert config["epochs"] == 300
        assert config["patience"] == 20


def test_llm2attr_pooling_ablation_configs_are_explicit() -> None:
    from ftrec.config import load_config

    root = Path(__file__).parents[1] / "configs" / "model"
    expected = {
        "sasrec_llm2attr_fused_mean": "mean_all",
        "sasrec_llm2attr_fused_soft_attention": "soft_attention",
        "sasrec_llm2attr_fused_domain_title_attention": (
            "domain_title_attention"
        ),
    }
    for name, pooling in expected.items():
        config = load_config(root / f"{name}.yaml")
        assert config["item_embedding_mode"] == "content_fused"
        assert config["attribute_pooling"] == pooling
    domain_config = load_config(
        root / "sasrec_llm2attr_fused_domain_title_attention.yaml"
    )
    assert domain_config["item_domain_file"].endswith("items.csv.gz")


def test_mymodel4_context_ablation_is_matched_and_monitors_test() -> None:
    from ftrec.config import load_config

    root = Path(__file__).parents[1]
    experiment = load_config(root / "configs" / "experiment" / "mymodel4_context_ablation.yaml")
    model = load_config(root / "configs" / "model" / "sasrec_llm2attr_mymodel4_behavior.yaml")
    script = (root / "scripts" / "run_mymodel4_context_ablation.sh").read_text(encoding="utf-8")

    assert experiment["patience"] == 5
    assert experiment["evaluate_test_each_epoch"] is True
    assert experiment["num_eval_negatives"] == 999
    assert model["item_embedding_mode"] == "mymodel4_behavior"
    assert "attribute_pooling" not in model
    assert model["shared_behavior_blocks"] == 1
    assert model["num_blocks"] == 2
    assert "single-domain|joint_domain" in script
    assert "mixed-domain|joint_mixed_matched" in script
