from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


CLI_MODULES = (
    "ftrec.cli.preprocess",
    "ftrec.cli.import_gmflowrec",
    "ftrec.cli.pretrain",
    "ftrec.cli.adapt",
    "ftrec.cli.analyze",
    "ftrec.cli.pilot",
    "ftrec.cli.smoke",
)


@pytest.mark.parametrize(
    "method", ("single_mixed", "joint_domain", "joint_mixed_matched", "joint_proportional")
)
def test_pretrain_cli_accepts_context_ablation_methods(method: str) -> None:
    """Catch the new experiment modes being implemented but unreachable on servers."""
    from ftrec.cli.pretrain import build_parser

    args = build_parser().parse_args(["--method", method])

    assert args.method == method


def test_pretrain_cli_accepts_in_place_resume() -> None:
    from ftrec.cli.pretrain import build_parser

    args = build_parser().parse_args(["--resume"])

    assert args.resume is True


@pytest.mark.parametrize("module", CLI_MODULES)
def test_all_commands_expose_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_preprocess_and_smoke_dry_run_do_not_create_outputs(tmp_path: Path) -> None:
    preprocess_output = tmp_path / "processed"
    preprocess = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftrec.cli.preprocess",
            "--input-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(preprocess_output),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert preprocess.returncode == 0, preprocess.stderr
    assert '"decision": "create"' in preprocess.stdout
    assert not preprocess_output.exists()

    smoke_output = tmp_path / "smoke"
    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftrec.cli.smoke",
            "--output-dir",
            str(smoke_output),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert '"model_runs": 67' in smoke.stdout
    assert not smoke_output.exists()


def test_server_scripts_are_fail_fast_and_stage_scoped() -> None:
    root = Path(__file__).parents[1]
    names = (
        "run_preprocess.sh",
        "run_import_gmflowrec.sh",
        "run_single.sh",
        "run_single_mixed.sh",
        "run_joint_domain.sh",
        "run_joint_mixed_matched.sh",
        "run_joint_proportional.sh",
        "run_joint.sh",
        "run_pcgrad.sh",
        "run_lora.sh",
        "run_lora_all.sh",
        "run_houlsby.sh",
        "run_pfeiffer.sh",
        "run_fullft.sh",
        "run_analysis.sh",
        "run_pilot.sh",
        "check_environment.sh",
    )
    for name in names:
        text = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert 'BASH_SOURCE[0]' in text


def test_production_training_configs_enable_bf16() -> None:
    root = Path(__file__).parents[1]
    for name in (
        "single.yaml",
        "single_mixed.yaml",
        "joint_domain.yaml",
        "joint_mixed_matched.yaml",
        "joint_proportional.yaml",
        "joint.yaml",
        "pcgrad.yaml",
        "lora.yaml",
        "lora_all.yaml",
        "houlsby.yaml",
        "pfeiffer.yaml",
        "fullft.yaml",
    ):
        config = yaml.safe_load(
            (root / "configs" / "experiment" / name).read_text(encoding="utf-8")
        )
        assert config["bf16"] is True


def test_context_ablation_configs_match_formal_training_protocol() -> None:
    root = Path(__file__).parents[1] / "configs" / "experiment"
    single_mixed = yaml.safe_load(
        (root / "single_mixed.yaml").read_text(encoding="utf-8")
    )
    joint_domain = yaml.safe_load(
        (root / "joint_domain.yaml").read_text(encoding="utf-8")
    )
    joint_mixed_matched = yaml.safe_load(
        (root / "joint_mixed_matched.yaml").read_text(encoding="utf-8")
    )

    assert single_mixed["method"] == "single_mixed"
    assert single_mixed["domain"] == 0
    assert joint_domain["method"] == "joint_domain"
    assert joint_domain["domain"] is None
    assert joint_mixed_matched["method"] == "joint_mixed_matched"
    assert joint_mixed_matched["domain"] is None
    for config in (single_mixed, joint_domain, joint_mixed_matched):
        assert config["processed_dir"] == "data/processed/gmflowrec-amazon"
        assert config["device"] == "cuda"
        assert config["bf16"] is True
        assert config["epochs"] == 100
        assert config["patience"] == 10
        assert config["steps_per_epoch"] == "auto"
        assert config["lr"] == pytest.approx(0.001)
        assert config["embedding_lr"] == pytest.approx(0.001)
        assert config["evaluation_protocol"] == "sampled"
        assert config["evaluation_chunk_size"] == 4096
        assert config["evaluation_batch_size"] == 256
        assert config["num_eval_negatives"] == 999
        assert config["gradient_conflict"]["enabled"] is False


def test_attribute_sequence_context_ablation_keeps_one_matched_protocol() -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (root / "configs/experiment/attribute_context_ablation.yaml").read_text(
            encoding="utf-8"
        )
    )
    script = (root / "scripts/run_sequence_context_ablation.sh").read_text(
        encoding="utf-8"
    )

    assert config["method"] == "joint_domain"
    assert config["lr"] == pytest.approx(0.0001)
    assert config["epochs"] == 300
    assert config["patience"] == 20
    assert config["evaluation_protocol"] == "sampled"
    assert config["num_eval_negatives"] == 999
    assert "joint_domain joint_mixed_matched" in script
    assert "configs/model/sasrec.yaml" in script
    assert "configs/model/sasrec_llm2attr.yaml" in script
    assert "configs/model/sasrec_structured_title_fused.yaml" in script
    assert "--tune-adaptive-fusion" in script


def test_joint_proportional_config_is_the_lr1e4_control() -> None:
    root = Path(__file__).parents[1] / "configs" / "experiment"
    config = yaml.safe_load(
        (root / "joint_proportional.yaml").read_text(encoding="utf-8")
    )

    assert config["method"] == "joint_proportional"
    assert config["output_root"] == "runs-lr1e-4"
    assert config["batch_size"] == 256
    assert config["steps_per_epoch"] == "auto"
    assert config["lr"] == pytest.approx(0.0001)
    assert config["embedding_lr"] == pytest.approx(0.0001)
    assert config["gradient_conflict"]["enabled"] is False


def test_adapt_cli_accepts_patience_above_fixed_budget() -> None:
    from ftrec.cli.adapt import build_parser

    args = build_parser().parse_args(["--epochs", "12", "--patience", "13"])
    assert args.epochs == 12
    assert args.patience == 13


def test_context_ablation_scripts_encode_the_expected_run_matrix() -> None:
    root = Path(__file__).parents[1] / "scripts"
    single_mixed = (root / "run_single_mixed.sh").read_text(encoding="utf-8")
    joint_domain = (root / "run_joint_domain.sh").read_text(encoding="utf-8")
    joint_mixed = (root / "run_joint_mixed_matched.sh").read_text(
        encoding="utf-8"
    )

    assert "for domain in 0 1 2 3 4" in single_mixed
    assert '--domain "$domain"' in single_mixed
    assert single_mixed.count("ftrec-pretrain") == 1
    for script in (joint_domain, joint_mixed):
        assert "for domain in" not in script
        assert script.count("ftrec-pretrain") == 1
    for script in (single_mixed, joint_domain, joint_mixed):
        assert "--seed" in script
        assert 'extra_args+=("$1")' in script
        assert '"${extra_args[@]}"' in script


def test_adaptation_dry_run_reports_matrix_progress(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    with gzip.open(processed / "sequences.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(
            '{"domain_ids":[0,0,0],"item_ids":[1,2,3],"splits":["train","valid","test"],"timestamps":[1,2,3],"user_id":1}\n'
        )
    with gzip.open(processed / "items.csv.gz", "wt", encoding="utf-8") as stream:
        stream.write("item_id,domain_id,domain,parent_asin\n1,0,Health,item-1\n")
    (processed / "manifest.json").write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "fullft.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "processed_dir": str(processed),
                "base_root": str(tmp_path / "runs"),
                "output_root": str(tmp_path / "runs"),
                "method": "fullft",
                "pretrain_methods": ["joint"],
                "domains": [0],
                "seeds": [42],
                "device": "cpu",
                "bf16": False,
                "progress": True,
                "batch_size": 1,
                "steps_per_epoch": 1,
                "epochs": 1,
                "patience": 1,
                "lr": 0.001,
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "hidden_size": 4,
                "num_blocks": 1,
                "num_heads": 1,
                "dropout": 0.0,
                "maxlen": 3,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftrec.cli.adapt",
            "--config",
            str(config_path),
            "--model-config",
            str(model_path),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "fullft matrix" in result.stderr
    assert "100%" in result.stderr


def test_adaptation_rank_subset_limits_the_matrix(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    with gzip.open(processed / "sequences.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(
            '{"domain_ids":[0,0,0],"item_ids":[1,2,3],"splits":["train","valid","test"],"timestamps":[1,2,3],"user_id":1}\n'
        )
    with gzip.open(processed / "items.csv.gz", "wt", encoding="utf-8") as stream:
        stream.write("item_id,domain_id,domain,parent_asin\n1,0,Health,item-1\n")
    (processed / "manifest.json").write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "lora.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "processed_dir": str(processed),
                "base_root": str(tmp_path / "runs"),
                "output_root": str(tmp_path / "runs"),
                "method": "lora",
                "pretrain_methods": ["joint"],
                "domains": [0],
                "ranks": [1, 2, 4, 8, 16],
                "seeds": [42],
                "device": "cpu",
                "bf16": False,
                "progress": False,
                "batch_size": 1,
                "steps_per_epoch": 1,
                "epochs": 1,
                "patience": 1,
                "lr": 0.001,
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "hidden_size": 4,
                "num_blocks": 1,
                "num_heads": 1,
                "dropout": 0.0,
                "maxlen": 3,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftrec.cli.adapt",
            "--config",
            str(config_path),
            "--model-config",
            str(model_path),
            "--ranks",
            "1",
            "2",
            "4",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["combinations"] == 3
    assert [run["rank"] for run in payload["runs"]] == [1, 2, 4]


@pytest.mark.parametrize(
    "method",
    (
        "lora_all",
        "lora_all_content_adapter",
        "lora_all_embedding",
        "content_adapter",
        "embedding",
        "houlsby",
        "pfeiffer",
    ),
)
def test_adapt_cli_exposes_parameter_efficient_methods(method: str) -> None:
    from ftrec.cli.adapt import build_parser

    args = build_parser().parse_args(["--method", method])

    assert args.method == method


def test_adapt_cli_accepts_lr1e4_joint_proportional_backbone() -> None:
    from ftrec.cli.adapt import build_parser

    args = build_parser().parse_args(["--pretrain-method", "joint_proportional"])

    assert args.pretrain_method == "joint_proportional"


@pytest.mark.parametrize(
    ("extra_args", "expected_runs", "expected_stages"),
    [
        ([], 32, ["joint", "pcgrad", "lora"]),
        (["--with-fullft"], 42, ["joint", "pcgrad", "lora", "fullft"]),
    ],
)
def test_pilot_dry_run_plans_the_approved_single_seed_matrix(
    extra_args: list[str], expected_runs: int, expected_stages: list[str]
) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ftrec.cli.pilot", "--dry-run", *extra_args],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["seed"] == 42
    assert payload["ranks"] == [1, 2, 4]
    assert payload["model_runs"] == expected_runs
    assert [stage["stage"] for stage in payload["stages"]] == expected_stages
    assert payload["epochs"] == 3
    assert payload["steps_per_epoch"] == 100
    assert all("--epochs" in stage["argv"] for stage in payload["stages"])
    assert all("--steps-per-epoch" in stage["argv"] for stage in payload["stages"])


def test_pilot_expands_independent_runs_for_controlled_parallelism() -> None:
    """Catch a nominal worker option that still launches one serial matrix process."""
    from ftrec.cli.pilot import build_pilot_stages, expand_pilot_jobs

    stages = build_pilot_stages(
        seed=42,
        ranks=(1, 4),
        with_fullft=False,
        epochs=3,
        steps_per_epoch=100,
    )
    phases = expand_pilot_jobs(stages, ranks=(1, 4))

    assert [phase.name for phase in phases] == ["pretrain", "lora"]
    assert [len(phase.jobs) for phase in phases] == [2, 20]
    assert {job.stage for job in phases[0].jobs} == {"joint", "pcgrad"}
    assert all("--pretrain-method" in job.argv for job in phases[1].jobs)
    assert all("--domain" in job.argv for job in phases[1].jobs)
    assert all("--rank" in job.argv for job in phases[1].jobs)
    assert all("--ranks" not in job.argv for job in phases[1].jobs)


def test_pilot_dry_run_reports_parallel_worker_limits() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftrec.cli.pilot",
            "--dry-run",
            "--pretrain-workers",
            "2",
            "--adapt-workers",
            "3",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["pretrain_workers"] == 2
    assert payload["adapt_workers"] == 3
    assert [phase["jobs"] for phase in payload["phases"]] == [2, 30]


def test_parallel_pilot_interrupt_terminates_inflight_children(monkeypatch) -> None:
    """Catch Ctrl+C leaving GPU jobs running or allowing the next phase to start."""
    import threading

    from ftrec.cli.pilot import PilotJob, PilotPhase, _run_phase

    peer_started = threading.Event()
    peer_terminated = threading.Event()
    processes = []

    class FakeProcess:
        def __init__(self, _argv) -> None:
            self.index = len(processes)
            self.terminated = False
            processes.append(self)

        def wait(self, timeout=None) -> int:
            if timeout is not None:
                return -15
            if self.index == 0:
                assert peer_started.wait(1)
                raise KeyboardInterrupt
            peer_started.set()
            assert peer_terminated.wait(1)
            return -15

        def terminate(self) -> None:
            self.terminated = True
            peer_terminated.set()

        def kill(self) -> None:
            self.terminated = True
            peer_terminated.set()

    monkeypatch.setattr("ftrec.cli.pilot.subprocess.Popen", FakeProcess)
    phase = PilotPhase(
        "pretrain",
        (PilotJob("joint", ("joint",)), PilotJob("pcgrad", ("pcgrad",))),
    )

    with pytest.raises(KeyboardInterrupt):
        _run_phase(phase, workers=2)

    assert len(processes) == 2
    assert processes[1].terminated


def test_analysis_prefers_best_checkpoint_gradient_records(tmp_path: Path) -> None:
    """Catch conflict figures silently using post-best training trajectory records."""
    import json

    from ftrec.cli.analyze import _read_gradient_records

    run = tmp_path / "pretrain" / "joint" / "all-domains" / "seed-42"
    run.mkdir(parents=True)
    (run / "gradient_conflicts.jsonl").write_text(
        json.dumps({"profile_scope": "training_trajectory", "step": 47}) + "\n",
        encoding="utf-8",
    )
    (run / "gradient_conflicts_best_checkpoint.jsonl").write_text(
        json.dumps({"profile_scope": "best_checkpoint", "step": 0}) + "\n",
        encoding="utf-8",
    )

    records = _read_gradient_records(tmp_path)

    assert records == ({"profile_scope": "best_checkpoint", "step": 0},)
