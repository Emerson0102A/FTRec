"""Complete, tiny CPU execution of every scientific experiment branch."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from ftrec.analysis.recovery import compute_recovery_rows
from ftrec.analysis.results import collect_result_rows
from ftrec.artifacts import sha256_file
from ftrec.config import canonical_hash
from ftrec.data.amazon import AMAZON5_DOMAINS
from ftrec.data.datasets import SequenceStore
from ftrec.data.preprocessing import PreprocessSettings, preprocess_amazon5
from ftrec.models.sasrec import SASRecConfig
from ftrec.training.adapt import AdaptSettings, train_adaptation
from ftrec.training.pretrain import PretrainSettings, train_pretraining


LORA_RANKS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class SmokeReport:
    ok: bool
    output_dir: Path
    completed: dict[str, int]
    model_runs: int
    lora_ranks: tuple[int, ...]
    figure_count: int
    evaluation_protocols: frozenset[str]
    recovery_rows: int
    semantic_hash: str


def _write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(
            text,
            fieldnames=("user_id", "parent_asin", "rating", "timestamp"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    finally:
        text.close()
        compressed.close()
        raw.close()


def generate_synthetic_amazon5(output_dir: str | Path) -> dict[str, int]:
    """Create deterministic cross-domain reviews plus a cascading k-core tail."""
    root = Path(output_dir)
    by_domain: dict[int, list[dict[str, object]]] = {
        domain.domain_id: [] for domain in AMAZON5_DOMAINS
    }
    for user_index in range(10):
        user = f"user-{user_index:02d}"
        assigned_domain = user_index % 5
        for domain in AMAZON5_DOMAINS:
            by_domain[domain.domain_id].append(
                {
                    "user_id": user,
                    "parent_asin": f"d{domain.domain_id}-shared-{user_index % 2}",
                    "rating": 5,
                    "timestamp": domain.domain_id // 2 + 1,
                }
            )
        for item_index, timestamp in ((2, 10), (3, 11), (4, 12)):
            by_domain[assigned_domain].append(
                {
                    "user_id": user,
                    "parent_asin": f"d{assigned_domain}-item-{item_index}",
                    "rating": 5,
                    "timestamp": timestamp,
                }
            )
    by_domain[0].append(
        {
            "user_id": "user-00",
            "parent_asin": "d0-shared-0",
            "rating": 1,
            "timestamp": 99,
        }
    )
    for noise_user, suffixes in (("tail-a", ("a", "b")), ("tail-b", ("c", "d"))):
        by_domain[0].append(
            {
                "user_id": noise_user,
                "parent_asin": "tail-shared",
                "rating": 1,
                "timestamp": 1,
            }
        )
        for timestamp, suffix in enumerate(suffixes, start=2):
            by_domain[0].append(
                {
                    "user_id": noise_user,
                    "parent_asin": f"tail-{suffix}",
                    "rating": 1,
                    "timestamp": timestamp,
                }
            )
    for domain in AMAZON5_DOMAINS:
        _write_gzip_csv(root / domain.filename, by_domain[domain.domain_id])
    return {
        "raw_rows": sum(len(rows) for rows in by_domain.values()),
        "main_users": 10,
        "retained_interactions": 80,
        "retained_items": 25,
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_finite_run_metrics(runs_root: Path) -> None:
    for path in runs_root.rglob("metrics.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            for key in ("loss", "gradient_norm"):
                if key in value and not math.isfinite(float(value[key])):
                    raise AssertionError(f"non-finite {key} in {path}")


def _semantic_hash(
    processed_dir: Path,
    runs_root: Path,
    initialization_hashes: dict[str, str],
) -> str:
    rows = collect_result_rows(runs_root)
    comparable_rows = [
        {
            "seed": row.seed,
            "domain": row.domain,
            "pretrain_method": row.pretrain_method,
            "adapt_method": row.adapt_method,
            "lora_rank": row.lora_rank,
            "protocol": row.evaluation_protocol,
            "hr": row.hr_at_10,
            "ndcg": row.ndcg_at_10,
            "eval": row.num_eval_users,
            "skipped": row.num_skipped_users,
            "trainable": row.num_trainable_params,
            "total": row.num_total_params,
        }
        for row in rows
    ]
    batches = {
        path.relative_to(runs_root).as_posix(): sha256_file(path)
        for path in sorted(runs_root.rglob("batch_manifest.json"))
    }
    payload = {
        "batches": batches,
        "data_manifest": sha256_file(processed_dir / "manifest.json"),
        "initializations": initialization_hashes,
        "results": comparable_rows,
    }
    return canonical_hash(payload)


def run_smoke(
    output_dir: str | Path, *, seed: int = 42, force: bool = False
) -> SmokeReport:
    root = Path(output_dir).resolve()
    if root.exists():
        if not force:
            raise FileExistsError(f"smoke output already exists: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    runs_root = root / "runs"
    analysis_dir = root / "analysis"
    fixture = generate_synthetic_amazon5(raw_dir)
    if fixture != {
        "raw_rows": 87,
        "main_users": 10,
        "retained_interactions": 80,
        "retained_items": 25,
    }:
        raise AssertionError(f"synthetic fixture contract changed: {fixture}")
    preprocess = preprocess_amazon5(
        PreprocessSettings(
            input_dir=raw_dir,
            output_dir=processed_dir,
            min_user_interactions=3,
            min_item_interactions=2,
            batch_size=32,
        )
    )
    if (
        preprocess.export.users != 10
        or preprocess.export.items != 25
        or preprocess.export.interactions != 80
        or preprocess.kcore.rounds < 3
    ):
        raise AssertionError("synthetic preprocessing or cascading k-core contract failed")
    data_hash = str(_read_json(processed_dir / "COMPLETE.json")["data_hash"])
    store = SequenceStore.from_processed(processed_dir)
    model_config = SASRecConfig(
        num_items=25,
        hidden_size=8,
        num_blocks=1,
        num_heads=1,
        dropout=0.0,
        maxlen=8,
    )
    completed = {
        "preprocess": 1,
        "single": 0,
        "joint": 0,
        "pcgrad": 0,
        "lora": 0,
        "fullft": 0,
        "analysis": 0,
    }
    initialization_hashes: dict[str, str] = {}
    for domain in range(5):
        train_pretraining(
            store,
            model_config,
            PretrainSettings(
                method="single",
                domain=domain,
                output_dir=runs_root
                / "pretrain"
                / "single"
                / f"domain-{domain}"
                / f"seed-{seed}",
                seed=seed,
                batch_size=2,
                steps_per_epoch=1,
                epochs=1,
                patience=1,
                device="cpu",
                evaluation_protocol="sampled",
                num_eval_negatives=2,
                gradient_log_interval=1,
                data_hash=data_hash,
            ),
        )
        completed["single"] += 1
    for method in ("joint", "pcgrad"):
        result = train_pretraining(
            store,
            model_config,
            PretrainSettings(
                method=method,
                output_dir=runs_root
                / "pretrain"
                / method
                / "all-domains"
                / f"seed-{seed}",
                seed=seed,
                batch_size=2,
                steps_per_epoch=1,
                epochs=1,
                patience=1,
                device="cpu",
                evaluation_protocol="sampled",
                num_eval_negatives=2,
                gradient_log_interval=1,
                data_hash=data_hash,
            ),
        )
        initialization_hashes[method] = result.initialization_hash
        completed[method] += 1
    if initialization_hashes["joint"] != initialization_hashes["pcgrad"]:
        raise AssertionError("Joint and PCGrad did not share initialization")
    joint_manifest = (
        runs_root
        / "pretrain"
        / "joint"
        / "all-domains"
        / f"seed-{seed}"
        / "batch_manifest.json"
    )
    pcgrad_manifest = (
        runs_root
        / "pretrain"
        / "pcgrad"
        / "all-domains"
        / f"seed-{seed}"
        / "batch_manifest.json"
    )
    if sha256_file(joint_manifest) != sha256_file(pcgrad_manifest):
        raise AssertionError("Joint and PCGrad batch manifests differ")

    lora_counts: dict[tuple[str, int], list[int]] = {}
    for pretrain_method in ("joint", "pcgrad"):
        base = (
            runs_root
            / "pretrain"
            / pretrain_method
            / "all-domains"
            / f"seed-{seed}"
            / "best.pt"
        )
        for domain in range(5):
            for rank in LORA_RANKS:
                result = train_adaptation(
                    store,
                    model_config,
                    base,
                    AdaptSettings(
                        method="lora",
                        pretrain_method=pretrain_method,
                        domain=domain,
                        rank=rank,
                        alpha=rank,
                        output_dir=runs_root
                        / "adapt"
                        / "lora"
                        / pretrain_method
                        / f"domain-{domain}"
                        / f"rank-{rank}"
                        / f"seed-{seed}",
                        seed=seed,
                        batch_size=2,
                        steps_per_epoch=1,
                        epochs=1,
                        patience=1,
                        lr=1e-2,
                        device="cpu",
                        evaluation_protocol="sampled",
                        num_eval_negatives=2,
                        data_hash=data_hash,
                    ),
                )
                if not all(
                    ("q_proj" in name or "v_proj" in name) and ".lora_" in name
                    for name in result.trainable_names
                ):
                    raise AssertionError("smoke LoRA exposed a non-Q/V parameter")
                lora_counts.setdefault((pretrain_method, domain), []).append(
                    result.num_trainable_params
                )
                completed["lora"] += 1
            fullft = train_adaptation(
                store,
                model_config,
                base,
                AdaptSettings(
                    method="fullft",
                    pretrain_method=pretrain_method,
                    domain=domain,
                    output_dir=runs_root
                    / "adapt"
                    / "fullft"
                    / pretrain_method
                    / f"domain-{domain}"
                    / f"seed-{seed}",
                    seed=seed,
                    batch_size=2,
                    steps_per_epoch=1,
                    epochs=1,
                    patience=1,
                    lr=1e-3,
                    embedding_lr=1e-3,
                    device="cpu",
                    evaluation_protocol="sampled",
                    num_eval_negatives=2,
                    data_hash=data_hash,
                ),
            )
            if fullft.num_trainable_params != fullft.num_total_params:
                raise AssertionError("smoke FullFT did not expose all parameters")
            completed["fullft"] += 1
    if any(counts != sorted(counts) or len(set(counts)) != len(counts) for counts in lora_counts.values()):
        raise AssertionError("LoRA parameter count is not strictly monotonic")
    _assert_finite_run_metrics(runs_root)

    from ftrec.cli.analyze import main as analyze_main

    status = analyze_main(
        ["--runs-root", str(runs_root), "--output-dir", str(analysis_dir)]
    )
    if status:
        raise AssertionError(f"analysis exited with status {status}")
    completed["analysis"] = 1
    rows = collect_result_rows(runs_root)
    protocols = frozenset(row.evaluation_protocol for row in rows)
    recoveries = compute_recovery_rows(rows)
    figure_count = len(tuple((analysis_dir / "figures").glob("figure*.*")))
    if len(rows) != 75 or len({row.key for row in rows}) != 75:
        raise AssertionError("canonical smoke results are incomplete or duplicated")
    if not tuple(runs_root.rglob("gradient_conflicts.jsonl")):
        raise AssertionError("gradient conflict logs are absent")
    model_runs = completed["single"] + completed["joint"] + completed["pcgrad"] + completed["lora"] + completed["fullft"]
    report = SmokeReport(
        ok=True,
        output_dir=root,
        completed=completed,
        model_runs=model_runs,
        lora_ranks=LORA_RANKS,
        figure_count=figure_count,
        evaluation_protocols=protocols,
        recovery_rows=len(recoveries),
        semantic_hash=_semantic_hash(
            processed_dir, runs_root, initialization_hashes
        ),
    )
    (root / "smoke_report.json").write_text(
        json.dumps(
            {
                **report.__dict__,
                "output_dir": str(report.output_dir),
                "evaluation_protocols": sorted(report.evaluation_protocols),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
