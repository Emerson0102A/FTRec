"""Canonical result rows, validation, aggregation, and run discovery."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from ftrec.data.amazon import DOMAIN_BY_ID


class ResultSchemaError(ValueError):
    """Raised when experiment results cannot be compared safely."""


@dataclass(frozen=True)
class ResultRow:
    seed: int
    domain: str
    pretrain_method: str
    adapt_method: str
    lora_rank: int | None
    split: str
    evaluation_protocol: str
    hr_at_10: float
    ndcg_at_10: float
    num_eval_users: int
    num_skipped_users: int
    num_trainable_params: int
    num_total_params: int
    checkpoint_path: str
    config_hash: str
    data_hash: str
    contributing_domains: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ResultRow":
        required = {
            "seed",
            "domain",
            "pretrain_method",
            "adapt_method",
            "lora_rank",
            "split",
            "evaluation_protocol",
            "HR@10",
            "NDCG@10",
            "num_eval_users",
            "num_skipped_users",
            "num_trainable_params",
            "num_total_params",
            "checkpoint_path",
            "config_hash",
            "data_hash",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ResultSchemaError(f"result row is missing: {', '.join(missing)}")
        rank = value["lora_rank"]
        if rank in (None, "", "null"):
            rank = None
        contributing = value.get("contributing_domains")
        if contributing in (None, "", "null"):
            contributing = None
        try:
            row = cls(
                seed=int(value["seed"]),
                domain=str(value["domain"]),
                pretrain_method=str(value["pretrain_method"]),
                adapt_method=str(value["adapt_method"]),
                lora_rank=int(rank) if rank is not None else None,
                split=str(value["split"]),
                evaluation_protocol=str(value["evaluation_protocol"]),
                hr_at_10=float(value["HR@10"]),
                ndcg_at_10=float(value["NDCG@10"]),
                num_eval_users=int(value["num_eval_users"]),
                num_skipped_users=int(value["num_skipped_users"]),
                num_trainable_params=int(value["num_trainable_params"]),
                num_total_params=int(value["num_total_params"]),
                checkpoint_path=str(value["checkpoint_path"]),
                config_hash=str(value["config_hash"]),
                data_hash=str(value["data_hash"]),
                contributing_domains=(
                    int(contributing) if contributing is not None else None
                ),
            )
        except (TypeError, ValueError) as error:
            raise ResultSchemaError(f"invalid result row: {error}") from error
        row.validate()
        return row

    def validate(self) -> None:
        if self.evaluation_protocol not in {"full", "sampled"}:
            raise ResultSchemaError("evaluation_protocol must be full or sampled")
        if self.adapt_method == "lora" and self.lora_rank is None:
            raise ResultSchemaError("LoRA result requires lora_rank")
        if self.adapt_method != "lora" and self.lora_rank is not None:
            raise ResultSchemaError("only LoRA results may contain lora_rank")
        if self.num_trainable_params < 0 or self.num_total_params < 1:
            raise ResultSchemaError("parameter counts must be non-negative and nonzero")
        if self.num_trainable_params > self.num_total_params:
            raise ResultSchemaError("trainable parameter count exceeds total")
        if self.num_eval_users < 0 or self.num_skipped_users < 0:
            raise ResultSchemaError("cohort counts must be non-negative")

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.seed,
            self.domain,
            self.pretrain_method,
            self.adapt_method,
            self.lora_rank,
            self.split,
            self.evaluation_protocol,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "domain": self.domain,
            "pretrain_method": self.pretrain_method,
            "adapt_method": self.adapt_method,
            "lora_rank": self.lora_rank,
            "split": self.split,
            "evaluation_protocol": self.evaluation_protocol,
            "HR@10": self.hr_at_10,
            "NDCG@10": self.ndcg_at_10,
            "num_eval_users": self.num_eval_users,
            "num_skipped_users": self.num_skipped_users,
            "num_trainable_params": self.num_trainable_params,
            "num_total_params": self.num_total_params,
            "checkpoint_path": self.checkpoint_path,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "contributing_domains": self.contributing_domains,
        }


@dataclass(frozen=True)
class SummaryRow:
    domain: str
    pretrain_method: str
    adapt_method: str
    lora_rank: int | None
    split: str
    evaluation_protocol: str
    metric: str
    mean: float
    std: float
    seeds: int

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def validate_result_rows(rows: Iterable[ResultRow]) -> tuple[ResultRow, ...]:
    rows = tuple(rows)
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        row.validate()
        if row.key in seen:
            raise ResultSchemaError(f"duplicate result key: {row.key}")
        seen.add(row.key)
    return rows


def add_macro_rows(rows: Iterable[ResultRow]) -> tuple[ResultRow, ...]:
    original = tuple(rows)
    grouped: dict[tuple[object, ...], list[ResultRow]] = {}
    for row in original:
        if row.domain == "Macro":
            continue
        key = (
            row.seed,
            row.pretrain_method,
            row.adapt_method,
            row.lora_rank,
            row.split,
            row.evaluation_protocol,
        )
        grouped.setdefault(key, []).append(row)
    macros: list[ResultRow] = []
    for values in grouped.values():
        contributing = [
            row
            for row in values
            if row.num_eval_users > 0
            and math.isfinite(row.hr_at_10)
            and math.isfinite(row.ndcg_at_10)
        ]
        if not contributing:
            continue
        first = contributing[0]
        macros.append(
            replace(
                first,
                domain="Macro",
                hr_at_10=statistics.fmean(row.hr_at_10 for row in contributing),
                ndcg_at_10=statistics.fmean(row.ndcg_at_10 for row in contributing),
                num_eval_users=sum(row.num_eval_users for row in contributing),
                num_skipped_users=sum(row.num_skipped_users for row in values),
                contributing_domains=len(contributing),
            )
        )
    return original + tuple(sorted(macros, key=lambda row: row.key))


def aggregate_results(rows: Iterable[ResultRow]) -> tuple[SummaryRow, ...]:
    grouped: dict[tuple[object, ...], list[ResultRow]] = {}
    for row in rows:
        key = (
            row.domain,
            row.pretrain_method,
            row.adapt_method,
            row.lora_rank,
            row.split,
            row.evaluation_protocol,
        )
        grouped.setdefault(key, []).append(row)
    result: list[SummaryRow] = []
    for key, values in grouped.items():
        for metric, attribute in (("HR@10", "hr_at_10"), ("NDCG@10", "ndcg_at_10")):
            observations = [
                float(getattr(row, attribute))
                for row in values
                if row.num_eval_users > 0 and math.isfinite(float(getattr(row, attribute)))
            ]
            if not observations:
                continue
            result.append(
                SummaryRow(
                    *key,
                    metric=metric,
                    mean=statistics.fmean(observations),
                    std=statistics.stdev(observations) if len(observations) > 1 else 0.0,
                    seeds=len(observations),
                )
            )
    return tuple(
        sorted(
            result,
            key=lambda row: (
                row.domain,
                row.pretrain_method,
                row.adapt_method,
                row.lora_rank or 0,
                row.metric,
            ),
        )
    )


RESULT_COLUMNS = tuple(ResultRow(0, "d", "joint", "none", None, "test", "full", 0, 0, 0, 0, 0, 1, "p", "c", "h").to_dict())


def write_result_rows(path: str | Path, rows: Iterable[ResultRow]) -> None:
    rows = validate_result_rows(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item.key):
            writer.writerow(row.to_dict())


def write_summary_rows(path: str | Path, rows: Iterable[SummaryRow]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(SummaryRow("d", "p", "a", None, "test", "full", "m", 0, 0, 0).to_dict())
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _domain_name(value: object) -> str:
    try:
        return DOMAIN_BY_ID[int(value)].name
    except (KeyError, TypeError, ValueError):
        return str(value)


def collect_result_rows(root: str | Path) -> tuple[ResultRow, ...]:
    rows: list[ResultRow] = []
    for path in sorted(Path(root).rglob("result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = str(path.parent / "best.pt")
        if "pretrain_method" in value:
            metrics_by_domain = {str(value["domain"]): value["test_metrics"]}
            pretrain_method = str(value["pretrain_method"])
            adapt_method = str(value["method"])
            rank = value.get("rank")
        elif value.get("method") in {
            "single",
            "single_mixed",
            "joint_domain",
            "joint_mixed_matched",
            "joint",
            "pcgrad",
        }:
            metrics_by_domain = value["test_metrics"]
            pretrain_method = str(value["method"])
            adapt_method = "none"
            rank = None
        else:
            continue
        for domain, metrics in metrics_by_domain.items():
            row_value = {
                "seed": value["seed"],
                "domain": _domain_name(domain),
                "pretrain_method": pretrain_method,
                "adapt_method": adapt_method,
                "lora_rank": rank,
                "split": "test",
                "evaluation_protocol": metrics["evaluation_protocol"],
                "HR@10": metrics["HR@10"],
                "NDCG@10": metrics["NDCG@10"],
                "num_eval_users": metrics["num_eval_users"],
                "num_skipped_users": metrics["num_skipped_users"],
                "num_trainable_params": value["num_trainable_params"],
                "num_total_params": value["num_total_params"],
                "checkpoint_path": checkpoint,
                "config_hash": value["config_hash"],
                "data_hash": value["data_hash"],
            }
            rows.append(ResultRow.from_dict(row_value))
    return validate_result_rows(rows)
