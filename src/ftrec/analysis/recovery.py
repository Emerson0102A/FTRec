"""Unclipped adaptation Recovery metrics and comparison warnings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .results import ResultRow


@dataclass(frozen=True)
class RecoveryValue:
    value: float
    undefined_recovery: bool
    warning: str = ""


def recovery(
    *, pretrain: float, lora: float, fullft: float, epsilon: float = 1e-12
) -> RecoveryValue:
    denominator = fullft - pretrain
    warnings: list[str] = []
    if abs(denominator) < epsilon:
        return RecoveryValue(
            float("nan"),
            True,
            "FullFT-pretrain denominator is below epsilon; Recovery is undefined",
        )
    if fullft < pretrain:
        warnings.append("FullFT is below the pretrained baseline")
    return RecoveryValue((lora - pretrain) / denominator, False, "; ".join(warnings))


@dataclass(frozen=True)
class RecoveryRow:
    seed: int
    domain: str
    pretrain_method: str
    lora_rank: int
    metric: str
    evaluation_protocol: str
    pretrain_value: float
    lora_value: float
    fullft_value: float
    recovery: float
    undefined_recovery: bool
    warning: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def compute_recovery_rows(
    rows: Iterable[ResultRow], *, epsilon: float = 1e-12
) -> tuple[RecoveryRow, ...]:
    rows = tuple(row for row in rows if row.split == "test" and row.domain != "Macro")
    baseline = {
        (row.seed, row.domain, row.pretrain_method, row.evaluation_protocol): row
        for row in rows
        if row.adapt_method == "none"
    }
    fullft = {
        (row.seed, row.domain, row.pretrain_method, row.evaluation_protocol): row
        for row in rows
        if row.adapt_method == "fullft"
    }
    results: list[RecoveryRow] = []
    for lora in rows:
        if lora.adapt_method != "lora" or lora.lora_rank is None:
            continue
        key = (lora.seed, lora.domain, lora.pretrain_method, lora.evaluation_protocol)
        if key not in baseline or key not in fullft:
            continue
        for metric, attribute in (("HR@10", "hr_at_10"), ("NDCG@10", "ndcg_at_10")):
            start = float(getattr(baseline[key], attribute))
            adapted = float(getattr(lora, attribute))
            upper = float(getattr(fullft[key], attribute))
            value = recovery(
                pretrain=start, lora=adapted, fullft=upper, epsilon=epsilon
            )
            results.append(
                RecoveryRow(
                    lora.seed,
                    lora.domain,
                    lora.pretrain_method,
                    lora.lora_rank,
                    metric,
                    lora.evaluation_protocol,
                    start,
                    adapted,
                    upper,
                    value.value,
                    value.undefined_recovery,
                    value.warning,
                )
            )
    return tuple(
        sorted(
            results,
            key=lambda row: (
                row.seed,
                row.domain,
                row.pretrain_method,
                row.lora_rank,
                row.metric,
            ),
        )
    )


def analysis_warnings(
    rows: Iterable[ResultRow], recovery_rows: Iterable[RecoveryRow]
) -> tuple[str, ...]:
    rows = tuple(rows)
    recovery_rows = tuple(recovery_rows)
    warnings: set[str] = set()
    protocols = {row.evaluation_protocol for row in rows}
    if len(protocols) > 1:
        warnings.add(f"mixed evaluation protocols: {sorted(protocols)}")
    for row in recovery_rows:
        if row.warning:
            warnings.add(
                f"{row.pretrain_method}/{row.domain}/rank-{row.lora_rank}/{row.metric}: {row.warning}"
            )
    expected_ranks = {1, 2, 4, 8, 16}
    for method in ("joint", "pcgrad"):
        present = {
            row.lora_rank
            for row in rows
            if row.pretrain_method == method and row.adapt_method == "lora"
        }
        missing = expected_ranks - present
        if present and missing:
            warnings.add(f"{method} is missing LoRA ranks: {sorted(missing)}")
    seeds = {row.seed for row in rows}
    if len(seeds) < 3:
        warnings.add(f"only {len(seeds)} seed(s) available; production inference needs at least 3")
    return tuple(sorted(warnings))
