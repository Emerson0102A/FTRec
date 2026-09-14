"""Deterministic headless figures generated from canonical result records."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .recovery import RecoveryRow
from .results import ResultRow


COLORS = {"joint": "#4472C4", "pcgrad": "#ED7D31", "single": "#70AD47"}


def _save(fig: plt.Figure, root: Path, stem: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    png = root / f"{stem}.png"
    pdf = root / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", metadata={"Software": "FTRec"})
    fig.savefig(pdf, bbox_inches="tight", metadata={"Creator": "FTRec"})
    plt.close(fig)
    return png, pdf


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def _figure_pretraining(rows: Sequence[ResultRow]) -> plt.Figure:
    selected = [row for row in rows if row.adapt_method == "none" and row.domain != "Macro"]
    domains = sorted({row.domain for row in selected}) or ["No data"]
    methods = [method for method in ("single", "joint", "pcgrad") if any(row.pretrain_method == method for row in selected)] or ["joint"]
    fig, axis = plt.subplots(figsize=(max(6, len(domains) * 1.2), 4))
    width = 0.8 / len(methods)
    x = np.arange(len(domains))
    for index, method in enumerate(methods):
        values = [
            _mean([row.ndcg_at_10 for row in selected if row.domain == domain and row.pretrain_method == method])
            for domain in domains
        ]
        axis.bar(x + (index - (len(methods) - 1) / 2) * width, values, width, label=method, color=COLORS.get(method))
    axis.set_xticks(x, domains, rotation=20, ha="right")
    axis.set_ylabel("NDCG@10")
    axis.set_title("Pretraining comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    return fig


def _figure_lora(rows: Sequence[ResultRow]) -> plt.Figure:
    selected = [row for row in rows if row.adapt_method == "lora" and row.domain != "Macro"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for method in ("joint", "pcgrad"):
        for domain in sorted({row.domain for row in selected}):
            ranks = sorted({row.lora_rank for row in selected if row.pretrain_method == method and row.domain == domain and row.lora_rank is not None})
            if not ranks:
                continue
            values = [
                _mean([row.ndcg_at_10 for row in selected if row.pretrain_method == method and row.domain == domain and row.lora_rank == rank])
                for rank in ranks
            ]
            axis.plot(ranks, values, marker="o", color=COLORS[method], alpha=0.65, label=f"{method}: {domain}")
    fullft = [row for row in rows if row.adapt_method == "fullft"]
    for method in ("joint", "pcgrad"):
        values = [row.ndcg_at_10 for row in fullft if row.pretrain_method == method]
        if values:
            axis.axhline(_mean(values), color=COLORS[method], linestyle="--", alpha=0.5)
    axis.set_xscale("log", base=2)
    ticks = sorted({row.lora_rank for row in selected if row.lora_rank is not None})
    if ticks:
        axis.set_xticks(ticks, [str(rank) for rank in ticks])
    axis.set_xlabel("LoRA rank")
    axis.set_ylabel("NDCG@10")
    axis.set_title("Low-rank adaptation")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    return fig


def _figure_recovery(rows: Sequence[RecoveryRow]) -> plt.Figure:
    selected = [row for row in rows if row.metric == "NDCG@10"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for method in ("joint", "pcgrad"):
        ranks = sorted({row.lora_rank for row in selected if row.pretrain_method == method})
        values = [
            _mean([row.recovery for row in selected if row.pretrain_method == method and row.lora_rank == rank])
            for rank in ranks
        ]
        if ranks:
            axis.plot(ranks, values, marker="o", linewidth=2, color=COLORS[method], label=method)
    axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    axis.set_xscale("log", base=2)
    ticks = sorted({row.lora_rank for row in selected})
    if ticks:
        axis.set_xlim(ticks[0] / 1.5, ticks[-1] * 1.5)
        axis.set_xticks(ticks, [str(rank) for rank in ticks])
    axis.set_xlabel("LoRA rank")
    axis.set_ylabel("Recovery")
    axis.set_title("Adaptation capacity recovery")
    axis.grid(alpha=0.25)
    axis.legend()
    return fig


def _figure_gradients(records: Sequence[Mapping[str, object]]) -> plt.Figure:
    matrices: list[np.ndarray] = []
    names: Sequence[str] = ()
    for record in records:
        raw = record.get("raw_cosine", {})
        if isinstance(raw, Mapping) and "full" in raw:
            matrix = np.asarray(raw["full"], dtype=float)
            if matrix.ndim == 2:
                matrices.append(matrix)
                names = tuple(str(name) for name in record.get("domain_names", ()))
    matrix = np.nanmean(np.stack(matrices), axis=0) if matrices else np.zeros((1, 1))
    if not names or len(names) != matrix.shape[0]:
        names = tuple(str(index) for index in range(matrix.shape[0]))
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
    axis.set_yticks(range(len(names)), names)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("Raw task-gradient cosine")
    fig.colorbar(image, ax=axis, fraction=0.046)
    return fig


def generate_figures(
    result_rows: Iterable[ResultRow],
    recovery_rows: Iterable[RecoveryRow],
    gradient_records: Iterable[Mapping[str, object]],
    output_dir: str | Path,
) -> tuple[Path, ...]:
    rows = tuple(result_rows)
    recoveries = tuple(recovery_rows)
    gradients = tuple(gradient_records)
    root = Path(output_dir)
    outputs: list[Path] = []
    outputs.extend(_save(_figure_pretraining(rows), root, "figure1_pretraining"))
    outputs.extend(_save(_figure_lora(rows), root, "figure2_lora_rank"))
    outputs.extend(_save(_figure_recovery(recoveries), root, "figure3_recovery"))
    outputs.extend(_save(_figure_gradients(gradients), root, "figure4_gradient_conflict"))
    return tuple(outputs)
