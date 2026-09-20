"""Streaming helpers shared by attribute providers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator


def catalog_manifest_path(catalog_path: str | Path) -> Path:
    return Path(catalog_path).with_suffix("").with_suffix(".manifest.json")


def load_catalog_manifest(catalog_path: str | Path) -> dict[str, Any]:
    path = catalog_manifest_path(catalog_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "ftrec-item-catalog":
        raise ValueError(f"not an FTRec item catalog manifest: {path}")
    return manifest


def iter_catalog(catalog_path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(catalog_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def batched(rows: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def move_to_device(features: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) for key, value in features.items()}
