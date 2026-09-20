"""Provider-neutral, item-ID-aligned attribute embedding artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class AttributeArtifact:
    root: Path
    provider: str
    item_count: int
    attribute_count: int
    embedding_dim: int
    catalog_sha256: str
    title_embeddings: np.ndarray
    attribute_embeddings: np.ndarray
    present_mask: np.ndarray
    metadata: dict[str, Any]


def create_embedding_arrays(
    output_dir: str | Path,
    *,
    item_count: int,
    attribute_count: int,
    embedding_dim: int,
    dtype: str,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    title = np.lib.format.open_memmap(
        root / "title_emb.npy", mode="w+", dtype=dtype,
        shape=(item_count + 1, embedding_dim),
    )
    attributes = np.lib.format.open_memmap(
        root / "attr_emb.npy", mode="w+", dtype=dtype,
        shape=(item_count + 1, attribute_count, embedding_dim),
    )
    present = np.lib.format.open_memmap(
        root / "present_mask.npy", mode="w+", dtype=np.bool_, shape=(item_count + 1,)
    )
    title[:] = 0
    attributes[:] = 0
    present[:] = False
    return title, attributes, present


def finish_artifact(
    output_dir: str | Path,
    *,
    provider: str,
    item_count: int,
    attribute_count: int,
    embedding_dim: int,
    catalog_sha256: str,
    title: np.memmap,
    attributes: np.memmap,
    present: np.memmap,
    provider_config: dict[str, Any],
) -> Path:
    title.flush()
    attributes.flush()
    present.flush()
    manifest = {
        "format": "ftrec-attribute-embeddings",
        "version": ARTIFACT_VERSION,
        "provider": provider,
        "item_id_convention": "one-based; row zero is padding",
        "item_count": int(item_count),
        "attribute_count": int(attribute_count),
        "embedding_dim": int(embedding_dim),
        "catalog_sha256": catalog_sha256,
        "provider_config": provider_config,
    }
    path = Path(output_dir) / "artifact.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_attribute_artifact(
    root: str | Path,
    *,
    expected_item_count: int | None = None,
    mmap_mode: str | None = "r",
    allow_partial: bool = False,
) -> AttributeArtifact:
    """Load and validate a provider-neutral, item-ID-aligned artifact."""

    root = Path(root)
    if root.name == "artifact.json":
        root = root.parent
    manifest_path = root / "artifact.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        metadata.get("format") != "ftrec-attribute-embeddings"
        or metadata.get("version") != ARTIFACT_VERSION
    ):
        raise ValueError(f"unsupported attribute artifact: {manifest_path}")

    item_count = int(metadata["item_count"])
    attribute_count = int(metadata["attribute_count"])
    embedding_dim = int(metadata["embedding_dim"])
    if expected_item_count is not None and item_count != expected_item_count:
        raise ValueError(
            f"attribute artifact has {item_count} items; model expects "
            f"{expected_item_count}"
        )
    processed_items = metadata.get("provider_config", {}).get("processed_items")
    if (
        processed_items is not None
        and int(processed_items) != item_count
        and not allow_partial
    ):
        raise ValueError(
            "attribute artifact is partial "
            f"({processed_items}/{item_count} items); regenerate in full mode"
        )

    title = np.load(root / "title_emb.npy", mmap_mode=mmap_mode)
    attributes = np.load(root / "attr_emb.npy", mmap_mode=mmap_mode)
    present = np.load(root / "present_mask.npy", mmap_mode=mmap_mode)
    expected_title_shape = (item_count + 1, embedding_dim)
    expected_attribute_shape = (
        item_count + 1,
        attribute_count,
        embedding_dim,
    )
    if title.shape != expected_title_shape:
        raise ValueError(
            f"title_emb.npy shape {title.shape}; expected {expected_title_shape}"
        )
    if attributes.shape != expected_attribute_shape:
        raise ValueError(
            "attr_emb.npy shape "
            f"{attributes.shape}; expected {expected_attribute_shape}"
        )
    if present.shape != (item_count + 1,):
        raise ValueError(
            f"present_mask.npy shape {present.shape}; expected {(item_count + 1,)}"
        )
    if np.any(title[0]) or np.any(attributes[0]) or bool(present[0]):
        raise ValueError("attribute artifact padding row zero must contain only zeros")
    if not np.isfinite(title).all() or not np.isfinite(attributes).all():
        raise ValueError("attribute artifact contains non-finite embeddings")

    return AttributeArtifact(
        root=root,
        provider=str(metadata["provider"]),
        item_count=item_count,
        attribute_count=attribute_count,
        embedding_dim=embedding_dim,
        catalog_sha256=str(metadata["catalog_sha256"]),
        title_embeddings=title,
        attribute_embeddings=attributes,
        present_mask=present,
        metadata=metadata,
    )
