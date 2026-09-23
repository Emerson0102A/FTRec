"""Frozen content-bank controls for item/semantic alignment experiments."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ftrec.attributes.artifacts import AttributeArtifact
from ftrec.models.content_encoder import load_item_domain_ids


def _random_directions_like(source: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep each vector's norm and missing slots while replacing its direction."""

    result = np.empty(source.shape, dtype=source.dtype)
    result[0] = 0
    for start in range(1, source.shape[0], 1025):
        stop = min(start + 1024, source.shape[0])
        original = np.asarray(source[start:stop], dtype=np.float32)
        draws = rng.standard_normal(original.shape, dtype=np.float32)
        original_norm = np.linalg.norm(original, axis=-1, keepdims=True)
        draw_norm = np.linalg.norm(draws, axis=-1, keepdims=True)
        result[start:stop] = (
            draws * (original_norm / np.maximum(draw_norm, 1e-12))
        ).astype(source.dtype)
    return result


def ablate_content_artifact(
    artifact: AttributeArtifact,
    *,
    control: str,
    seed: int,
    domain_file: str | None,
) -> AttributeArtifact:
    """Return a content bank with one reproducible, explicit control applied."""

    if control in {"none", "attribute_only"}:
        return artifact
    rng = np.random.default_rng(seed)
    if control == "random":
        return replace(
            artifact,
            title_embeddings=_random_directions_like(artifact.title_embeddings, rng),
            attribute_embeddings=_random_directions_like(
                artifact.attribute_embeddings, rng
            ),
        )
    if control != "shuffled":
        raise ValueError(f"unknown content ablation: {control}")
    if domain_file is None:
        raise ValueError("shuffled content requires content_ablation_domain_file")
    domains, _ = load_item_domain_ids(domain_file, artifact.item_count)
    domain_values = domains.numpy()
    present = np.asarray(artifact.present_mask)
    permutation = np.arange(artifact.item_count + 1)
    for domain in np.unique(domain_values[1:]):
        for is_present in (False, True):
            indices = np.flatnonzero((domain_values == domain) & (present == is_present))
            if len(indices) < 2:
                continue
            rng.shuffle(indices)
            permutation[indices] = np.roll(indices, 1)
    return replace(
        artifact,
        title_embeddings=artifact.title_embeddings[permutation],
        attribute_embeddings=artifact.attribute_embeddings[permutation],
        present_mask=artifact.present_mask[permutation],
    )
