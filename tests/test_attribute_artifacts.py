from __future__ import annotations

import json

import numpy as np
import pytest

from ftrec.attributes.artifacts import (
    create_embedding_arrays,
    finish_artifact,
    load_attribute_artifact,
)


def _artifact(tmp_path, *, processed_items=2):
    title, attributes, present = create_embedding_arrays(
        tmp_path,
        item_count=2,
        attribute_count=3,
        embedding_dim=4,
        dtype="float32",
    )
    title[1:] = 1
    attributes[1:] = 2
    present[1:] = True
    finish_artifact(
        tmp_path,
        provider="test-provider",
        item_count=2,
        attribute_count=3,
        embedding_dim=4,
        catalog_sha256="catalog",
        title=title,
        attributes=attributes,
        present=present,
        provider_config={"processed_items": processed_items},
    )


def test_load_attribute_artifact_validates_shapes_and_padding(tmp_path):
    _artifact(tmp_path)
    artifact = load_attribute_artifact(tmp_path, expected_item_count=2)
    assert artifact.provider == "test-provider"
    assert artifact.title_embeddings.shape == (3, 4)
    assert artifact.attribute_embeddings.shape == (3, 3, 4)
    assert not artifact.present_mask[0]


def test_partial_artifact_cannot_be_used_for_training(tmp_path):
    _artifact(tmp_path, processed_items=1)
    with pytest.raises(ValueError, match="partial"):
        load_attribute_artifact(tmp_path)


def test_mismatched_item_count_is_rejected(tmp_path):
    _artifact(tmp_path)
    with pytest.raises(ValueError, match="model expects 3"):
        load_attribute_artifact(tmp_path, expected_item_count=3)


def test_bad_padding_row_is_rejected(tmp_path):
    _artifact(tmp_path)
    title = np.load(tmp_path / "title_emb.npy", mmap_mode="r+")
    title[0, 0] = 1
    title.flush()
    with pytest.raises(ValueError, match="padding row zero"):
        load_attribute_artifact(tmp_path)
