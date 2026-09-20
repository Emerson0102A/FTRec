import gzip
import json
import pickle

import numpy as np
import pytest

from ftrec.attributes.catalog import build_catalog
from ftrec.attributes.checkpoints import resolve_checkpoint_paths
from ftrec.attributes.runtime import PINNED_RUNTIME, recommended_batch_size


def _touch(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _checkpoint_tree(tmp_path):
    root = tmp_path / "LLM2Attr"
    _touch(root, "LLM2Attr.py")
    _touch(root, "PLoRAModel.py")
    mntp = root / "output/mntp/Qwen2.5-0.5B/checkpoint-10000"
    attr = root / "output/attr/Qwen2.5-0.5B/checkpoint-3000"
    for relative in ("adapter_config.json", "adapter_model.safetensors"):
        _touch(mntp, relative)
        _touch(attr, relative)
    for prompt in ("title", "attr"):
        for name in ("adapter_config.json", "prompt_tokens.pt", "pytorch_model.bin"):
            _touch(attr, f"{prompt}_prompt_encoder/{name}")
    return root


def test_checkpoint_paths_are_inferred_from_llm2attr_root(tmp_path):
    root = _checkpoint_tree(tmp_path)
    paths = resolve_checkpoint_paths(root)
    assert paths.mntp.name == "checkpoint-10000"
    assert paths.attribute.name == "checkpoint-3000"


def test_checkpoint_validation_requires_both_prompt_encoders(tmp_path):
    root = _checkpoint_tree(tmp_path)
    (root / "output/attr/Qwen2.5-0.5B/checkpoint-3000/attr_prompt_encoder/prompt_tokens.pt").unlink()
    with pytest.raises(FileNotFoundError, match="attr_prompt_encoder/prompt_tokens.pt"):
        resolve_checkpoint_paths(root)


def test_memory_based_batch_recommendation_is_conservative():
    assert recommended_batch_size(80) == 32
    assert recommended_batch_size(24) == 16
    assert recommended_batch_size(16) == 8


def test_llm2attr_runtime_versions_match_upstream_constraint():
    assert PINNED_RUNTIME == {
        "transformers": "4.44.2",
        "peft": "0.18.1",
        "llm2vec": "0.2.3",
    }


def test_catalog_keeps_explicit_one_based_item_ids(tmp_path):
    dataset = tmp_path / "Dataset"
    dataset.mkdir()
    mappings = {
        "domain": {"Alpha": 0},
        "domain_offset": {0: (0, 2)},
        "item": {"A": np.int64(0), "B": np.int64(1)},
    }
    mapping_path = tmp_path / "mappings.pkl"
    with mapping_path.open("wb") as handle:
        pickle.dump(mappings, handle, protocol=4)
    with gzip.open(dataset / "meta_Alpha.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"parent_asin": "B", "title": " Second item "}) + "\n")
        handle.write(json.dumps({"parent_asin": "A", "title": "First item"}) + "\n")
    output = tmp_path / "catalog.jsonl.gz"
    result = build_catalog(dataset, mapping_path, output)
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        rows = {row["item_id"]: row for row in map(json.loads, handle)}
    assert rows[1]["parent_asin"] == "A"
    assert rows[2]["parent_asin"] == "B"
    assert result.item_count == 2
