"""Checks for the CGRec adapter against GMFlowRec's Parquet protocol."""

from __future__ import annotations

import pickle
import json
import math
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FTRec-CGRec"))

from parquet_data import (  # noqa: E402
    CGRecCategoryMapping,
    CGRecDomainCategoryMapping,
    CGRecEvaluationDataset,
    CGRecTrainDataset,
    domain_remap,
    load_parquet_data,
)
from parquet_model import CGRecParquetModel  # noqa: E402
from parquet_eval import rank_of_positive, summarize_ranks  # noqa: E402


def _write_split(path: Path, items: list[list[int]], domains: list[list[int]]) -> None:
    pq.write_table(
        pa.table({
            "item_id": pa.array(items, type=pa.list_(pa.int64())),
            "domain_id": pa.array(domains, type=pa.list_(pa.int64())),
        }),
        path,
    )


def make_fixture(tmp_path: Path) -> Path:
    mapping = {
        "user": {f"u{i}": i for i in range(5)},
        "item": {f"i{i}": i for i in range(50)},
        "domain_offset": {d: (10 * d, 10 * (d + 1)) for d in range(5)},
    }
    with (tmp_path / "mappings.pkl").open("wb") as stream:
        pickle.dump(mapping, stream)
    train_items = [[10 * d, 10 * d + 1, 10 * ((d + 1) % 5), 10 * d + 2] for d in range(5)]
    train_domains = [[d, d, (d + 1) % 5, d] for d in range(5)]
    valid_items = [[10 * ((d + 1) % 5), 10 * ((d + 1) % 5) + 1, 10 * d] for d in range(5)]
    valid_domains = [[(d + 1) % 5, (d + 1) % 5, d] for d in range(5)]
    test_items = [row + [10 * d + 1] for d, row in enumerate(valid_items)]
    test_domains = [row + [d] for d, row in enumerate(valid_domains)]
    _write_split(tmp_path / "train_new.parquet", train_items, train_domains)
    _write_split(tmp_path / "valid_new.parquet", valid_items, valid_domains)
    _write_split(tmp_path / "test_new.parquet", test_items, test_domains)
    with gzip.open(tmp_path / "catalog.jsonl.gz", "wt", encoding="utf-8") as stream:
        for item_id in range(50):
            stream.write(json.dumps({
                "item_id": item_id + 1,
                "domain_id": item_id // 10,
                "parent_asin": f"i{item_id}",
                "categories": (
                    [f"domain-{item_id // 10}", f"group-{item_id % 3}", f"leaf-{item_id % 5}"]
                    if item_id != 7 else []
                ),
            }) + "\n")
    return tmp_path


def test_training_uses_all_training_interactions_and_reserved_ids(tmp_path: Path) -> None:
    data = load_parquet_data(make_fixture(tmp_path))
    dataset = CGRecTrainDataset(data.train, data.metadata, target_domain=0, maxlen=5, seed=7)
    dataset.set_epoch(2)
    items, positives, negatives, _, _, _, _, _, _, domains, row_id = dataset[0]

    assert row_id == 0
    assert items.tolist() == [0, 0, 5, 6, 15]
    assert positives.tolist() == [0, 0, 6, 15, 7]
    assert domains.tolist() == [0, 0, 5, 5, 6]
    assert np.array_equal(negatives, dataset[0][2])
    assert set(negatives[negatives != 0]).isdisjoint({5, 6, 15, 7})
    assert np.all(negatives[negatives != 0] >= 5)


def test_evaluation_reuses_gmflowrec_candidates_and_target_filter(tmp_path: Path) -> None:
    data = load_parquet_data(make_fixture(tmp_path))
    dataset = CGRecEvaluationDataset(
        data.valid, data.metadata, target_domain=0, maxlen=4,
        num_negatives=3, eval_seed=11,
    )
    assert len(dataset) == 1
    items, _, _, domains, candidates, row_id = dataset[0]
    assert row_id == 0
    assert items.tolist() == [0, 0, 15, 16]
    assert domains.tolist() == [0, 0, 6, 6]
    assert candidates[0] == 5
    assert len(set(candidates.tolist())) == 4
    assert all(5 <= candidate < 15 for candidate in candidates)
    assert set(candidates[1:]).isdisjoint({5})
    assert np.array_equal(candidates, dataset[0][4])
    dataset.precompute()
    assert np.array_equal(candidates, dataset[0][4])


def test_each_target_has_stable_original_domain_mapping() -> None:
    assert domain_remap(0) == {0: 5, 1: 6, 2: 7, 3: 8, 4: 9}
    assert domain_remap(3) == {3: 5, 0: 6, 1: 7, 2: 8, 4: 9}


def test_catalog_restores_two_category_levels_and_checks_item_alignment(tmp_path: Path) -> None:
    root = make_fixture(tmp_path)
    data = load_parquet_data(root)
    categories = CGRecCategoryMapping(root / "catalog.jsonl.gz", root / "mappings.pkl", data.metadata)
    assert categories.cat1_size > categories.cat2_size > 1
    assert categories.cat2_by_item[5] == categories.cat2_by_item[8]
    assert categories.cat1_by_item[5] != categories.cat1_by_item[8]
    assert categories.cat1_by_item[12] > 0  # missing category uses its own token

    train = CGRecTrainDataset(data.train, data.metadata, 0, 5, 7, categories=categories)
    item, pos, neg, cat1, cat1_pos, cat1_neg, cat2, cat2_pos, cat2_neg, _, _ = train[0]
    assert np.array_equal(cat1, categories.cat1_by_item[item])
    assert np.array_equal(cat2, categories.cat2_by_item[item])
    assert np.array_equal(cat1_pos, categories.cat1_by_item[pos])
    assert np.array_equal(cat2_pos, categories.cat2_by_item[pos])
    assert np.all(cat1_neg[item > 0] > 0)
    assert np.all(cat2_neg[item > 0] > 0)
    assert np.array_equal(neg, train[0][2])

    evaluation = CGRecEvaluationDataset(data.valid, data.metadata, 0, 4, 3, 11, categories=categories)
    context, eval_cat1, eval_cat2, _, _, _ = evaluation[0]
    assert np.array_equal(eval_cat1, categories.cat1_by_item[context])
    assert np.array_equal(eval_cat2, categories.cat2_by_item[context])

    broken = tmp_path / "broken.jsonl.gz"
    with gzip.open(broken, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"item_id": 1, "domain_id": 0, "parent_asin": "WRONG", "categories": []}) + "\n")
    with pytest.raises(ValueError, match="ASIN|catalog"):
        CGRecCategoryMapping(broken, root / "mappings.pkl", data.metadata)


def test_released_code_domain_categories_mode(tmp_path: Path) -> None:
    data = load_parquet_data(make_fixture(tmp_path))
    categories = CGRecDomainCategoryMapping(data.metadata, target_domain=3)
    assert categories.cat1_size == 12
    assert categories.cat2_size == 11
    assert categories.negative_min == 5
    assert np.array_equal(categories.cat1_by_item, categories.cat2_by_item)
    assert categories.cat1_by_item[5] == 6  # original domain 0 becomes source 6
    assert categories.cat1_by_item[35] == 5  # original domain 3 becomes target 5
    sample = CGRecTrainDataset(data.train, data.metadata, 3, 5, 7, categories=categories)[0]
    assert np.all(sample[5][sample[0] > 0] >= 5)
    assert np.all(sample[8][sample[0] > 0] >= 5)


def test_original_cgrec_shapley_path_trains_and_scores_without_categories(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    data = load_parquet_data(make_fixture(tmp_path))
    dataset = CGRecTrainDataset(data.train, data.metadata, target_domain=0, maxlen=5, seed=7)
    items, positive, negative, _, _, _, _, _, _, domains, _ = dataset[0]
    model = CGRecParquetModel(
        item_count=50, maxlen=5, hidden_size=8, num_layers=1,
        num_heads=2, dropout=0.0, device="cpu", shapley=True,
    )
    batch = lambda array: torch.as_tensor(array).unsqueeze(0)
    zeros = torch.zeros(1, len(items), dtype=torch.long)
    loss = model.train_loss(
        batch(items), batch(positive), batch(negative),
        zeros, zeros, zeros, zeros, zeros, zeros, batch(domains),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.item_embeddings.weight.grad is not None
    assert torch.isfinite(model.item_embeddings.weight.grad).all()
    assert "shaply_values_update" in model.state_dict()

    evaluation = CGRecEvaluationDataset(
        data.valid, data.metadata, target_domain=0, maxlen=4,
        num_negatives=3, eval_seed=11,
    )
    context, _, _, context_domains, candidates, _ = evaluation[0]
    eval_zeros = torch.zeros(1, len(context), dtype=torch.long)
    scores = model.score(
        batch(context), eval_zeros, eval_zeros,
        batch(context_domains), batch(candidates),
    )
    assert scores.shape == (1, 4)
    assert torch.isfinite(scores).all()


def test_hierarchical_cgrec_uses_real_category_inputs(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    root = make_fixture(tmp_path)
    data = load_parquet_data(root)
    categories = CGRecCategoryMapping(root / "catalog.jsonl.gz", root / "mappings.pkl", data.metadata)
    sample = CGRecTrainDataset(data.train, data.metadata, 0, 5, 7, categories=categories)[0]
    model = CGRecParquetModel(
        item_count=50, maxlen=5, hidden_size=8, num_layers=1,
        num_heads=2, dropout=0.0, device="cpu", shapley=True,
        cat1_size=categories.cat1_size, cat2_size=categories.cat2_size,
        hierarchical=True,
    )
    tensors = [torch.as_tensor(array).unsqueeze(0) for array in sample[:-1]]
    loss = model.train_loss(*tensors)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.cat1_embeddings.weight.grad is not None
    assert model.cat2_embeddings.weight.grad is not None
    assert model.item_embeddings.weight.grad is not None
    assert torch.count_nonzero(model.cat1_embeddings.weight.grad) > 0
    assert torch.count_nonzero(model.cat2_embeddings.weight.grad) > 0


def test_ranking_uses_deterministic_item_id_tie_break() -> None:
    scores = torch.tensor([[0.5, 0.6, 0.5, 0.1], [0.5, 0.5, 0.6, 0.1]])
    candidates = torch.tensor([[5, 6, 7, 8], [7, 5, 6, 8]])
    ranks = rank_of_positive(scores, candidates)
    assert ranks.tolist() == [1, 2]
    metrics = summarize_ranks(ranks.tolist())
    assert metrics["count"] == 2
    assert metrics["hr@5"] == 1.0
    assert math.isclose(metrics["ndcg@5"], (1 / math.log2(3) + 1 / math.log2(4)) / 2)


def test_parquet_cli_trains_selects_checkpoint_and_writes_result(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    data_dir = make_fixture(tmp_path / "data")
    run_dir = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "FTRec-CGRec" / "train_parquet.py"),
            "--parquet_dir", str(data_dir),
            "--category_catalog", str(data_dir / "catalog.jsonl.gz"),
            "--run_dir", str(run_dir),
            "--target_domain", "0",
            "--device", "cpu",
            "--seed", "7",
            "--epochs", "1",
            "--batch_size", "2",
            "--eval_batch_size", "2",
            "--maxlen", "4",
            "--hidden_size", "8",
            "--num_layers", "1",
            "--num_eval_negatives", "3",
            "--max_train_examples", "2",
            "--max_eval_examples", "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = run_dir / "domain-0" / "seed-7"
    report = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert (output / "best.pt").is_file()
    assert report["best_epoch"] == 1
    assert report["validation"]["count"] == 1
    assert report["test"]["count"] == 1
    assert report["protocol"]["category_features"] == "Amazon metadata categories; hierarchical CGRec"
