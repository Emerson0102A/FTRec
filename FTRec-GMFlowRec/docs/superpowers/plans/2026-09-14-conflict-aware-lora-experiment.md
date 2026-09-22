# Conflict-Aware LoRA Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, reproducible five-domain SASRec experiment pipeline comparing Joint and PCGrad pretraining under Q/V LoRA rank sweeps, with a full CPU smoke run and server-ready production configuration.

**Architecture:** A typed `src/ftrec` package owns deterministic preprocessing, task construction, an explicit-Q/K/V SASRec, shared training/evaluation engines, sparse-aware PCGrad, Q/V LoRA, structured result analysis, and stage CLIs. Every stage publishes fingerprinted artifacts atomically and the smoke command composes the same production components on synthetic data.

**Tech Stack:** Python 3.12, PyTorch 2.9+ locally and PyTorch 2.13.0+cu130 on the server, NumPy, PyArrow, PyYAML, Matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-09-14-conflict-aware-lora-experiment-design.md`

## Global Constraints

- Preserve the preprocessing semantics in `docs/superpowers/specs/2026-09-14-amazon-five-domain-preprocessing-design.md`.
- Use explicit `q_proj`, `k_proj`, `v_proj`, and `out_proj`; do not use fused `nn.MultiheadAttention` in the new model.
- Keep Joint and PCGrad architecture, initialization, batch manifests, optimizer settings, schedule, clipping, validation, and checkpoint rules identical.
- Apply standard PCGrad to every trainable SASRec parameter, including sparse item-embedding gradients, without densifying the complete embedding matrix.
- Insert LoRA only into Q/V projections and assert that only adapter parameters are trainable.
- Use validation NDCG@10 for checkpoint selection; never use test metrics for model selection.
- Run only synthetic CPU smoke training locally; provide server commands for real-data preprocessing and training.
- Preserve raw per-seed results and report mean ± std for the production three-seed experiment.
- Do not infer scientific support from synthetic smoke metrics.

## File Map

- `pyproject.toml`: package metadata, runtime/test dependencies, CLI entry points, and pytest settings.
- `environment.yml`, `requirements-cu130.txt`: server Python environment and pinned CUDA 13.0 Torch installation.
- `legacy/pmixer_sasrec/`: preserved original root implementation and provenance note.
- `src/ftrec/config.py`: typed configuration loading, validation, canonical hashes, and overrides.
- `src/ftrec/reproducibility.py`: seed setup, deterministic worker seeds, device/AMP resolution, environment capture.
- `src/ftrec/artifacts.py`: atomic run directories, JSON/JSONL serialization, file hashes, completion markers.
- `src/ftrec/data/amazon.py`: domain constants and raw row parsing.
- `src/ftrec/data/preprocessing.py`: SQLite ingestion, deduplication, k-core, mapping, split, export, and validation.
- `src/ftrec/data/datasets.py`: processed sequence loading and Single/mixed target-example construction.
- `src/ftrec/data/sampling.py`: same-domain negative sampling and persistent balanced batch manifests.
- `src/ftrec/models/attention.py`: explicit attention and SASRec block.
- `src/ftrec/models/sasrec.py`: embeddings, encoder, scoring, initialization, and parameter groups.
- `src/ftrec/models/lora.py`: Q/V LoRA injection, freezing, state extraction, and parameter counting.
- `src/ftrec/training/objectives.py`: sampled binary next-item objective.
- `src/ftrec/training/checkpoint.py`: fingerprinted checkpoint save/load.
- `src/ftrec/training/engine.py`: optimizer construction, epochs, validation, early stopping, and common train loop.
- `src/ftrec/training/pcgrad.py`: dense/sparse gradient algebra, PCGrad, gradient groups, and logging.
- `src/ftrec/training/pretrain.py`: Single, Joint, and PCGrad orchestration.
- `src/ftrec/training/adapt.py`: domain LoRA and FullFT orchestration.
- `src/ftrec/evaluation/metrics.py`: HR@10/NDCG@10 accumulators.
- `src/ftrec/evaluation/ranking.py`: sampled and chunked full-catalog evaluation.
- `src/ftrec/analysis/results.py`: canonical result rows and aggregation.
- `src/ftrec/analysis/recovery.py`: Recovery calculations and warnings.
- `src/ftrec/analysis/plotting.py`: four required figure families.
- `src/ftrec/cli/*.py`: preprocess, pretrain, adapt, analyze, and smoke entry points.
- `src/ftrec/smoke.py`: deterministic synthetic data generator and complete smoke orchestration.
- `configs/**/*.yaml`: production and smoke configurations.
- `scripts/*.sh`: resumable server stage launchers.
- `tests/`: unit, integration, and end-to-end tests mirroring package modules.
- `docs/environment.md`, `docs/experiments.md`: server setup and experiment operation guide.

---

### Task 1: Package Foundation, Environment, and Legacy Isolation

**Files:**
- Create: `pyproject.toml`
- Create: `environment.yml`
- Create: `requirements-cu130.txt`
- Create: `src/ftrec/__init__.py`
- Create: `tests/test_package.py`
- Move: `main.py` to `legacy/pmixer_sasrec/main.py`
- Move: `model.py` to `legacy/pmixer_sasrec/model.py`
- Move: `utils.py` to `legacy/pmixer_sasrec/utils.py`
- Create: `legacy/pmixer_sasrec/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Python 3.12 and an installed compatible PyTorch.
- Produces: importable `ftrec`, version string, stable dependency declarations, and preserved legacy source.

- [ ] **Step 1: Write the failing package import test**

```python
def test_package_exposes_version() -> None:
    import ftrec
    assert ftrec.__version__ == "0.1.0"
```

- [ ] **Step 2: Verify the package test fails**

Run: `python -m pytest tests/test_package.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'ftrec'`.

- [ ] **Step 3: Add package metadata and editable installation**

Configure setuptools with `package-dir = {"" = "src"}` and dependencies `numpy>=1.26`, `pyarrow>=14`, `PyYAML>=6`, `matplotlib>=3.8`, `torch>=2.9,<2.14`, and `tqdm>=4.66`. Add a `test` optional dependency containing `pytest>=8`.

Run: `python -m pip install -e . --no-deps`

Expected: installation succeeds without replacing local Torch 2.9.1+cpu.

- [ ] **Step 4: Preserve legacy code and document provenance**

Move the three root files with `git mv`. The legacy README must state that the code derives from the pmixer PyTorch SASRec implementation, is preserved only for comparison, and is not imported by `ftrec`.

- [ ] **Step 5: Define server environment files**

`environment.yml` creates `ftrec` with Python 3.12 and pip. `requirements-cu130.txt` uses PyPI as the primary source, adds the official CUDA 13.0 wheel source with `--extra-index-url https://download.pytorch.org/whl/cu130`, and pins `torch==2.13.0+cu130`; it also pins compatible major/minor ranges for the project dependencies. This keeps PyArrow, PyYAML, Matplotlib, and pytest resolving from PyPI while making the required Torch build unambiguous.

- [ ] **Step 6: Ignore generated data and run artifacts**

Preserve the existing `Dataset` rule and add `/data/processed/`, `/runs/`, `/results/`, `*.sqlite`, `*.sqlite-wal`, and `*.sqlite-shm`.

- [ ] **Step 7: Run package checks**

Run: `python -m pytest tests/test_package.py -v`

Expected: PASS.

Run: `python -c "import ftrec, torch; print(ftrec.__version__, torch.__version__)"`

Expected: prints `0.1.0` and the current Torch version.

- [ ] **Step 8: Commit the foundation**

```bash
git add pyproject.toml environment.yml requirements-cu130.txt src/ftrec tests/test_package.py legacy .gitignore
git commit -m "build: establish FTRec experiment package"
```

---

### Task 2: Typed Configuration, Reproducibility, and Atomic Artifacts

**Files:**
- Create: `src/ftrec/config.py`
- Create: `src/ftrec/reproducibility.py`
- Create: `src/ftrec/artifacts.py`
- Create: `tests/test_config.py`
- Create: `tests/test_reproducibility.py`
- Create: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: YAML paths, dotted CLI overrides, seeds, and target directories.
- Produces: `load_config(path, overrides) -> dict`, `canonical_hash(value) -> str`, `seed_everything(seed, deterministic)`, `resolve_device(name)`, `RunDirectory`, and deterministic JSON writers.

- [ ] **Step 1: Write failing tests for recursive config merge and stable hashes**

```python
def test_overrides_merge_recursively_and_hash_canonically(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, {"model": {"hidden_size": 64, "dropout": 0.2}})
    cfg = load_config(path, ["model.dropout=0.0", "seed=42"])
    assert cfg["model"] == {"hidden_size": 64, "dropout": 0.0}
    assert canonical_hash(cfg) == canonical_hash(dict(reversed(list(cfg.items()))))
```

- [ ] **Step 2: Run the config test and verify it fails**

Run: `python -m pytest tests/test_config.py::test_overrides_merge_recursively_and_hash_canonically -v`

Expected: FAIL importing `ftrec.config`.

- [ ] **Step 3: Implement YAML loading, scalar parsing, validation helpers, and SHA-256 canonical hashes**

Serialize canonical values as sorted compact JSON with normalized `Path` strings. Reject unknown override paths and non-finite floats.

- [ ] **Step 4: Write failing deterministic-seed and atomic-publication tests**

```python
def test_seed_everything_repeats_torch_and_numpy() -> None:
    seed_everything(42, deterministic=True)
    first = (torch.rand(3), np.random.rand(3))
    seed_everything(42, deterministic=True)
    second = (torch.rand(3), np.random.rand(3))
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_run_directory_publishes_only_after_complete(tmp_path: Path) -> None:
    final = tmp_path / "run"
    with RunDirectory(final) as run:
        run.write_json("metrics.json", {"value": 1})
        assert not final.exists()
        run.complete({"config_hash": "abc"})
    assert (final / "COMPLETE.json").is_file()
```

- [ ] **Step 5: Implement deterministic runtime setup and artifact lifecycle**

Seed Python, NumPy, Torch CPU/CUDA, and DataLoader workers. `RunDirectory` writes to a unique sibling staging directory, refuses an existing complete run unless forced, rolls back an exact-target replacement on failure, and publishes with `os.replace` only after `complete()`.

- [ ] **Step 6: Run and commit foundation utilities**

Run: `python -m pytest tests/test_config.py tests/test_reproducibility.py tests/test_artifacts.py -v`

Expected: all tests PASS.

```bash
git add src/ftrec/config.py src/ftrec/reproducibility.py src/ftrec/artifacts.py tests/test_config.py tests/test_reproducibility.py tests/test_artifacts.py
git commit -m "feat: add reproducible experiment foundations"
```

---

### Task 3: Streaming Amazon Ingestion, Deduplication, and Joint k-Core

**Files:**
- Create: `src/ftrec/data/__init__.py`
- Create: `src/ftrec/data/amazon.py`
- Create: `src/ftrec/data/preprocessing.py`
- Create: `tests/data_helpers.py`
- Create: `tests/test_preprocessing_core.py`

**Interfaces:**
- Consumes: five gzip CSV files and `PreprocessSettings(input_dir, output_dir, min_user_interactions, min_item_interactions, batch_size, sqlite_path, force)`.
- Produces: `ingest_amazon5(conn, settings) -> IngestReport`, `run_joint_k_core(conn, user_min, item_min) -> KCoreReport`, and a converged SQLite `interactions` table.

- [ ] **Step 1: Write a failing earliest-duplicate ingestion test**

```python
def test_ingest_keeps_earliest_namespaced_interaction(tmp_path: Path) -> None:
    write_amazon5_fixture(tmp_path, duplicate=("Health", "u1", "a", 3000, 1000))
    with open_database(tmp_path / "stage.sqlite") as conn:
        report = ingest_amazon5(conn, fixture_settings(tmp_path))
        row = conn.execute(
            "SELECT timestamp FROM interactions WHERE user_raw='u1' AND domain_id=0 AND item_raw='a'"
        ).fetchone()
    assert row == (1000,)
    assert report.duplicate_rows == 1
```

- [ ] **Step 2: Verify the ingestion test fails**

Run: `python -m pytest tests/test_preprocessing_core.py::test_ingest_keeps_earliest_namespaced_interaction -v`

Expected: FAIL importing preprocessing functions.

- [ ] **Step 3: Implement fixed domains, header validation, streaming UPSERT, and invalid-row counters**

Use primary key `(user_raw, domain_id, item_raw)`, retain minimum `(timestamp, source_ordinal)`, parse no rating semantics, commit in bounded batches, and create degree indexes after ingestion.

- [ ] **Step 4: Write a failing multi-round joint k-core test**

```python
def test_joint_k_core_recomputes_both_sides_until_stable(sqlite_graph) -> None:
    insert_edges(sqlite_graph, cascading_fixture_edges())
    report = run_joint_k_core(sqlite_graph, user_min=2, item_min=2)
    assert report.iterations[-1].deleted_edges == 0
    assert report.rounds >= 2
    assert minimum_user_degree(sqlite_graph) >= 2
    assert minimum_item_degree(sqlite_graph) >= 2
```

- [ ] **Step 5: Implement simultaneous per-round low-user/low-item deletion and invariant queries**

Materialize both low-node sets from the same pre-deletion graph, delete their incident edges in one statement, record every round, and stop only on zero deleted edges.

- [ ] **Step 6: Cover malformed rows, missing columns, ASIN collisions across domains, and empty convergence**

Add explicit tests asserting per-reason counters and error messages naming the offending file/column.

- [ ] **Step 7: Run and commit preprocessing core**

Run: `python -m pytest tests/test_preprocessing_core.py -v`

Expected: all tests PASS.

```bash
git add src/ftrec/data tests/data_helpers.py tests/test_preprocessing_core.py
git commit -m "feat: preprocess Amazon interactions with joint k-core"
```

---

### Task 4: Deterministic Processed Dataset Export and Validation

**Files:**
- Modify: `src/ftrec/data/preprocessing.py`
- Create: `tests/test_preprocessing_export.py`
- Create: `configs/data/amazon5.yaml`
- Create: `src/ftrec/cli/__init__.py`
- Create: `src/ftrec/cli/preprocess.py`

**Interfaces:**
- Consumes: converged SQLite interactions and ingest/k-core reports.
- Produces: `preprocess_amazon5(settings) -> PreprocessResult`, `validate_processed_dataset(path) -> ValidationReport`, Parquet/JSONL/CSV artifacts, and `ftrec-preprocess`.

- [ ] **Step 1: Write failing deterministic mapping and split tests**

```python
def test_export_maps_and_splits_deterministically(filtered_database, tmp_path: Path) -> None:
    first = export_processed_dataset(filtered_database, tmp_path / "a", fixture_reports())
    second = export_processed_dataset(filtered_database, tmp_path / "b", fixture_reports())
    assert first.artifact_hashes == second.artifact_hashes
    rows = read_interactions(tmp_path / "a" / "interactions.parquet")
    assert [row["position"] for row in rows if row["user_id"] == 1] == list(range(user_length(rows, 1)))
    assert splits_for(rows, 1)[-2:] == ["valid", "test"]
```

- [ ] **Step 2: Verify export tests fail**

Run: `python -m pytest tests/test_preprocessing_export.py::test_export_maps_and_splits_deterministically -v`

Expected: FAIL importing `export_processed_dataset`.

- [ ] **Step 3: Implement stable maps, sorted streaming export, and deterministic compression**

Map users lexically from 1, map items by `(domain_id, parent_asin)` from 1, sort rows by `(user_id, timestamp, domain_id, parent_asin)`, and assign train/valid/test. Write fixed-schema Parquet, deterministic gzip JSONL, deterministic gzip CSV maps, domain statistics, 25 overlap rows, summary, and manifest hashes.

- [ ] **Step 4: Write failing cross-artifact validator tests**

```python
def test_validator_detects_hash_and_split_corruption(processed_fixture: Path) -> None:
    assert validate_processed_dataset(processed_fixture).ok
    corrupt_domain_stats(processed_fixture)
    report = validate_processed_dataset(processed_fixture)
    assert not report.ok
    assert any("SHA-256" in error for error in report.errors)
```

- [ ] **Step 5: Implement streaming validation and atomic publication**

Check IDs, degrees, uniqueness, sorted positions, exactly one valid/test per user, cross-file counts, sequence JSONL equality, domain stats, overlap symmetry/diagonal, and every manifest hash. Publish only validated staging output; exact target replacement requires `force=True`.

- [ ] **Step 6: Add CLI and production data config**

Expose input/output, thresholds, batch size, SQLite path, and force options. `--validate-only` validates an existing output and returns nonzero on failure.

- [ ] **Step 7: Run and commit exports**

Run: `python -m pytest tests/test_preprocessing_core.py tests/test_preprocessing_export.py -v`

Expected: all tests PASS.

```bash
git add src/ftrec/data/preprocessing.py src/ftrec/cli configs/data tests/test_preprocessing_export.py
git commit -m "feat: export validated mixed-domain datasets"
```

---

### Task 5: Sequence Datasets, Target Tasks, Negatives, and Batch Manifests

**Files:**
- Create: `src/ftrec/data/datasets.py`
- Create: `src/ftrec/data/sampling.py`
- Create: `tests/test_datasets.py`
- Create: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `sequences.jsonl.gz`, item map, split, maxlen, domain, seed, and batch size.
- Produces: `SequenceStore`, `TargetExample`, `build_mixed_examples`, `build_single_domain_examples`, `SameDomainNegativeSampler`, and `BalancedBatchManifest`.

- [ ] **Step 1: Write failing mixed-versus-single context tests**

```python
def test_target_domain_uses_mixed_context_but_single_filters_context(sequence_store) -> None:
    mixed = build_mixed_examples(sequence_store, split="train", target_domain=1, maxlen=4)
    single = build_single_domain_examples(sequence_store, split="train", domain=1, maxlen=4)
    assert any(0 in example.context_domains for example in mixed)
    assert all(set(example.context_domains) <= {1} for example in single)
    assert all(example.target_domain == 1 for example in mixed + single)
```

- [ ] **Step 2: Verify dataset tests fail**

Run: `python -m pytest tests/test_datasets.py::test_target_domain_uses_mixed_context_but_single_filters_context -v`

Expected: FAIL importing dataset builders.

- [ ] **Step 3: Implement lazy sequence loading and target-position indexing**

Build examples only from allowed training targets, left-pad context with item 0, retain seen-item sets for evaluation/negative rejection, and expose stable integer example IDs.

- [ ] **Step 4: Write failing negative and balanced-plan tests**

```python
def test_negative_is_unseen_and_from_target_domain(item_catalog, example) -> None:
    sampler = SameDomainNegativeSampler(item_catalog, seed=42)
    negative = sampler.sample(example)
    assert item_catalog.domain_of(negative) == example.target_domain
    assert negative not in example.seen_items


def test_joint_and_pcgrad_batch_manifests_are_byte_identical(domain_examples, tmp_path: Path) -> None:
    first = BalancedBatchManifest.create(domain_examples, batch_size=2, steps=3, seed=42)
    first.write(tmp_path / "joint.json")
    first.write(tmp_path / "pcgrad.json")
    assert (tmp_path / "joint.json").read_bytes() == (tmp_path / "pcgrad.json").read_bytes()
```

- [ ] **Step 5: Implement deterministic cycling domain batches and persisted manifests**

Each step contains one equal-sized micro-batch per domain. Smaller domains reshuffle deterministically at wraparound. Reject a domain with zero examples and report its name.

- [ ] **Step 6: Run and commit data-task construction**

Run: `python -m pytest tests/test_datasets.py tests/test_sampling.py -v`

Expected: all tests PASS.

```bash
git add src/ftrec/data/datasets.py src/ftrec/data/sampling.py tests/test_datasets.py tests/test_sampling.py
git commit -m "feat: construct deterministic domain training tasks"
```

---

### Task 6: Explicit-Q/K/V Vanilla SASRec

**Files:**
- Create: `src/ftrec/models/__init__.py`
- Create: `src/ftrec/models/attention.py`
- Create: `src/ftrec/models/sasrec.py`
- Create: `tests/test_attention.py`
- Create: `tests/test_sasrec.py`

**Interfaces:**
- Consumes: padded item tensors `[batch, length]` and `SASRecConfig`.
- Produces: `CausalSelfAttention`, `SASRecBlock`, `SASRec.encode(items)`, `SASRec.score(items, candidates)`, `initialize_sasrec`, and named dense/sparse parameter groups.

- [ ] **Step 1: Write failing causal and padding-mask tests**

```python
def test_future_and_padding_tokens_cannot_change_previous_states() -> None:
    attention = deterministic_attention(hidden_size=4, heads=1)
    base = torch.tensor([[0, 1, 2, 3]])
    changed = torch.tensor([[0, 1, 2, 4]])
    base_states = encode_ids(attention, base)
    changed_states = encode_ids(attention, changed)
    torch.testing.assert_close(base_states[:, :3], changed_states[:, :3])
    assert torch.count_nonzero(base_states[:, 0]) == 0
```

- [ ] **Step 2: Verify attention tests fail**

Run: `python -m pytest tests/test_attention.py::test_future_and_padding_tokens_cannot_change_previous_states -v`

Expected: FAIL importing `CausalSelfAttention`.

- [ ] **Step 3: Implement explicit projections with scaled-dot-product attention**

Use batch-first tensors, reshape to heads, combine a causal allow-mask with key padding, use configured dropout only during training, zero padded query outputs after attention, then apply `out_proj`.

- [ ] **Step 4: Write failing model shape, initialization, and tied-scoring tests**

```python
def test_sasrec_shapes_padding_and_tied_item_weights() -> None:
    model = tiny_sasrec(num_items=20)
    items = torch.tensor([[0, 0, 2, 3], [0, 4, 5, 6]])
    states = model.encode(items)
    assert states.shape == (2, 4, 8)
    assert torch.count_nonzero(states[items == 0]) == 0
    assert model.item_embedding.weight.data_ptr() == model.scoring_weight().data_ptr()
    assert torch.count_nonzero(model.item_embedding.weight[0]) == 0
```

- [ ] **Step 5: Implement embeddings, pre-norm blocks, ReLU point-wise FFN, final norm, and last-state scoring**

Positions start at 1 only for non-padding tokens. Use hidden-to-hidden FFN layers. `parameter_groups()` returns sparse item embedding separately from dense parameters and categorizes block attention/FFN groups for gradient logs.

- [ ] **Step 6: Run model tests on CPU in float32**

Run: `python -m pytest tests/test_attention.py tests/test_sasrec.py -v`

Expected: all tests PASS with no CUDA requirement.

- [ ] **Step 7: Commit the model**

```bash
git add src/ftrec/models tests/test_attention.py tests/test_sasrec.py
git commit -m "feat: implement explicit-projection SASRec"
```

---

### Task 7: Objective and Deterministic Ranking Evaluation

**Files:**
- Create: `src/ftrec/training/__init__.py`
- Create: `src/ftrec/training/objectives.py`
- Create: `src/ftrec/evaluation/__init__.py`
- Create: `src/ftrec/evaluation/metrics.py`
- Create: `src/ftrec/evaluation/ranking.py`
- Create: `tests/test_objectives.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_ranking.py`

**Interfaces:**
- Consumes: SASRec scores, positive/negative IDs, evaluation examples, candidate catalogs, and protocol config.
- Produces: `sampled_bce_loss`, `RankingMetrics`, `rank_ground_truth_chunked`, and `evaluate_model`.

- [ ] **Step 1: Write failing objective and metric tests**

```python
def test_sampled_bce_matches_manual_logits() -> None:
    positive = torch.tensor([2.0, 0.0])
    negative = torch.tensor([-1.0, 1.0])
    expected = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
    expected += F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
    torch.testing.assert_close(sampled_bce_loss(positive, negative), expected)


@pytest.mark.parametrize("rank,hr,ndcg", [(0, 1.0, 1.0), (9, 1.0, 1 / math.log2(11)), (10, 0.0, 0.0)])
def test_top10_metrics(rank, hr, ndcg):
    assert metrics_for_rank(rank, 10) == pytest.approx((hr, ndcg))
```

- [ ] **Step 2: Verify tests fail and implement objective/accumulator**

Run the two test files, then implement mean BCE sums and an accumulator tracking users, skipped users, HR, and NDCG.

- [ ] **Step 3: Write a failing chunked rank/tie/exclusion test**

```python
def test_chunked_full_rank_excludes_seen_and_breaks_ties_by_item_id() -> None:
    scores = {1: 0.5, 2: 0.9, 3: 0.5, 4: 0.4}
    rank = rank_fixture(scores, target=3, seen={2}, chunk_size=2)
    assert rank == 1  # item 1 ties and wins by smaller ID; seen item 2 is excluded
```

- [ ] **Step 4: Implement full-catalog and persisted sampled-candidate protocols**

Compute rank without a global sort: count eligible scores greater than target plus equal-score candidates with smaller item ID. Chunk candidate scoring on the active device. Sampled candidates are serialized and fingerprinted; protocol names are mandatory in result rows.

Add an integration test in which mixed evaluation and Single evaluation receive the same target user IDs, while Single reports users with no prior target-domain history through `num_skipped_users` instead of silently changing the target cohort.

- [ ] **Step 5: Run and commit evaluation**

Run: `python -m pytest tests/test_objectives.py tests/test_metrics.py tests/test_ranking.py -v`

Expected: all tests PASS.

```bash
git add src/ftrec/training src/ftrec/evaluation tests/test_objectives.py tests/test_metrics.py tests/test_ranking.py
git commit -m "feat: add next-item objective and ranking evaluation"
```

---

### Task 8: Sparse-Aware Gradient Algebra, PCGrad, and Conflict Logs

**Files:**
- Create: `src/ftrec/training/pcgrad.py`
- Create: `tests/test_pcgrad.py`
- Create: `tests/test_gradient_logging.py`

**Interfaces:**
- Consumes: ordered model parameters, five task losses, parameter-group names, seed, and global step.
- Produces: `TaskGradients`, `gradient_dot`, `gradient_norm`, `project_pcgrad`, `assign_mean_gradients`, `GradientConflictLogger`, and raw/projected cosine records.

- [ ] **Step 1: Write failing dense PCGrad geometry tests**

```python
def test_pcgrad_removes_negative_component_against_original_peer() -> None:
    task_a = gradients(torch.tensor([1.0, -1.0]))
    task_b = gradients(torch.tensor([-1.0, 0.0]))
    projected = project_pcgrad([task_a, task_b], seed=7, step=0)
    assert gradient_dot(projected[0], task_b) >= -1e-7
```

- [ ] **Step 2: Verify dense PCGrad test fails**

Run: `python -m pytest tests/test_pcgrad.py::test_pcgrad_removes_negative_component_against_original_peer -v`

Expected: FAIL importing PCGrad functions.

- [ ] **Step 3: Implement parameter-wise dense gradient algebra and standard projection**

Treat `None` as zero. Clone every task gradient. For each task use a deterministic random permutation of original peers, project only on negative dot, retain original peers unchanged, and average projected tasks after all projections.

- [ ] **Step 4: Write failing sparse embedding tests**

```python
def test_sparse_projection_stays_sparse_and_matches_dense_reference() -> None:
    sparse_a, sparse_b = sparse_gradient_pair()
    result = project_pcgrad([gradients(sparse_a), gradients(sparse_b)], seed=3, step=2)
    assert result[0].values[0].is_sparse
    torch.testing.assert_close(result[0].values[0].to_dense(), dense_reference_projection(sparse_a, sparse_b))
```

- [ ] **Step 5: Implement coalesced sparse dot, norm, add, scale, and mean without whole-matrix densification**

Intersect sparse row indices for dot products; merge row indices for linear combinations; coalesce after every operation. Tests monkeypatch `Tensor.to_dense` to raise inside production PCGrad, proving it is used only by the test reference.

- [ ] **Step 6: Write and implement gradient-group cosine logging**

Test exact 5×5 symmetry, unit diagonal for nonzero tasks, NaN marker for both-zero groups, negative-pair ratio over the ten unique off-diagonal pairs, and group names `full`, `item_embedding`, `position_embedding`, `block_0_attention`, and `block_0_ffn`.

- [ ] **Step 7: Run and commit PCGrad**

Run: `python -m pytest tests/test_pcgrad.py tests/test_gradient_logging.py -v`

Expected: all tests PASS and sparse tests never densify full embeddings.

```bash
git add src/ftrec/training/pcgrad.py tests/test_pcgrad.py tests/test_gradient_logging.py
git commit -m "feat: implement sparse-aware PCGrad diagnostics"
```

---

### Task 9: Checkpoints and Shared Training Engine

**Files:**
- Create: `src/ftrec/training/checkpoint.py`
- Create: `src/ftrec/training/engine.py`
- Create: `tests/test_checkpoint.py`
- Create: `tests/test_training_engine.py`

**Interfaces:**
- Consumes: model, dense/sparse parameter groups, data/config fingerprints, optimizer state, epochs, and validation callback.
- Produces: `save_checkpoint`, `load_checkpoint`, `build_optimizers`, `TrainingState`, `train_epochs`, early stopping, `best.pt`, and `last.pt`.

- [ ] **Step 1: Write a failing checkpoint fingerprint round-trip test**

```python
def test_checkpoint_rejects_wrong_data_fingerprint(tmp_path: Path) -> None:
    path = save_fixture_checkpoint(tmp_path, data_hash="data-a")
    with pytest.raises(CheckpointMismatchError, match="data_hash"):
        load_checkpoint(path, tiny_sasrec(), expected={"data_hash": "data-b"})
```

- [ ] **Step 2: Implement CPU-portable checkpoints with RNG and optimizer states**

Store schema version, plain model state dict, optimizer/scheduler states, epoch/step, best metric, resolved config hash, data hash, initialization hash, and Python/NumPy/Torch RNG state. Load with explicit `map_location` and strict keys.

- [ ] **Step 3: Write failing optimizer and early-stop tests**

```python
def test_engine_uses_sparseadam_for_item_embedding_and_adamw_for_dense() -> None:
    optimizers = build_optimizers(tiny_sasrec(), optimizer_settings())
    assert isinstance(optimizers.sparse, torch.optim.SparseAdam)
    assert isinstance(optimizers.dense, torch.optim.AdamW)


def test_best_checkpoint_uses_validation_ndcg_only(tmp_path: Path) -> None:
    history = run_scripted_metrics(tmp_path, validation=[0.2, 0.4, 0.3], test=[0.9, 0.1, 1.0])
    assert history.best_epoch == 2
```

- [ ] **Step 4: Implement AMP/device handling, post-aggregation clipping, validation, and early stopping**

Use BF16 autocast only when configured and supported. Keep parameters/grads FP32. Clip dense and sparse aggregate gradients by the same configured global norm after Joint averaging or PCGrad projection. Never invoke test evaluation in the epoch-selection loop.

- [ ] **Step 5: Run and commit engine**

Run: `python -m pytest tests/test_checkpoint.py tests/test_training_engine.py -v`

Expected: all tests PASS.

```bash
git add src/ftrec/training/checkpoint.py src/ftrec/training/engine.py tests/test_checkpoint.py tests/test_training_engine.py
git commit -m "feat: add recoverable shared training engine"
```

---

### Task 10: Single, Joint, and PCGrad Pretraining

**Files:**
- Create: `src/ftrec/training/pretrain.py`
- Create: `tests/test_pretrain.py`
- Create: `src/ftrec/cli/pretrain.py`
- Create: `configs/model/sasrec.yaml`
- Create: `configs/experiment/single.yaml`
- Create: `configs/experiment/joint.yaml`
- Create: `configs/experiment/pcgrad.yaml`

**Interfaces:**
- Consumes: processed dataset, training config, batch manifest, initialization checkpoint, seed, and method.
- Produces: `train_single_domain`, `train_multidomain(method='joint'|'pcgrad')`, per-domain loss histories, conflict logs, and pretrained checkpoints.

- [ ] **Step 1: Write a failing Joint/PCGrad fairness test**

```python
def test_joint_and_pcgrad_share_initialization_and_batches(pretrain_fixture) -> None:
    joint = prepare_multidomain_run(pretrain_fixture, method="joint", seed=42)
    pcgrad = prepare_multidomain_run(pretrain_fixture, method="pcgrad", seed=42)
    assert joint.initialization_hash == pcgrad.initialization_hash
    assert joint.batch_manifest.read_bytes() == pcgrad.batch_manifest.read_bytes()
```

- [ ] **Step 2: Implement seed-specific initialization checkpoint and five Single runs**

Single models use domain-filtered examples and item candidates while retaining global item IDs. Save per-domain best/last checkpoints and valid/test cohort counts.

- [ ] **Step 3: Write a failing one-step gradient-behavior test**

```python
def test_joint_averages_raw_tasks_and_pcgrad_projects_only_conflicts(pretrain_fixture) -> None:
    joint_step = run_one_step(pretrain_fixture, method="joint")
    pcgrad_step = run_one_step(pretrain_fixture, method="pcgrad")
    assert joint_step.domain_losses.keys() == set(DOMAIN_NAMES)
    assert pcgrad_step.domain_losses == pytest.approx(joint_step.domain_losses)
    assert pcgrad_step.raw_cosines == pytest.approx(joint_step.raw_cosines)
    assert pcgrad_step.projected_conflict_count > 0
```

- [ ] **Step 4: Implement Joint and PCGrad step functions through the shared engine**

Run five independent forward/loss/`autograd.grad` calls from the same parameter state. Joint averages raw task gradients; PCGrad projects then averages. Both clip and step the same optimizer groups, schedule, and logging cadence.

- [ ] **Step 5: Implement pretrain CLI, dry-run, resume, and configs**

Support `single`, `joint`, and `pcgrad`; a domain is required only for Single. Dry-run prints resolved run IDs, sample counts, config/data hashes, and skip/resume decisions without creating checkpoints.

- [ ] **Step 6: Run and commit pretraining**

Run: `python -m pytest tests/test_pretrain.py tests/test_training_engine.py tests/test_pcgrad.py -v`

Expected: all tests PASS.

```bash
git add src/ftrec/training/pretrain.py src/ftrec/cli/pretrain.py configs/model configs/experiment tests/test_pretrain.py
git commit -m "feat: train Single Joint and PCGrad SASRec"
```

---

### Task 11: Q/V LoRA and Full Fine-Tuning

**Files:**
- Create: `src/ftrec/models/lora.py`
- Create: `src/ftrec/training/adapt.py`
- Create: `src/ftrec/cli/adapt.py`
- Create: `tests/test_lora.py`
- Create: `tests/test_adapt.py`
- Create: `configs/experiment/lora.yaml`
- Create: `configs/experiment/fullft.yaml`

**Interfaces:**
- Consumes: Joint/PCGrad best checkpoint, domain examples, rank, seed, and adaptation config.
- Produces: `LoRALinear`, `inject_qv_lora`, `freeze_for_lora`, `lora_state_dict`, `train_lora`, `train_fullft`, and adaptation checkpoints/results.

- [ ] **Step 1: Write failing LoRA initialization and equivalence tests**

```python
@pytest.mark.parametrize("rank", [1, 2, 4, 8, 16])
def test_zero_initialized_qv_lora_preserves_base_output_and_trainable_set(rank: int) -> None:
    base = tiny_sasrec().eval()
    adapted = inject_qv_lora(copy.deepcopy(base), rank=rank, alpha=rank).eval()
    torch.testing.assert_close(adapted.encode(INPUTS), base.encode(INPUTS))
    assert all(("q_proj" in name or "v_proj" in name) and ".lora_" in name
               for name, parameter in adapted.named_parameters() if parameter.requires_grad)
```

- [ ] **Step 2: Implement `LoRALinear` and Q/V-only module replacement**

Keep the frozen base `nn.Linear`, initialize A with Kaiming and B with zero, use scale `alpha/rank`, and expose only adapter tensors in adapter checkpoints.

- [ ] **Step 3: Write failing rank-count and fingerprint tests**

```python
def test_qv_lora_parameter_count_is_exact_and_monotonic() -> None:
    counts = [count_trainable(inject_qv_lora(tiny_sasrec(blocks=2, hidden=8), r, r)) for r in RANKS]
    assert counts == [4 * 2 * 8 * rank for rank in RANKS]
    assert counts == sorted(counts)


def test_adapter_rejects_wrong_base_checkpoint(tmp_path: Path) -> None:
    adapter = save_adapter(tmp_path, base_hash="base-a")
    with pytest.raises(CheckpointMismatchError, match="base_hash"):
        load_adapter(adapter, base_hash="base-b")
```

After one optimizer step, compare every frozen base tensor with a pre-step clone and assert byte equality; assert at least one LoRA B tensor changed. This proves training does not merely expose the correct parameter names but actually preserves the base checkpoint.

- [ ] **Step 4: Implement domain adaptation and FullFT through the shared engine**

LoRA asserts the exact trainable-name set before training. FullFT asserts every expected base parameter is trainable. Both use identical domain examples/cohorts and validation selection, with separate configured learning rates.

- [ ] **Step 5: Implement rank/domain/method matrix CLI with resume and dry-run**

Default LoRA matrix is 2 pretrain methods × 5 domains × 5 ranks. Default FullFT matrix is 2 × 5 domains. A single combination can be selected for debugging without changing run IDs.

- [ ] **Step 6: Run and commit adaptation**

Run: `python -m pytest tests/test_lora.py tests/test_adapt.py -v`

Expected: all tests PASS.

```bash
git add src/ftrec/models/lora.py src/ftrec/training/adapt.py src/ftrec/cli/adapt.py configs/experiment/lora.yaml configs/experiment/fullft.yaml tests/test_lora.py tests/test_adapt.py
git commit -m "feat: add QV LoRA and full fine-tuning"
```

---

### Task 12: Result Schema, Recovery, Aggregation, and Figures

**Files:**
- Create: `src/ftrec/analysis/__init__.py`
- Create: `src/ftrec/analysis/results.py`
- Create: `src/ftrec/analysis/recovery.py`
- Create: `src/ftrec/analysis/plotting.py`
- Create: `src/ftrec/cli/analyze.py`
- Create: `tests/test_results.py`
- Create: `tests/test_recovery.py`
- Create: `tests/test_plotting.py`

**Interfaces:**
- Consumes: completed run metrics and gradient logs.
- Produces: canonical `ResultRow`, `results.csv`, `summary.csv`, `recovery.csv`, warnings JSON, and Figures 1–4 as PNG/PDF.

- [ ] **Step 1: Write failing result validation and aggregation tests**

```python
def test_result_row_requires_protocol_and_parameter_counts() -> None:
    with pytest.raises(ResultSchemaError):
        ResultRow.from_dict({"seed": 42, "domain": "Health", "NDCG@10": 0.2})


def test_aggregation_reports_sample_std_over_seeds() -> None:
    summary = aggregate_results(metric_rows([0.2, 0.3, 0.4]))
    assert summary.mean == pytest.approx(0.3)
    assert summary.std == pytest.approx(0.1)
```

- [ ] **Step 2: Implement strict result rows, deduplication keys, CSV ordering, and macro rows**

Reject duplicate `(seed, domain, pretrain_method, adapt_method, lora_rank, split, evaluation_protocol)` keys. Macro averages exclude domains with zero evaluable users and record the contributing domain count.

- [ ] **Step 3: Write failing Recovery edge-case tests**

```python
def test_recovery_is_signed_unclipped_and_marks_zero_denominator() -> None:
    assert recovery(pretrain=0.2, lora=0.5, fullft=0.4).value == pytest.approx(1.5)
    undefined = recovery(pretrain=0.2, lora=0.3, fullft=0.2)
    assert math.isnan(undefined.value)
    assert undefined.undefined_recovery
```

- [ ] **Step 4: Implement HR/NDCG Recovery for every domain/rank/method/seed and analysis warnings**

Warn on undefined denominators, FullFT below pretrain, missing ranks/seeds, mixed evaluation protocols, and gains fully explained by pretrain gaps.

- [ ] **Step 5: Write failing figure-generation test**

```python
def test_analysis_generates_all_required_figures(complete_result_fixture, tmp_path: Path) -> None:
    outputs = generate_figures(complete_result_fixture, tmp_path)
    assert {path.stem for path in outputs} >= {
        "figure1_pretraining", "figure2_lora_rank", "figure3_recovery", "figure4_gradient_conflict"
    }
    assert all(path.stat().st_size > 1000 for path in outputs)
```

- [ ] **Step 6: Implement deterministic headless plots and analyze CLI**

Use Matplotlib Agg, fixed styles/domain colors, shared rank ticks, FullFT horizontal lines, labeled heatmaps, and both PNG/PDF output. Read numbers only from canonical structured files.

- [ ] **Step 7: Run and commit analysis**

Run: `python -m pytest tests/test_results.py tests/test_recovery.py tests/test_plotting.py -v`

Expected: all tests PASS.

```bash
git add src/ftrec/analysis src/ftrec/cli/analyze.py tests/test_results.py tests/test_recovery.py tests/test_plotting.py
git commit -m "feat: analyze recovery and gradient conflicts"
```

---

### Task 13: Complete Synthetic CPU Smoke Orchestration

**Files:**
- Create: `src/ftrec/smoke.py`
- Create: `src/ftrec/cli/smoke.py`
- Create: `configs/smoke.yaml`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: one smoke config and output root.
- Produces: deterministic synthetic raw files, all pipeline runs, canonical results, Recovery, gradients, figures, and `SmokeReport`.

- [ ] **Step 1: Write the failing end-to-end smoke contract test**

```python
def test_complete_smoke_runs_every_experiment_branch(tmp_path: Path) -> None:
    report = run_smoke(tmp_path / "smoke", seed=42)
    assert report.ok
    assert report.completed == {
        "preprocess": 1, "single": 5, "joint": 1, "pcgrad": 1,
        "lora": 50, "fullft": 10, "analysis": 1,
    }
    assert report.lora_ranks == (1, 2, 4, 8, 16)
    assert report.figure_count >= 8  # PNG and PDF for four families
    assert report.evaluation_protocols == {"sampled"}
```

- [ ] **Step 2: Verify smoke test fails before orchestrator exists**

Run: `python -m pytest tests/test_smoke.py::test_complete_smoke_runs_every_experiment_branch -v`

Expected: FAIL importing `run_smoke`.

- [ ] **Step 3: Implement a deterministic five-domain synthetic raw generator**

Include overlapping users, domain-specific items, duplicate user-item rows, equal timestamps, cascading k-core edges, sufficient train/valid/test targets per domain, and fixed sampled candidates. Assert the fixture itself has expected counts.

- [ ] **Step 4: Compose real stage APIs with tiny CPU settings**

Use hidden size 8, one block/head, maxlen 8, batch size 2, one optimizer step per epoch, one epoch, one seed, all ranks, and no CUDA. Do not add smoke-only model/trainer implementations.

- [ ] **Step 5: Add semantic smoke assertions**

Assert finite losses/metrics, Joint/PCGrad initialization and batch hash equality, raw gradient log presence, Q/V-only LoRA trainables, monotonic rank parameter counts, all-parameter FullFT trainables, checkpoint fingerprint validity, 67 model runs, result uniqueness, Recovery rows, and all figure outputs. Do not assert PCGrad beats Joint.

- [ ] **Step 6: Implement `ftrec-smoke` CLI and rerun determinism**

Two clean smoke runs with the same seed must produce identical data, batch, checkpoint-initialization, candidate, and canonical result structure hashes. Exclude measured wall time and absolute output paths from compared semantic hashes.

- [ ] **Step 7: Run full smoke and test suite**

Run: `python -m pytest tests/test_smoke.py -v`

Expected: PASS within a practical CPU test budget.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit smoke orchestration**

```bash
git add src/ftrec/smoke.py src/ftrec/cli/smoke.py configs/smoke.yaml tests/test_smoke.py
git commit -m "test: add complete CPU experiment smoke run"
```

---

### Task 14: Server Configurations, Resumable Scripts, and Operator Guides

**Files:**
- Create: `scripts/run_preprocess.sh`
- Create: `scripts/run_single.sh`
- Create: `scripts/run_joint.sh`
- Create: `scripts/run_pcgrad.sh`
- Create: `scripts/run_lora.sh`
- Create: `scripts/run_fullft.sh`
- Create: `scripts/run_analysis.sh`
- Create: `scripts/check_environment.sh`
- Create: `docs/environment.md`
- Create: `docs/experiments.md`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: server checkout, Conda, real Amazon files, and stage configs.
- Produces: documented setup, environment diagnosis, dry-run/resume commands, and sequential production workflow.

- [ ] **Step 1: Write failing CLI-help and dry-run tests**

```python
@pytest.mark.parametrize("command", ["ftrec-preprocess", "ftrec-pretrain", "ftrec-adapt", "ftrec-analyze", "ftrec-smoke"])
def test_all_commands_expose_help(command: str) -> None:
    result = subprocess.run([command, "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
```

- [ ] **Step 2: Complete pyproject entry points and make every shell script fail-fast/resumable**

Scripts use `set -euo pipefail`, resolve repository root from script location, accept additional CLI overrides, run a dry-run first when requested, and invoke only one stage. They do not delete completed runs.

- [ ] **Step 3: Implement environment self-check**

Print Python/Torch/Torch-CUDA versions, GPU name, compute capability, driver-visible CUDA, BF16 support, PyArrow version, free disk space, and a one-step CUDA tensor calculation. Exit nonzero if Torch is not 2.13.0+cu130, CUDA is unavailable, GPU is not visible, or BF16 is unsupported.

- [ ] **Step 4: Write exact environment and experiment guides**

Document Conda creation, official cu130 pip installation, driver minimum, editable package install, raw input placement, preprocessing, the seven production stages, resume/dry-run behavior, output schemas, three seeds 42/43/44, BF16, full-catalog evaluation, expected run matrix, and failure diagnostics.

- [ ] **Step 5: Run documentation-facing checks**

Run: `python -m pytest tests/test_cli.py -v`

Expected: all commands return valid help and dry-run output.

Run: `python -m compileall -q src tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: exit code 0.

- [ ] **Step 6: Commit server operation support**

```bash
git add pyproject.toml scripts docs/environment.md docs/experiments.md tests/test_cli.py
git commit -m "docs: add resumable server experiment workflow"
```

---

### Task 15: Final Verification and Scope Audit

**Files:**
- Modify only when a verification failure has a reproduced test: files owned by Tasks 1–14.

**Interfaces:**
- Consumes: complete repository implementation.
- Produces: fresh verification evidence and a clean handoff.

- [ ] **Step 1: Run all automated tests from a clean process**

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 2: Run complete smoke outside pytest**

Run: `ftrec-smoke --config configs/smoke.yaml --output-dir results/smoke-final --force`

Expected: completion report lists preprocessing 1, Single 5, Joint 1, PCGrad 1, LoRA 50, FullFT 10, analysis 1, and no failures.

- [ ] **Step 3: Validate generated artifact contracts**

Run the processed-data validator, checkpoint fingerprint validator, result-schema validator, and analysis command against `results/smoke-final`. Expected: each exits 0; all expected CSV/JSON/checkpoint/PNG/PDF artifacts exist.

- [ ] **Step 4: Audit scientific fairness metadata**

Confirm Joint/PCGrad initialization hashes and batch manifest hashes match, evaluation protocol is sampled only for smoke, LoRA trainable-name lists contain only Q/V adapter names, FullFT trainable counts equal total counts, and Recovery warnings are explicit rather than clipped.

- [ ] **Step 5: Audit scope and repository state**

Run: `git status --short`

Expected: generated `data/`, `runs/`, and `results/` artifacts are ignored; source work is committed; no Amazon full-data training was launched.

- [ ] **Step 6: Prepare handoff evidence**

Report test count, smoke duration, implementation commits, smoke result/figure locations, server environment command, first production command, and explicitly state that scientific conclusions require the three-seed server experiment.
