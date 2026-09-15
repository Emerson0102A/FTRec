# FTRec Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove confirmed preprocessing and training bottlenecks while preserving deterministic experiment behavior.

**Architecture:** Keep exact SQLite preprocessing, but batch Parquet output and add phase progress. Replace materialized training work with deterministic lazy plans, reusable samplers/lookups, batched ranking, and device-side gradient reductions.

**Tech Stack:** Python 3.12, PyTorch, PyArrow, SQLite, pytest

**Spec:** `docs/superpowers/specs/2026-09-15-performance-design.md`

## Global Constraints

- Preserve dataset semantics, ranking tie-breaking, reproducibility, and checkpoint compatibility.
- Do not enable concurrent GPU experiments by default on one RTX 4090D.
- Every behavioral change follows a failing-test-first cycle.

---

### Task 1: Buffered preprocessing and progress

**Files:**
- Modify: `src/ftrec/data/preprocessing.py`
- Modify: `src/ftrec/cli/preprocess.py`
- Test: `tests/test_preprocessing_export.py`

**Interfaces:**
- Consumes: `PreprocessSettings.batch_size` and the existing user-group iterator.
- Produces: bounded Parquet row-group counts plus structured progress lines.

- [ ] Add a failing export test asserting many users do not create one row group each.
- [ ] Run the test and confirm the old writer fails it.
- [ ] Buffer interaction rows to the configured export batch size and flush final rows.
- [ ] Add phase, row-rate, elapsed, and k-core iteration progress.
- [ ] Run preprocessing tests.

### Task 2: Reusable sampling, lookup caches, and compact batch plans

**Files:**
- Modify: `src/ftrec/data/sampling.py`
- Modify: `src/ftrec/training/pretrain.py`
- Modify: `src/ftrec/training/adapt.py`
- Test: `tests/test_sampling.py`
- Test: `tests/test_pretrain.py`

**Interfaces:**
- Produces: `BalancedBatchPlan.iter_steps(limit=None)`, bounded `to_dict()`, and reusable `SameDomainNegativeSampler` instances.

- [ ] Add failing tests for bounded compact plans, deterministic slices, dense-catalog fallback, and no per-step hash.
- [ ] Confirm failures against current code.
- [ ] Implement deterministic lazy generation and rejection sampling.
- [ ] Cache lookup maps and pass the run initialization hash into step results.
- [ ] Run sampling and pretraining tests.

### Task 3: Batched full-catalog ranking

**Files:**
- Modify: `src/ftrec/evaluation/ranking.py`
- Modify: `src/ftrec/training/pretrain.py`
- Modify: `src/ftrec/training/adapt.py`
- Test: `tests/test_ranking.py`

**Interfaces:**
- Produces: `evaluate_model(..., batch_size=...)` that encodes one batch once and scores catalog chunks from final states.

- [ ] Add a failing counting-model test proving contexts are encoded once per batch.
- [ ] Confirm the old evaluator repeats encoding.
- [ ] Implement batched context encoding and chunk scoring with unchanged exclusions/ties.
- [ ] Run ranking tests and compare old/new literal ranks.

### Task 4: Device-side gradient reductions and epoch progress

**Files:**
- Modify: `src/ftrec/training/pcgrad.py`
- Modify: `src/ftrec/training/engine.py`
- Modify: `src/ftrec/training/pretrain.py`
- Modify: `src/ftrec/training/adapt.py`
- Test: `tests/test_pcgrad.py`
- Test: `tests/test_training_engine.py`

**Interfaces:**
- Produces: one host synchronization per dot/norm or cosine matrix and JSON epoch progress with ETA.

- [ ] Add reduction correctness tests covering dense and sparse gradients.
- [ ] Aggregate tensors before host conversion and batch cosine extraction.
- [ ] Emit epoch elapsed/rate/ETA fields without per-step logging.
- [ ] Run gradient and training tests.

### Task 5: Verification and server documentation

**Files:**
- Modify: `docs/experiments.md`

- [ ] Re-run targeted before/after benchmarks.
- [ ] Run the full pytest suite and complete CPU smoke.
- [ ] Document restart guidance, progress output, manifest format, and recommended single-GPU execution.
- [ ] Commit and push the existing feature branch.
