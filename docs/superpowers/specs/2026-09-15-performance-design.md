# FTRec Performance Optimization Design

## Goal

Reduce preprocessing wall time and remove avoidable CPU, memory, disk, and GPU synchronization overhead from the production experiment matrix without changing dataset semantics, batch determinism, ranking tie-breaking, or checkpoint compatibility.

## Design

Preprocessing remains SQLite-backed and exact, but emits phase/rate progress and buffers Parquet rows into large row groups instead of writing one row group per user. This is a low-risk change that preserves stable mappings and exact joint k-core output while eliminating a confirmed file-layout bottleneck.

Training uses reusable per-domain negative samplers, cached example-id lookups, and one initialization hash per run. Batch plans are represented by a compact deterministic specification and generated epoch-by-epoch, so Joint and PCGrad still consume byte-for-byte equivalent plans without materializing millions of identifiers. Full-catalog ranking batches users, encodes every context once, and scores item chunks from cached states. PCGrad reductions aggregate on-device and transfer one scalar or matrix per reduction rather than one scalar per parameter.

Progress is written once per preprocessing interval and once per training epoch with elapsed time and ETA. Multi-process GPU concurrency remains opt-in and is not enabled by default on a single RTX 4090D.

## Compatibility and correctness

- Existing full manifests remain readable.
- Compact plan generation is deterministic across processes and independent of Python hash randomization.
- Negative samples remain in-domain and unseen; rejection sampling has a deterministic exhaustive fallback for dense users.
- Full ranking preserves the existing score/tie rule and seen-item filtering.
- Output filenames and checkpoint metadata remain compatible.
- CPU smoke remains the acceptance test; targeted benchmarks report before/after evidence.

