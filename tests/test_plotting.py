from __future__ import annotations

from pathlib import Path


def test_analysis_generates_all_required_figures(tmp_path: Path) -> None:
    from ftrec.analysis.plotting import generate_figures
    from ftrec.analysis.recovery import RecoveryRow
    from ftrec.analysis.results import ResultRow

    rows = []
    recoveries = []
    for method_index, pretrain in enumerate(("joint", "pcgrad")):
        for domain_index, domain in enumerate(("Health", "Beauty")):
            base = 0.15 + method_index * 0.02 + domain_index * 0.01
            rows.append(
                ResultRow(42, domain, pretrain, "none", None, "test", "sampled", 0.3, base, 2, 0, 100, 100, "base.pt", "c", "d")
            )
            rows.append(
                ResultRow(42, domain, pretrain, "fullft", None, "test", "sampled", 0.6, base + 0.2, 2, 0, 100, 100, "full.pt", "c", "d")
            )
            for rank in (1, 2, 4):
                value = base + rank * 0.02
                rows.append(
                    ResultRow(42, domain, pretrain, "lora", rank, "test", "sampled", 0.4, value, 2, 0, rank * 16, 100, "lora.pt", "c", "d")
                )
                recoveries.append(
                    RecoveryRow(42, domain, pretrain, rank, "NDCG@10", "sampled", base, value, base + 0.2, rank * 0.1, False, "")
                )
    gradients = [
        {
            "domain_names": ["Health", "Beauty"],
            "method": "joint",
            "raw_cosine": {"full": [[1.0, -0.25], [-0.25, 1.0]]},
        }
    ]

    outputs = generate_figures(rows, recoveries, gradients, tmp_path)

    assert {path.stem for path in outputs} >= {
        "figure1_pretraining",
        "figure2_lora_rank",
        "figure3_recovery",
        "figure4_gradient_conflict",
    }
    assert len(outputs) == 8
    assert all(path.stat().st_size > 1000 for path in outputs)


def test_pretraining_figure_includes_context_ablation_methods() -> None:
    """Catch the canonical comparison plot omitting completed ablations."""
    from ftrec.analysis.plotting import _figure_pretraining
    from ftrec.analysis.results import ResultRow

    rows = tuple(
        ResultRow(
            42,
            "Health",
            method,
            "none",
            None,
            "test",
            "sampled",
            0.3,
            0.2,
            10,
            0,
            100,
            100,
            "best.pt",
            "config",
            "data",
        )
        for method in (
            "single",
            "single_mixed",
            "joint_domain",
            "joint_mixed_matched",
            "joint",
            "joint_proportional",
            "pcgrad",
        )
    )

    figure = _figure_pretraining(rows)
    labels = figure.axes[0].get_legend_handles_labels()[1]

    assert labels == [
        "single",
        "single_mixed",
        "joint_domain",
        "joint_mixed_matched",
        "joint",
        "joint_proportional",
        "pcgrad",
    ]
