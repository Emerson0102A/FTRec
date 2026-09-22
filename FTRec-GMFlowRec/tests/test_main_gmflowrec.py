from __future__ import annotations

import sys

import main_gmflowrec


def test_default_learning_rate_matches_paper_main_text(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_gmflowrec.py"])
    assert main_gmflowrec.parse_args().lr == 1e-4


def test_validation_evaluation_always_runs_test_evaluation(monkeypatch):
    calls = []

    def fake_evaluate(_model, loader, _device, steps):
        calls.append((loader, steps))
        return {"overall": {"ndcg@10": float(len(calls))}}

    monkeypatch.setattr(main_gmflowrec, "evaluate_gmflowrec", fake_evaluate)
    validation, test = main_gmflowrec.evaluate_validation_and_test(
        object(), "valid-loader", "test-loader", "cpu", steps=8
    )

    assert calls == [("valid-loader", 8), ("test-loader", 8)]
    assert validation["overall"]["ndcg@10"] == 1.0
    assert test["overall"]["ndcg@10"] == 2.0
