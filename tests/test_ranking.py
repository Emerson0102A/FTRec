import torch

from ftrec.data.datasets import TargetExample
from ftrec.models.sasrec import SASRec, SASRecConfig


class LookupModel:
    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores

    def score(self, contexts: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        del contexts
        values = torch.tensor(
            [[self.scores[int(item)] for item in row] for row in candidates.tolist()],
            dtype=torch.float32,
        )
        return values


def test_chunked_full_rank_excludes_seen_and_breaks_ties_by_item_id() -> None:
    from ftrec.evaluation.ranking import rank_ground_truth_chunked

    model = LookupModel({1: 0.5, 2: 0.9, 3: 0.5, 4: 0.4})
    rank = rank_ground_truth_chunked(
        model,
        context=torch.tensor([0, 1]),
        target_item=3,
        candidates=(1, 2, 3, 4),
        seen_items=frozenset({2}),
        chunk_size=2,
    )

    assert rank == 1


def test_chunk_size_does_not_change_rank() -> None:
    from ftrec.evaluation.ranking import rank_ground_truth_chunked

    model = LookupModel({1: 0.2, 2: 0.8, 3: 0.3, 4: 0.9, 5: 0.1})
    arguments = dict(
        model=model,
        context=torch.tensor([1, 2]),
        target_item=3,
        candidates=(1, 2, 3, 4, 5),
        seen_items=frozenset({1}),
    )

    assert rank_ground_truth_chunked(**arguments, chunk_size=1) == 2
    assert rank_ground_truth_chunked(**arguments, chunk_size=4) == 2


def test_full_catalog_evaluation_encodes_each_context_batch_once(capsys) -> None:
    from ftrec.evaluation.ranking import evaluate_model

    class CountingSASRec(SASRec):
        def __init__(self) -> None:
            super().__init__(
                SASRecConfig(
                    num_items=100,
                    hidden_size=8,
                    num_blocks=1,
                    num_heads=1,
                    dropout=0.0,
                    maxlen=2,
                )
            )
            self.encode_calls = 0

        def encode(self, item_ids: torch.Tensor) -> torch.Tensor:
            self.encode_calls += 1
            return super().encode(item_ids)

    model = CountingSASRec()
    examples = tuple(
        TargetExample(i, i, (0, 1), (-1, 0), i + 2, 0, frozenset({1}))
        for i in range(4)
    )

    result = evaluate_model(
        model,
        examples,
        {0: tuple(range(1, 101))},
        protocol="full",
        chunk_size=10,
        batch_size=4,
        progress=True,
        description="evaluate domain-0",
    )

    assert result["num_eval_users"] == 4
    assert model.encode_calls == 1
    stderr = capsys.readouterr().err
    assert "evaluate domain-0" in stderr
    assert "100%" in stderr


def test_sampled_evaluation_encodes_each_context_batch_once() -> None:
    """Catch regressions back to one GPU encode call per sampled user."""
    from ftrec.evaluation.ranking import evaluate_model

    class CountingSASRec(SASRec):
        def __init__(self) -> None:
            super().__init__(
                SASRecConfig(
                    num_items=100,
                    hidden_size=8,
                    num_blocks=1,
                    num_heads=1,
                    dropout=0.0,
                    maxlen=2,
                )
            )
            self.encode_calls = 0

        def encode(self, item_ids: torch.Tensor) -> torch.Tensor:
            self.encode_calls += 1
            return super().encode(item_ids)

    model = CountingSASRec()
    examples = tuple(
        TargetExample(i, i, (0, 1), (-1, 0), i + 2, 0, frozenset({1}))
        for i in range(4)
    )
    candidates = {
        example.example_id: (example.positive_item, 10, 11, 12)
        for example in examples
    }

    result = evaluate_model(
        model,
        examples,
        {0: tuple(range(1, 101))},
        protocol="sampled",
        sampled_candidates=candidates,
        batch_size=4,
    )

    assert result["num_eval_users"] == 4
    assert model.encode_calls == 1
