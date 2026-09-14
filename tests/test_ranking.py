import torch


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
