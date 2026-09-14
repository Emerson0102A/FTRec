import pytest
import torch


def _model():
    from ftrec.models.sasrec import SASRec, SASRecConfig

    torch.manual_seed(5)
    return SASRec(
        SASRecConfig(
            num_items=20,
            hidden_size=8,
            num_blocks=2,
            num_heads=2,
            dropout=0.0,
            maxlen=4,
        )
    )


def test_sasrec_shapes_zero_padding_and_tied_scoring() -> None:
    model = _model().eval()
    items = torch.tensor([[0, 0, 2, 3], [0, 4, 5, 6]])

    states = model.encode(items)

    assert states.shape == (2, 4, 8)
    assert torch.count_nonzero(states[items == 0]) == 0
    assert model.scoring_weight().data_ptr() == model.item_embedding.weight.data_ptr()
    assert torch.count_nonzero(model.item_embedding.weight[0]) == 0


def test_training_item_lookups_produce_sparse_embedding_gradient() -> None:
    model = _model().train()
    contexts = torch.tensor([[0, 1, 2, 3], [0, 3, 4, 5]])
    candidates = torch.tensor([[6, 7], [8, 9]])

    model.score(contexts, candidates).sum().backward()

    assert model.item_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad.is_sparse


def test_hidden_size_must_be_divisible_by_heads() -> None:
    from ftrec.models.sasrec import SASRecConfig

    with pytest.raises(ValueError, match="divisible"):
        SASRecConfig(num_items=10, hidden_size=7, num_heads=2)
