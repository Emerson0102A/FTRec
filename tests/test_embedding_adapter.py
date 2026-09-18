from __future__ import annotations

import copy

import torch


def _tiny_sasrec():
    from ftrec.models.sasrec import SASRec, SASRecConfig

    torch.manual_seed(42)
    return SASRec(
        SASRecConfig(
            num_items=10,
            hidden_size=4,
            num_blocks=1,
            num_heads=1,
            dropout=0,
            maxlen=3,
        )
    )


def test_zero_initialized_target_embedding_adapter_preserves_outputs() -> None:
    from ftrec.models.embedding_adapter import inject_target_embedding_adapter

    base = _tiny_sasrec().eval()
    adapted = copy.deepcopy(base).eval()
    adapter = inject_target_embedding_adapter(
        adapted, (2, 4, 6), freeze_existing=True
    )
    contexts = torch.tensor([[1, 2, 3], [4, 5, 6]])
    candidates = torch.tensor([[4, 7], [6, 8]])

    assert adapter.num_target_items == 3
    torch.testing.assert_close(adapted.encode(contexts), base.encode(contexts))
    torch.testing.assert_close(
        adapted.score(contexts, candidates), base.score(contexts, candidates)
    )
    assert tuple(
        name for name, parameter in adapted.named_parameters() if parameter.requires_grad
    ) == ("item_embedding_adapter.delta.weight",)


def test_embedding_adapter_updates_only_target_residual_rows() -> None:
    from ftrec.models.embedding_adapter import inject_target_embedding_adapter
    from ftrec.training.engine import OptimizerSettings, build_optimizers

    model = _tiny_sasrec()
    adapter = inject_target_embedding_adapter(model, (2, 4, 6), freeze_existing=True)
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    before = adapter.delta.weight.detach().clone()
    optimizers = build_optimizers(
        model, OptimizerSettings(lr=1e-2, embedding_lr=1e-2)
    )
    contexts = torch.tensor([[1, 2, 3]])
    candidates = torch.tensor([[4, 5]])

    model.score(contexts, candidates).sum().backward()
    optimizers.step()

    parameters = dict(model.named_parameters())
    for name, expected in frozen.items():
        assert torch.equal(parameters[name], expected), name
    assert not torch.equal(adapter.delta.weight[1], before[1])  # item 2
    assert not torch.equal(adapter.delta.weight[2], before[2])  # item 4
    assert torch.equal(adapter.delta.weight[3], before[3])  # unseen item 6
    assert torch.equal(adapter.delta.weight[0], before[0])  # padding/non-target


def test_target_embedding_adapter_uses_sparse_optimizer_group() -> None:
    from ftrec.models.embedding_adapter import inject_target_embedding_adapter
    from ftrec.training.engine import OptimizerSettings, build_optimizers

    model = _tiny_sasrec()
    inject_target_embedding_adapter(model, (2, 4), freeze_existing=True)
    optimizers = build_optimizers(model, OptimizerSettings(embedding_lr=1e-3))

    assert optimizers.dense is None
    assert isinstance(optimizers.sparse, torch.optim.SparseAdam)
