import torch


def _model():
    from ftrec.models.sasrec import SASRec, SASRecConfig

    return SASRec(SASRecConfig(num_items=10, hidden_size=4, num_blocks=1, num_heads=1, dropout=0, maxlen=3))


def test_engine_uses_sparseadam_for_item_embedding_and_adamw_for_dense() -> None:
    from ftrec.training.engine import OptimizerSettings, build_optimizers

    optimizers = build_optimizers(_model(), OptimizerSettings(lr=1e-3))

    assert isinstance(optimizers.sparse, torch.optim.SparseAdam)
    assert isinstance(optimizers.dense, torch.optim.AdamW)


def test_global_gradient_clipping_scales_dense_and_sparse_together() -> None:
    from ftrec.training.engine import clip_global_grad_norm

    dense = torch.nn.Parameter(torch.zeros(1))
    sparse = torch.nn.Parameter(torch.zeros(3, 1))
    dense.grad = torch.tensor([3.0])
    sparse.grad = torch.sparse_coo_tensor(
        torch.tensor([[1]]), torch.tensor([[4.0]]), (3, 1)
    ).coalesce()

    original_norm = clip_global_grad_norm([dense, sparse], max_norm=2.5)

    assert original_norm == 5.0
    torch.testing.assert_close(dense.grad, torch.tensor([1.5]))
    torch.testing.assert_close(sparse.grad.values(), torch.tensor([[2.0]]))


def test_early_stopping_selects_validation_metric_only() -> None:
    from ftrec.training.engine import EarlyStopping

    stopping = EarlyStopping(patience=2)
    test_metrics = [0.9, 0.1, 1.0]
    for epoch, (validation, test) in enumerate(zip([0.2, 0.4, 0.3], test_metrics, strict=True), 1):
        del test
        stopping.update(epoch, validation)

    assert stopping.best_epoch == 2
    assert stopping.best_metric == 0.4
    assert not stopping.should_stop
