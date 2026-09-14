from __future__ import annotations

import torch


def _task(value: torch.Tensor):
    from ftrec.training.pcgrad import TaskGradients

    return TaskGradients(("weight",), (value,))


def test_pcgrad_removes_negative_component_against_original_peer() -> None:
    from ftrec.training.pcgrad import gradient_dot, project_pcgrad

    task_a = _task(torch.tensor([1.0, -1.0]))
    task_b = _task(torch.tensor([-1.0, 0.0]))

    projected = project_pcgrad((task_a, task_b), seed=7, step=0)

    assert gradient_dot(projected[0], task_b) >= -1e-7
    assert gradient_dot(projected[1], task_a) >= -1e-7


def test_nonconflicting_gradients_are_unchanged() -> None:
    from ftrec.training.pcgrad import project_pcgrad

    first = _task(torch.tensor([1.0, 0.0]))
    second = _task(torch.tensor([0.0, 1.0]))

    projected = project_pcgrad((first, second), seed=1, step=0)

    torch.testing.assert_close(projected[0].values[0], first.values[0])
    torch.testing.assert_close(projected[1].values[0], second.values[0])


def test_sparse_projection_matches_dense_reference_without_densifying(monkeypatch) -> None:
    from ftrec.training.pcgrad import project_pcgrad

    sparse_a = torch.sparse_coo_tensor(
        torch.tensor([[0, 2]]), torch.tensor([[1.0, -1.0], [2.0, 0.0]]), (4, 2)
    ).coalesce()
    sparse_b = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]]), torch.tensor([[-1.0, 0.0], [0.0, 3.0]]), (4, 2)
    ).coalesce()
    dense_a = sparse_a.to_dense()
    dense_b = sparse_b.to_dense()
    expected_a = dense_a - (dense_a.mul(dense_b).sum() / dense_b.square().sum()) * dense_b

    def forbid_to_dense(*args, **kwargs):
        raise AssertionError("production PCGrad densified a sparse gradient")

    monkeypatch.setattr(torch.Tensor, "to_dense", forbid_to_dense)
    result = project_pcgrad((_task(sparse_a), _task(sparse_b)), seed=2, step=0)
    assert result[0].values[0].is_sparse
    monkeypatch.undo()

    torch.testing.assert_close(result[0].values[0].to_dense(), expected_a)


def test_assign_mean_gradients_preserves_sparse_parameter_gradient() -> None:
    from ftrec.training.pcgrad import assign_mean_gradients

    parameter = torch.nn.Parameter(torch.zeros(4, 2))
    first = torch.sparse_coo_tensor(
        torch.tensor([[0]]), torch.tensor([[2.0, 4.0]]), (4, 2)
    ).coalesce()
    second = torch.sparse_coo_tensor(
        torch.tensor([[1]]), torch.tensor([[6.0, 8.0]]), (4, 2)
    ).coalesce()

    assign_mean_gradients({"weight": parameter}, (_task(first), _task(second)))

    assert parameter.grad.is_sparse
    torch.testing.assert_close(
        parameter.grad.to_dense(),
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]),
    )

