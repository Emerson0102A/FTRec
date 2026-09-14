import math

import pytest
import torch


def test_cosine_matrix_is_symmetric_and_negative_ratio_uses_unique_pairs() -> None:
    from ftrec.training.pcgrad import TaskGradients, cosine_matrix, negative_pair_ratio

    tasks = (
        TaskGradients(("w",), (torch.tensor([1.0, 0.0]),)),
        TaskGradients(("w",), (torch.tensor([-1.0, 0.0]),)),
        TaskGradients(("w",), (torch.tensor([0.0, 1.0]),)),
    )

    matrix = cosine_matrix(tasks)

    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(-1.0)
    assert matrix[0][2] == pytest.approx(0.0)
    assert matrix[1][0] == pytest.approx(matrix[0][1])
    assert negative_pair_ratio(matrix) == pytest.approx(1 / 3)


def test_zero_gradient_cosine_is_nan_not_a_fake_zero() -> None:
    from ftrec.training.pcgrad import TaskGradients, cosine_matrix

    tasks = (
        TaskGradients(("w",), (torch.zeros(2),)),
        TaskGradients(("w",), (torch.ones(2),)),
    )

    matrix = cosine_matrix(tasks)

    assert math.isnan(matrix[0][0])
    assert math.isnan(matrix[0][1])


def test_parameter_prefix_selects_layer_group() -> None:
    from ftrec.training.pcgrad import TaskGradients, cosine_matrix

    tasks = (
        TaskGradients(
            ("item_embedding.weight", "blocks.0.attention.q_proj.weight"),
            (torch.tensor([1.0]), torch.tensor([1.0])),
        ),
        TaskGradients(
            ("item_embedding.weight", "blocks.0.attention.q_proj.weight"),
            (torch.tensor([-1.0]), torch.tensor([1.0])),
        ),
    )

    full = cosine_matrix(tasks)
    attention = cosine_matrix(tasks, prefixes=("blocks.0.attention",))

    assert full[0][1] == pytest.approx(0.0)
    assert attention[0][1] == pytest.approx(1.0)
