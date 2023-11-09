"""Tests for Linear module."""
import numpy as np
import pytest
import torch

from pytorch_lattice import Monotonicity
from pytorch_lattice.layers import Linear

from ..utils import train_calibrated_module


@pytest.mark.parametrize(
    "input_dim,monotonicities,use_bias,weighted_average",
    [(5, None, True, False), (5, None, True, True)],
)
def test_initialization(input_dim, monotonicities, use_bias, weighted_average) -> None:
    """Tests that Linear initialization works properly"""
    linear = Linear(input_dim, monotonicities, use_bias, weighted_average)
    assert linear.input_dim == input_dim
    assert linear.monotonicities == (
        monotonicities
        if not weighted_average
        else [Monotonicity.INCREASING] * input_dim
    )
    assert linear.use_bias == (use_bias if not weighted_average else False)
    assert linear.weighted_average == weighted_average
    assert torch.allclose(
        linear.kernel.data,
        torch.tensor([[1.0 / input_dim] * input_dim]).double(),
    )
    if use_bias:
        assert linear.bias.data.size() == torch.Size([1])
        assert torch.all(linear.bias.data == 0.0)


@pytest.mark.parametrize(
    "kernel_data,bias_data,inputs,expected_outputs",
    [
        (
            torch.tensor([[1.0], [2.0], [3.0]]).double(),
            None,
            torch.tensor([[1.0, 1.0, 1.0], [3.0, 2.0, 1.0], [1.0, -2.0, 3.0]]).double(),
            torch.tensor([[6.0], [10.0], [6.0]]).double(),
        ),
        (
            torch.tensor([[1.0], [2.0], [1.0]]).double(),
            torch.tensor([-1.0]).double(),
            torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [4.0, -1.0, 2.0]]).double(),
            torch.tensor([[7.0], [8.0], [3.0]]).double(),
        ),
    ],
)
def test_forward(kernel_data, bias_data, inputs, expected_outputs) -> None:
    """Tests that forward properly combined inputs."""
    linear = Linear(kernel_data.size()[0], use_bias=bias_data is not None)
    linear.kernel.data = kernel_data
    if bias_data is not None:
        linear.bias.data = bias_data
    outputs = linear(inputs)
    assert torch.allclose(outputs, expected_outputs)


@pytest.mark.parametrize(
    "monotonicities,kernel_data,expected_out",
    [
        (
            [Monotonicity.INCREASING, Monotonicity.INCREASING, Monotonicity.INCREASING],
            torch.tensor([[0.2], [0.1], [0.2]]).double(),
            [],
        ),
        (
            [Monotonicity.INCREASING, Monotonicity.INCREASING, Monotonicity.INCREASING],
            torch.tensor([[0.2], [-0.01], [0.2]]).double(),
            [],
        ),
        (
            [Monotonicity.DECREASING, Monotonicity.DECREASING, Monotonicity.DECREASING],
            torch.tensor([[-0.2], [-0.1], [-0.2]]).double(),
            [],
        ),
        (
            [Monotonicity.DECREASING, Monotonicity.DECREASING, Monotonicity.DECREASING],
            torch.tensor([[-0.2], [0.01], [-0.2]]).double(),
            [],
        ),
        (
            [Monotonicity.INCREASING, Monotonicity.DECREASING, Monotonicity.INCREASING],
            torch.tensor([[-0.2], [0.2], [-0.2]]).double(),
            ["Monotonicity violated at: [0, 1, 2]"],
        ),
        (
            [
                Monotonicity.NONE,
                Monotonicity.NONE,
                Monotonicity.DECREASING,
                Monotonicity.INCREASING,
            ],
            torch.tensor([[1.5], [-1.5], [0.01], [-0.01]]).double(),
            [],
        ),
    ],
)
def test_assert_constraints_monotonicty(
    monotonicities, kernel_data, expected_out
) -> None:
    """Tests that assert_constraints properly checks monotonicity."""
    linear = Linear(
        kernel_data.size()[0], monotonicities=monotonicities, weighted_average=False
    )
    linear.kernel.data = kernel_data
    assert linear.assert_constraints(eps=0.05) == expected_out


@pytest.mark.parametrize(
    "monotonicities,kernel_data,expected_out",
    [
        (
            [Monotonicity.INCREASING, Monotonicity.INCREASING],
            torch.tensor([[0.4], [0.6]]).double(),
            [],
        ),
        (
            [Monotonicity.INCREASING, Monotonicity.INCREASING],
            torch.tensor([[0.4], [0.61]]).double(),
            [],
        ),
        (
            [Monotonicity.INCREASING, Monotonicity.DECREASING],
            torch.tensor([[1.5], [-0.5]]).double(),
            [],
        ),
        (
            [Monotonicity.INCREASING, Monotonicity.DECREASING],
            torch.tensor([[1.5], [-0.51]]).double(),
            [],
        ),
        (
            [Monotonicity.INCREASING, Monotonicity.DECREASING, Monotonicity.NONE],
            torch.tensor([[1.5], [-2.2], [1.7]]).double(),
            [],
        ),
        (
            [Monotonicity.INCREASING, Monotonicity.DECREASING, Monotonicity.NONE],
            torch.tensor([[1.5], [-2.2], [2.7]]).double(),
            ["Weights do not sum to 1."],
        ),
    ],
)
def test_assert_constraints_weighted_average(
    monotonicities, kernel_data, expected_out
) -> None:
    """Tests assert_constraints checks weights sum to 1 when weighted_average=True."""
    linear = Linear(
        kernel_data.size()[0], monotonicities=monotonicities, weighted_average=True
    )
    linear.kernel.data = kernel_data
    linear.monotonicities = monotonicities
    assert linear.assert_constraints(eps=0.05) == expected_out


@pytest.mark.parametrize(
    "monotonicities,kernel_data,expected_out",
    [
        (
            [Monotonicity.INCREASING, Monotonicity.INCREASING],
            torch.tensor([[0.4], [0.6]]).double(),
            [],
        ),
        (
            [Monotonicity.DECREASING, Monotonicity.NONE, Monotonicity.DECREASING],
            torch.tensor([[0.4], [0.01], [0.6]]).double(),
            ["Monotonicity violated at: [0, 2]"],
        ),
        (
            [Monotonicity.INCREASING, Monotonicity.NONE, Monotonicity.DECREASING],
            torch.tensor([[-0.5], [2.0], [0.5]]).double(),
            ["Weights do not sum to 1.", "Monotonicity violated at: [0, 2]"],
        ),
    ],
)
def test_assert_constraints_combo(monotonicities, kernel_data, expected_out) -> None:
    """Tests asserts_constraints for both monotonicity and weighed_average."""
    linear = Linear(
        kernel_data.size()[0], monotonicities=monotonicities, weighted_average=True
    )
    linear.kernel.data = kernel_data
    linear.monotonicities = monotonicities
    assert linear.assert_constraints(eps=0.05) == expected_out


@pytest.mark.parametrize(
    "monotonicities,kernel_data,bias_data",
    [
        (None, torch.tensor([[1.2], [2.5], [3.1]]).double(), None),
        (
            None,
            torch.tensor([[1.2], [2.5], [3.1]]).double(),
            torch.tensor([1.0]).double(),
        ),
        (
            [Monotonicity.NONE, Monotonicity.NONE, Monotonicity.NONE],
            torch.tensor([[1.2], [2.5], [3.1]]).double(),
            torch.tensor([1.0]).double(),
        ),
        (
            [Monotonicity.NONE, Monotonicity.NONE, Monotonicity.NONE],
            torch.tensor([[1.2], [2.5], [3.1]]).double(),
            torch.tensor([1.0]).double(),
        ),
    ],
)
def test_constrain_no_constraints(monotonicities, kernel_data, bias_data) -> None:
    """Tests that constrain does nothing when there are no constraints."""
    linear = Linear(kernel_data.size()[0], monotonicities=monotonicities)
    linear.kernel.data = kernel_data
    if bias_data is not None:
        linear.bias.data = bias_data
    linear.constrain()
    assert torch.allclose(linear.kernel.data, kernel_data)
    if bias_data is not None:
        assert torch.allclose(linear.bias.data, bias_data)


@pytest.mark.parametrize(
    "monotonicities,kernel_data,expected_projected_kernel_data",
    [
        (
            [
                Monotonicity.NONE,
                Monotonicity.INCREASING,
                Monotonicity.DECREASING,
            ],
            torch.tensor([[1.0], [-0.2], [0.2]]).double(),
            torch.tensor([[1.0], [0.0], [0.0]]).double(),
        ),
        (
            [
                Monotonicity.NONE,
                Monotonicity.INCREASING,
                Monotonicity.NONE,
            ],
            torch.tensor([[1.0], [0.2], [-2.0]]).double(),
            torch.tensor([[1.0], [0.2], [-2.0]]).double(),
        ),
        (
            [
                Monotonicity.DECREASING,
                Monotonicity.DECREASING,
            ],
            torch.tensor([[-1.0], [0.2]]).double(),
            torch.tensor([[-1.0], [0.0]]).double(),
        ),
        (
            [
                Monotonicity.INCREASING,
                Monotonicity.INCREASING,
            ],
            torch.tensor([[-1.0], [1.0]]).double(),
            torch.tensor([[0.0], [1.0]]).double(),
        ),
    ],
)
def test_constrain_monotonicities(
    monotonicities, kernel_data, expected_projected_kernel_data
) -> None:
    """Tests that constrain properly projects kernel according to monotonicies."""
    linear = Linear(kernel_data.size()[0], monotonicities=monotonicities)
    linear.kernel.data = kernel_data
    linear.constrain()
    assert torch.allclose(linear.kernel.data, expected_projected_kernel_data)


@pytest.mark.parametrize(
    "kernel_data,expected_projected_kernel_data",
    [
        (
            torch.tensor([[1.0], [2.0], [3.0]]).double(),
            torch.tensor([[1 / 6], [2 / 6], [0.5]]).double(),
        ),
        (
            torch.tensor([[2.0], [-1.0], [1.0], [3.0]]).double(),
            torch.tensor([[2 / 6], [0.0], [1 / 6], [0.5]]).double(),
        ),
    ],
)
def test_constrain_weighted_average(
    kernel_data, expected_projected_kernel_data
) -> None:
    """Tests that constrain properly projects kernel to be a weighted average."""
    linear = Linear(kernel_data.size()[0], weighted_average=True)
    linear.kernel.data = kernel_data
    linear.constrain()
    assert torch.allclose(linear.kernel.data, expected_projected_kernel_data)


def test_training() -> None:
    """Tests that the `Linear` module can learn f(x_1,x_2) = 2x_1 + 3x_2"""
    num_examples = 1000
    input_min, input_max = 0.0, 10.0
    training_examples = torch.from_numpy(
        np.random.uniform(input_min, input_max, size=(1000, 2))
    )
    linear_coefficients = torch.tensor([2.0, 3.0]).double()
    training_labels = torch.sum(
        linear_coefficients * training_examples, dim=1, keepdim=True
    )

    linear = Linear(2, use_bias=False)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-2)

    train_calibrated_module(
        linear,
        training_examples,
        training_labels,
        loss_fn,
        optimizer,
        300,
        num_examples // 10,
    )

    assert torch.allclose(torch.squeeze(linear.kernel.data), linear_coefficients)
