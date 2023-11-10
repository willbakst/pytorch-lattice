"""Tests for Lattice module."""
import numpy as np
import pytest
import torch

from pytorch_lattice.enums import Interpolation, Monotonicity
from pytorch_lattice.layers import Lattice


@pytest.mark.parametrize(
    "lattice_sizes, expected_size",
    [
        ((2,), torch.Size([2, 1])),
        ([2], torch.Size([2, 1])),
        ((2, 4), torch.Size([8, 1])),
        ([2, 3, 4], torch.Size([24, 1])),
    ],
)
def test_initialization(
    lattice_sizes,
    expected_size,
):
    """Tests that Lattice Initialization works properly."""
    lattice = Lattice(lattice_sizes)
    assert lattice.kernel.data.size() == expected_size
    assert lattice.interpolation == Interpolation.HYPERCUBE
    assert lattice.clip_inputs
    assert lattice.units == 1


@pytest.mark.parametrize(
    "lattice_sizes, output_min, output_max, units, expected_out",
    [
        ([2], 0.0, 1.0, 1, torch.Tensor([[0], [1]])),
        ([2, 2], 0.0, 1.0, 1, torch.Tensor([[0], [0.5], [0.5], [1]])),
        (
            [2, 3],
            0.2,
            3.1,
            2,
            torch.Tensor(
                [
                    [0.2, 0.2],
                    [0.925, 0.925],
                    [1.65, 1.65],
                    [1.65, 1.65],
                    [2.375, 2.375],
                    [3.1, 3.1],
                ]
            ),
        ),
    ],
)
def test_linear_initialization(
    lattice_sizes,
    output_min,
    output_max,
    units,
    expected_out,
):
    """Tests that linear initialization generates correct values."""
    lattice = Lattice(
        lattice_sizes, output_min=output_min, output_max=output_max, units=units
    )
    assert torch.allclose(lattice.kernel, expected_out.double())


@pytest.mark.parametrize(
    "input_point, lattice_size, expected_out",
    [
        (torch.tensor([[0.2, 0.7]]), (2, 2), torch.tensor([[1.1]]).double()),
        (torch.tensor([[0.0, 0.0]]), (3, 3), torch.tensor([[0.0]]).double()),
        (torch.tensor([[1.0, 1.0]]), (3, 3), torch.tensor([[4.0]]).double()),
        (torch.tensor([[2.0, 2.0]]), (3, 3), torch.tensor([[8.0]]).double()),
        (torch.tensor([[0.0, 0.4, 0.8]]), (2, 2, 2), torch.tensor([[1.6]]).double()),
        (torch.tensor([[0.2, 0.4, 1.0]]), (2, 2, 2), torch.tensor([[2.6]]).double()),
        (
            torch.tensor([[0.2, 0.7, 1.8, 2.1]]),
            (2, 2, 3, 4),
            torch.tensor([[22.5]]).double(),
        ),
    ],
)
def test_forward_hypercube(
    input_point,
    lattice_size,
    expected_out,
):
    """Tests complete lattice layer for units=1 and hypercube interpolation."""
    lattice = Lattice(lattice_size)
    vertices = np.prod(lattice_size)
    lattice.kernel.data = torch.arange(0, vertices, dtype=torch.double).view(-1, 1)
    assert torch.allclose(lattice.forward(input_point), expected_out, atol=1e-4)


@pytest.mark.parametrize(
    "lattice_size, inputs, expected_out",
    [
        ((2, 2), torch.tensor([[0.3, 0.6]]), torch.tensor([[1.2]]).double()),
        ((3, 3), torch.tensor([[0.7, 1.2]]), torch.tensor([[3.3]]).double()),
        ((2, 2, 2), torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([[1.1]]).double()),
        ((2, 3, 4), torch.tensor([[0.1, 1.2, 2.3]]), torch.tensor([[8.3]]).double()),
    ],
)
def test_forward_simplex(
    lattice_size,
    inputs,
    expected_out,
):
    """Tests complete lattice layer for units=1 and simplex interpolation."""
    lattice = Lattice(lattice_size)
    lattice.interpolation = Interpolation.SIMPLEX
    vertices = np.prod(lattice_size)
    lattice.kernel.data = torch.arange(0, vertices, dtype=torch.double).view(-1, 1)
    assert torch.allclose(lattice.forward(inputs), expected_out, atol=1e-4)


@pytest.mark.parametrize(
    "lattice_size, inputs, expected_out",
    [
        (
            (2, 2),
            torch.tensor([[0.3, 0.6], [0.3, 0.6]]),
            torch.tensor([[1.2, 2.2]]).double(),
        ),
        (
            (2, 3, 4),
            torch.tensor([[0.1, 1.2, 2.3], [0.1, 1.2, 2.3]]),
            torch.tensor([[8.3, 9.3]]).double(),
        ),
    ],
)
def test_forward_simplex_2_units(
    lattice_size,
    inputs,
    expected_out,
):
    """Tests complete lattice layer for units=2 and simplex interpolation."""
    lattice = Lattice(lattice_size)
    lattice.units = 2
    lattice.interpolation = Interpolation.SIMPLEX
    vertices = np.prod(lattice_size)
    tensor = torch.arange(0, vertices, dtype=torch.double).view(-1, 1)
    lattice.kernel.data = torch.cat([tensor, tensor + 1], dim=1)
    assert torch.allclose(lattice.forward(inputs), expected_out, atol=1e-4)


@pytest.mark.parametrize(
    "input_point, lattice_size, expected_out",
    [
        (
            torch.tensor([[0.2, 0.7], [0.2, 0.7]]),
            (2, 2),
            torch.tensor([[1.1, 2.1]]).double(),
        ),
        (
            torch.tensor([[0.2, 0.7, 1.8, 2.1], [0.2, 0.7, 1.8, 2.1]]),
            (2, 2, 3, 4),
            torch.tensor([[22.5, 23.5]]).double(),
        ),
    ],
)
def test_forward_hypercube_2_units(
    input_point,
    lattice_size,
    expected_out,
):
    """Tests lattice layer for units=2 and hypercube interpolation."""
    lattice = Lattice(lattice_size, units=2)
    vertices = np.prod(lattice_size)
    weights1 = torch.arange(0, vertices, dtype=torch.double).view(-1, 1)
    weights2 = torch.arange(1, vertices + 1, dtype=torch.double).view(-1, 1)
    lattice.kernel.data = torch.stack((weights1, weights2), dim=-1).squeeze(1)
    assert torch.allclose(lattice.forward(input_point), expected_out, atol=1e-4)


@pytest.mark.parametrize(
    "lattice_size, input1, input2",
    [
        ((2, 2), torch.tensor([[0.3, 1.6]]), torch.tensor([[0.3, 1.0]])),
        ((2, 3, 4), torch.tensor([[3.1, -0.2, 5.2]]), torch.tensor([[2.0, 0.0, 4.0]])),
    ],
)
@pytest.mark.parametrize(
    "interpolation",
    [
        Interpolation.HYPERCUBE,
        Interpolation.SIMPLEX,
    ],
)
@pytest.mark.parametrize(
    "units",
    [
        1,
        2,
        5,
    ],
)
def test_clip_onto_lattice_range(
    lattice_size,
    input1,
    input2,
    interpolation,
    units,
):
    """Tests clipping for coordinates out of lattice bounds."""
    lattice = Lattice(lattice_size)
    lattice.interpolation = interpolation
    lattice.units = units
    vertices = np.prod(lattice_size)
    tensor = torch.arange(0, vertices, dtype=torch.double).view(-1, 1)
    if units == 1:
        lattice.kernel.data = tensor
    else:
        lattice.kernel.data = torch.cat([tensor for _ in range(units)], dim=1)
        input1 = input1.repeat(units, 1)
        input2 = input2.repeat(units, 1)
    assert torch.allclose(lattice.forward(input1), lattice.forward(input2), atol=1e-4)


@pytest.mark.parametrize(
    "inputs, expected_out",
    [
        (
            [torch.tensor([[0.8, 0.2]]), torch.tensor([[0.7, 0.3]])],
            torch.tensor([[0.56, 0.24, 0.14, 0.06]]),
        ),
        (
            [torch.tensor([[0.8, 0.2, 0.0]]), torch.tensor([[0.0, 0.7, 0.3]])],
            torch.tensor([[0.0, 0.56, 0.24, 0.0, 0.14, 0.06, 0.0, 0.0, 0.0]]),
        ),
        (
            [
                torch.tensor([[0.8, 0.2]]),
                torch.tensor([[0.7, 0.3]]),
                torch.tensor([[0.6, 0.4]]),
            ],
            torch.tensor([[0.336, 0.224, 0.144, 0.096, 0.084, 0.056, 0.036, 0.024]]),
        ),
        (
            [
                torch.tensor([[0.8, 0.2]]),
                torch.tensor([[0.0, 0.7, 0.3]]),
                torch.tensor([[0.0, 0.0, 0.9, 0.1]]),
            ],
            torch.tensor(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.504,
                        0.056,
                        0.0,
                        0.0,
                        0.216,
                        0.024,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.126,
                        0.014,
                        0.0,
                        0.0,
                        0.054,
                        0.006,
                    ]
                ]
            ),
        ),
        (
            [
                torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
                torch.tensor([[0.7, 0.3], [0.3, 0.7]]),
            ],
            torch.tensor([[0.56, 0.24, 0.14, 0.06], [0.06, 0.14, 0.24, 0.56]]),
        ),
    ],
)
def test_batch_outer_operation(
    inputs,
    expected_out,
):
    """Tests _batch_outer_operation works correctly"""
    assert torch.allclose(
        Lattice._batch_outer_operation(inputs), expected_out, atol=1e-4
    )


@pytest.mark.parametrize(
    "lattice_sizes, expected_out",
    [
        ((2, 2, 2, 2), [(torch.tensor([[0.1, 0.2, 0.3, 0.4]]), 4, 2)]),
        (
            (2, 3, 4, 5),
            [
                (torch.tensor([[0.1]]), 1, 2),
                (torch.tensor([[0.2]]), 1, 3),
                (torch.tensor([[0.3]]), 1, 4),
                (torch.tensor([[0.4]]), 1, 5),
            ],
        ),
        (
            (2, 2, 3, 4),
            [
                (torch.tensor([[0.1, 0.2]]), 2, 2),
                (torch.tensor([[0.3]]), 1, 3),
                (torch.tensor([[0.4]]), 1, 4),
            ],
        ),
        (
            (3, 2, 2, 3),
            [
                (torch.tensor([[0.1]]), 1, 3),
                (torch.tensor([[0.2, 0.3]]), 2, 2),
                (torch.tensor([[0.4]]), 1, 3),
            ],
        ),
    ],
)
def test_bucketize_consecutive_inputs(
    lattice_sizes,
    expected_out,
):
    """Tests _bucketize_consecutive_inputs works correctly"""
    lattice = Lattice(lattice_sizes)
    actual_out = list(
        lattice._bucketize_consecutive_equal_dims(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    )
    for (expected_tensor, expected_int1, expected_int2), (
        actual_tensor,
        actual_int1,
        actual_int2,
    ) in zip(expected_out, actual_out):
        assert torch.allclose(expected_tensor, actual_tensor)
        assert expected_int1 == actual_int1
        assert expected_int2 == actual_int2


@pytest.mark.parametrize(
    "monotonicities, expected_out",
    [
        (
            [Monotonicity.INCREASING, Monotonicity.INCREASING],
            torch.tensor([[2.5, 2.5], [2.5, 2.5]]),
        ),
        (
            [Monotonicity.INCREASING, None],
            torch.tensor([[3.0, 2.0], [3.0, 2.0]]),
        ),
        (
            [None, Monotonicity.INCREASING],
            torch.tensor([[3.5, 3.5], [1.5, 1.5]]),
        ),
        (
            [None, None],
            torch.tensor([[4.0, 3.0], [2.0, 1.0]]),
        ),
    ],
)
def test_project_monotonicity_simple(
    monotonicities,
    expected_out,
):
    """Tests _project_monotonicity with a simple 2x2, 1-unit example."""
    lattice = Lattice(lattice_sizes=[2, 2], monotonicities=monotonicities)
    lattice.kernel.data = torch.tensor([[4], [3], [2], [1]]).double()
    lattice.constrain()
    assert torch.allclose(lattice.kernel.data, expected_out.view(-1, 1).double())


@pytest.mark.parametrize(
    "monotonicities, expected_out",
    [
        (
            [Monotonicity.INCREASING, Monotonicity.INCREASING, Monotonicity.INCREASING],
            torch.tensor(
                [
                    [
                        [[6.5, 18.5], [6.5, 18.5], [6.5, 18.5]],
                        [[6.5, 18.5], [6.5, 18.5], [6.5, 18.5]],
                    ],
                    [
                        [[6.5, 18.5], [6.5, 18.5], [6.5, 18.5]],
                        [[6.5, 18.5], [6.5, 18.5], [6.5, 18.5]],
                    ],
                ]
            ),
        ),
        (
            [Monotonicity.INCREASING, None, None],
            torch.tensor(
                [
                    [
                        [[9.0, 21.0], [8.0, 20.0], [7.0, 19.0]],
                        [[6.0, 18.0], [5.0, 17.0], [4.0, 16.0]],
                    ],
                    [
                        [[9.0, 21.0], [8.0, 20.0], [7.0, 19.0]],
                        [[6.0, 18.0], [5.0, 17.0], [4.0, 16.0]],
                    ],
                ]
            ),
        ),
        (
            [Monotonicity.INCREASING, None, Monotonicity.INCREASING],
            torch.tensor(
                [
                    [
                        [[8.0, 20.0], [8.0, 20.0], [8.0, 20.0]],
                        [[5.0, 17.0], [5.0, 17.0], [5.0, 17.0]],
                    ],
                    [
                        [[8.0, 20.0], [8.0, 20.0], [8.0, 20.0]],
                        [[5.0, 17.0], [5.0, 17.0], [5.0, 17.0]],
                    ],
                ]
            ),
        ),
    ],
)
def test_project_monotonicity_complex(
    monotonicities,
    expected_out,
):
    """Tests _project_monotonicity with more complex 2x2x3, 2-unit example."""
    lattice = Lattice(lattice_sizes=[2, 2, 3], monotonicities=monotonicities, units=2)
    col1 = torch.arange(12, 0, -1).reshape(-1, 1).double()
    col2 = torch.arange(24, 12, -1).reshape(-1, 1).double()
    lattice.kernel.data = torch.cat((col1, col2), dim=1)
    lattice.constrain()
    assert torch.allclose(lattice.kernel.data, expected_out.view(-1, 2).double())


@pytest.mark.parametrize(
    "lattice_size, monotonicities, kernel_data, expected_out",
    [
        (
            (2, 2),
            None,
            torch.tensor([[4.0], [3.0], [2.0], [1.0]]),
            [],
        ),
        (
            (2, 2),
            [Monotonicity.INCREASING, Monotonicity.INCREASING],
            torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            [],
        ),
        (
            (2, 2),
            [Monotonicity.INCREASING, None],
            torch.tensor([[4.0], [3.0], [2.0], [1.0]]),
            ["Monotonicity violated at feature index 0."],
        ),
        (
            (2, 2),
            [None, Monotonicity.INCREASING],
            torch.tensor([[4.0], [3.0], [2.0], [1.0]]),
            ["Monotonicity violated at feature index 1."],
        ),
        (
            (2, 2, 3),
            [Monotonicity.INCREASING, None, Monotonicity.INCREASING],
            torch.tensor(
                [
                    [4.0, 2.8],
                    [3.0, 4.2],
                    [7.0, 1.2],
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                ]
            ),
            [
                "Monotonicity violated at feature index 0.",
                "Monotonicity violated at feature index 2.",
                "Monotonicity violated at feature index 2.",
            ],
        ),
    ],
)
def test_assert_constraints_monotonicity(
    lattice_size, monotonicities, kernel_data, expected_out
):
    """Tests lattice layer's assert_constraints() function."""
    lattice = Lattice(lattice_size, monotonicities=monotonicities)
    lattice.kernel.data = kernel_data
    assert lattice.assert_constraints() == expected_out


@pytest.mark.parametrize(
    "out_min, out_max, kernel_data, expected_out",
    [
        (
            0.0,
            1.0,
            torch.tensor([[0.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [-1.0, 2.0]]),
            ["Max weight greater than output_max.", "Min weight less than output_min."],
        ),
        (
            None,
            1.0,
            torch.tensor([[0.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [-1.0, 2.0]]),
            ["Max weight greater than output_max."],
        ),
        (
            0.0,
            None,
            torch.tensor([[0.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [-1.0, 2.0]]),
            ["Min weight less than output_min."],
        ),
        (
            0.0,
            1.0,
            torch.tensor(
                [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.0 - 1e-7, 1.0 + 1e-7]]
            ),
            [],
        ),
    ],
)
def test_assert_constraints_bounds(out_min, out_max, kernel_data, expected_out):
    """Tests that assert_constraints() detects out of output_min/max bounds."""
    lattice = Lattice(lattice_sizes=[2, 2], output_max=out_max, output_min=out_min)
    lattice.kernel.data = kernel_data
    assert lattice.assert_constraints() == expected_out


@pytest.mark.parametrize(
    "out_min, out_max, kernel_data, expected_out",
    [
        (
            0.0,
            1.0,
            torch.tensor([[0.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [-1.0, 2.0]]),
            torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        ),
        (
            None,
            1.0,
            torch.tensor([[0.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [-1.0, 2.0]]),
            torch.tensor([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]),
        ),
        (
            0.0,
            None,
            torch.tensor([[0.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [-1.0, 2.0]]),
            torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 2.0], [0.0, 2.0]]),
        ),
    ],
)
def test_constrain_clipping_functionality(out_min, out_max, kernel_data, expected_out):
    """Tests that constrain() method clips out of bounds weights to output min/max."""
    lattice = Lattice([2, 2], units=2, output_max=out_max, output_min=out_min)
    lattice.kernel.data = kernel_data
    lattice.constrain()
    assert torch.allclose(lattice.kernel.data, expected_out)
