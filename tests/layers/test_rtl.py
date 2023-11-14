"""Tests for RTL layer."""
from itertools import cycle
from unittest.mock import Mock, patch

import pytest
import torch

from pytorch_lattice import Interpolation, LatticeInit, Monotonicity
from pytorch_lattice.layers import RTL, Lattice


@pytest.mark.parametrize(
    "monotonicities, num_lattices, lattice_rank, output_min, output_max, kernel_init,"
    "clip_inputs, interpolation, average_outputs",
    [
        (
            [
                None,
                None,
                None,
                None,
            ],
            3,
            3,
            None,
            2.0,
            LatticeInit.LINEAR,
            True,
            Interpolation.HYPERCUBE,
            True,
        ),
        (
            [
                Monotonicity.INCREASING,
                Monotonicity.INCREASING,
                None,
                None,
            ],
            3,
            3,
            -1.0,
            4.0,
            LatticeInit.LINEAR,
            False,
            Interpolation.SIMPLEX,
            False,
        ),
        (
            [Monotonicity.INCREASING, None] * 25,
            20,
            5,
            None,
            None,
            LatticeInit.LINEAR,
            True,
            Interpolation.HYPERCUBE,
            True,
        ),
    ],
)
def test_initialization(
    monotonicities,
    num_lattices,
    lattice_rank,
    output_min,
    output_max,
    kernel_init,
    clip_inputs,
    interpolation,
    average_outputs,
):
    """Tests that RTL Initialization works properly."""
    rtl = RTL(
        monotonicities=monotonicities,
        num_lattices=num_lattices,
        lattice_rank=lattice_rank,
        output_min=output_min,
        output_max=output_max,
        kernel_init=kernel_init,
        clip_inputs=clip_inputs,
        interpolation=interpolation,
        average_outputs=average_outputs,
    )
    assert rtl.monotonicities == monotonicities
    assert rtl.num_lattices == num_lattices
    assert rtl.lattice_rank == lattice_rank
    assert rtl.output_min == output_min
    assert rtl.output_max == output_max
    assert rtl.kernel_init == kernel_init
    assert rtl.interpolation == interpolation
    assert rtl.average_outputs == average_outputs

    total_lattices = 0
    for monotonic_count, (lattice, group) in rtl._lattice_layers.items():
        # test monotonic features have been sorted to front of list for lattice indices
        for single_lattice_indices in group:
            for i in range(lattice_rank):
                if i < monotonic_count:
                    assert (
                        rtl.monotonicities[single_lattice_indices[i]]
                        == Monotonicity.INCREASING
                    )
                else:
                    assert rtl.monotonicities[single_lattice_indices[i]] == None

        assert len(lattice.monotonicities) == len(lattice.lattice_sizes)
        assert (
            sum(1 for _ in lattice.monotonicities if _ == Monotonicity.INCREASING)
            == monotonic_count
        )
        assert lattice.output_min == rtl.output_min
        assert lattice.output_max == rtl.output_max
        assert lattice.kernel_init == rtl.kernel_init
        assert lattice.clip_inputs == rtl.clip_inputs
        assert lattice.interpolation == rtl.interpolation

        # test number of lattices created is correct
        total_lattices += lattice.units

    assert total_lattices == num_lattices


@pytest.mark.parametrize(
    "monotonicities, num_lattices, lattice_rank",
    [
        ([None] * 9, 2, 2),
        ([Monotonicity.INCREASING] * 10, 3, 3),
    ],
)
def test_initialization_invalid(
    monotonicities,
    num_lattices,
    lattice_rank,
):
    """Tests that RTL Initialization raises error when RTL is too small."""
    with pytest.raises(ValueError) as exc_info:
        RTL(
            monotonicities=monotonicities,
            num_lattices=num_lattices,
            lattice_rank=lattice_rank,
        )
    assert (
        str(exc_info.value)
        == f"RTL with {num_lattices}x{lattice_rank}D structure cannot support "
        + f"{len(monotonicities)} input features."
    )


@pytest.mark.parametrize(
    "num_features, num_lattices, lattice_rank, units, expected_lattice_args,"
    "expected_result, expected_avg",
    [
        (
            6,
            6,
            3,
            [3, 2, 1],
            [
                torch.tensor([[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.0, 0.1, 0.2]]]),
                torch.tensor([[[0.3, 0.4, 0.5], [0.0, 0.1, 0.2]]]),
                torch.tensor([[0.3, 0.4, 0.5]]),
            ],
            torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 2.0]]),
            torch.tensor([[2 / 3]]),
        ),
        (
            3,
            3,
            2,
            [1, 1, 1],
            [
                torch.tensor([[0.0, 0.1]]),
                torch.tensor([[0.2, 0.0]]),
                torch.tensor([[0.1, 0.2]]),
            ],
            torch.tensor([[0.0, 1.0, 2.0]]),
            torch.tensor([[1.0]]),
        ),
        (
            6,
            7,
            5,
            [2, 3, 2],
            [
                torch.tensor([[[0.0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.0, 0.1, 0.2, 0.3]]]),
                torch.tensor(
                    [
                        [
                            [0.4, 0.5, 0.0, 0.1, 0.2],
                            [0.3, 0.4, 0.5, 0.0, 0.1],
                            [0.2, 0.3, 0.4, 0.5, 0.0],
                        ]
                    ]
                ),
                torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 0.1, 0.2, 0.3, 0.4]]]),
            ],
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]]),
            torch.tensor([[1.0]]),
        ),
    ],
)
def test_forward(
    num_features,
    num_lattices,
    lattice_rank,
    units,
    expected_lattice_args,
    expected_result,
    expected_avg,
):
    """Tests forward function of RTL Lattice."""
    rtl = RTL(
        monotonicities=[None, Monotonicity.INCREASING] * (num_features // 2),
        num_lattices=num_lattices,
        lattice_rank=lattice_rank,
    )
    # generate indices for each lattice in cyclic fashion based off units
    groups = []
    feature_indices = cycle(range(num_features))
    for lattice_units in units:
        group = [
            [next(feature_indices) for _ in range(lattice_rank)]
            for _ in range(lattice_units)
        ]
        groups.append(group)
    lattice_indices = {i: groups[i % len(groups)] for i in range(len(units))}
    rtl._lattice_layers = {
        i: (Lattice(lattice_sizes=[2] * lattice_rank, units=unit), lattice_indices[i])
        for i, unit in enumerate(units)
    }

    mock_forwards = []
    for monotonic_count, (lattice, _) in rtl._lattice_layers.items():
        mock_forward = Mock()
        lattice.forward = mock_forward
        mock_forward.return_value = torch.full(
            (1, units[monotonic_count]),
            float(monotonic_count),
            dtype=torch.float32,
        )
        mock_forwards.append(mock_forward)

    x = torch.arange(0, num_features * 0.1, 0.1).unsqueeze(0)
    result = rtl.forward(x)

    # Assert the calls and results for each mock_forward based on expected_outs
    for i, mock_forward in enumerate(mock_forwards):
        mock_forward.assert_called_once()
        assert torch.allclose(
            mock_forward.call_args[0][0], expected_lattice_args[i], atol=1e-6
        )
    assert torch.allclose(result, expected_result)
    rtl.average_outputs = True
    result = rtl.forward(x)
    assert torch.allclose(result, expected_avg)


@pytest.mark.parametrize(
    "monotonic_counts, units, expected_out",
    [
        (
            [0, 1, 2, 3],
            [2, 1, 1, 1],
            [None] * 2 + [Monotonicity.INCREASING] * 3,
        ),
        (
            [0, 4, 5, 7],
            [1, 2, 3, 4],
            [None] + [Monotonicity.INCREASING] * 9,
        ),
        ([0], [3], [None] * 3),
        ([1, 2, 3], [1, 1, 1], [Monotonicity.INCREASING] * 3),
    ],
)
def test_output_monotonicities(
    monotonic_counts,
    units,
    expected_out,
):
    """Tests output_monotonicities function."""
    rtl = RTL(
        monotonicities=[None, Monotonicity.INCREASING],
        num_lattices=3,
        lattice_rank=3,
    )
    rtl._lattice_layers = {
        monotonic_count: (Lattice(lattice_sizes=[2, 2], units=units[i]), [])
        for i, monotonic_count in enumerate(monotonic_counts)
    }
    assert rtl.output_monotonicities() == expected_out


def test_constrain():
    """Tests RTL constrain function."""
    rtl = RTL(
        monotonicities=[None, Monotonicity.INCREASING],
        num_lattices=3,
        lattice_rank=3,
    )
    mock_constrains = []
    for lattice, _ in rtl._lattice_layers.values():
        mock_constrain = Mock()
        lattice.constrain = mock_constrain
        mock_constrains.append(mock_constrain)

    rtl.constrain()
    for mock_constrain in mock_constrains:
        mock_constrain.assert_called_once()


def test_assert_constraints():
    """Tests RTL assert_constraints function."""
    rtl = RTL(
        monotonicities=[None, Monotonicity.INCREASING],
        num_lattices=3,
        lattice_rank=3,
    )
    mock_asserts = []
    for lattice, _ in rtl._lattice_layers.values():
        mock_assert = Mock()
        lattice.assert_constraints = mock_assert
        mock_assert.return_value = "violation"
        mock_asserts.append(mock_assert)

    violations = rtl.assert_constraints()
    for mock_assert in mock_asserts:
        mock_assert.assert_called_once()

    assert violations == ["violation"] * len(rtl._lattice_layers)


@pytest.mark.parametrize(
    "rtl_indices",
    [
        [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [6, 6]],
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        [[1, 1, 1], [2, 2, 2], [1, 2, 3], [3, 3, 3]],
        [
            [1, 1, 2],
            [2, 3, 4],
            [1, 5, 5],
            [4, 6, 7],
            [1, 3, 4],
            [2, 3, 3],
            [4, 5, 6],
            [6, 6, 6],
        ],
    ],
)
def test_ensure_unique_sublattices_possible(rtl_indices):
    """Tests _ensure_unique_sublattices removes duplicates from groups when possible."""
    swapped_indices = RTL._ensure_unique_sublattices(rtl_indices)
    for group in swapped_indices:
        assert len(set(group)) == len(group)


@pytest.mark.parametrize(
    "rtl_indices, max_swaps",
    [
        ([[1, 1], [1, 2], [1, 3]], 100),
        ([[1, 1], [2, 2], [3, 3], [4, 4]], 2),
    ],
)
def test_ensure_unique_sublattices_impossible(rtl_indices, max_swaps):
    """Tests _ensure_unique_sublattices logs when it can't remove duplicates."""
    with patch("logging.info") as mock_logging_info:
        RTL._ensure_unique_sublattices(rtl_indices, max_swaps)
        mock_logging_info.assert_called_with(
            "Some lattices in RTL may use the same feature multiple times."
        )
