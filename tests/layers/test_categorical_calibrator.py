"""Tests for CategoricalCalibrator module."""
import numpy as np
import pytest
import torch
from pytorch_lattice.enums import CategoricalCalibratorInit
from pytorch_lattice.layers import CategoricalCalibrator

from ..utils import train_calibrated_module


@pytest.mark.parametrize(
    "num_categories,missing_input_value,output_min,output_max,monotonicity_pairs,kernel_init",
    [
        (4, None, None, None, None, CategoricalCalibratorInit.CONSTANT),
        (4, None, None, None, None, CategoricalCalibratorInit.UNIFORM),
        (5, 100.0, 0.0, None, [(1, 2)], CategoricalCalibratorInit.CONSTANT),
        (5, 100.0, 0.0, None, [(1, 2)], CategoricalCalibratorInit.UNIFORM),
        (6, -10.0, None, 5.0, [(2, 1), (3, 1)], CategoricalCalibratorInit.CONSTANT),
        (6, -10.0, None, 5.0, [(2, 1), (3, 1)], CategoricalCalibratorInit.UNIFORM),
        (4, -1.0, -1.0, 2.0, [(0, 1), (1, 2)], CategoricalCalibratorInit.CONSTANT),
        (4, -1.0, -1.0, 2.0, [(0, 1), (1, 2)], CategoricalCalibratorInit.UNIFORM),
    ],
)
def test_initialization(
    num_categories,
    missing_input_value,
    output_min,
    output_max,
    monotonicity_pairs,
    kernel_init,
):
    """Tests that CategoricalCalibrator initialization works properly."""
    calibrator = CategoricalCalibrator(
        num_categories,
        missing_input_value,
        output_min,
        output_max,
        monotonicity_pairs,
        kernel_init,
    )
    if missing_input_value is None:
        assert calibrator.num_categories == num_categories
    else:
        assert calibrator.num_categories == num_categories + 1
    assert calibrator.missing_input_value == missing_input_value
    assert calibrator.output_min == output_min
    assert calibrator.output_max == output_max
    assert calibrator.monotonicity_pairs == monotonicity_pairs
    assert calibrator.kernel_init == kernel_init
    is_constant = torch.all(calibrator.kernel == calibrator.kernel[0])
    if kernel_init == CategoricalCalibratorInit.CONSTANT:
        assert is_constant
    elif kernel_init == CategoricalCalibratorInit.UNIFORM:
        assert not is_constant
    if output_min is not None:
        assert torch.all(calibrator.kernel >= output_min)
    if output_max is not None:
        assert torch.all(calibrator.kernel <= output_max)


@pytest.mark.parametrize(
    "missing_input_value,kernel_data,inputs,expected_outputs",
    [
        (
            None,
            torch.tensor([[1.0], [1.5], [2.0]]).double(),
            torch.tensor([[0], [1], [2]]).long(),
            torch.tensor([[1.0], [1.5], [2.0]]).double(),
        ),
        (
            None,
            torch.tensor([[-1.0], [1.5], [2.0]]).double(),
            torch.tensor([[0], [0], [2], [1]]).long(),
            torch.tensor([[-1.0], [-1.0], [2.0], [1.5]]).double(),
        ),
        (
            -1.0,
            torch.tensor([[1.0], [1.5], [0.5]]).double(),
            torch.tensor([[-1], [0], [1], [-1]]).long(),
            torch.tensor([[0.5], [1.0], [1.5], [0.5]]).double(),
        ),
    ],
)
def test_forward(missing_input_value, kernel_data, inputs, expected_outputs):
    """Tests that forward properly calibrated inputs."""
    num_categories = kernel_data.size()[0]
    if missing_input_value is not None:
        num_categories -= 1
    calibrator = CategoricalCalibrator(num_categories, missing_input_value)
    calibrator.kernel.data = kernel_data
    outputs = calibrator(inputs)
    assert torch.allclose(outputs, expected_outputs)


@pytest.mark.parametrize(
    "kernel_data,monotonicity_pairs,expected_out",
    [
        (torch.tensor([[1.0], [0.8], [0.7], [1.5]]), [(0, 1), (2, 0), (0, 3)], []),
        (
            torch.tensor([[1.0], [0.8], [1.5], [2.0]]),
            [(0, 1), (2, 1), (3, 2)],
            ["Monotonicity violated at: [(2, 1), (3, 2)]."],
        ),
        (
            torch.tensor([[1.0], [0.8], [0.5], [2.5]]),
            [(0, 2), (1, 2), (3, 2)],
            ["Monotonicity violated at: [(0, 2), (1, 2), (3, 2)]."],
        ),
    ],
)
def test_assert_constraints_monotonicity(kernel_data, monotonicity_pairs, expected_out):
    """Tests assert_constraints for monotonicity pairs with a tolerance of eps."""
    calibrator = CategoricalCalibrator(kernel_data.size()[0])
    calibrator.kernel.data = kernel_data
    calibrator.monotonicity_pairs = monotonicity_pairs
    assert calibrator.assert_constraints(eps=0.25) == expected_out


@pytest.mark.parametrize(
    "kernel_data,monotonicity_pairs,expected_out",
    [
        (torch.tensor([[1.0], [0.8], [0.7], [1.5]]), [], []),
        (torch.tensor([[1.0], [-0.1], [0.7], [2.1]]), [], []),
        (
            torch.tensor([[1.0], [0.8], [0.5], [2.5]]),
            [],
            ["Max weight greater than output_max."],
        ),
        (
            torch.tensor([[1.0], [0.8], [-0.5], [2.0]]),
            [],
            ["Min weight less than output_min."],
        ),
        (
            torch.tensor([[1.0], [0.8], [-0.5], [2.5]]),
            [],
            ["Max weight greater than output_max.", "Min weight less than output_min."],
        ),
    ],
)
def test_assert_constraints_output_bounds(
    kernel_data, monotonicity_pairs, expected_out
):
    """Tests assert_constraints for output bounds with a tolerance of eps."""
    calibrator = CategoricalCalibrator(kernel_data.size()[0])
    calibrator.kernel.data = kernel_data
    calibrator.monotonicity_pairs = monotonicity_pairs
    calibrator.output_min = 0.0
    calibrator.output_max = 2.0
    assert calibrator.assert_constraints(eps=0.25) == expected_out


@pytest.mark.parametrize(
    "kernel_data,monotonicity_pairs,expected_out",
    [
        (torch.tensor([[1.0], [0.8], [0.5], [1.8]]), [(1, 0), (2, 3)], []),
        (
            torch.tensor([[1.0], [0.8], [-0.5], [2.0]]),
            [(1, 0), (3, 2), (2, 1)],
            ["Min weight less than output_min.", "Monotonicity violated at: [(3, 2)]."],
        ),
        (
            torch.tensor([[1.0], [0.8], [-0.5], [2.5]]),
            [(1, 0), (0, 3)],
            ["Max weight greater than output_max.", "Min weight less than output_min."],
        ),
        (
            torch.tensor([[1.0], [0.8], [-0.5], [2.5]]),
            [(1, 0), (3, 0), (1, 2)],
            [
                "Max weight greater than output_max.",
                "Min weight less than output_min.",
                "Monotonicity violated at: [(3, 0), (1, 2)].",
            ],
        ),
    ],
)
def test_assert_constraints_combo(kernel_data, monotonicity_pairs, expected_out):
    """Tests assert_constraints for output bounds and monotonicity together"""
    calibrator = CategoricalCalibrator(kernel_data.size()[0])
    calibrator.kernel.data = kernel_data
    calibrator.monotonicity_pairs = monotonicity_pairs
    calibrator.output_min = 0.0
    calibrator.output_max = 2.0
    assert calibrator.assert_constraints(eps=0.25) == expected_out


def test_constrain_no_constraints():
    """Tests that constrain does nothing when there are no constraints."""
    calibrator = CategoricalCalibrator(
        3, kernel_init=CategoricalCalibratorInit.CONSTANT
    )
    calibrator.constrain()
    expected_kernel_data = torch.tensor([[0.0], [0.0], [0.0]]).double()
    assert torch.allclose(calibrator.kernel.data, expected_kernel_data)


@pytest.mark.parametrize(
    "output_min,kernel_data",
    [
        (2.0, torch.tensor([[1.0], [-2.0], [3.0]])),
        (-2.0, torch.tensor([[3.0], [-5.0]]).double()),
    ],
)
def test_constrain_only_output_min(output_min, kernel_data):
    """Tests that constrain properly projects kernel into output_min constraint."""
    calibrator = CategoricalCalibrator(kernel_data.size()[0], output_min=output_min)
    calibrator.kernel.data = kernel_data
    calibrator.constrain()
    assert torch.all(calibrator.keypoints_outputs() >= output_min)


@pytest.mark.parametrize(
    "output_max,kernel_data",
    [
        (2.0, torch.tensor([[1.0], [-2.0], [3.0]])),
        (-2.0, torch.tensor([[3.0], [-5.0]]).double()),
    ],
)
def test_constrain_only_output_max(output_max, kernel_data):
    """Tests that constrain properly projects kernel into output_max constraint."""
    calibrator = CategoricalCalibrator(kernel_data.size()[0], output_max=output_max)
    calibrator.kernel.data = kernel_data
    calibrator.constrain()
    assert torch.all(calibrator.keypoints_outputs() <= output_max)


@pytest.mark.parametrize(
    "output_min,output_max,kernel_data",
    [
        (0.0, 1.0, torch.tensor([[-1.0], [0.0], [1.0]]).double()),
        (-2.0, -1.0, torch.tensor([[-1.0], [-3.0], [1.0]]).double()),
        (-1.0, 1.0, torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]).double()),
    ],
)
def test_constrain_bounds(output_min, output_max, kernel_data):
    """Tests that constrain properly projects kernel into output bounds."""
    calibrator = CategoricalCalibrator(
        kernel_data.size()[0], output_min=output_min, output_max=output_max
    )
    # pylint: disable=R0801
    calibrator.kernel.data = kernel_data
    calibrator.constrain()
    keypoints_outputs = calibrator.keypoints_outputs()
    assert torch.all(keypoints_outputs >= output_min)
    assert torch.all(keypoints_outputs <= output_max)
    # pylint: enable=R0801


@pytest.mark.parametrize(
    "monotonicity_pairs,kernel_data",
    [
        ([(0, 1)], torch.tensor([[1.0], [0.8]]).double()),
        ([(0, 1), (1, 2)], torch.tensor([[1.0], [0.8], [0.6]]).double()),
        ([(0, 1), (0, 2)], torch.tensor([[1.0], [0.8], [0.6]]).double()),
        ([(1, 0)], torch.tensor([[0.8], [1.0]]).double()),
        ([(2, 1), (1, 0)], torch.tensor([[0.6], [0.8], [1.0]]).double()),
        ([(2, 1), (2, 0)], torch.tensor([[0.6], [0.8], [1.0]]).double()),
    ],
)
def test_constrain_monotonicity_pairs(monotonicity_pairs, kernel_data):
    """Tests that contrain properly projects kernel to match monotonicity pairs."""
    calibrator = CategoricalCalibrator(
        kernel_data.size()[0], monotonicity_pairs=monotonicity_pairs
    )
    calibrator.kernel.data = kernel_data
    calibrator.constrain()
    keypoints_outputs = calibrator.keypoints_outputs()
    for i, j in monotonicity_pairs:
        assert keypoints_outputs[i] <= keypoints_outputs[j]


@pytest.mark.parametrize(
    "output_min,monotonicity_pairs,kernel_data",
    [
        (-1.0, [(0, 1)], torch.tensor([[-1.5], [-1.8]]).double()),
        (1.0, [(0, 1), (0, 2)], torch.tensor([[1.2], [1.0], [0.9]]).double()),
        (0.0, [(0, 1), (1, 2)], torch.tensor([[1.0], [0.8], [0.6], [-1.0]]).double()),
    ],
)
def test_constrain_output_min_with_monotonicity_pairs(
    output_min, monotonicity_pairs, kernel_data
):
    """Tests constaining output min with monotonicity pairs."""
    calibrator = CategoricalCalibrator(
        kernel_data.size()[0],
        output_min=output_min,
        monotonicity_pairs=monotonicity_pairs,
    )
    calibrator.kernel.data = kernel_data
    calibrator.constrain()
    keypoints_outputs = calibrator.keypoints_outputs()
    assert torch.all(keypoints_outputs >= output_min)
    for i, j in monotonicity_pairs:
        assert keypoints_outputs[i] <= keypoints_outputs[j]


@pytest.mark.parametrize(
    "output_max,monotonicity_pairs,kernel_data",
    [
        (-1.0, [(0, 1)], torch.tensor([[-0.8], [-1.0]]).double()),
        (1.0, [(0, 1), (0, 2)], torch.tensor([[1.2], [1.0], [0.9]]).double()),
        (0.0, [(0, 1), (1, 2)], torch.tensor([[0.1], [0.0], [-0.1]]).double()),
    ],
)
def test_constrain_output_max_with_monotonicity_pairs(
    output_max, monotonicity_pairs, kernel_data
):
    """Tests constaining output max with monotonicity pairs."""
    calibrator = CategoricalCalibrator(
        kernel_data.size()[0],
        output_max=output_max,
        monotonicity_pairs=monotonicity_pairs,
    )
    calibrator.kernel.data = kernel_data
    calibrator.constrain()
    keypoints_outputs = calibrator.keypoints_outputs()
    assert torch.all(keypoints_outputs <= output_max)
    for i, j in monotonicity_pairs:
        assert keypoints_outputs[i] == keypoints_outputs[j]


@pytest.mark.parametrize(
    "output_min,output_max,monotonicity_pairs,kernel_data",
    [
        (-1.0, 1.0, [(0, 1)], torch.tensor([[-0.8], [-1.0], [2.0]]).double()),
        (
            1.0,
            5.0,
            [(0, 1), (0, 2)],
            torch.tensor([[1.2], [1.1], [0.9], [6.0]]).double(),
        ),
        (
            -1.0,
            2.0,
            [(0, 1), (1, 2)],
            torch.tensor([[2.2], [1.8], [1.7], [-2.0]]).double(),
        ),
    ],
)
def test_constrain_bounds_with_monotonicity_pairs(
    output_min, output_max, monotonicity_pairs, kernel_data
):
    """Tests constaining bounds with monotonicity pairs."""
    calibrator = CategoricalCalibrator(
        kernel_data.size()[0],
        output_min=output_min,
        output_max=output_max,
        monotonicity_pairs=monotonicity_pairs,
    )
    # pylint: disable=R0801
    calibrator.kernel.data = kernel_data
    calibrator.constrain()
    keypoints_outputs = calibrator.keypoints_outputs()
    assert torch.all(keypoints_outputs >= output_min)
    assert torch.all(keypoints_outputs <= output_max)
    for i, j in monotonicity_pairs:
        assert keypoints_outputs[i] <= keypoints_outputs[j]
    # pylint: enable=R0801


@pytest.mark.parametrize(
    "num_categories,missing_input_value,expected_keypoints_inputs",
    [
        (3, None, torch.tensor([0, 1, 2]).long()),
        (3, -1, torch.tensor([0, 1, 2, -1]).long()),
    ],
)
def test_keypoints_inputs(
    num_categories, missing_input_value, expected_keypoints_inputs
):
    """Tests that the correct keypoint inputs are returned."""
    calibrator = CategoricalCalibrator(num_categories, missing_input_value)
    assert torch.allclose(calibrator.keypoints_inputs(), expected_keypoints_inputs)


@pytest.mark.parametrize(
    "kernel_data,expected_keypoints_outputs",
    [
        (
            torch.tensor([[0.0], [1.0], [2.0]]).double(),
            torch.tensor([0.0, 1.0, 2.0]).double(),
        ),
        (
            torch.tensor([[-1.0], [3.0], [0.5], [-4.2]]).double(),
            torch.tensor([-1.0, 3.0, 0.5, -4.2]).double(),
        ),
    ],
)
def test_keypoints_outputs(kernel_data, expected_keypoints_outputs):
    """Tests that the correct keypoint outputs are returned."""
    calibrator = CategoricalCalibrator(kernel_data.size()[0])
    calibrator.kernel.data = kernel_data
    assert torch.allclose(calibrator.keypoints_outputs(), expected_keypoints_outputs)


@pytest.mark.parametrize(
    "monotonicity_pairs,kernel_data,expected_projected_kernel_data",
    [
        (
            [(0, 1)],
            torch.tensor([[1.0], [0.8]]).double(),
            torch.tensor([[0.9], [0.9]]).double(),
        ),
        (
            [(0, 1), (0, 2)],
            torch.tensor([[1.0], [0.8], [0.6]]).double(),
            torch.tensor([[0.775], [0.85], [0.775]]).double(),
        ),
        (
            [(0, 1), (1, 2)],
            torch.tensor([[1.0], [0.8], [0.6]]).double(),
            torch.tensor([[0.8], [0.8], [0.8]]).double(),
        ),
    ],
)
def test_approximately_project_monotonicity_pairs(
    monotonicity_pairs, kernel_data, expected_projected_kernel_data
):
    """Tests that kernel is properly projected to match monotonicity pairs."""
    calibrator = CategoricalCalibrator(
        kernel_data.size()[0], monotonicity_pairs=monotonicity_pairs
    )
    # pylint: disable=protected-access
    projected_kernel_data = calibrator._approximately_project_monotonicity_pairs(
        kernel_data
    )
    # pylint: enable=protected-access
    assert torch.allclose(projected_kernel_data, expected_projected_kernel_data)


def test_training():  # pylint: disable=too-many-locals
    """Tests that the `CategoricalCalibrator` module can learn a mapping."""
    num_categories, num_examples_per_category = 5, 200
    training_examples = np.concatenate(
        [[c] * num_examples_per_category for c in range(num_categories)]
    )
    np.random.shuffle(training_examples)
    training_examples = torch.from_numpy(np.expand_dims(training_examples, 1))
    training_labels = training_examples.double()

    num_examples = num_categories * num_examples_per_category

    calibrator = CategoricalCalibrator(
        num_categories,
        output_min=0.0,
        output_max=num_categories - 1,
        monotonicity_pairs=[(i, i + 1) for i in range(num_categories - 1)],
    )

    # pylint: disable=R0801
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=1e-2)

    train_calibrated_module(
        calibrator,
        training_examples,
        training_labels,
        loss_fn,
        optimizer,
        300,
        num_examples // 10,
    )

    keypoints_inputs = calibrator.keypoints_inputs()
    keypoints_outputs = calibrator.keypoints_outputs()
    assert torch.allclose(keypoints_inputs.double(), keypoints_outputs, atol=2e-2)
    # pylint: enable=R0801
