"""Tests for NumericalCalibrator module."""
import numpy as np
import pytest
import torch

from pytorch_lattice import Monotonicity, NumericalCalibratorInit
from pytorch_lattice.layers import NumericalCalibrator

from ..utils import train_calibrated_module


@pytest.mark.parametrize(
    "input_keypoints,missing_input_value,output_min,output_max,monotonicity,"
    "kernel_init,projection_iterations,expected_kernel",
    [
        (
            np.linspace(1.0, 4.0, num=5),
            None,
            None,
            None,
            None,
            NumericalCalibratorInit.EQUAL_HEIGHTS,
            10,
            torch.tensor([[[-2.0]] + ([[1.0]] * 4)]).double(),
        ),
        (
            np.linspace(-2.0, 8.0, num=11),
            -1.0,
            -1.0,
            1.0,
            Monotonicity.INCREASING,
            NumericalCalibratorInit.EQUAL_SLOPES,
            4,
            torch.tensor([[-1.0]] + ([[0.2]] * 10)).double(),
        ),
        (
            np.linspace(-2.0, 8.0, num=17),
            20,
            2.0,
            None,
            Monotonicity.DECREASING,
            NumericalCalibratorInit.EQUAL_HEIGHTS,
            1,
            torch.tensor([[6.0]] + ([[-0.25]] * 16)).double(),
        ),
    ],
)
def test_initialization(
    input_keypoints,
    missing_input_value,
    output_min,
    output_max,
    monotonicity,
    kernel_init,
    projection_iterations,
    expected_kernel,
):
    """Tests that NumericalCalibrator class initialization works properly."""
    calibrator = NumericalCalibrator(
        input_keypoints,
        missing_input_value,
        output_min,
        output_max,
        monotonicity,
        kernel_init,
        projection_iterations,
    )
    assert torch.allclose(calibrator.kernel.data, expected_kernel)
    assert (calibrator.input_keypoints == input_keypoints).all()
    assert calibrator.missing_input_value == missing_input_value
    assert calibrator.output_min == output_min
    assert calibrator.output_max == output_max
    assert calibrator.monotonicity == monotonicity
    assert calibrator.kernel_init == kernel_init
    assert calibrator.projection_iterations == projection_iterations


@pytest.mark.parametrize(
    "input_keypoints,kernel_init,kernel_data,inputs,expected_outputs",
    [
        (
            np.linspace(1.0, 5.0, num=5),
            NumericalCalibratorInit.EQUAL_HEIGHTS,
            None,
            torch.tensor(
                [
                    [0.5],
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0],
                    [5.5],
                ]
            ).double(),
            torch.tensor(
                [
                    [-2.0],
                    [-2.0],
                    [-1.0],
                    [0.0],
                    [1.0],
                    [2.0],
                    [2.0],
                ]
            ).double(),
        ),
        (
            np.linspace(1.0, 5.0, num=5),
            NumericalCalibratorInit.EQUAL_HEIGHTS,
            torch.tensor([[2.0], [-4.0], [2.0], [-1.0], [2.0]]).double(),
            torch.tensor([[1.5], [2.5], [3.5], [4.5]]).double(),
            torch.tensor([[0.0], [-1.0], [-0.5], [0.0]]).double(),
        ),
        (
            np.array([1.0, 3.0, 4.0, 5.0, 7.0, 9.0]),
            NumericalCalibratorInit.EQUAL_SLOPES,
            None,
            torch.tensor(
                [
                    [1.0],
                    [1.5],
                    [2.0],
                    [2.5],
                    [3.0],
                    [3.5],
                    [4.0],
                    [4.5],
                    [5.0],
                    [5.5],
                    [6.0],
                    [6.5],
                    [7.0],
                    [7.5],
                    [8.0],
                    [8.5],
                    [9.0],
                ]
            ).double(),
            torch.tensor(
                [
                    [-2.0],
                    [-1.75],
                    [-1.5],
                    [-1.25],
                    [-1.0],
                    [-0.75],
                    [-0.5],
                    [-0.25],
                    [0.0],
                    [0.25],
                    [0.5],
                    [0.75],
                    [1.0],
                    [1.25],
                    [1.5],
                    [1.75],
                    [2.0],
                ]
            ).double(),
        ),
    ],
)
def test_forward(input_keypoints, kernel_init, kernel_data, inputs, expected_outputs):
    """Tests that forward properly calibrated inputs."""
    calibrator = NumericalCalibrator(input_keypoints, kernel_init=kernel_init)
    if kernel_data is not None:
        calibrator.kernel.data = kernel_data
    outputs = calibrator.forward(inputs)
    assert torch.allclose(outputs, expected_outputs)


@pytest.mark.parametrize(
    "kernel_data,monotonicity, expected_out",
    [
        (torch.tensor([[0.0], [0.1], [0.2], [0.3]]), Monotonicity.INCREASING, []),
        (torch.tensor([[0.0], [0.1], [-0.01], [0.3]]), Monotonicity.INCREASING, []),
        (
            torch.tensor([[-1.0], [-0.3], [0.2], [-0.1]]),
            Monotonicity.INCREASING,
            ["Monotonicity violated at: [(0, 1), (2, 3)]."],
        ),
        (torch.tensor([[3.0], [-0.2], [-0.1], [0.0]]), Monotonicity.DECREASING, []),
        (torch.tensor([[3.0], [0.01], [-0.1], [0.0]]), Monotonicity.DECREASING, []),
        (
            torch.tensor([[1.0], [0.1], [-0.2], [0.3]]),
            Monotonicity.DECREASING,
            ["Monotonicity violated at: [(0, 1), (2, 3)]."],
        ),
        (torch.tensor([[-0.4], [1.0], [-1.0], [0.0]]), None, []),
    ],
)
def test_assert_constraints_monotonicity(kernel_data, monotonicity, expected_out):
    """Tests assert_constraints maintains monotonicity with a tolerance of eps."""
    calibrator = NumericalCalibrator(np.linspace(1.0, 4.0, num=4))
    calibrator.kernel.data = kernel_data
    calibrator.monotonicity = monotonicity
    assert calibrator.assert_constraints(eps=0.05) == expected_out


@pytest.mark.parametrize(
    "kernel_data,expected_out",
    [
        (torch.tensor([[1.0], [-0.1], [-0.1], [-0.1]]), []),
        (torch.tensor([[1.0], [0.0], [0.0], [0.05]]), []),
        (torch.tensor([[0.0], [0.0], [0.0], [-0.05]]), []),
        (
            torch.tensor([[0.0], [0.5], [0.5], [0.2]]),
            ["Max weight greater than output_max."],
        ),
        (
            torch.tensor([[1.0], [-0.5], [-0.5], [-0.2]]),
            ["Min weight less than output_min."],
        ),
        (
            torch.tensor([[0.5], [0.8], [-1.5], [0.0]]),
            ["Max weight greater than output_max.", "Min weight less than output_min."],
        ),
    ],
)
def test_assert_constraints_output_bounds(kernel_data, expected_out):
    """Tests assert_constraints for output bounds with a tolerance of eps."""
    calibrator = NumericalCalibrator(np.linspace(1.0, 4.0, num=4))
    calibrator.kernel.data = kernel_data
    calibrator.monotonicity = None
    calibrator.output_min = 0.0
    calibrator.output_max = 1.0
    assert calibrator.assert_constraints(eps=0.1) == expected_out


@pytest.mark.parametrize(
    "kernel_data,monotonicity, expected_out",
    [
        (
            torch.tensor([[0.0], [2.0], [-0.5], [2.5]]),
            Monotonicity.INCREASING,
            [
                "Max weight greater than output_max.",
                "Monotonicity violated at: [(1, 2)].",
            ],
        ),
        (
            torch.tensor([[-1.0], [3.0], [0.5], [-1.0]]),
            Monotonicity.INCREASING,
            [
                "Max weight greater than output_max.",
                "Min weight less than output_min.",
                "Monotonicity violated at: [(2, 3)].",
            ],
        ),
        (
            torch.tensor([[2.0], [-2.5], [0.5], [-1.0]]),
            Monotonicity.DECREASING,
            ["Min weight less than output_min.", "Monotonicity violated at: [(1, 2)]."],
        ),
        (torch.tensor([[0.3], [0.2], [0.1], [-0.1]]), Monotonicity.DECREASING, []),
    ],
)
def test_assert_constraints_combo(kernel_data, monotonicity, expected_out):
    """Tests assert_constraints for monotonicity and output bounds together."""
    calibrator = NumericalCalibrator(np.linspace(1.0, 4.0, num=4))
    calibrator.monotonicity = monotonicity
    calibrator.kernel.data = kernel_data
    calibrator.output_min = 0.0
    calibrator.output_max = 2.0
    assert calibrator.assert_constraints(eps=0.25) == expected_out


def test_constrain_no_constraints():
    """Tests that constrain does nothing when there are no constraints."""
    calibrator = NumericalCalibrator(np.linspace(1.0, 5.0, num=5))
    calibrator.apply_constraints()
    expected_kernel_data = torch.tensor([[-2.0], [1.0], [1.0], [1.0], [1.0]]).double()
    assert torch.allclose(calibrator.kernel.data, expected_kernel_data)


@pytest.mark.parametrize(
    "output_min,kernel_data",
    [
        (2.0, torch.tensor([[-3.0], [1.0], [1.0], [1.0], [1.0]]).double()),
        (-3.0, torch.tensor([[-3.0], [1.0], [-12.0], [8.0], [-1.0]]).double()),
    ],
)
def test_constrain_only_output_min(output_min, kernel_data):
    """Tests that constrain properly projects kernel into output_min constraint."""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 5.0, num=5), output_min=output_min
    )
    calibrator.kernel.data = kernel_data
    calibrator.apply_constraints()
    assert torch.all(calibrator.keypoints_outputs() >= output_min)


@pytest.mark.parametrize(
    "output_max,kernel_data",
    [
        (-2.0, torch.tensor([[-3.0], [1.0], [1.0], [1.0], [1.0]]).double()),
        (5.0, torch.tensor([[-3.0], [1.0], [6.0], [9.0], [-11.0]]).double()),
    ],
)
def test_constrain_only_output_max(output_max, kernel_data):
    """Tests that constrain properly projects kernel into output_max constraint."""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 5.0, num=5), output_max=output_max
    )
    calibrator.kernel.data = kernel_data
    calibrator.apply_constraints()
    assert torch.all(calibrator.keypoints_outputs() <= output_max)


@pytest.mark.parametrize(
    "output_min,output_max,kernel_data",
    [
        (1.0, 5.0, torch.tensor([[-2.0], [1.0], [1.0], [1.0], [1.0], [3.0]]).double()),
        (
            -5.0,
            0.0,
            torch.tensor([[-2.0], [-8.0], [12.0], [3.0], [-30.0], [3.0]]).double(),
        ),
    ],
)
def test_constrain_bounds(output_min, output_max, kernel_data):
    """Tests that constrain properly projects kernel into output bounds."""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 5.0, num=6), output_min=output_min, output_max=output_max
    )
    calibrator.kernel.data = kernel_data
    calibrator.apply_constraints()
    keypoints_outputs = calibrator.keypoints_outputs()
    assert torch.all(keypoints_outputs >= output_min)
    assert torch.all(keypoints_outputs <= output_max)


@pytest.mark.parametrize(
    "kernel_data",
    [
        (torch.tensor([[0.0], [1.0], [-2.0], [3.0], [1.0]]).double()),
        (torch.tensor([[-2.0], [-2.0], [1.0], [-3.0], [-4.0]]).double()),
    ],
)
def test_constrain_increasing_monotonicity(kernel_data):
    """Tests that contrain properly projects kernel to be increasingly monotonic."""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 5.0, num=5), monotonicity=Monotonicity.INCREASING
    )
    calibrator.kernel.data = kernel_data
    calibrator.apply_constraints()
    heights = calibrator.kernel.data[1:]
    assert torch.all(heights >= 0)


@pytest.mark.parametrize(
    "kernel_data",
    [
        (torch.tensor([[0.0], [1.0], [-2.0], [3.0]]).double()),
        (torch.tensor([[-2.0], [-2.0], [1.0], [-3.0]]).double()),
    ],
)
def test_constrain_decreasing_monotonicity(kernel_data):
    """Tests that contrain properly projects kernel to be decreasingly monotonic."""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 5.0, num=4), monotonicity=Monotonicity.DECREASING
    )
    calibrator.kernel.data = kernel_data
    calibrator.apply_constraints()
    heights = calibrator.kernel.data[1:]
    assert torch.all(heights <= 0)


@pytest.mark.parametrize(
    "output_min,monotonicity,kernel_data",
    [
        (
            -2.0,
            Monotonicity.INCREASING,
            torch.tensor([[-3.0], [1.0], [-2.5], [2.0]]).double(),
        ),
        (
            3.0,
            Monotonicity.INCREASING,
            torch.tensor([[-3.0], [-1.0], [4.0], [-2.0]]).double(),
        ),
        (
            -3.0,
            Monotonicity.DECREASING,
            torch.tensor([[-3.0], [-1.0], [4.0], [-2.0]]).double(),
        ),
        (
            2.0,
            Monotonicity.DECREASING,
            torch.tensor([[-1.0], [-2.0], [-4.0], [2.0]]).double(),
        ),
    ],
)
def test_constrain_output_min_monotonicity(output_min, monotonicity, kernel_data):
    """Tests contraining output min with monotonicity constraints."""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 5.0, num=4),
        output_min=output_min,
        monotonicity=monotonicity,
    )
    calibrator.kernel.data = kernel_data
    calibrator.apply_constraints()
    keypoints_outputs = calibrator.keypoints_outputs()
    assert torch.all(keypoints_outputs >= output_min)
    heights = calibrator.kernel.data[1:]
    if monotonicity == Monotonicity.INCREASING:
        assert torch.all(heights >= 0)
    else:
        assert torch.all(heights <= 0)


@pytest.mark.parametrize(
    "output_max,monotonicity,kernel_data",
    [
        (
            -2.0,
            Monotonicity.INCREASING,
            torch.tensor([[-1.0], [1.0], [-2.5], [2.0]]).double(),
        ),
        (
            3.0,
            Monotonicity.INCREASING,
            torch.tensor([[4.0], [-1.0], [4.0], [-2.0]]).double(),
        ),
        (
            3.0,
            Monotonicity.DECREASING,
            torch.tensor([[4.0], [-1.0], [4.0], [-2.0]]).double(),
        ),
        (
            -2.0,
            Monotonicity.DECREASING,
            torch.tensor([[3.0], [-2.0], [3.0], [-4.0]]).double(),
        ),
    ],
)
def test_constrain_output_max_with_monotonicity(output_max, monotonicity, kernel_data):
    """Tests contraining output max with monotonicity constraints."""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 5.0, num=4),
        output_max=output_max,
        monotonicity=monotonicity,
    )
    calibrator.kernel.data = kernel_data
    calibrator.apply_constraints()
    keypoints_outputs = calibrator.keypoints_outputs()
    assert torch.all(keypoints_outputs <= output_max)
    heights = calibrator.kernel.data[1:]
    if monotonicity == Monotonicity.INCREASING:
        assert torch.all(heights >= 0)
    else:
        assert torch.all(heights <= 0)


@pytest.mark.parametrize(
    "output_min,output_max,monotonicity,kernel_data",
    [
        (
            -1.0,
            1.0,
            Monotonicity.INCREASING,
            torch.tensor([[-1.5], [1.5], [1.5], [-1.0]]).double(),
        ),
        (
            -1.0,
            1.0,
            Monotonicity.DECREASING,
            torch.tensor([[1.5], [-1.5], [-1.5], [1.0]]).double(),
        ),
    ],
)
def test_constrain_bounds_with_monotonicity(
    output_min, output_max, monotonicity, kernel_data
):
    """Tests constraining output bounds with monotonicity constraints."""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 5.0, num=4),
        output_min=output_min,
        output_max=output_max,
        monotonicity=monotonicity,
    )
    calibrator.kernel.data = kernel_data
    calibrator.apply_constraints()
    keypoints_outputs = calibrator.keypoints_outputs()
    assert torch.all(keypoints_outputs >= output_min)
    assert torch.all(keypoints_outputs <= output_max)
    heights = calibrator.kernel.data[1:]
    if monotonicity == Monotonicity.INCREASING:
        assert torch.all(heights >= 0)
    else:
        assert torch.all(heights <= 0)


@pytest.mark.parametrize(
    "input_keypoints",
    [(np.linspace(1.0, 5.0, num=5)), (np.linspace(1.0, 10.0, num=34))],
)
def test_keypoints_inputs(input_keypoints):
    """Tests that the correct keypoint inputs are returned."""
    calibrator = NumericalCalibrator(input_keypoints)
    assert torch.allclose(calibrator.keypoints_inputs(), torch.tensor(input_keypoints))


@pytest.mark.parametrize(
    "num_keypoints,kernel_data,expected_keypoints_outputs",
    [
        (
            5,
            torch.tensor([[0.0], [0.2], [0.7], [1.5], [4.8]]).double(),
            torch.tensor([0.0, 0.2, 0.9, 2.4, 7.2]).double(),
        ),
        (
            6,
            torch.tensor([[-2.0], [4.0], [-2.0], [0.5], [-1.7], [3.4]]).double(),
            torch.tensor([-2.0, 2.0, 0.0, 0.5, -1.2, 2.2]).double(),
        ),
    ],
)
def test_keypoints_outputs(num_keypoints, kernel_data, expected_keypoints_outputs):
    """Tests that the correct keypoint outputs are returned."""
    calibrator = NumericalCalibrator(np.linspace(1.0, 5.0, num=num_keypoints))
    calibrator.kernel.data = kernel_data
    assert torch.allclose(calibrator.keypoints_outputs(), expected_keypoints_outputs)


@pytest.mark.parametrize(
    "input_keypoints,output_min,output_max,monotonicity,kernel_data,"
    "expected_projected_kernel_data",
    [
        (
            np.linspace(1.0, 4.0, num=4),
            None,
            None,
            Monotonicity.INCREASING,
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            1.0,
            None,
            Monotonicity.INCREASING,
            torch.tensor([[1.0], [1.0], [2.0], [3.0]]).double(),
            torch.tensor([[1.0], [1.0], [2.0], [3.0]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            1.0,
            None,
            Monotonicity.INCREASING,
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
            torch.tensor([[1.0], [1.0], [1.0], [1.0]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            None,
            5.0,
            Monotonicity.INCREASING,
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            None,
            5.0,
            Monotonicity.INCREASING,
            torch.tensor([[3.0], [1.0], [2.0], [2.0]]).double(),
            torch.tensor([[2.25], [0.25], [1.25], [1.25]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            3.0,
            5.0,
            Monotonicity.INCREASING,
            torch.tensor([[3.0], [1.0], [2.0], [2.0]]).double(),
            torch.tensor([[3.0], [0.0], [1.0], [1.0]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            None,
            None,
            Monotonicity.DECREASING,
            torch.tensor([[0.0], [-1.0], [-1.0], [-1.0]]).double(),
            torch.tensor([[0.0], [-1.0], [-1.0], [-1.0]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            -5.0,
            None,
            Monotonicity.DECREASING,
            torch.tensor([[0.0], [-1.0], [-1.0], [-1.0]]).double(),
            torch.tensor([[0.0], [-1.0], [-1.0], [-1.0]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            -5.0,
            None,
            Monotonicity.DECREASING,
            torch.tensor([[-3.0], [-1.0], [-2.0], [-2.0]]).double(),
            torch.tensor([[-2.25], [-0.25], [-1.25], [-1.25]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            None,
            -1.0,
            Monotonicity.DECREASING,
            torch.tensor([[1.0], [-1.0], [-2.0], [-3.0]]).double(),
            torch.tensor([[-1.0], [-1.0], [-2.0], [-3.0]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            None,
            -1.0,
            Monotonicity.DECREASING,
            torch.tensor([[0.0], [-1.0], [-1.0], [-1.0]]).double(),
            torch.tensor([[-1.0], [-1.0], [-1.0], [-1.0]]).double(),
        ),
        (
            np.linspace(1.0, 4.0, num=4),
            -5.0,
            -3.0,
            Monotonicity.DECREASING,
            torch.tensor([[-3.0], [-1.0], [-2.0], [-2.0]]).double(),
            torch.tensor([[-3.0], [0.0], [-1.0], [-1.0]]).double(),
        ),
    ],
)
def test_project_monotonic_bounds(
    input_keypoints,
    output_min,
    output_max,
    monotonicity,
    kernel_data,
    expected_projected_kernel_data,
):
    """Tests that kernel is properly projected into bounds with monotonicity."""
    calibrator = NumericalCalibrator(
        input_keypoints,
        output_min=output_min,
        output_max=output_max,
        monotonicity=monotonicity,
    )
    bias, heights = kernel_data[0:1], kernel_data[1:]
    (
        projected_bias,
        projected_heights,
    ) = calibrator._project_monotonic_bounds(bias, heights)
    projected_kernel_data = torch.cat((projected_bias, projected_heights), 0)
    assert torch.allclose(projected_kernel_data, expected_projected_kernel_data)


@pytest.mark.parametrize(
    "output_min,output_max,kernel_data,expected_projected_kernel_data",
    [
        (
            None,
            None,
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
        ),
        (
            1.0,
            None,
            torch.tensor([[1.0], [1.0], [2.0], [3.0]]).double(),
            torch.tensor([[1.0], [1.0], [2.0], [3.0]]).double(),
        ),
        (
            1.0,
            None,
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
            torch.tensor([[1.0], [0.0], [1.0], [1.0]]).double(),
        ),
        (
            None,
            5.0,
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
        ),
        (
            None,
            5.0,
            torch.tensor([[4.0], [1.0], [2.0], [1.0]]).double(),
            torch.tensor([[4.0], [1.0], [0.0], [0.0]]).double(),
        ),
        (
            3.0,
            5.0,
            torch.tensor([[4.0], [1.0], [-3.0], [1.0]]).double(),
            torch.tensor([[4.0], [1.0], [-2.0], [0.0]]).double(),
        ),
    ],
)
def test_approximately_project_bounds_only(
    output_min,
    output_max,
    kernel_data,
    expected_projected_kernel_data,
):
    """Tests that bounds are properly projected when monotonicity is NONE."""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 4.0, num=4),
        output_min=output_min,
        output_max=output_max,
        monotonicity=None,
    )
    bias, heights = kernel_data[0:1], kernel_data[1:]
    (
        projected_bias,
        projected_heights,
    ) = calibrator._approximately_project_bounds_only(bias, heights)
    projected_kernel_data = torch.cat((projected_bias, projected_heights), 0)
    assert torch.allclose(projected_kernel_data, expected_projected_kernel_data)


@pytest.mark.parametrize(
    "monotonicity,heights,expected_projected_heights",
    [
        (
            Monotonicity.INCREASING,
            torch.tensor([[1.0], [-2.0], [3.0], [-1.0], [0.5]]).double(),
            torch.tensor([[1.0], [0.0], [3.0], [0.0], [0.5]]).double(),
        ),
        (
            Monotonicity.DECREASING,
            torch.tensor([[-1.0], [2.0], [-3.0], [1.0], [-0.5]]).double(),
            torch.tensor([[-1.0], [0.0], [-3.0], [0.0], [-0.5]]).double(),
        ),
    ],
)
def test_project_monotonicity(
    monotonicity,
    heights,
    expected_projected_heights,
):
    """Tests that monotonicity is properly projected"""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 4.0, num=4), monotonicity=monotonicity
    )
    projected_heights = calibrator._project_monotonicity(heights)
    assert torch.allclose(projected_heights, expected_projected_heights)


@pytest.mark.parametrize(
    "monotonicity,output_min,output_max,kernel_data,expected_projected_kernel_data",
    [
        (
            Monotonicity.INCREASING,
            None,
            None,
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
            torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
        ),
        (
            Monotonicity.INCREASING,
            1.0,
            None,
            torch.tensor([[1.0], [1.0], [2.0], [3.0]]).double(),
            torch.tensor([[1.0], [1.0], [2.0], [3.0]]).double(),
        ),
        (
            Monotonicity.INCREASING,
            None,
            5.0,
            torch.tensor([[1.0], [1.0], [2.0], [3.0]]).double(),
            torch.tensor(
                [[1.0], [0.6666666666666666], [1.3333333333333333], [2.0]]
            ).double(),
        ),
        (
            Monotonicity.DECREASING,
            None,
            -1.0,
            torch.tensor([[-1.0], [-1.0], [-2.0], [-3.0]]).double(),
            torch.tensor([[-1.0], [-1.0], [-2.0], [-3.0]]).double(),
        ),
        (
            Monotonicity.DECREASING,
            -5.0,
            None,
            torch.tensor([[-1.0], [-1.0], [-2.0], [-3.0]]).double(),
            torch.tensor(
                [[-1.0], [-0.6666666666666666], [-1.3333333333333333], [-2.0]]
            ).double(),
        ),
    ],
)
def test_squeeze_by_scaling(
    monotonicity,
    output_min,
    output_max,
    kernel_data,
    expected_projected_kernel_data,
):
    """Tests that kernel is scaled into bound constraints properly."""
    calibrator = NumericalCalibrator(
        np.linspace(1.0, 4.0, num=4),
        output_min=output_min,
        output_max=output_max,
        monotonicity=monotonicity,
    )
    bias, heights = kernel_data[0:1], kernel_data[1:]
    projected_bias, projected_heights = calibrator._squeeze_by_scaling(bias, heights)
    projected_kernel_data = torch.cat((projected_bias, projected_heights), 0)
    assert torch.allclose(projected_kernel_data, expected_projected_kernel_data)


def test_training():
    """Tests that the `NumericalCalibrator` module can learn f(x) = |x|."""
    num_examples = 1000
    output_min, output_max = 0.0, 2.0
    training_examples = torch.from_numpy(
        np.random.uniform(-output_max, output_max, size=num_examples)
    )[:, None]
    training_labels = torch.absolute(training_examples)

    calibrator = NumericalCalibrator(
        np.linspace(-2.0, 2.0, num=21),
        output_min=output_min,
        output_max=output_max,
        monotonicity=None,
    )

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
    assert torch.allclose(
        torch.absolute(keypoints_inputs), keypoints_outputs, atol=2e-2
    )
