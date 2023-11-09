"""Tests for calibrated lattice model."""
from unittest.mock import patch, Mock
import numpy as np
import pytest
import torch

from pytorch_lattice import Monotonicity, Interpolation, LatticeInit
from pytorch_lattice.models.features import CategoricalFeature, NumericalFeature
from pytorch_lattice.models import CalibratedLattice
from pytorch_lattice.layers import Lattice, NumericalCalibrator

from ..utils import train_calibrated_module


def test_init_required_args():
    """Tests `CalibratedLattice` initialization with only required arguments."""
    calibrated_lattice = CalibratedLattice(
        features=[
            NumericalFeature(
                feature_name="numerical_feature",
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                num_keypoints=5,
                monotonicity=Monotonicity.NONE,
            ),
            CategoricalFeature(
                feature_name="categorical_feature",
                categories=["a", "b", "c"],
                monotonicity_pairs=[("a", "b")],
            ),
        ]
    )
    assert calibrated_lattice.clip_inputs
    assert calibrated_lattice.output_min is None
    assert calibrated_lattice.output_max is None
    assert calibrated_lattice.kernel_init == LatticeInit.LINEAR
    assert calibrated_lattice.interpolation == Interpolation.HYPERCUBE
    assert calibrated_lattice.lattice.lattice_sizes == [2, 2]
    assert calibrated_lattice.output_calibration_num_keypoints is None
    assert calibrated_lattice.output_calibrator is None
    for calibrator in calibrated_lattice.calibrators.values():
        assert calibrator.output_min == 0.0
        assert calibrator.output_max == 1.0


@pytest.mark.parametrize(
    "features, output_min, output_max, interpolation, output_num_keypoints,"
    "expected_monotonicity, expected_lattice_sizes, expected_output_monotonicity",
    [
        (
            [
                NumericalFeature(
                    feature_name="numerical_feature",
                    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    num_keypoints=5,
                    monotonicity=Monotonicity.DECREASING,
                    lattice_size=3,
                ),
                CategoricalFeature(
                    feature_name="categorical_feature",
                    categories=["a", "b", "c"],
                    monotonicity_pairs=[("a", "b")],
                    lattice_size=2,
                ),
            ],
            0.5,
            2.0,
            Interpolation.SIMPLEX,
            4,
            [Monotonicity.INCREASING, Monotonicity.INCREASING],
            [3, 2],
            Monotonicity.INCREASING,
        ),
        (
            [
                NumericalFeature(
                    feature_name="numerical_feature",
                    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    num_keypoints=5,
                    lattice_size=4,
                ),
                CategoricalFeature(
                    feature_name="categorical_feature",
                    categories=["a", "b", "c"],
                    lattice_size=4,
                ),
            ],
            -0.5,
            8.0,
            Interpolation.HYPERCUBE,
            5,
            [Monotonicity.NONE, Monotonicity.NONE],
            [4, 4],
            Monotonicity.NONE,
        ),
    ],
)
def test_init_full_args(
    features,
    output_min,
    output_max,
    interpolation,
    output_num_keypoints,
    expected_monotonicity,
    expected_lattice_sizes,
    expected_output_monotonicity,
):
    """Tests `CalibratedLattice` initialization with all arguments."""
    calibrated_lattice = CalibratedLattice(
        features=features,
        output_min=output_min,
        output_max=output_max,
        interpolation=interpolation,
        output_calibration_num_keypoints=output_num_keypoints,
    )
    assert calibrated_lattice.clip_inputs
    assert calibrated_lattice.output_min == output_min
    assert calibrated_lattice.output_max == output_max
    assert calibrated_lattice.interpolation == interpolation
    assert calibrated_lattice.output_calibration_num_keypoints == output_num_keypoints
    assert calibrated_lattice.output_calibrator.output_min == output_min
    assert calibrated_lattice.output_calibrator.output_max == output_max
    assert (
        calibrated_lattice.output_calibrator.monotonicity
        == expected_output_monotonicity
    )
    assert calibrated_lattice.monotonicities == expected_monotonicity
    assert calibrated_lattice.lattice.lattice_sizes == expected_lattice_sizes
    for calibrator, lattice_dim in zip(
        calibrated_lattice.calibrators.values(), expected_lattice_sizes
    ):
        assert calibrator.output_min == 0.0
        assert calibrator.output_max == lattice_dim - 1


def test_forward():
    """Tests all parts of calibrated lattice forward pass are called."""
    calibrated_lattice = CalibratedLattice(
        features=[
            NumericalFeature(
                feature_name="n",
                data=np.array([1.0, 2.0]),
            ),
            CategoricalFeature(
                feature_name="c",
                categories=["a", "b", "c"],
            ),
        ],
        output_calibration_num_keypoints=10,
    )
    with patch(
        "pytorch_lattice.models.calibrated_lattice.calibrate_and_stack",
    ) as mock_calibrate_and_stack, patch.object(
        calibrated_lattice.lattice,
        "forward",
    ) as mock_lattice_forward, patch.object(
        calibrated_lattice.output_calibrator,
        "forward",
    ) as mock_output_calibrator:
        mock_calibrate_and_stack.return_value = torch.rand((1, 1))
        mock_lattice_forward.return_value = torch.rand((1, 1))
        mock_output_calibrator.return_value = torch.rand((1, 1))
        input_tensor = torch.rand((1, 2))

        result = calibrated_lattice.forward(input_tensor)

        mock_calibrate_and_stack.assert_called_once()
        assert torch.allclose(mock_calibrate_and_stack.call_args[0][0], input_tensor)
        assert (
            mock_calibrate_and_stack.call_args[0][1] == calibrated_lattice.calibrators
        )
        mock_lattice_forward.assert_called_once()
        assert torch.allclose(
            mock_lattice_forward.call_args[0][0], mock_calibrate_and_stack.return_value
        )
        mock_output_calibrator.assert_called_once()
        assert torch.allclose(
            mock_output_calibrator.call_args[0][0], mock_lattice_forward.return_value
        )
        assert torch.allclose(result, mock_output_calibrator.return_value)


@pytest.mark.parametrize(
    "interpolation",
    [
        Interpolation.HYPERCUBE,
        Interpolation.SIMPLEX,
    ],
)
@pytest.mark.parametrize(
    "lattice_dim",
    [
        2,
        3,
        4,
    ],
)
def test_training(interpolation, lattice_dim):  # pylint: disable=too-many-locals
    """Tests `CalibratedLattice` training on data from f(x) = 0.7|x_1| + 0.3x_2."""
    num_examples, num_categories = 3000, 3
    output_min, output_max = 0.0, num_categories - 1
    x_1_numpy = np.random.uniform(-output_max, output_max, size=num_examples)
    x_1 = torch.from_numpy(x_1_numpy)[:, None]
    num_examples_per_category = num_examples // num_categories
    x2_numpy = np.concatenate(
        [[c] * num_examples_per_category for c in range(num_categories)]
    )
    x_2 = torch.from_numpy(x2_numpy)[:, None]
    training_examples = torch.column_stack((x_1, x_2))
    linear_coefficients = torch.tensor([0.7, 0.3]).double()
    training_labels = torch.sum(
        torch.column_stack((torch.absolute(x_1), x_2)) * linear_coefficients,
        dim=1,
        keepdim=True,
    )
    randperm = torch.randperm(training_examples.size()[0])
    training_examples = training_examples[randperm]
    training_labels = training_labels[randperm]

    calibrated_lattice = CalibratedLattice(
        features=[
            NumericalFeature(
                "x1", x_1_numpy, num_keypoints=4, lattice_size=lattice_dim
            ),
            CategoricalFeature(
                "x2",
                [0, 1, 2],
                monotonicity_pairs=[(0, 1), (1, 2)],
                lattice_size=lattice_dim,
            ),
        ],
        output_min=output_min,
        output_max=output_max,
        interpolation=interpolation,
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(calibrated_lattice.parameters(recurse=True), lr=1e-1)

    with torch.no_grad():
        initial_predictions = calibrated_lattice(training_examples)
        initial_loss = loss_fn(initial_predictions, training_labels)

    train_calibrated_module(
        calibrated_lattice,
        training_examples,
        training_labels,
        loss_fn,
        optimizer,
        500,
        num_examples // 10,
    )

    with torch.no_grad():
        trained_predictions = calibrated_lattice(training_examples)
        trained_loss = loss_fn(trained_predictions, training_labels)

    # calibrated_lattice.constrain()
    assert not calibrated_lattice.assert_constraints()
    assert trained_loss < initial_loss
    assert trained_loss < 0.08


@patch.object(Lattice, "assert_constraints")
@patch.object(NumericalCalibrator, "assert_constraints")
def test_assert_constraints(
    mock_lattice_assert_constraints, mock_output_assert_constraints
):
    """Tests `assert_constraints()` method calls internal assert_constraints."""
    calibrated_lattice = CalibratedLattice(
        features=[
            NumericalFeature(
                feature_name="numerical_feature",
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                num_keypoints=5,
                monotonicity=Monotonicity.NONE,
            ),
            CategoricalFeature(
                feature_name="categorical_feature",
                categories=["a", "b", "c"],
                monotonicity_pairs=[("a", "b")],
            ),
        ],
        output_calibration_num_keypoints=5,
    )

    mock_asserts = []
    for calibrator in calibrated_lattice.calibrators.values():
        mock_assert = Mock()
        calibrator.assert_constraints = mock_assert
        mock_asserts.append(mock_assert)

    calibrated_lattice.assert_constraints()

    mock_lattice_assert_constraints.assert_called_once()
    for mock_assert in mock_asserts:
        mock_assert.assert_called_once()
    mock_output_assert_constraints.assert_called_once()


@patch.object(Lattice, "constrain")
@patch.object(NumericalCalibrator, "constrain")
def test_constrain(mock_lattice_constrain, mock_output_calibrator_constrain):
    """Tests `constrain()` method calls internal constrain functions."""
    calibrated_lattice = CalibratedLattice(
        features=[
            NumericalFeature(
                feature_name="numerical_feature",
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                num_keypoints=5,
                monotonicity=Monotonicity.NONE,
            ),
            CategoricalFeature(
                feature_name="categorical_feature",
                categories=["a", "b", "c"],
                monotonicity_pairs=[("a", "b")],
            ),
        ],
        output_calibration_num_keypoints=2,
    )
    mock_constrains = []
    for calibrator in calibrated_lattice.calibrators.values():
        mock_calibrator_constrain = Mock()
        calibrator.constrain = mock_calibrator_constrain
        mock_constrains.append(mock_calibrator_constrain)

    calibrated_lattice.constrain()

    mock_lattice_constrain.assert_called_once()
    mock_output_calibrator_constrain.assert_called_once()
    for mock_constrain in mock_constrains:
        mock_constrain.assert_called_once()
