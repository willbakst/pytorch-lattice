"""Tests for calibrated linear model."""
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from pytorch_lattice import Monotonicity
from pytorch_lattice.layers import Linear, NumericalCalibrator
from pytorch_lattice.models import (
    CalibratedLinear,
    CategoricalFeature,
    NumericalFeature,
)

from ..test_utils import train_calibrated_module


@pytest.mark.parametrize(
    "features,output_min,output_max,output_calibration_num_keypoints,"
    "expected_linear_monotonicities,expected_output_calibrator_monotonicity",
    [
        (
            [
                NumericalFeature(
                    feature_name="numerical_feature",
                    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    num_keypoints=5,
                    monotonicity=None,
                ),
                CategoricalFeature(
                    feature_name="categorical_feature",
                    categories=["a", "b", "c"],
                    monotonicity_pairs=[("a", "b")],
                ),
            ],
            None,
            None,
            None,
            [
                None,
                Monotonicity.INCREASING,
            ],
            Monotonicity.INCREASING,
        ),
        (
            [
                NumericalFeature(
                    feature_name="numerical_feature",
                    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    num_keypoints=5,
                    monotonicity=None,
                ),
                CategoricalFeature(
                    feature_name="categorical_feature",
                    categories=["a", "b", "c"],
                    monotonicity_pairs=None,
                ),
            ],
            -1.0,
            1.0,
            10,
            [
                Monotonicity.INCREASING,
                Monotonicity.INCREASING,
            ],
            None,
        ),
        (
            [
                NumericalFeature(
                    feature_name="numerical_feature",
                    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    num_keypoints=5,
                    monotonicity=Monotonicity.DECREASING,
                ),
                CategoricalFeature(
                    feature_name="categorical_feature",
                    categories=["a", "b", "c"],
                    monotonicity_pairs=None,
                ),
            ],
            0.0,
            None,
            None,
            [
                Monotonicity.INCREASING,
                Monotonicity.INCREASING,
            ],
            Monotonicity.INCREASING,
        ),
    ],
)
def test_initialization(
    features,
    output_min,
    output_max,
    output_calibration_num_keypoints,
    expected_linear_monotonicities,
    expected_output_calibrator_monotonicity,
):
    """Tests that `CalibratedLinear` initialization works."""
    calibrated_linear = CalibratedLinear(
        features=features,
        output_min=output_min,
        output_max=output_max,
        output_calibration_num_keypoints=output_calibration_num_keypoints,
    )
    assert calibrated_linear.features == features
    assert calibrated_linear.output_min == output_min
    assert calibrated_linear.output_max == output_max
    assert (
        calibrated_linear.output_calibration_num_keypoints
        == output_calibration_num_keypoints
    )
    assert len(calibrated_linear.calibrators) == len(features)
    for calibrator in calibrated_linear.calibrators.values():
        assert calibrator.output_min == output_min
        assert calibrator.output_max == output_max
    assert calibrated_linear.linear.monotonicities == expected_linear_monotonicities
    if (
        output_min is not None
        or output_max is not None
        or output_calibration_num_keypoints
    ):
        assert not calibrated_linear.linear.use_bias
        assert calibrated_linear.linear.weighted_average
    else:
        assert calibrated_linear.linear.use_bias
        assert not calibrated_linear.linear.weighted_average
    if not output_calibration_num_keypoints:
        assert calibrated_linear.output_calibrator is None
    else:
        assert calibrated_linear.output_calibrator.output_min == output_min
        assert calibrated_linear.output_calibrator.output_max == output_max
        assert (
            calibrated_linear.output_calibrator.monotonicity
            == expected_output_calibrator_monotonicity
        )


def test_forward():
    """Tests all parts of calibrated lattice forward pass are called."""
    calibrated_linear = CalibratedLinear(
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
        "pytorch_lattice.models.calibrated_linear.calibrate_and_stack",
    ) as mock_calibrate_and_stack, patch.object(
        calibrated_linear.linear,
        "forward",
    ) as mock_linear_forward, patch.object(
        calibrated_linear.output_calibrator,
        "forward",
    ) as mock_output_calibrator:
        mock_calibrate_and_stack.return_value = torch.rand((1, 1))
        mock_linear_forward.return_value = torch.rand((1, 1))
        mock_output_calibrator.return_value = torch.rand((1, 1))
        input_tensor = torch.rand((1, 2))

        result = calibrated_linear.forward(input_tensor)

        mock_calibrate_and_stack.assert_called_once()
        assert torch.allclose(mock_calibrate_and_stack.call_args[0][0], input_tensor)
        assert mock_calibrate_and_stack.call_args[0][1] == calibrated_linear.calibrators
        mock_linear_forward.assert_called_once()
        assert torch.allclose(
            mock_linear_forward.call_args[0][0], mock_calibrate_and_stack.return_value
        )
        mock_output_calibrator.assert_called_once()
        assert torch.allclose(
            mock_output_calibrator.call_args[0][0], mock_linear_forward.return_value
        )
        assert torch.allclose(result, mock_output_calibrator.return_value)


@patch.object(Linear, "assert_constraints")
@patch.object(NumericalCalibrator, "assert_constraints")
def test_assert_constraints(
    mock_linear_assert_constraints, mock_output_assert_constraints
):
    """Tests `assert_constraints()` method calls internal assert_constraints."""
    calibrated_linear = CalibratedLinear(
        features=[
            NumericalFeature(
                feature_name="numerical_feature",
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                num_keypoints=5,
                monotonicity=None,
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
    for calibrator in calibrated_linear.calibrators.values():
        mock_assert = Mock()
        calibrator.assert_constraints = mock_assert
        mock_asserts.append(mock_assert)

    calibrated_linear.assert_constraints()

    mock_linear_assert_constraints.assert_called_once()
    mock_output_assert_constraints.assert_called_once()
    for mock_assert in mock_asserts:
        mock_assert.assert_called_once()


@patch.object(Linear, "apply_constraints")
@patch.object(NumericalCalibrator, "apply_constraints")
def test_constrain(
    mock_linear_apply_constraints, mock_output_calibrator_apply_constraints
):
    """Tests `apply_constraints()` method calls internal constraint functions."""
    calibrated_linear = CalibratedLinear(
        features=[
            NumericalFeature(
                feature_name="numerical_feature",
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                num_keypoints=5,
                monotonicity=None,
            ),
            CategoricalFeature(
                feature_name="categorical_feature",
                categories=["a", "b", "c"],
                monotonicity_pairs=[("a", "b")],
            ),
        ],
        output_calibration_num_keypoints=2,
    )
    mock_apply_constraints_fns = []
    for calibrator in calibrated_linear.calibrators.values():
        mock_calibrator_apply_constraints = Mock()
        calibrator.apply_constraints = mock_calibrator_apply_constraints
        mock_apply_constraints_fns.append(mock_calibrator_apply_constraints)

    calibrated_linear.apply_constraints()

    mock_linear_apply_constraints.assert_called_once()
    mock_output_calibrator_apply_constraints.assert_called_once()
    for mock_constrain in mock_apply_constraints_fns:
        mock_constrain.assert_called_once()


def test_training():
    """Tests `CalibratedLinear` training on data from f(x) = 0.7|x_1| + 0.3x_2."""
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

    calibrated_linear = CalibratedLinear(
        features=[
            NumericalFeature(
                "x1",
                x_1_numpy,
                num_keypoints=4,
            ),
            CategoricalFeature("x2", [0, 1, 2], monotonicity_pairs=[(0, 1), (1, 2)]),
        ],
        output_min=output_min,
        output_max=output_max,
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(calibrated_linear.parameters(recurse=True), lr=1e-1)

    with torch.no_grad():
        initial_predictions = calibrated_linear(training_examples)
        initial_loss = loss_fn(initial_predictions, training_labels)

    train_calibrated_module(
        calibrated_linear,
        training_examples,
        training_labels,
        loss_fn,
        optimizer,
        500,
        num_examples // 10,
    )

    with torch.no_grad():
        trained_predictions = calibrated_linear(training_examples)
        trained_loss = loss_fn(trained_predictions, training_labels)

    assert not calibrated_linear.assert_constraints()
    assert trained_loss < initial_loss
    assert trained_loss < 0.02
