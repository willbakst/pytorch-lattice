"""Tests for util functions for calibrated models."""
from unittest.mock import Mock

import torch
import numpy as np
import pytest

from pytorch_lattice import (
    Monotonicity,
    NumericalCalibratorInit,
    CategoricalCalibratorInit,
    FeatureType,
)
from pytorch_lattice.models.features import NumericalFeature, CategoricalFeature
from pytorch_lattice.models.model_utils import (
    initialize_feature_calibrators,
    initialize_monotonicities,
    initialize_output_calibrator,
    calibrate_and_stack,
)


@pytest.mark.parametrize(
    "num_feat, cat_feat",
    [
        (
            NumericalFeature(
                feature_name="numerical_feature",
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                monotonicity=Monotonicity.NONE,
            ),
            CategoricalFeature(
                feature_name="categorical_feature",
                categories=["a", "b"],
            ),
        ),
        (
            NumericalFeature(
                feature_name="numerical_feature",
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                num_keypoints=5,
                monotonicity=Monotonicity.DECREASING,
            ),
            CategoricalFeature(
                feature_name="categorical_feature",
                categories=["a", "b", "c"],
                monotonicity_pairs=[("a", "b"), ("c", "b")],
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "output_min, output_max, expected_output_max",
    [
        (None, None, [None, None]),
        (None, 1, [1, 1]),
        (0, None, [None, None]),
        (None, [2, 3], [2, 3]),
        (0, [2, 2], [2, 2]),
    ],
)
def test_initialize_feature_calibrators(
    num_feat, cat_feat, output_min, output_max, expected_output_max
) -> None:
    """Test for calibrator initialization helper function."""
    features = [num_feat, cat_feat]
    calibrators_dict = initialize_feature_calibrators(
        features=features,
        output_min=output_min,
        output_max=output_max,
    )

    np.testing.assert_allclose(
        calibrators_dict["numerical_feature"].input_keypoints,
        num_feat.input_keypoints,
        rtol=1e-5,
    )
    assert (
        calibrators_dict["numerical_feature"].missing_input_value
        == num_feat.missing_input_value
    )
    assert calibrators_dict["numerical_feature"].output_min == output_min
    assert calibrators_dict["numerical_feature"].output_max == expected_output_max[0]
    assert calibrators_dict["numerical_feature"].monotonicity == num_feat.monotonicity
    assert (
        calibrators_dict["numerical_feature"].kernel_init
        == NumericalCalibratorInit.EQUAL_SLOPES
    )
    assert (
        calibrators_dict["numerical_feature"].projection_iterations
        == num_feat.projection_iterations
    )

    assert calibrators_dict["categorical_feature"].num_categories == len(
        cat_feat.categories
    )
    assert (
        calibrators_dict["categorical_feature"].missing_input_value
        == cat_feat.missing_input_value
    )
    assert calibrators_dict["categorical_feature"].output_min == output_min
    assert calibrators_dict["categorical_feature"].output_max == expected_output_max[1]
    assert (
        calibrators_dict["categorical_feature"].monotonicity_pairs
        == cat_feat.monotonicity_index_pairs
    )
    assert (
        calibrators_dict["categorical_feature"].kernel_init
        == CategoricalCalibratorInit.UNIFORM
    )


def test_initialize_feature_calibrators_invalid() -> None:
    """Test for calibrator initialization helper function for invalid feature type."""
    with pytest.raises(ValueError, match=r"Unknown feature type unknown for feature a"):
        features = [NumericalFeature(feature_name="a", data=np.array([1.0]))]
        features[0].feature_type = FeatureType.UNKNOWN
        initialize_feature_calibrators(features)


@pytest.mark.parametrize(
    "features, expected_monotonicities",
    [
        (
            [
                NumericalFeature(
                    feature_name="n",
                    data=np.array([1.0]),
                    monotonicity=Monotonicity.NONE,
                ),
                CategoricalFeature(
                    feature_name="c",
                    categories=["a", "b"],
                ),
            ],
            [Monotonicity.NONE, Monotonicity.NONE],
        ),
        (
            [
                NumericalFeature(
                    feature_name="n",
                    data=np.array([1.0]),
                    monotonicity=Monotonicity.NONE,
                ),
                CategoricalFeature(
                    feature_name="c",
                    categories=["a", "b"],
                    monotonicity_pairs=[("a", "b")],
                ),
            ],
            [Monotonicity.NONE, Monotonicity.INCREASING],
        ),
        (
            [
                NumericalFeature(
                    feature_name="n",
                    data=np.array([1.0]),
                    monotonicity=Monotonicity.INCREASING,
                ),
                CategoricalFeature(
                    feature_name="c",
                    categories=["a", "b"],
                ),
            ],
            [Monotonicity.INCREASING, Monotonicity.NONE],
        ),
        (
            [
                NumericalFeature(
                    feature_name="n",
                    data=np.array([1.0]),
                    monotonicity=Monotonicity.DECREASING,
                ),
                CategoricalFeature(
                    feature_name="c",
                    categories=["a", "b"],
                    monotonicity_pairs=[("a", "b")],
                ),
            ],
            [Monotonicity.INCREASING, Monotonicity.INCREASING],
        ),
    ],
)
def test_initialize_monotonicities(features, expected_monotonicities) -> None:
    """Tests for monotonicity initialization logic in helper function."""
    monotonicities = initialize_monotonicities(features)
    for mono, expected_mono in zip(monotonicities, expected_monotonicities):
        assert mono == expected_mono


@pytest.mark.parametrize(
    "output_calibration_num_keypoints, monotonic, output_min, output_max",
    [
        (None, None, None, None),
        (0, None, None, None),
    ],
)
def test_initialize_output_calibrator_none(
    output_calibration_num_keypoints, monotonic, output_min, output_max
) -> None:
    """Tests helper function for initializing output calibrator when not initialized."""
    output_cal = initialize_output_calibrator(
        output_calibration_num_keypoints=output_calibration_num_keypoints,
        monotonic=monotonic,
        output_min=output_min,
        output_max=output_max,
    )
    assert output_cal is None


@pytest.mark.parametrize(
    "output_calibration_num_keypoints, monotonic, output_min, output_max",
    [
        (4, True, None, None),
        (5, False, 0.0, 1.0),
    ],
)
def test_initialize_output_calibrator(
    output_calibration_num_keypoints, monotonic, output_min, output_max
) -> None:
    """Tests helper function for initializing output calibrator when initialized."""
    output_cal = initialize_output_calibrator(
        output_calibration_num_keypoints=output_calibration_num_keypoints,
        monotonic=monotonic,
        output_min=output_min,
        output_max=output_max,
    )
    assert len(output_cal.input_keypoints) == output_calibration_num_keypoints
    assert output_cal.missing_input_value is None
    assert output_cal.output_max == output_max
    assert output_cal.output_min == output_min
    if monotonic:
        assert output_cal.monotonicity == Monotonicity.INCREASING
    else:
        assert output_cal.monotonicity == Monotonicity.NONE


@pytest.mark.parametrize(
    "data,expected_args",
    [
        (
            torch.tensor([[1.0, 2.0, 3.0]]),
            [torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor([[3.0]])],
        ),
        (
            torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
            [torch.tensor([[4.0], [6.0]]), torch.tensor([[5.0], [7.0]])],
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            [
                torch.tensor([[1.0], [4.0], [7.0]]),
                torch.tensor([[2.0], [5.0], [8.0]]),
                torch.tensor([[3.0], [6.0], [9.0]]),
            ],
        ),
    ],
)
def test_calibrate_and_stack(data, expected_args):
    """Tests slicing logic of calibrate_and_stack function used in forward passes."""
    mock_calibrators = {
        f"calibrator_{i}": Mock(
            spec=torch.nn.Module, return_value=torch.tensor([[0.0]])
        )
        for i in range(data.shape[1])
    }
    calibrators = torch.nn.ModuleDict(mock_calibrators)

    result = calibrate_and_stack(data, calibrators)

    for mock_calibrator, expected_arg in zip(calibrators.values(), expected_args):
        mock_calibrator.assert_called_once()
        assert torch.allclose(mock_calibrator.call_args[0][0], expected_arg)
    expected_result = torch.zeros(data.shape[0], data.shape[1])
    assert torch.allclose(result, expected_result)
