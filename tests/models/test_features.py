"""Tests for configuration objects."""
import numpy as np
import pytest

from pytorch_lattice import FeatureType, InputKeypointsInit, Monotonicity
from pytorch_lattice.models import CategoricalFeature, NumericalFeature


@pytest.mark.parametrize(
    "data,num_keypoints,input_keypoints_init,missing_input_value,monotonicity,"
    "expected_input_keypoints",
    [
        (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            5,
            InputKeypointsInit.QUANTILES,
            None,
            Monotonicity.NONE,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ),
        (
            np.array([1.0] * 10 + [2.0] * 10 + [3.0] * 10 + [4.0] * 10 + [5.0] * 10),
            5,
            InputKeypointsInit.QUANTILES,
            None,
            Monotonicity.INCREASING,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ),
        (
            np.array([1.0] * 10 + [2.0] * 10 + [3.0] * 10 + [4.0] * 10 + [5.0] * 10),
            5,
            InputKeypointsInit.UNIFORM,
            None,
            Monotonicity.INCREASING,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ),
        (
            np.array(
                [1.0] * 10 + [2.0] * 8 + [3.0] * 6 + [4.0] * 4 + [5.0] * 2 + [6.0]
            ),
            4,
            InputKeypointsInit.QUANTILES,
            None,
            Monotonicity.INCREASING,
            np.array([1.0, 3.0, 4.0, 6.0]),
        ),
        (
            np.array(
                [1.0] * 10 + [2.0] * 8 + [3.0] * 6 + [4.0] * 4 + [5.0] * 2 + [6.0]
            ),
            10,
            InputKeypointsInit.QUANTILES,
            None,
            Monotonicity.NONE,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
    ],
)
def test_numerical_feature_config_initialization(
    data,
    num_keypoints,
    input_keypoints_init,
    missing_input_value,
    monotonicity,
    expected_input_keypoints,
) -> None:
    """Tests that numerical feature configs initialize properly."""
    feature_name = "test_feature"
    feature = NumericalFeature(
        feature_name,
        data,
        num_keypoints,
        input_keypoints_init,
        missing_input_value,
        monotonicity,
    )
    assert feature.feature_type == FeatureType.NUMERICAL
    assert feature.feature_name == feature_name
    assert (feature.data == data).all()
    assert feature.num_keypoints == num_keypoints
    assert feature.input_keypoints_init == input_keypoints_init
    assert feature.missing_input_value == missing_input_value
    assert feature.monotonicity == monotonicity
    assert np.allclose(feature.input_keypoints, expected_input_keypoints)


@pytest.mark.parametrize(
    "categories,missing_input_value,monotonicity_pairs,expected_category_indices,"
    "expected_monotonicity_index_pairs",
    [
        (["a", "b", "c"], None, None, {"a": 0, "b": 1, "c": 2}, []),
        (
            ["a", "b", "c", "d"],
            -1.0,
            [("a", "b"), ("c", "d")],
            {"a": 0, "b": 1, "c": 2, "d": 3},
            [(0, 1), (2, 3)],
        ),
    ],
)
def test_categorical_feature_config_initialization(
    categories,
    missing_input_value,
    monotonicity_pairs,
    expected_category_indices,
    expected_monotonicity_index_pairs,
):
    """Tests that categorical feature configs initialize properly."""
    feature_name = "test_feature"
    feature = CategoricalFeature(
        feature_name, categories, missing_input_value, monotonicity_pairs
    )
    assert feature.feature_type == FeatureType.CATEGORICAL
    assert feature.feature_name == feature_name
    assert feature.categories == categories
    assert feature.missing_input_value == missing_input_value
    assert feature.monotonicity_pairs == monotonicity_pairs
    assert feature.category_indices == expected_category_indices
    assert feature.monotonicity_index_pairs == expected_monotonicity_index_pairs
