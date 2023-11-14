"""Tests for the `FeatureConfig` class."""

from pytorch_lattice.feature_config import FeatureConfig


def test_initialization():
    """Tests that the feature config initializes properly with default values."""
    name = "name"
    fc = FeatureConfig(name)
    assert fc.name == name
    assert fc._categories is None
    assert fc._num_keypoints == 5
    assert fc._input_keypoints_init == "quantiles"
    assert fc._input_keypoints_type == "fixed"
    assert fc._monotonicity is None
    assert fc._projection_iterations == 8
    assert fc._lattice_size == 2


def test_setters():
    """Tests that setting configuration values through methods works."""
    fc = FeatureConfig("name")

    # Categories
    categories = ["a", "b"]
    fc.categories(["a", "b"])
    assert fc._categories == categories

    # Num Keypoints
    num_keypoints = 10
    fc.num_keypoints(num_keypoints)
    assert fc._num_keypoints == num_keypoints

    # Input Keypoints Init
    input_keypoints_init = "uniform"
    fc.input_keypoints_init(input_keypoints_init)
    assert fc._input_keypoints_init == input_keypoints_init

    # Input Keypoints Type (LEARNED not yet implemented)
    # input_keypoints_type = "learned_interior"
    # fc.input_keypoints_type(input_keypoints_type)
    # assert fc._input_keypoints_type == input_keypoints_type

    # Monotonicity
    monotonicity = "increasing"
    fc.monotonicity(monotonicity)
    assert fc._monotonicity == monotonicity

    # Projection Iterations
    projection_iterations = 10
    fc.projection_iterations(projection_iterations)
    assert fc._projection_iterations == projection_iterations

    # Lattice Size
    lattice_size = 10
    fc.lattice_size(lattice_size)
    assert fc._lattice_size == lattice_size
