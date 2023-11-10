"""Configuration objects for the PyTorch Lattice library."""
from __future__ import annotations

from typing import Optional, Union

from .enums import InputKeypointsInit, InputKeypointsType, Monotonicity


class FeatureConfig:
    """A configuration object for a feature in a calibrated model.

    This configuration object handles both numerical and categorical features. If the
    `categeories` attribute is `None`, then this feature will be handled as numerical.
    Otherwise, it will be handled as categorical.

    Attributes:
        name: The name of the feature.
        categories: The categories for a categorical feature.
        num_keypoints: The number of keypoints to use for a numerical feature.
        input_keypoints_init: The method for initializing the input keypoints for a
            numerical calibrator.
        input_keypoints_type: The type of input keypoints to use for a numerical
            calibrator.
        monotonicity: The monotonicity constraint, if any. For numerical features, this
            should be an instance of `Monotonicity`. For categorical features, this
            should be a list of pairs of categories that should be constrained to be
            increasing such that for the pair `(a, b)` the category `b` will always have
            a higher output compared to `a`, all else being equal.
        projection_iterations: Number of times to run Dykstra's projection algorithm
            when applying constraints to a numerical calibrator. The more iterations,
            the more accurate the projection will be, but the longer it will take to
            run. Should a monotonicity constraint not be satisfied after training, try
            increasing this value.
    """

    def __init__(
        self,
        name: str,
        categories: Optional[list[str]] = None,
        num_keypoints: int = 5,
        input_keypoints_init: InputKeypointsInit = InputKeypointsInit.QUANTILES,
        input_keypoints_type: InputKeypointsType = InputKeypointsType.FIXED,
        monotonicity: Optional[Union[Monotonicity, list[tuple[str, str]]]] = None,
        projection_iterations: int = 8,
    ):
        """Initializes an instance of `FeatureConfig`."""
        self.name = name
        self._categories = categories
        self._num_keypoints = num_keypoints
        self._input_keypoints_init = input_keypoints_init
        self._input_keypoints_type = input_keypoints_type
        self._monotonicity = monotonicity
        self._projection_iterations = projection_iterations

    def categories(self, categories: list[str]) -> FeatureConfig:
        """Sets the categories for a categorical feature."""
        self._categories = categories
        return self

    def num_keypoints(self, num_keypoints: int) -> FeatureConfig:
        """Sets the categories for a categorical feature."""
        self._num_keypoints = num_keypoints
        return self

    def input_keypoints_init(
        self, input_keypoints_init: InputKeypointsInit
    ) -> FeatureConfig:
        """Sets the input keypoints initialization method for a numerical calibrator."""
        self._input_keypoints_init = input_keypoints_init
        return self

    def input_keypoints_type(
        self, input_keypoints_type: InputKeypointsType
    ) -> FeatureConfig:
        """Sets the input keypoints type for a numerical calibrator."""
        self._input_keypoints_type = input_keypoints_type
        return self

    def monotonicity(
        self, monotonicity: Optional[Union[Monotonicity, list[tuple[str, str]]]]
    ) -> FeatureConfig:
        """Sets the monotonicity constraint for a feature."""
        self._monotonicity = monotonicity
        return self

    def projection_iterations(self, projection_iterations: int) -> FeatureConfig:
        """Sets the number of projection iterations for a numerical calibrator."""
        self._projection_iterations = projection_iterations
        return self
