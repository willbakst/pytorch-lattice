"""Configuration objects for the PyTorch Lattice library."""
from __future__ import annotations

from typing import Optional, Union

from .enums import InputKeypointsInit, InputKeypointsType, Monotonicity


class FeatureConfig:
    """A configuration object for a feature in a calibrated model.

    This configuration object handles both numerical and categorical features. If the
    `categeories` attribute is `None`, then this feature will be handled as numerical.
    Otherwise, it will be handled as categorical.

    Example:
    ```python
    fc = FeatureConfig(name="feature_name").num_keypoints(10).monotonicity("increasing")
    ```

    Attributes:
        name: The name of the feature.
    """

    def __init__(self, name: str):
        """Initializes an instance of `FeatureConfig` with default values."""
        self.name = name
        self._categories: Optional[list[str]] = None
        self._num_keypoints: int = 5
        self._input_keypoints_init: InputKeypointsInit = InputKeypointsInit.QUANTILES
        self._input_keypoints_type: InputKeypointsType = InputKeypointsType.FIXED
        self._monotonicity: Optional[Union[Monotonicity, list[tuple[str, str]]]] = None
        self._projection_iterations: int = 8
        self._lattice_size: int = 2  # only used in lattice models

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

    def lattice_size(self, lattice_size: int) -> FeatureConfig:
        """Sets the lattice size for a feature."""
        self._lattice_size = lattice_size
        return self
