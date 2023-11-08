"""Feature objects for use in models.

To construct a calibrated model, create the calibrated model configuration and pass it
in to the corresponding calibrated model constructor.

Example:
```python
feature_configs = [...]
linear_config = CalibratedLinearConfig(feature_configs, ...)
linear_model = CalibratedLinear(linear_config)
```
"""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from .enums import FeatureType, InputKeypointsInit, Monotonicity


# pylint: disable-next=too-few-public-methods
class _Feature:
    """Base configuration for numerical and categorical features.

    Attributes:
        - All `__init__` arguments.
    """

    def __init__(self, feature_name: str, feature_type: FeatureType):
        """Initializes a `_FeatureConfig` instance."""
        self.feature_name = feature_name
        self.feature_type = feature_type


# pylint: disable-next=too-few-public-methods
# pylint: disable-next=too-many-instance-attributes
class NumericalFeature(_Feature):
    """Feature configuration for numerical features.

    Attributes:
        - All `__init__` arguments.
        feature_type: The type of this feature. Always `FeatureType.NUMERICAL`.
        input_keypoints: The input keypoints used for this feature's calibrator. These
            keypoints will be initialized using the given `data` under the desired
            `input_keypoints_init` scheme.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        feature_name: str,
        data: np.ndarray,
        num_keypoints: int = 10,
        input_keypoints_init: InputKeypointsInit = InputKeypointsInit.QUANTILES,
        missing_input_value: Optional[float] = None,
        monotonicity: Monotonicity = Monotonicity.NONE,
        projection_iterations: int = 8,
        lattice_size: int = 2,
    ) -> None:
        """Initializes a `NumericalFeatureConfig` instance.

        Args:
            feature_name: The name of the feature. This should match the header for the
                column in the dataset representing this feature.
            data: Numpy array of float-valued data used for calculating keypoint inputs
                and initializing keypoint outputs.
            num_keypoints: The number of keypoints used by the underlying piece-wise
                linear function of a NumericalCalibrator. There will be
                `num_keypoints - 1` total segments.
            input_keypoints_init: The scheme to use for initializing the input
                keypoints. See `InputKeypointsInit` for more details.
            missing_input_value: If provided, this feature's calibrator will learn to
                map all instances of this missing input value to a learned output value.
            monotonicity: Monotonicity constraint for this feature, if any.
            projection_iterations: Number of times to run Dykstra's projection
                algorithm when applying constraints.
            lattice_size: The default number of keypoints outputted by the
                calibrator. Only used within `Lattice` models.

        Raises:
            ValueError: If `data` contains NaN values.
            ValueError: If `input_keypoints_init` is invalid.
        """
        super().__init__(feature_name, FeatureType.NUMERICAL)

        if np.isnan(data).any():
            raise ValueError("Data contains NaN values.")

        self.data = data
        self.num_keypoints = num_keypoints
        self.input_keypoints_init = input_keypoints_init
        self.missing_input_value = missing_input_value
        self.monotonicity = monotonicity
        self.projection_iterations = projection_iterations
        self.lattice_size = lattice_size

        sorted_unique_values = np.unique(data)

        if input_keypoints_init == InputKeypointsInit.QUANTILES:
            if sorted_unique_values.size < num_keypoints:
                logging.info(
                    "Observed fewer unique values for feature %s than %d desired "
                    "keypoints. Using the observed %d unique values as keypoints.",
                    feature_name,
                    num_keypoints,
                    sorted_unique_values.size,
                )
                self.input_keypoints = sorted_unique_values
            else:
                quantiles = np.linspace(0.0, 1.0, num=num_keypoints)
                self.input_keypoints = np.quantile(
                    sorted_unique_values, quantiles, method="nearest"
                )
        elif input_keypoints_init == InputKeypointsInit.UNIFORM:
            self.input_keypoints = np.linspace(
                sorted_unique_values[0], sorted_unique_values[-1], num=num_keypoints
            )
        else:
            raise ValueError(f"Unknown input keypoints init: {input_keypoints_init}")


# pylint: disable-next=too-few-public-methods
class CategoricalFeature(_Feature):
    """Feature configuration for categorical features.

    Attributes:
        - All `__init__` arguments.
        feature_type: The type of this feature. Always `FeatureType.CATEGORICAL`.
        category_indices: A dictionary mapping string categories to their index.
        monotonicity_index_pairs: A conversion of `monotonicity_pairs` from string
            categories to category indices. Only available if `monotonicity_pairs` are
            provided.
    """

    def __init__(
        self,
        feature_name: str,
        categories: Union[List[int], List[str]],
        missing_input_value: Optional[float] = None,
        monotonicity_pairs: Optional[List[Tuple[str, str]]] = None,
        lattice_size: int = 2,
    ) -> None:
        """Initializes a `CategoricalFeatureConfig` instance.

        Args:
            feature_name: The name of the feature. This should match the header for the
                column in the dataset representing this feature.
            categories: The categories that should be used for this feature. Any
                categories not contained will be considered missing or unknown. If you
                expect to have such missing categories, make sure to
            missing_input_value: If provided, this feature's calibrator will learn to
                map all instances of this missing input value to a learned output value.
            monotonicity_pairs: List of pairs of categories `(category_a, category_b)`
                indicating that the calibrator output for `category_b` should be greater
                than or equal to that of `category_a`.
            lattice_size: The default number of keypoints outputted by the
            calibrator. Only used within `Lattice` models.
        """
        super().__init__(feature_name, FeatureType.CATEGORICAL)

        self.categories = categories
        self.missing_input_value = missing_input_value
        self.monotonicity_pairs = monotonicity_pairs
        self.lattice_size = lattice_size

        self.category_indices = {category: i for i, category in enumerate(categories)}
        self.monotonicity_index_pairs = [
            (self.category_indices[a], self.category_indices[b])
            for a, b in monotonicity_pairs or []
        ]
