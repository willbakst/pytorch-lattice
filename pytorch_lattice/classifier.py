"""A class for training classifiers on tabular data using calibrated models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .configs import FeatureConfig


class Classifier:
    """A classifier for tabular data using calibrated models.

    Note: currently only handles binary classification targets.

    Example:
    ```python
    X, y = pyl.datasets.heart()
    clf = pyl.Classifier(X.columns)
    clf.configure("age").num_keypoints(10).monotonicity("increasing")
    clf.fit(X, y)
    ```

    Attributes:
        - All `__init__` arguments.
    """

    def __init__(self, features: list[str]):
        """Initializes an instance of `Classifier`."""
        self.features = features
        self.feature_configs = {
            feature: FeatureConfig(name=feature) for feature in features
        }

    def configure(self, feature_name: str):
        """Returns a `FeatureConfig` object for the given feature name."""
        return self.feature_configs[feature_name]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Classifier:
        """Returns this classifier fit to the given data."""
        raise NotImplementedError()
