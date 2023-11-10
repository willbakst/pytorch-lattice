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
        features: A dict mapping feature names to their corresponding `FeatureConfig`
            instances.
    """

    def __init__(self, feature_names: list[str]):
        """Initializes an instance of `Classifier`."""
        self.features = {
            feature_name: FeatureConfig(name=feature_name)
            for feature_name in feature_names
        }

    def configure(self, feature_name: str):
        """Returns a `FeatureConfig` object for the given feature name."""
        return self.features[feature_name]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Classifier:
        """Returns this classifier fit to the given data."""
        raise NotImplementedError()
