"""A class for training classifiers on tabular data using calibrated models."""
from .configs import FeatureConfig


class Classifier:
    """A classifier for tabular data using calibrated models.

    Note: currently only handles binary classification targets.

    Example:
    ```python
    X, y = pyl.datasets.heart()
    clf = pyl.Classifier(X.columns).fit(X, y)
    ```

    Attributes:
        - All `__init__` arguments.
    """

    def __init__(self, features: list[str]):
        """Initializes an instance of `Classifier`."""
        self.features = features
        self.feature_configs = [FeatureConfig(name=feature) for feature in features]
