"""Utility functions and classes for handling data."""
from typing import Union

import numpy as np
import pandas as pd
import torch

from ..models.features import CategoricalFeature, NumericalFeature


def prepare_features(
    X: pd.DataFrame, features: list[Union[NumericalFeature, CategoricalFeature]]
):
    """Maps categorical features to their integer indices in place."""
    for feature in features:
        feature_data = X[feature.feature_name]

        if isinstance(feature, CategoricalFeature):
            feature_data = feature_data.map(feature.category_indices)

        if feature.missing_input_value is not None:
            feature_data = feature_data.fillna(feature.missing_input_value)

        X[feature.feature_name] = feature_data


class Dataset(torch.utils.data.Dataset):
    """A class for loading a dataset for a calibrated model."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        features: list[Union[NumericalFeature, CategoricalFeature]],
    ):
        """Initializes an instance of `Dataset`."""
        self.X = X.copy()
        self.y = y.copy()

        selected_features = [feature.feature_name for feature in features]
        unavailable_features = set(selected_features) - set(self.X.columns)
        if len(unavailable_features) > 0:
            raise ValueError(f"Features {unavailable_features} not found in dataset.")

        drop_features = list(set(self.X.columns) - set(selected_features))
        self.X.drop(drop_features, axis=1, inplace=True)
        prepare_features(self.X, features)

        self.data = torch.from_numpy(self.X.values).double()
        self.labels = torch.from_numpy(self.y).double()[:, None]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.data[idx], self.labels[idx]]
