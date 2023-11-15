"""A class for training classifiers on tabular data using calibrated models."""
from __future__ import annotations

import os
import pickle
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from .enums import Monotonicity
from .feature_config import FeatureConfig
from .model_configs import LatticeConfig, LinearConfig
from .models import (
    CalibratedLattice,
    CalibratedLinear,
)
from .models.features import CategoricalFeature, NumericalFeature
from .utils.data import Dataset, prepare_features

MISSING_INPUT_VALUE = -123456789


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
        model_config: The model configuration to use for fitting the classifier.
        self.model: The fitted model. This will be `None` until `fit` is called.
    """

    def __init__(
        self,
        feature_names: list[str],
        model_config: Optional[Union[LinearConfig, LatticeConfig]] = None,
    ):
        """Initializes an instance of `Classifier`."""
        self.features = {
            feature_name: FeatureConfig(name=feature_name)
            for feature_name in feature_names
        }
        self.model_config = model_config if model_config is not None else LinearConfig()
        self.model: Optional[Union[CalibratedLinear, CalibratedLattice]] = None

    def configure(self, feature_name: str):
        """Returns a `FeatureConfig` object for the given feature name."""
        return self.features[feature_name]

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        shuffle: bool = False,
    ) -> Classifier:
        """Returns this classifier after fitting a model to the given data.

        Note that calling this function will overwrite any existing model and train a
        new model from scratch.

        Args:
            X: A `pd.DataFrame` containing the features for the training data.
            y: A `np.ndarray` containing the labels for the training data.
            epochs: The number of epochs for which to fit the classifier.
            batch_size: The batch size to use for fitting.
            learning_rate: The learning rate to use for fitting the model.
            shuffle: Whether to shuffle the data before fitting.
        """
        model = self._create_model(X)
        optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        dataset = Dataset(X, y, model.features)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        for _ in trange(epochs, desc="Training Progress"):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                model.apply_constraints()

        self.model = model
        return self

    def predict(self, X: pd.DataFrame, logits: bool = False) -> np.ndarray:
        """Returns predictions for the given data.

        Args:
            X: a `pd.DataFrame` containing to data for which to generate predictions.
            logits: If `True`, returns the logits of the predictions. Otherwise, returns
                probabilities.
        """
        if self.model is None:
            raise RuntimeError("Cannot predict before fitting the model.")

        self.model.eval()
        X_copy = X[[feature.feature_name for feature in self.model.features]].copy()
        prepare_features(X_copy, self.model.features)
        X_tensor = torch.tensor(X_copy.values).double()
        with torch.no_grad():
            preds = self.model(X_tensor).numpy()

        if logits:
            return preds
        else:
            return 1.0 / (1.0 + np.exp(-preds))

    def save(self, filepath: str):
        """Saves the classifier to the specified path.

        Args:
            filepath: The directory where the classifier will be saved. If the directory
                does not exist, this function will attempt to create it. If the
                directory already exists, this function will overwrite any existing
                content with conflicting filenames.
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with open(os.path.join(filepath, "clf_attrs.pkl"), "wb") as f:
            attrs = {key: self.__dict__[key] for key in ["features", "model_config"]}
            pickle.dump(attrs, f)
        if self.model is not None:
            model_path = os.path.join(filepath, "model.pt")
            torch.save(self.model, model_path)

    @classmethod
    def load(cls, filepath: str) -> Classifier:
        """Loads a `Classifier` from the specified path.

        Args:
            filepath: The filepath from which to load the classifier. The filepath
                should point to the filepath used in the `save` method when saving the
                classifier.

        Returns:
            A `Classifier` instance.
        """
        with open(os.path.join(filepath, "clf_attrs.pkl"), "rb") as f:
            attrs = pickle.load(f)

        clf = cls([])
        clf.__dict__.update(attrs)

        model_path = os.path.join(filepath, "model.pt")
        if os.path.exists(model_path):
            clf.model = torch.load(model_path)

        return clf

    ################################################################################
    ############################## PRIVATE METHODS #################################
    ################################################################################

    def _create_model(
        self, X: pd.DataFrame
    ) -> Union[CalibratedLinear, CalibratedLattice]:
        """Returns a model based on `self.features` and `self.model_config`."""
        features: list[Union[CategoricalFeature, NumericalFeature]] = []

        for feature_name, feature in self.features.items():
            if X[feature_name].dtype.kind in ["S", "O", "b"]:  # string, object, bool
                if feature._categories is None:
                    categories = X[feature_name].unique().tolist()
                    feature.categories(categories)
                else:
                    categories = feature._categories
                if feature._monotonicity is not None and isinstance(
                    feature._monotonicity, list
                ):
                    monotonicity_pairs = feature._monotonicity
                else:
                    monotonicity_pairs = None
                features.append(
                    CategoricalFeature(
                        feature_name=feature_name,
                        categories=categories,
                        missing_input_value=MISSING_INPUT_VALUE,
                        monotonicity_pairs=monotonicity_pairs,
                        lattice_size=feature._lattice_size,
                    )
                )
            else:  # numerical feature
                if feature._monotonicity is not None and isinstance(
                    feature._monotonicity, str
                ):
                    monotonicity = feature._monotonicity
                else:
                    monotonicity = None
                features.append(
                    NumericalFeature(
                        feature_name=feature_name,
                        data=np.array(X[feature_name].values),
                        num_keypoints=feature._num_keypoints,
                        input_keypoints_init=feature._input_keypoints_init,
                        missing_input_value=MISSING_INPUT_VALUE,
                        monotonicity=monotonicity,
                        projection_iterations=feature._projection_iterations,
                        lattice_size=feature._lattice_size,
                    )
                )

        if isinstance(self.model_config, LinearConfig):
            return CalibratedLinear(
                features,
                self.model_config.output_min,
                self.model_config.output_max,
                self.model_config.use_bias,
                self.model_config.output_calibration_num_keypoints,
            )
        else:
            return CalibratedLattice(
                features,
                True,
                self.model_config.output_min,
                self.model_config.output_max,
                self.model_config.kernel_init,
                self.model_config.interpolation,
                self.model_config.output_calibration_num_keypoints,
            )
