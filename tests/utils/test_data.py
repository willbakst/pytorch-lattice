"""Tests for data utilities."""
import numpy as np
import pandas as pd
import pytest

from pytorch_lattice.models.features import CategoricalFeature, NumericalFeature
from pytorch_lattice.utils.data import Dataset


@pytest.fixture(name="X")
def fixture_X():
    """Returns a `pd.DataFrame` fixture for testing."""
    return pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": ["a", "b", "c"], "c": [4.0, 5.0, 6.0]}
    )


@pytest.fixture(name="features")
def fixture_features(X):
    """Returns a list of model features for testing."""
    return [
        NumericalFeature(feature_name="a", data=X["a"].values),
        CategoricalFeature(feature_name="b", categories=list(X["b"].values)),
    ]


@pytest.fixture(name="dataset")
def fixture_dataset(X, features):
    """Returns a `Dataset` fixture for testing."""
    y = np.array([1.0, 2.0, 3.0])
    return Dataset(X, y, features)


def test_prepare_features(X, features):
    """Tests that the `prepare_features` function works as expected."""
    raise NotImplementedError()


def test_initialization(dataset):
    """Tests that `Dataset` initialization work as expected."""
    expected_X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0, 1, 2]})
    assert dataset.X.equals(expected_X)


def test_len(dataset):
    """Tests that `Dataset` __len__ is correct."""
    assert len(dataset) == 3


def test_get_item(dataset):
    """Tests that `Dataset` __getitem__ is correct."""
    inputs, labels = dataset[:2]
    assert np.array_equal(inputs, np.array([[1.0, 0], [2.0, 1]]))
    assert np.array_equal(labels, np.array([[1.0], [2.0]]))
