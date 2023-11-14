"""Tests for the `Classifier` class."""
import numpy as np
import pandas as pd
import pytest

from pytorch_lattice.classifier import Classifier
from pytorch_lattice.feature_config import FeatureConfig
from pytorch_lattice.model_configs import LatticeConfig, LinearConfig
from pytorch_lattice.models import CalibratedLattice, CalibratedLinear


def test_initialization():
    """Tests that the classifier initializes properly."""
    expected_features = {"a": FeatureConfig(name="a"), "b": FeatureConfig(name="b")}
    clf = Classifier(list(expected_features.keys()))
    assert list(clf.features.keys()) == list(expected_features.keys())
    for name, config in clf.features.items():
        assert config.name == expected_features[name].name
    assert isinstance(clf.model_config, LinearConfig)
    assert clf.model is None


def test_configure():
    """Tests that configure returns the correct feature configs."""
    feature_names = ["a", "b"]
    clf = Classifier(feature_names)
    for name in feature_names:
        config = clf.configure(name)
        assert isinstance(config, FeatureConfig)
        assert config.name == name


@pytest.fixture(name="X")
def fixture_data():
    """Randomized training data for fitting a classifier."""
    return pd.DataFrame(
        {
            "numerical": np.random.rand(100),
            "categorical": np.random.choice(["a", "b", "c"], 100),
        }
    )


@pytest.fixture(name="y")
def fixture_labels(X):
    """Randomized training labels for fitting a classifier."""
    return np.random.randint(0, 2, len(X))


@pytest.mark.parametrize("model_config", [LinearConfig(), LatticeConfig()])
def test_fit_and_predict(model_config, X, y):
    """Tests that the classifier can be fit and generate predictions."""
    clf = Classifier(X.columns, model_config).fit(X, y, epochs=1)
    assert clf.model is not None
    preds = clf.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (100, 1)


def test_fit_linear(X, y):
    """Tests that a linear config will fit a calibrated linear model."""
    clf = Classifier(X.columns, LinearConfig()).fit(X, y, epochs=1)
    assert clf.model is not None
    assert isinstance(clf.model, CalibratedLinear)


def test_fit_lattice(X, y):
    """Tests that a lattice config will fit a calibrated lattice model."""
    clf = Classifier(X.columns, LatticeConfig()).fit(X, y, epochs=1)
    assert clf.model is not None
    assert isinstance(clf.model, CalibratedLattice)


@pytest.mark.parametrize("model_config", [LinearConfig(), LatticeConfig()])
def test_save_and_load(model_config, X, y, tmp_path):
    """Tests that the classifier can be saved and loaded."""
    clf = Classifier(X.columns, model_config).fit(X, y, epochs=1)
    clf.save(tmp_path)
    loaded_clf = Classifier.load(tmp_path)
    assert list(loaded_clf.__dict__.keys()) == list(clf.__dict__.keys())
    for name, config in clf.features.items():
        assert loaded_clf.features[name].__dict__ == config.__dict__
    assert loaded_clf.model_config.__dict__ == clf.model_config.__dict__
    assert isinstance(loaded_clf.model, type(clf.model))
