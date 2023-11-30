"""Functions for loading datasets to use with the PyTorch Lattice package."""
import numpy as np
import pandas as pd


def heart() -> tuple[pd.DataFrame, np.ndarray]:
    """Loads the UCI Statlog (Heart) dataset.

    The UCI Statlog (Heart) dataset is a classification dataset with 303 rows and 14
    columns. The target is binary, with 0 indicating no heart disease and 1 indicating
    heart disease. The features are a mix of categorical and numerical features. For
    more information, see https://archive.ics.uci.edu/ml/datasets/heart+Disease.

    Returns:
        A tuple `(X, y)` of the features and target.
    """
    X = pd.read_csv(
        "https://raw.githubusercontent.com/ControlAI/datasets/main/heart.csv"
    )
    y = np.array(X.pop("target").values)
    return X, y


def adult() -> tuple[pd.DataFrame, np.ndarray]:
    """Loads the UCI Adult Income dataset.

    The UCI Adult Income dataset is a classification dataset with 48,842 rows and 14
    columns. The target is binary, with 0 indicating an income of less than $50k and 1
    indicating an income of at least $50k. The features are a mix of categorical and
    numerical features. For more information, see
    https://archive.ics.uci.edu/dataset/2/adult

    Returns:
        A tuple `(X, y)` of the features and target.
    """
    X = pd.read_csv(
        "https://raw.githubusercontent.com/ControlAI/datasets/main/adult.csv"
    )
    y = np.array(X.pop("label").values)
    return X, y
