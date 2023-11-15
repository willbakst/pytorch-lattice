"""Plotting functions for PyTorch Lattice calibrated models using matplotlib."""
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from .layers import CategoricalCalibrator
from .models import CalibratedLattice, CalibratedLinear
from .models.features import CategoricalFeature


def calibrator(
    model: Union[CalibratedLinear, CalibratedLattice],
    feature_name: str,
) -> None:
    """Plots the calibrator for the given feature and calibrated model.

    Args:
        model: The calibrated model for which to plot calibrators.
        feature_name: The name of the feature for which to plot the calibrator.
    """
    if feature_name not in model.calibrators:
        raise ValueError(f"Feature {feature_name} not found in model.")

    calibrator = model.calibrators[feature_name]
    input_keypoints = calibrator.keypoints_inputs().numpy()
    output_keypoints = calibrator.keypoints_outputs().numpy()

    if isinstance(calibrator, CategoricalCalibrator):
        model_feature = next(
            (x for x in model.features if x.feature_name == feature_name), None
        )
        if isinstance(model_feature, CategoricalFeature):
            input_keypoints = np.array(
                [
                    model_feature.categories[i]
                    if i < len(input_keypoints) - 1
                    else "<Missing>"
                    for i, ik in enumerate(input_keypoints)
                ]
            )
        plt.xticks(rotation=45)
        plt.bar(input_keypoints, output_keypoints)
    else:
        plt.plot(input_keypoints, output_keypoints)

    plt.title(f"Calibrator: {feature_name}")
    plt.xlabel("Input Keypoints")
    plt.ylabel("Output Keypoints")
    plt.show()


def linear_coefficients(model: CalibratedLinear) -> None:
    """Plots the coefficients for the linear layer of a calibrated linear model."""
    if not isinstance(model, CalibratedLinear):
        raise ValueError(
            "Model must be a `CalibratedLinear` model to plot linear coefficients."
        )
    linear_coefficients = dict(
        zip(
            [feature.feature_name for feature in model.features],
            model.linear.kernel.detach().numpy().flatten(),
        )
    )
    if model.use_bias:
        linear_coefficients["bias"] = model.linear.bias.detach().numpy()[0]

    plt.bar(list(linear_coefficients.keys()), list(linear_coefficients.values()))
    plt.title("Linear Coefficients")
    plt.xlabel("Feature Name")
    plt.xticks(rotation=45)
    plt.ylabel("Coefficient Value")
    plt.show()
