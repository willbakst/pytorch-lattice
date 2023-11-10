"""Class for easily constructing a calibrated linear model."""
from typing import Optional, Union

import torch

from ..layers import Linear
from .features import CategoricalFeature, NumericalFeature
from .model_utils import (
    calibrate_and_stack,
    initialize_feature_calibrators,
    initialize_monotonicities,
    initialize_output_calibrator,
)


class CalibratedLinear(torch.nn.Module):
    """PyTorch Calibrated Linear Model.

    Creates a `torch.nn.Module` representing a calibrated linear model, which will be
    constructed using the provided model configuration. Note that the model inputs
    should match the order in which they are defined in the `feature_configs`.

    Attributes:
        - All `__init__` arguments.
        calibrators: A dictionary that maps feature names to their calibrators.
        linear: The `Linear` layer of the model.
        output_calibrator: The output `NumericalCalibrator` calibration layer. This
            will be `None` if no output calibration is desired.

    Example:

    ```python
    csv_data = CSVData(...)

    feature_configs = [...]
    calibrated_model = CalibratedLinear(feature_configs, ...)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(calibrated_model.parameters(recurse=True), lr=1e-1)

    csv_data.prepare(feature_configs, "target", ...)
    for epoch in range(100):
        for examples, targets in csv_data.batch(64):
            optimizer.zero_grad()
            outputs = calibrated_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            calibrated_model.constrain()
    ```
    """

    def __init__(
        self,
        features: list[Union[NumericalFeature, CategoricalFeature]],
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        use_bias: bool = True,
        output_calibration_num_keypoints: Optional[int] = None,
    ) -> None:
        """Initializes an instance of `CalibratedLinear`.

        Args:
            features: A list of numerical and/or categorical feature configs.
            output_min: The minimum output value for the model. If `None`, the minimum
                output value will be unbounded.
            output_max: The maximum output value for the model. If `None`, the maximum
                output value will be unbounded.
            use_bias: Whether to use a bias term for the linear combination. If any of
                `output_min`, `output_max`, or `output_calibration_num_keypoints` are
                set, a bias term will not be used regardless of the setting here.
            output_calibration_num_keypoints: The number of keypoints to use for the
                output calibrator. If `None`, no output calibration will be used.

        Raises:
            ValueError: If any feature configs are not `NUMERICAL` or `CATEGORICAL`.
        """
        super().__init__()

        self.features = features
        self.output_min = output_min
        self.output_max = output_max
        self.use_bias = use_bias
        self.output_calibration_num_keypoints = output_calibration_num_keypoints
        self.monotonicities = initialize_monotonicities(features)
        self.calibrators = initialize_feature_calibrators(
            features=features, output_min=output_min, output_max=output_max
        )

        self.linear = Linear(
            input_dim=len(features),
            monotonicities=self.monotonicities,
            use_bias=use_bias,
            weighted_average=bool(
                output_min is not None
                or output_max is not None
                or output_calibration_num_keypoints
            ),
        )

        self.output_calibrator = initialize_output_calibrator(
            output_calibration_num_keypoints=output_calibration_num_keypoints,
            monotonic=not all(m is None for m in self.monotonicities),
            output_min=output_min,
            output_max=output_max,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs an input through the network to produce a calibrated linear output.

        Args:
            x: The input tensor of feature values of shape `(batch_size, num_features)`.

        Returns:
            torch.Tensor of shape `(batch_size, 1)` containing the model output result.
        """
        result = calibrate_and_stack(x, self.calibrators)
        result = self.linear(result)
        if self.output_calibrator is not None:
            result = self.output_calibrator(result)

        return result

    @torch.no_grad()
    def assert_constraints(self) -> dict[str, list[str]]:
        """Asserts all layers within model satisfied specified constraints.

        Asserts monotonicity pairs and output bounds for categorical calibrators,
        monotonicity and output bounds for numerical calibrators, and monotonicity and
        weights summing to 1 if weighted_average for linear layer.

        Returns:
            A dict where key is feature_name for calibrators and 'linear' for the linear
            layer, and value is the error messages for each layer. Layers with no error
            messages are not present in the dictionary.
        """
        messages = {}

        for name, calibrator in self.calibrators.items():
            calibrator_messages = calibrator.assert_constraints()
            if calibrator_messages:
                messages[f"{name}_calibrator"] = calibrator_messages
        linear_messages = self.linear.assert_constraints()
        if linear_messages:
            messages["linear"] = linear_messages
        if self.output_calibrator:
            output_calibrator_messages = self.output_calibrator.assert_constraints()
            if output_calibrator_messages:
                messages["output_calibrator"] = output_calibrator_messages

        return messages

    @torch.no_grad()
    def constrain(self) -> None:
        """Constrains the model into desired constraints specified by the config."""
        for calibrator in self.calibrators.values():
            calibrator.constrain()
        self.linear.constrain()
        if self.output_calibrator:
            self.output_calibrator.constrain()
