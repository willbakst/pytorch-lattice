"""Class for easily constructing a calibrated lattice model."""
from typing import Optional, Union

import torch

from ..constrained_module import ConstrainedModule
from ..enums import (
    Interpolation,
    LatticeInit,
)
from ..layers import Lattice
from .features import CategoricalFeature, NumericalFeature
from .model_utils import (
    calibrate_and_stack,
    initialize_feature_calibrators,
    initialize_monotonicities,
    initialize_output_calibrator,
)


class CalibratedLattice(ConstrainedModule):
    """PyTorch Calibrated Lattice Model.

    Creates a `torch.nn.Module` representing a calibrated lattice model, which will be
    constructed using the provided model configuration. Note that the model inputs
    should match the order in which they are defined in the `feature_configs`.

    Attributes:
        - All `__init__` arguments.
        calibrators: A dictionary that maps feature names to their calibrators.
        lattice: The `Lattice` layer of the model.
        output_calibrator: The output `NumericalCalibrator` calibration layer. This
            will be `None` if no output calibration is desired.

    Example:

    ```python
    csv_data = CSVData(...)

    feature_configs = [...]
    calibrated_model = CalibratedLattice(feature_configs, ...)

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
            calibrated_model.apply_constraints()
    ```
    """

    def __init__(
        self,
        features: list[Union[NumericalFeature, CategoricalFeature]],
        clip_inputs: bool = True,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        kernel_init: LatticeInit = LatticeInit.LINEAR,
        interpolation: Interpolation = Interpolation.HYPERCUBE,
        output_calibration_num_keypoints: Optional[int] = None,
    ) -> None:
        """Initializes an instance of `CalibratedLattice`.

        Args:
            features: A list of numerical and/or categorical feature configs.
            clip_inputs: Whether to restrict inputs to the bounds of lattice.
            output_min: The minimum output value for the model. If `None`, the minimum
                output value will be unbounded.
            output_max: The maximum output value for the model. If `None`, the maximum
                output value will be unbounded.
            kernel_init: the method of initializing kernel weights. If otherwise
                unspecified, will default to `LatticeInit.LINEAR`.
            interpolation: the method of interpolation in the lattice's forward pass.
                If otherwise unspecified, will default to `Interpolation.HYPERCUBE`.
            output_calibration_num_keypoints: The number of keypoints to use for the
                output calibrator. If `None`, no output calibration will be used.

        Raises:
            ValueError: If any feature configs are not `NUMERICAL` or `CATEGORICAL`.
        """
        super().__init__()

        self.features = features
        self.clip_inputs = clip_inputs
        self.output_min = output_min
        self.output_max = output_max
        self.kernel_init = kernel_init
        self.interpolation = interpolation
        self.output_calibration_num_keypoints = output_calibration_num_keypoints
        self.monotonicities = initialize_monotonicities(features)
        self.calibrators = initialize_feature_calibrators(
            features=features,
            output_min=0,
            output_max=[feature.lattice_size - 1 for feature in features],
        )

        self.lattice = Lattice(
            lattice_sizes=[feature.lattice_size for feature in features],
            monotonicities=self.monotonicities,
            clip_inputs=self.clip_inputs,
            output_min=self.output_min,
            output_max=self.output_max,
            interpolation=interpolation,
            kernel_init=kernel_init,
        )

        self.output_calibrator = initialize_output_calibrator(
            output_calibration_num_keypoints=output_calibration_num_keypoints,
            monotonic=not all(m is None for m in self.monotonicities),
            output_min=output_min,
            output_max=output_max,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs an input through the network to produce a calibrated lattice output.

        Args:
            x: The input tensor of feature values of shape `(batch_size, num_features)`.

        Returns:
            torch.Tensor of shape `(batch_size, 1)` containing the model output result.
        """
        result = calibrate_and_stack(x, self.calibrators)
        result = self.lattice(result)
        if self.output_calibrator is not None:
            result = self.output_calibrator(result)

        return result

    @torch.no_grad()
    def apply_constraints(self) -> None:
        """Constrains the model into desired constraints specified by the config."""
        for calibrator in self.calibrators.values():
            calibrator.apply_constraints()
        self.lattice.apply_constraints()
        if self.output_calibrator:
            self.output_calibrator.apply_constraints()

    @torch.no_grad()
    def assert_constraints(self, eps=1e-6) -> dict[str, list[str]]:
        """Asserts all layers within model satisfied specified constraints.

        Asserts monotonicity pairs and output bounds for categorical calibrators,
        monotonicity and output bounds for numerical calibrators, and monotonicity and
        weights summing to 1 if weighted_average for linear layer.

        Args:
            eps: the margin of error allowed

        Returns:
            A dict where key is feature_name for calibrators and 'linear' for the linear
            layer, and value is the error messages for each layer. Layers with no error
            messages are not present in the dictionary.
        """
        messages = {}

        for name, calibrator in self.calibrators.items():
            calibrator_messages = calibrator.assert_constraints(eps)
            if calibrator_messages:
                messages[f"{name}_calibrator"] = calibrator_messages
        lattice_messages = self.lattice.assert_constraints(eps)
        if lattice_messages:
            messages["lattice"] = lattice_messages
        if self.output_calibrator:
            output_calibrator_messages = self.output_calibrator.assert_constraints(eps)
            if output_calibrator_messages:
                messages["output_calibrator"] = output_calibrator_messages

        return messages
