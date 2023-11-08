"""Numerical calibration module.

PyTorch implementation of the numerical calibration module. This module takes in a
single-dimensional input and transforms it using piece-wise linear functions that
satisfy desired bounds and monotonicity constraints.
"""
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..enums import Monotonicity, NumericalCalibratorInit


# pylint: disable-next=too-many-instance-attributes
class NumericalCalibrator(torch.nn.Module):
    """A numerical calibrator.

    This module takes an input of shape `(batch_size, 1)` and calibrates it using a
    piece-wise linear function that conforms to any provided constraints. The output
    will have the same shape as the input.

    Attributes:
        - All `__init__` arguments.
        kernel: `torch.nn.Parameter` that stores the piece-wise linear function weights.
        missing_output: `torch.nn.Parameter` that stores the output learned for any
            missing inputs. Only available if `missing_input_value` is provided.

    Example:
    ```python
    inputs = torch.tensor(...)  # shape: (batch_size, 1)
    calibrator = NumericalCalibrator(
        input_keypoints=np.linspace(1., 5., num=5),
        output_min=0.0,
        output_max=1.0,
        monotonicity=Monotonicity.INCREASING,
        kernel_init=NumericalCalibratorInit.EQUAL_HEIGHTS,
    )
    outputs = calibrator(inputs)
    ```
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_keypoints: np.ndarray,
        missing_input_value: Optional[float] = None,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        monotonicity: Monotonicity = Monotonicity.NONE,
        kernel_init: NumericalCalibratorInit = NumericalCalibratorInit.EQUAL_HEIGHTS,
        projection_iterations: int = 8,
    ) -> None:
        """Initializes an instance of `NumericalCalibrator`.

        Args:
            input_keypoints: Ordered list of float-valued keypoints for the underlying
                piece-wise linear function.
            missing_input_value: If provided, the calibrator will learn to map all
                instances of this missing input value to a learned output value.
            output_min: Minimum output value. If `None`, the minimum output value will
                be unbounded.
            output_max: Maximum output value. If `None`, the maximum output value will
                be unbounded.
            monotonicity: Monotonicity constraint for the underlying piece-wise linear
                function.
            kernel_init: Initialization scheme to use for the kernel.
            projectionion_iterations: Number of times to run Dykstra's projection
                algorithm when applying constraints.

        Raises:
            ValueError: If `kernel_init` is invalid.
        """
        super().__init__()

        self.input_keypoints = input_keypoints
        self.missing_input_value = missing_input_value
        self.output_min = output_min
        self.output_max = output_max
        self.monotonicity = monotonicity
        self.kernel_init = kernel_init
        self.projection_iterations = projection_iterations

        # Determine default output initialization values if bounds are not fully set.
        if output_min is not None and output_max is not None:
            output_init_min, output_init_max = self.output_min, self.output_max
        elif output_min is not None:
            output_init_min, output_init_max = self.output_min, self.output_min + 4.0
        elif output_max is not None:
            output_init_min, output_init_max = self.output_max - 4.0, self.output_max
        else:
            output_init_min, output_init_max = -2.0, 2.0
        self._output_init_min, self._output_init_max = output_init_min, output_init_max

        self._interpolation_keypoints = torch.from_numpy(input_keypoints[:-1])
        self._lengths = torch.from_numpy(input_keypoints[1:] - input_keypoints[:-1])

        # First row of the kernel represents the bias. The remaining rows represent
        # the y-value delta compared to the previous point i.e. the segment heights.
        @torch.no_grad()
        def initialize_kernel() -> torch.Tensor:
            output_init_range = self._output_init_max - self._output_init_min
            if kernel_init == NumericalCalibratorInit.EQUAL_HEIGHTS:
                num_segments = self._interpolation_keypoints.size()[0]
                segment_height = output_init_range / num_segments
                heights = torch.tensor([[segment_height]] * num_segments)
            elif kernel_init == NumericalCalibratorInit.EQUAL_SLOPES:
                heights = (
                    self._lengths * output_init_range / torch.sum(self._lengths)
                )[:, None]
            else:
                raise ValueError(f"Unknown kernel init: {self.kernel_init}")

            if monotonicity == Monotonicity.DECREASING:
                bias = self._output_init_max
                heights = -heights
            else:
                bias = self._output_init_min
            bias = torch.tensor([[bias]])
            return torch.cat((bias, heights), 0).double()

        self.kernel = torch.nn.Parameter(initialize_kernel())

        if missing_input_value:
            self.missing_output = torch.nn.Parameter(torch.Tensor(1))
            torch.nn.init.constant_(
                self.missing_output,
                (self._output_init_min + self._output_init_max) / 2.0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        """Calibrates numerical inputs through piece-wise linear interpolation.

        Args:
            x: The input tensor of shape `(batch_size, 1)`.

        Returns:
            torch.Tensor of shape `(batch_size, 1)` containing calibrated input values.
        """
        interpolation_weights = (x - self._interpolation_keypoints) / self._lengths
        interpolation_weights = torch.minimum(interpolation_weights, torch.tensor(1.0))
        interpolation_weights = torch.maximum(interpolation_weights, torch.tensor(0.0))
        interpolation_weights = torch.cat(
            (torch.ones_like(x), interpolation_weights), -1
        )
        result = torch.mm(interpolation_weights, self.kernel)

        if self.missing_input_value is not None:
            missing_mask = torch.eq(x, self.missing_input_value).long()
            result = missing_mask * self.missing_output + (1.0 - missing_mask) * result

        return result

    @torch.no_grad()
    def assert_constraints(self, eps=1e-6) -> List[str]:
        """Asserts that layer satisfies specified constraints.

        This checks that weights follow monotonicity constraints and that the output is
        within bounds.

        Args:
            eps: the margin of error allowed

        Returns:
            A list of messages describing violated constraints including indices of
            monotonicity violations. If no constraints violated, the list will be empty.
        """
        weights = torch.squeeze(self.kernel.data)
        messages = []

        if (
            self.output_max is not None
            and torch.max(self.keypoints_outputs()) > self.output_max + eps
        ):
            messages.append("Max weight greater than output_max.")
        if (
            self.output_min is not None
            and torch.min(self.keypoints_outputs()) < self.output_min - eps
        ):
            messages.append("Min weight less than output_min.")

        diffs = weights[1:]
        violation_indices = []

        if self.monotonicity == Monotonicity.INCREASING:
            violation_indices = (diffs < -eps).nonzero().tolist()
        elif self.monotonicity == Monotonicity.DECREASING:
            violation_indices = (diffs > eps).nonzero().tolist()

        violation_indices = [(i[0], i[0] + 1) for i in violation_indices]
        if violation_indices:
            messages.append(f"Monotonicity violated at: {str(violation_indices)}.")

        return messages

    @torch.no_grad()
    def constrain(self) -> None:
        """Jointly projects kernel into desired constraints.

        Uses Dykstra's alternating projection algorithm to jointly project onto all
        given constraints. This algorithm projects with respect to the L2 norm, but it
        approached the norm from the "wrong" side. To ensure that all constraints are
        strictly met, we do final approximate projections that project strictly into the
        feasible space, but this is not an exact projection with respect to the L2 norm.
        Enough iterations make the impact of this approximation negligible.
        """
        constrain_bounds = self.output_min is not None or self.output_max is not None
        constrain_monotonicity = self.monotonicity != Monotonicity.NONE
        num_constraints = sum([constrain_bounds, constrain_monotonicity])

        # We do nothing to the weights in this case
        if num_constraints == 0:
            return

        original_bias, original_heights = self.kernel.data[0:1], self.kernel.data[1:]
        previous_bias_delta = defaultdict(lambda: torch.zeros_like(original_bias))
        previous_heights_delta = defaultdict(lambda: torch.zeros_like(original_heights))

        def apply_bound_constraints(bias, heights):
            previous_bias = bias - previous_bias_delta["BOUNDS"]
            previous_heights = heights - previous_heights_delta["BOUNDS"]
            if constrain_monotonicity:
                bias, heights = self._project_monotonic_bounds(
                    previous_bias, previous_heights
                )
            else:
                bias, heights = self._approximately_project_bounds_only(
                    previous_bias, previous_heights
                )
            previous_bias_delta["BOUNDS"] = bias - previous_bias
            previous_heights_delta["BOUNDS"] = heights - previous_heights
            return bias, heights

        def apply_monotonicity_constraints(heights):
            previous_heights = heights - previous_bias_delta["MONOTONICITY"]
            heights = self._project_monotonicity(previous_heights)
            previous_heights_delta["MONOTONICITY"] = heights - previous_heights
            return heights

        def apply_dykstras_projection(bias, heights):
            if constrain_bounds:
                bias, heights = apply_bound_constraints(bias, heights)
            if constrain_monotonicity:
                heights = apply_monotonicity_constraints(heights)
            return bias, heights

        def finalize_constraints(bias, heights):
            if constrain_monotonicity:
                heights = self._project_monotonicity(heights)
            if constrain_bounds:
                if constrain_monotonicity:
                    bias, heights = self._squeeze_by_scaling(bias, heights)
                else:
                    bias, heights = self._approximately_project_bounds_only(
                        bias, heights
                    )
            return bias, heights

        projected_bias, projected_heights = apply_dykstras_projection(
            original_bias, original_heights
        )
        if num_constraints > 1:
            for _ in range(self.projection_iterations - 1):
                projected_bias, projected_heights = apply_dykstras_projection(
                    projected_bias, projected_heights
                )
            projected_bias, projected_heights = finalize_constraints(
                projected_bias, projected_heights
            )

        self.kernel.data = torch.cat((projected_bias, projected_heights), 0)

    @torch.no_grad()
    def keypoints_inputs(self) -> torch.Tensor:
        """Returns tensor of keypoint inputs."""
        return torch.cat(
            (
                self._interpolation_keypoints,
                self._interpolation_keypoints[-1:] + self._lengths[-1:],
            ),
            0,
        )

    @torch.no_grad()
    def keypoints_outputs(self) -> torch.Tensor:
        """Returns tensor of keypoint outputs."""
        return torch.cumsum(self.kernel.data, 0).T[0]

    ################################################################################
    ############################## PRIVATE METHODS #################################
    ################################################################################

    def _project_monotonic_bounds(
        self, bias: torch.Tensor, heights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects bias and heights into bounds considering monotonicity.

        For computation simplification in the case of decreasing monotonicity, we mirror
        bias and heights and swap-mirror the output bounds. After doing the standard
        projection with resepct to increasing monotonicity, we then mirror everything
        back to get the correct projection.

        Args:
            bias: The bias of the underlying piece-wise linear function.
            heights: The heights of each segment of the underlying piece-wise linear
                function.

        Returns:
            A tuple containing the projected bias and projected heights.
        """
        output_min, output_max = self.output_min, self.output_max
        decreasing = self.monotonicity == Monotonicity.DECREASING
        if decreasing:
            bias, heights = -bias, -heights
            output_min = None if self.output_max is None else -1 * self.output_max
            output_max = None if self.output_min is None else -1 * self.output_min
        if output_max is not None:
            num_heights = heights.size()[0]
            output_max_diffs = output_max - (bias + torch.sum(heights, 0))
            bias_delta = output_max_diffs / (num_heights + 1)
            bias_delta = torch.minimum(bias_delta, torch.tensor(0.0))
            if output_min is not None:
                bias = torch.maximum(bias + bias_delta, torch.tensor(output_min))
                heights_delta = output_max_diffs / num_heights
            else:
                bias += bias_delta
                heights_delta = bias_delta
            heights += torch.minimum(heights_delta, torch.tensor(0.0))
        elif output_min is not None:
            bias = torch.maximum(bias, torch.tensor(output_min))
        if decreasing:
            bias, heights = -bias, -heights
        return bias, heights

    def _approximately_project_bounds_only(
        self, bias: torch.Tensor, heights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects bias and heights without considering monotonicity.

        It is worth noting that this projection is an approximation and is not an exact
        projection with respect to the L2 norm; however, it is sufficiently accurate and
        efficient in practice for non-monotonic functions.

        Args:
            bias: The bias of the underlying piece-wise linear function.
            heights: The heights of each segment of the underlying piece-wise linear
                function.

        Returns:
            A tuple containing the projected bias and projected heights.
        """
        sums = torch.cumsum(torch.cat((bias, heights), 0), 0)
        if self.output_min is not None:
            sums = torch.maximum(sums, torch.tensor(self.output_min))
        if self.output_max is not None:
            sums = torch.minimum(sums, torch.tensor(self.output_max))
        bias = sums[0:1]
        heights = sums[1:] - sums[:-1]
        return bias, heights

    def _project_monotonicity(self, heights: torch.Tensor) -> torch.Tensor:
        """Returns bias and heights projected into desired monotonicity constraints."""
        if self.monotonicity == Monotonicity.INCREASING:
            return torch.maximum(heights, torch.tensor(0.0))
        if self.monotonicity == Monotonicity.DECREASING:
            return torch.minimum(heights, torch.tensor(0.0))
        return heights

    def _squeeze_by_scaling(
        self, bias: torch.Tensor, heights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Squeezes monotonic calibrators by scaling them into bound constraints.

        It is worth noting that this is not an exact projection with respect to the L2
        norm; however, it maintains convexity, which projection by shift does not.

        Args:
            bias: The bias of the underlying piece-wise linear function.
            heights: The heights of each segment of the underlying piece-wise linear
                function.

        Returns:
            A tuple containing the projected bias and projected heights.
        """
        decreasing = self.monotonicity == Monotonicity.DECREASING
        output_max = self.output_max
        if decreasing:
            if self.output_min is None:
                return bias, heights
            bias, heights = -bias, -heights
            output_max = None if self.output_min is None else -1 * self.output_min
        if output_max is None:
            return bias, heights
        delta = output_max - bias
        scaling_factor = torch.where(
            delta > 0.0001, torch.sum(heights, 0) / delta, torch.ones_like(delta)
        )
        heights /= torch.maximum(scaling_factor, torch.tensor(1.0))
        if decreasing:
            bias, heights = -bias, -heights
        return bias, heights
