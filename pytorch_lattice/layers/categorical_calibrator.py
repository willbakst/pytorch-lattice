"""Categorical calibration module.

PyTorch implementation of the categorical calibration module. This module takes in a
single-dimensional input of categories represented as indices and transforms it by
mapping a given category to its learned output value.
"""
from collections import defaultdict
from graphlib import CycleError, TopologicalSorter
from typing import List, Optional, Tuple

import torch

from ..enums import CategoricalCalibratorInit


# pylint: disable-next=too-many-instance-attributes
class CategoricalCalibrator(torch.nn.Module):
    """A categorical calibrator.

    This module takes an input of shape `(batch_size, 1)` and calibrates it by mapping a
    given category to its learned output value. The output will have the same shape as
    the input.

    Attributes:
        - All `__init__` arguments.
        kernel: `torch.nn.Parameter` that stores the categorical mapping weights.

    Example:
    ```python
    inputs = torch.tensor(...)  # shape: (batch_size, 1)
    calibrator = CategoricalCalibrator(
        num_categories=5,
        missing_input_value=-1,
        output_min=0.0
        output_max=1.0,
        monotonicity_pairs=[(0, 1), (1, 2)],
        kernel_init=CateegoricalCalibratorInit.UNIFORM,
    )
    outputs = calibrator(inputs)
    ```
    """

    # pylint: disable-next=too-many-branches
    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_categories: int,
        missing_input_value: Optional[float] = None,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        monotonicity_pairs: Optional[List[Tuple[int, int]]] = None,
        kernel_init: CategoricalCalibratorInit = CategoricalCalibratorInit.UNIFORM,
    ) -> None:
        """Initializes an instance of `CategoricalCalibrator`.

        Args:
            num_categories: The number of known categories.
            missing_input_value: If provided, the calibrator will learn to map all
                instances of this missing input value to a learned output value just
                the same as it does for known categories. Note that `num_categories`
                will be one greater to include this missing category.
            output_min: Minimum output value. If `None`, the minimum output value will
                be unbounded.
            output_max: Maximum output value. If `None`, the maximum output value will
                be unbounded.
            monotonicity_pairs: List of pairs of indices `(i,j)` indicating that the
                calibrator output for index `j` should be greater than or equal to that
                of index `i`.
            kernel_init: Initialization scheme to use for the kernel.

        Raises:
            ValueError: If `monotonicity_pairs` is cyclic.
            ValueError: If `kernel_init` is invalid.
        """
        super().__init__()

        self.num_categories = (
            num_categories + 1 if missing_input_value is not None else num_categories
        )
        self.missing_input_value = missing_input_value
        self.output_min = output_min
        self.output_max = output_max
        self.monotonicity_pairs = monotonicity_pairs
        if monotonicity_pairs:
            self._monotonicity_graph = defaultdict(list)
            self._reverse_monotonicity_graph = defaultdict(list)
            for i, j in monotonicity_pairs:
                self._monotonicity_graph[i].append(j)
                self._reverse_monotonicity_graph[j].append(i)
            try:
                self._monotonically_sorted_indices = [
                    *TopologicalSorter(self._reverse_monotonicity_graph).static_order()
                ]
            except CycleError as exc:
                raise ValueError("monotonicity_pairs is cyclic") from exc
        self.kernel_init = kernel_init

        self.kernel = torch.nn.Parameter(torch.Tensor(self.num_categories, 1).double())
        if kernel_init == CategoricalCalibratorInit.CONSTANT:
            if output_min is not None and output_max is not None:
                init_value = (output_min + output_max) / 2
            elif output_min is not None:
                init_value = output_min
            elif output_max is not None:
                init_value = output_max
            else:
                init_value = 0.0
            torch.nn.init.constant_(self.kernel, init_value)
        elif kernel_init == CategoricalCalibratorInit.UNIFORM:
            if output_min is None and output_max is None:
                low, high = -0.05, 0.05
            elif output_min is None:
                low, high = output_max - 0.05, output_max
            elif output_max is None:
                low, high = output_min, output_min + 0.05
            else:
                low, high = output_min, output_max
            torch.nn.init.uniform_(self.kernel, low, high)
        else:
            raise ValueError(f"Unknown kernel init: {kernel_init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        """Calibrates categorical inputs through a learned mapping.

        Args:
            x: The input tensor of category indices of shape `(batch_size, 1)`.

        Returns:
            torch.Tensor of shape `(batch_size, 1)` containing calibrated input values.
        """
        if self.missing_input_value is not None:
            missing_category_tensor = torch.zeros_like(x) + (self.num_categories - 1)
            x = torch.where(x == self.missing_input_value, missing_category_tensor, x)
        # TODO: test if using torch.gather is faster than one-hot matmul.
        one_hot = torch.nn.functional.one_hot(
            torch.squeeze(x, -1).long(), num_classes=self.num_categories
        ).double()
        return torch.mm(one_hot, self.kernel)

    @torch.no_grad()
    def assert_constraints(self, eps=1e-6) -> List[str]:
        """Asserts that layer satisfies specified constraints.

        This checks that weights at the indexes of monotonicity pairs are in the correct
        order and that the output is within bounds.

        Args:
            eps: the margin of error allowed

        Returns:
            A list of messages describing violated constraints including violated
            monotonicity pairs. If no constraints  violated, the list will be empty.
        """
        weights = torch.squeeze(self.kernel.data)
        messages = []

        if self.output_max is not None and torch.max(weights) > self.output_max + eps:
            messages.append("Max weight greater than output_max.")
        if self.output_min is not None and torch.min(weights) < self.output_min - eps:
            messages.append("Min weight less than output_min.")

        if self.monotonicity_pairs:
            violation_indices = [
                (i, j)
                for (i, j) in self.monotonicity_pairs
                if weights[i] - weights[j] > eps
            ]
            if violation_indices:
                messages.append(f"Monotonicity violated at: {str(violation_indices)}.")

        return messages

    @torch.no_grad()
    def constrain(self) -> None:
        """Projects kernel into desired constraints."""
        projected_kernel_data = self.kernel.data
        if self.monotonicity_pairs:
            projected_kernel_data = self._approximately_project_monotonicity_pairs(
                projected_kernel_data
            )
        if self.output_min is not None:
            projected_kernel_data = torch.maximum(
                projected_kernel_data, torch.tensor(self.output_min)
            )
        if self.output_max is not None:
            projected_kernel_data = torch.minimum(
                projected_kernel_data, torch.tensor(self.output_max)
            )
        self.kernel.data = projected_kernel_data

    @torch.no_grad()
    def keypoints_inputs(self) -> torch.Tensor:
        """Returns a tensor of keypoint inputs (category indices)."""
        if self.missing_input_value is not None:
            return torch.cat(
                (
                    torch.arange(self.num_categories - 1),
                    torch.tensor([self.missing_input_value]),
                ),
                0,
            )
        return torch.arange(self.num_categories)

    @torch.no_grad()
    def keypoints_outputs(self) -> torch.Tensor:
        """Returns a tensor of keypoint outputs."""
        return torch.squeeze(self.kernel.data, -1)

    ################################################################################
    ############################## PRIVATE METHODS #################################
    ################################################################################

    def _approximately_project_monotonicity_pairs(self, kernel_data) -> torch.Tensor:
        """Projects kernel such that the monotonicity pairs are satisfied.

        The kernel will be projected such that `kernel_data[i] <= kernel_data[j]`. This
        results in calibrated outputs that adhere to the desired constraints.

        Args:
            kernel_data: The tensor of shape `(self.num_categories, 1)` to be projected
                into the constraints specified by `self.monotonicity pairs`.

        Returns:
            Projected kernel data. To prevent the kernel from drifting in one direction,
            the data returned is the average of the min/max and max/min projections.
        """
        projected_kernel_data = torch.unbind(kernel_data, 0)

        def project(data, monotonicity_graph, step, minimum):
            projected_data = list(data)
            sorted_indices = self._monotonically_sorted_indices
            if minimum:
                sorted_indices = sorted_indices[::-1]
            for i in sorted_indices:
                if i in monotonicity_graph:
                    projection = projected_data[i]
                    for j in monotonicity_graph[i]:
                        if minimum:
                            projection = torch.minimum(projection, projected_data[j])
                        else:
                            projection = torch.maximum(projection, projected_data[j])
                        if step == 1.0:
                            projected_data[i] = projection
                        else:
                            projected_data[i] = (
                                step * projection + (1 - step) * projected_data[i]
                            )
            return projected_data

        projected_kernel_min_max = project(
            projected_kernel_data, self._monotonicity_graph, 0.5, minimum=True
        )
        projected_kernel_min_max = project(
            projected_kernel_min_max,
            self._reverse_monotonicity_graph,
            1.0,
            minimum=False,
        )
        projected_kernel_min_max = torch.stack(projected_kernel_min_max)

        projected_kernel_max_min = project(
            projected_kernel_data, self._reverse_monotonicity_graph, 0.5, minimum=False
        )
        projected_kernel_max_min = project(
            projected_kernel_max_min, self._monotonicity_graph, 1.0, minimum=True
        )
        projected_kernel_max_min = torch.stack(projected_kernel_max_min)

        return (projected_kernel_min_max + projected_kernel_max_min) / 2
