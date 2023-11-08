"""Linear module for use in calibrated modeling.

PyTorch implementation of the calibrated linear module. This module takes in a
single-dimensional input and transforms it using a linear transformation and optionally
a bias term. This module supports monotonicity constraints.
"""
from typing import List, Optional

import torch

from ..enums import Monotonicity


class Linear(torch.nn.Module):
    """A linear module.

    This module takes an input of shape `(batch_size, input_dim)` and applied a linear
    transformation. The output will have the same shape as the input.

    Attributes:
      - All `__init__` arguments.
      kernel: `torch.nn.Parameter` that stores the linear combination weighting.
      bias: `torch.nn.Parameter` that stores the bias term. Only available is `use_bias`
        is true.

    Example:
    ```python
    input_dim = 3
    inputs = torch.tensor(...)  # shape: (batch_size, input_dim)
    linear = Linear(
      input_dim,
      monotonicities=[
        Monotonicity.NONE,
        Monotonicity.INCREASING,
        Monotonicity.DECREASING
      ],
      use_bias=False,
      weighted_average=True,
    )
    outputs = linear(inputs)
    ```
    """

    def __init__(
        self,
        input_dim,
        monotonicities: Optional[List[Monotonicity]] = None,
        use_bias: bool = True,
        weighted_average: bool = False,
    ) -> None:
        """Initializes an instance of `Linear`.

        Args:
            input_dim: The number of inputs that will be combined.
            monotonicities: If provided, specifies the monotonicity of each input
              dimension.
            use_bias: Whether to use a bias term for the linear combination.
            weighted_average: Whether to make the output a weighted average i.e. all
              coefficients are positive and add up to a total of 1.0. No bias term will
              be used, and `use_bias` will be set to false regardless of the original
              value. `monotonicities` will also be set to increasing for all input
              dimensions to ensure that all coefficients are positive.

        Raises:
            ValueError: If monotonicities does not have length input_dim (if provided).
        """
        super().__init__()

        self.input_dim = input_dim
        if monotonicities and len(monotonicities) != input_dim:
            raise ValueError("Monotonicities, if provided, must have length input_dim.")
        self.monotonicities = (
            monotonicities
            if not weighted_average
            else [Monotonicity.INCREASING] * input_dim
        )
        self.use_bias = use_bias if not weighted_average else False
        self.weighted_average = weighted_average

        self.kernel = torch.nn.Parameter(torch.Tensor(input_dim, 1).double())
        torch.nn.init.constant_(self.kernel, 1.0 / input_dim)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1).double())
            torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        """Transforms inputs using a linear combination.

        Args:
            x: The input tensor of shape `(batch_size, input_dim)`.

        Returns:
            torch.Tensor of shape `(batch_size, 1)` containing transformed input values.
        """
        result = torch.mm(x, self.kernel)
        if self.use_bias:
            result += self.bias
        return result

    @torch.no_grad()
    def assert_constraints(self, eps=1e-6) -> List[str]:
        """Asserts that layer satisfies specified constraints.

        This checks that decreasing monotonicity corresponds to negative weights,
        increasing monotonicity corresponds to positive weights, and weights sum to 1
        for weighted_average=True.

        Args:
            eps: the margin of error allowed

        Returns:
            A list of messages describing violated constraints. If no constraints
            violated, the list will be empty.
        """
        messages = []

        if self.weighted_average:
            total_weight = torch.sum(self.kernel.data)
            if torch.abs(total_weight - 1.0) > eps:
                messages.append("Weights do not sum to 1.")

        if self.monotonicities:
            monotonicities_constant = torch.tensor(
                [
                    1
                    if m == Monotonicity.INCREASING
                    else -1
                    if m == Monotonicity.DECREASING
                    else 0
                    for m in self.monotonicities
                ],
                device=self.kernel.device,
                dtype=self.kernel.dtype,
            ).view(-1, 1)

            violated_monotonicities = (self.kernel * monotonicities_constant) < -eps
            violation_indices = torch.where(violated_monotonicities)
            if violation_indices[0].numel() > 0:
                messages.append(
                    f"Monotonicity violated at: {violation_indices[0].tolist()}"
                )

        return messages

    @torch.no_grad()
    def constrain(self) -> None:
        """Projects kernel into desired constraints."""
        projected_kernel_data = self.kernel.data

        if self.monotonicities:
            if Monotonicity.INCREASING in self.monotonicities:
                increasing_mask = torch.tensor(
                    [
                        [0.0] if m == Monotonicity.INCREASING else [1.0]
                        for m in self.monotonicities
                    ]
                )
                projected_kernel_data = torch.maximum(
                    projected_kernel_data, projected_kernel_data * increasing_mask
                )
            if Monotonicity.DECREASING in self.monotonicities:
                decreasing_mask = torch.tensor(
                    [
                        [0.0] if m == Monotonicity.DECREASING else [1.0]
                        for m in self.monotonicities
                    ]
                )
                projected_kernel_data = torch.minimum(
                    projected_kernel_data, projected_kernel_data * decreasing_mask
                )

        if self.weighted_average:
            norm = torch.norm(projected_kernel_data, 1)
            norm = torch.where(norm < 1e-8, 1.0, norm)
            projected_kernel_data /= norm

        self.kernel.data = projected_kernel_data
