"""Lattice module for use in calibrated modeling.

PyTorch implementation of a lattice layer.
This layer takes one or more d-dimensional inputs and outputs the interpolated value
according the specified interpolation method.
"""
from typing import Iterator, Tuple, Union, List, Callable, Optional

import numpy as np
import torch

from ..enums import Interpolation, LatticeInit, Monotonicity


# pylint: disable=too-many-instance-attributes
class Lattice(torch.nn.Module):
    """A Lattice Module.

    Layer performs interpolation using one of 'units' d-dimensional lattices with
    arbitrary number of keypoints per dimension. Each lattice vertex has a trainable
    weight, and input is considered to be a d-dimensional point within the lattice.

    Attributes:
      - All `__init__` arguments.
      kernel: `torch.nn.Parameter` of shape `(prod(lattice_sizes), units)` which stores
        weights at each vertex of lattice.

    Example:
    `python
    lattice_sizes = [2, 2, 4, 3]
    inputs=torch.tensor(...) # shape: (batch_size, len(lattice_sizes))
    lattice=Lattice(
      lattice_sizes,
      clip_inputs=True,
      interpolation=Interpolation.HYPERCUBE,
      units=1,
    )
    outputs = Lattice(inputs)
    `
    """

    def __init__(
        self,
        lattice_sizes: Union[List[int], Tuple[int]],
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        kernel_init: LatticeInit = LatticeInit.LINEAR,
        monotonicities: Optional[List[Monotonicity]] = None,
        clip_inputs: bool = True,
        interpolation: Interpolation = Interpolation.HYPERCUBE,
        units: int = 1,
    ) -> None:
        """Initializes an instance of 'Lattice'.

        Args:
            lattice_sizes: `List` or `Tuple` of size of lattice along each dimension.
            output_min: Minimum output value for weights at vertices of lattice.
            output_max: Maximum output value for weights at vertices of lattice.
            kernel_init: Initialization scheme to use for the kernel.
            monotonicities: `None` or `List` of `Monotonicity.NONE` or
              `Monotonicity.INCREASING` of length `len(lattice_sizes)` specifying
              monotonicity of each feature of lattice. A monotonically decreasing
              feature should use `Monotonicity.INCREASING` in the lattice layer but
              `Monotonicity.DECREASING` in the calibrator.
            clip_inputs: Whether input points should be clipped to the range of lattice.
            interpolation: Interpolation scheme for a given input.
            units: Dimensionality of weights stored at each vertex of lattice.

        Raises:
            ValueError: if `kernel_init` is invalid.
            NotImplementedError: Random monotonic initialization not yet implemented.
        """
        super().__init__()

        self.lattice_sizes = list(lattice_sizes)
        self.output_min = output_min
        self.output_max = output_max
        self.kernel_init = kernel_init
        self.clip_inputs = clip_inputs
        self.interpolation = interpolation
        self.units = units

        if monotonicities is None:
            self.monotonicities = [Monotonicity.NONE] * len(lattice_sizes)
        else:
            self.monotonicities = monotonicities

        if output_min is not None and output_max is not None:
            output_init_min, output_init_max = self.output_min, self.output_max
        elif output_min is not None:
            output_init_min, output_init_max = self.output_min, self.output_min + 4.0
        elif output_max is not None:
            output_init_min, output_init_max = self.output_max - 4.0, self.output_max
        else:
            output_init_min, output_init_max = -2.0, 2.0
        self._output_init_min, self._output_init_max = output_init_min, output_init_max

        @torch.no_grad()
        def initialize_kernel() -> torch.Tensor:
            if self.kernel_init == LatticeInit.LINEAR:
                return self._linear_initializer()
            if self.kernel_init == LatticeInit.RANDOM_MONOTONIC:
                raise NotImplementedError(
                    "Random monotonic initialization not yet implemented."
                )
            raise ValueError(f"Unknown kernel init: {self.kernel_init}")

        self.kernel = torch.nn.Parameter(initialize_kernel())

    # pylint: disable-next=invalid-name
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Calculates interpolation from input, using method of self.interpolation.

        Args:
            x: input tensor. If `units == 1`, tensor of shape:
              `(batch_size, ..., len(lattice_size))` or list of `len(lattice_sizes)`
              tensors of same shape: `(batch_size, ..., 1)`. If `units > 1`, tensor of
              shape `(batch_size, ..., units, len(lattice_sizes))` or list of
              `len(lattice_sizes)` tensors of same shape `(batch_size, ..., units, 1)`.

        Returns:
            torch.Tensor of shape `(batch_size, ..., units)` containing interpolated
            values.

        Raises:
            ValueError: If the type of interpolation is unknown.
        """
        if self.interpolation == Interpolation.HYPERCUBE:
            return self._compute_hypercube_interpolation(x.double())
        if self.interpolation == Interpolation.SIMPLEX:
            return self._compute_simplex_interpolation(x.double())
        raise ValueError(f"Unknown interpolation type: {self.interpolation}")

    @torch.no_grad()
    def assert_constraints(self, eps=1e-6) -> List[str]:
        """Asserts that layer satisfies specified constraints.

        This checks that weights follow monotonicity and bounds constraints.

        Args:
            eps: the margin of error allowed

        Returns:
            A list of dicts describing violated constraints including indices of
            monotonicity violations. If no constraints violated, the list will be empty.
        """
        messages = []
        lattice_sizes = self.lattice_sizes
        monotonicities = self.monotonicities
        weights = self.kernel.data.clone()

        if weights.shape[1] > 1:
            lattice_sizes = lattice_sizes + [int(weights.shape[1])]
            if monotonicities:
                monotonicities = monotonicities + [Monotonicity.NONE]

        # Reshape weights to match lattice sizes
        weights = weights.reshape(*lattice_sizes)

        for i in range(len(monotonicities or [])):
            if monotonicities[i] != Monotonicity.INCREASING:
                continue
            weights_layers = torch.unbind(weights, dim=i)

            for j in range(1, len(weights_layers)):
                diff = torch.min(weights_layers[j] - weights_layers[j - 1])
                if diff.item() < -eps:
                    messages.append(f"Monotonicity violated at feature index {i}.")

        if self.output_max is not None and torch.max(weights) > self.output_max + eps:
            messages.append("Max weight greater than output_max.")
        if self.output_min is not None and torch.min(weights) < self.output_min - eps:
            messages.append("Min weight less than output_min.")

        return messages

    @torch.no_grad()
    def constrain(self) -> None:
        """Aggregate function for enforcing constraints of lattice."""
        weights = self.kernel.clone()

        if self._count_non_zeros(self.monotonicities):
            lattice_sizes = self.lattice_sizes
            monotonicities = self.monotonicities
            if self.units > 1:
                lattice_sizes = lattice_sizes + [int(self.units)]
                if self.monotonicities:
                    monotonicities = monotonicities + [Monotonicity.NONE]

            weights = weights.reshape(*lattice_sizes)
            weights = self._approximately_project_monotonicity(
                weights, lattice_sizes, monotonicities
            )

        if self.output_min is not None:
            weights = torch.clamp_min(weights, self.output_min)
        if self.output_max is not None:
            weights = torch.clamp_max(weights, self.output_max)

        self.kernel.data = weights.view(-1, self.units)

    ################################################################################
    ############################## PRIVATE METHODS #################################
    ################################################################################

    def _linear_initializer(
        self,
        unimodalities=None,
    ) -> torch.Tensor:
        """Creates initial weights tensor for linear initialization.

        Args:
            monotonicities: monotonicity constraints of lattice, enforced in
              initialization.
            unimodalities: unimodality constraints of lattice, enforced in
              initialization.

        Returns:
            `torch.Tensor` of shape `(prod(lattice_sizes), units)`
        """
        monotonicities = self.monotonicities[:]

        # TODO: convert counting logic of unimodalities to suit enums.
        if monotonicities is None:
            monotonicities = [Monotonicity.NONE] * len(self.lattice_sizes)
        if unimodalities is None:
            unimodalities = [0] * len(self.lattice_sizes)

        num_constraint_dims = self._count_non_zeros(monotonicities, unimodalities)

        if num_constraint_dims == 0:
            monotonicities = [1] * len(self.lattice_sizes)
            num_constraint_dims = len(self.lattice_sizes)

        dim_range = (
            float(self._output_init_max - self._output_init_min) / num_constraint_dims
        )
        one_d_weights = []

        for monotonicity, unimodality, dim_size in zip(
            monotonicities, unimodalities, self.lattice_sizes
        ):
            if monotonicity != 0:
                one_d = np.linspace(start=0.0, stop=dim_range, num=dim_size)
            elif unimodality != 0:
                decreasing = np.linspace(
                    start=dim_range, stop=0.0, num=(dim_size + 1) // 2
                )
                increasing = np.linspace(
                    start=0.0, stop=dim_range, num=(dim_size + 1) // 2
                )
                if unimodality == 1:
                    one_d = np.concatenate((decreasing, increasing[dim_size % 2 :]))
                else:
                    one_d = np.concatenate((increasing, decreasing[dim_size % 2 :]))
            else:
                one_d = np.array([0.0] * dim_size)

            one_d_weights.append(torch.tensor(one_d, dtype=torch.double).unsqueeze(0))

        weights = self._batch_outer_operation(one_d_weights, operation=torch.add)
        weights = (weights + self._output_init_min).view(-1, 1)
        if self.units > 1:
            weights = weights.repeat(1, self.units)

        return weights

    @staticmethod
    def _count_non_zeros(*iterables) -> int:
        """Returns total number of non 0/None enum elements in given iterables.

        Args:
            *iterables: Any number of the value `None` or iterables of numeric values
                        or `Monotonicity` enum values.
        """
        result = 0
        for iterable in iterables:
            if iterable is not None:
                for element in iterable:
                    if (
                        isinstance(element, Monotonicity)
                        and element != Monotonicity.NONE
                    ):
                        result += 1
                    elif not isinstance(element, Monotonicity) and element != 0:
                        result += 1
        return result

    # pylint: disable=too-many-locals
    def _compute_simplex_interpolation(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluates a lattice using simplex interpolation.

        Each `d`-dimensional unit hypercube of the lattice can be partitioned into `d!`
        disjoint simplices with `d+1` vertices. `S` is the unique simplex which contains
        input point `P`, and `S` has vertices `ABCD...`. For any vertex such as `A`, a
        new simplex `S'` can be created using the vertices `PBCD...`. The weight of `A`
        within the interpolation is then `vol(S')/vol(S)`. This process is repeated
        for every vertex in `S`, and the resulting values are summed.

        This interpolation can be computed in `O(D log(D))` time because it is only
        necessary to compute the volume of the simplex containing input point `P`. For
        context, the unit hypercube can be partitioned into `d!` simplices by starting
        at `(0,0,...,0)` and incrementing `0` to `1` dimension-by-dimensionuntil one
        reaches `(1,1,...,1)`. There are `d!` possible paths from `(0,0,...,0)` to
        `(1,1,...,1)`, which account for the number of unique, disjoint simplices
        created by the method. There are `d` steps for each possible path where each
        step comprises the vertices of one simplex. Thus, one can find the containing
        simplex for input `P` by argsorting the coordinates of `P` in descending order
        and pathing along said order. To compute the intepolation weights simply take
        the deltas from `[1, desc_sort(P_coords), 0]`.

        Args:
            inputs: input tensor. If `units == 1`, tensor of shape:
              `(batch_size, ..., len(lattice_size))` or list of `len(lattice_sizes)`
              tensors of same shape: `(batch_size, ..., 1)`. If `units > 1`, tensor of
              shape `(batch_size, ..., units, len(lattice_sizes))` or list of
              `len(lattice_sizes)` tensors of same shape `(batch_size, ..., units, 1)`

        Returns:
            `torch.Tensor` of shape `(batch_size, ..., units)` containing interpolated
            values.
        """
        if isinstance(inputs, list):
            inputs = torch.cat(inputs, dim=-1)

        if self.clip_inputs:
            inputs = self._clip_onto_lattice_range(inputs)

        lattice_rank = len(self.lattice_sizes)
        input_dim = len(inputs.shape)
        all_size_2 = all(size == 2 for size in self.lattice_sizes)

        # Strides are the index shift (with respect to flattened kernel data) of each
        # dimension, which can be used in a dot product with multi-dimensional
        # coordinates to give an index for the flattened lattice weights.
        # Ex): for lattice_sizes = [4, 3, 2], we get strides = [6, 2, 1]: when looking
        # at lattice coords (i, j, k) and kernel data flattened into 1-D, incrementing i
        # corresponds to a shift of 6 in flattened kernel data, j corresponds to a shift
        # of 2, and k corresponds to a shift of 1. Consequently, we can do
        # (coords * strides) for any coordinates to obtain the flattened index.
        strides = torch.tensor(
            np.cumprod([1] + self.lattice_sizes[::-1][:-1])[::-1].copy()
        )
        lower_corner_offset = None
        if not all_size_2:
            lower_corner_coordinates = inputs.int()
            lower_corner_coordinates = torch.min(
                lower_corner_coordinates, torch.tensor(self.lattice_sizes) - 2
            )
            lower_corner_offset = (lower_corner_coordinates * strides).sum(
                dim=-1, keepdim=True
            )
            inputs = inputs - lower_corner_coordinates.float()

        sorted_indices = torch.argsort(inputs, descending=True)
        sorted_inputs = torch.sort(inputs, descending=True).values

        # Pad the 1 and 0 onto the ends of sorted coordinates and compute deltas.
        no_padding_dims = [(0, 0)] * (input_dim - 1)
        flat_no_padding = [item for sublist in no_padding_dims for item in sublist]
        sorted_inputs_padded_left = torch.nn.functional.pad(
            sorted_inputs, [1, 0] + flat_no_padding, value=1.0
        )
        sorted_inputs_padded_right = torch.nn.functional.pad(
            sorted_inputs, [0, 1] + flat_no_padding, value=0.0
        )
        weights = sorted_inputs_padded_left - sorted_inputs_padded_right

        # Use strides to find indices of simplex vertices in flattened form.
        sorted_strides = torch.gather(strides, 0, sorted_indices.view(-1)).view(
            sorted_indices.shape
        )
        if all_size_2:
            corner_offset_and_sorted_strides = torch.nn.functional.pad(
                sorted_strides, [1, 0] + flat_no_padding
            )
        else:
            corner_offset_and_sorted_strides = torch.cat(
                [lower_corner_offset, sorted_strides], dim=-1
            )
        indices = torch.cumsum(corner_offset_and_sorted_strides, dim=-1)

        # Get kernel data from corresponding simplex vertices.
        if self.units == 1:
            gathered_params = torch.index_select(
                self.kernel.view(-1), 0, indices.view(-1)
            ).view(indices.shape)
        else:
            unit_offset = torch.tensor(
                [[i] * (lattice_rank + 1) for i in range(self.units)]
            )
            flat_indices = indices * self.units + unit_offset
            gathered_params = torch.index_select(
                self.kernel.view(-1), 0, flat_indices.view(-1)
            ).view(indices.shape)

        return (gathered_params * weights).sum(dim=-1, keepdim=self.units == 1)

    # pylint: enable=too-many-locals

    def _compute_hypercube_interpolation(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """Performs hypercube interpolation using the surrounding unit hypercube.

        Args:
            inputs: input tensor. If `units == 1`, tensor of shape:
              `(batch_size, ..., len(lattice_size))` or list of `len(lattice_sizes)`
              tensors of same shape: `(batch_size, ..., 1)`. If `units > 1`, tensor of
              shape `(batch_size, ..., units, len(lattice_sizes))` or list of
              `len(lattice_sizes)` tensors of same shape `(batch_size, ..., units, 1)`

        Returns:
            `torch.Tensor` of shape `(batch_size, ..., units)` containing interpolated
            value(s).
        """
        interpolation_weights = self._compute_hypercube_interpolation_weights(
            inputs=inputs, clip_inputs=self.clip_inputs
        )
        if self.units == 1:
            return torch.matmul(interpolation_weights, self.kernel)

        return torch.sum(interpolation_weights * self.kernel.t(), dim=-1)

    def _compute_hypercube_interpolation_weights(
        self, inputs: Union[torch.Tensor, List[torch.Tensor]], clip_inputs: bool = True
    ) -> torch.Tensor:
        """Computes weights for hypercube lattice interpolation.

        For each n-dim unit in "inputs," the weights matrix will generate the weights
        corresponding to the unit's location within its surrounding hypercube. These
        weights can then be multiplied by the lattice layer's kernel to compute the
        actual hypercube interpolation. Specifically, the outer product of the set
        `(1-x_i, x_i)` for all x_i in input unit x calculates the weights for each
        vertex in the surrounding hypercube, and every other vertex in the lattice is
        set to zero since it is not used. In addition, for consecutive dimensions of
        equal size in the lattice, broadcasting is used to speed up calculations.

        Args:
            inputs: torch.Tensor of shape `(batch_size, ..., len(lattice_sizes)` or list
              of `len(lattice_sizes)` tensors of same shape `(batch_size, ..., 1)`
            clip_inputs: Boolean to determine whether input values outside lattice
              bounds should be clipped to the min or max supported values.

        Returns:
            `torch.Tensor` of shape `(batch_size, ..., prod(lattice_sizes))` containing
            the weights which can be matrix multiplied with the kernel to perform
            hypercube interpolation.
        """
        if isinstance(inputs, list):
            input_dtype = inputs[0].dtype
        else:
            input_dtype = inputs.dtype

        # Special case: 2^d lattice with input passed in as a single tensor
        if all(size == 2 for size in self.lattice_sizes) and not isinstance(
            inputs, list
        ):
            # pylint: disable=invalid-name
            w = torch.stack([(1.0 - inputs), inputs], dim=-1)
            if clip_inputs:
                w = torch.clamp(w, min=0, max=1)
            # pylint: enable=invalid-name
            one_d_interpolation_weights = list(torch.unbind(w, dim=-2))
            return self._batch_outer_operation(one_d_interpolation_weights)

        if clip_inputs:
            inputs = self._clip_onto_lattice_range(inputs)

        # Set up buckets of consecutive equal dimensions for broadcasting later
        dim_keypoints = {}
        for dim_size in set(self.lattice_sizes):
            dim_keypoints[dim_size] = torch.tensor(
                list(range(dim_size)), dtype=input_dtype
            )
        bucketized_inputs = self._bucketize_consecutive_equal_dims(inputs)
        one_d_interpolation_weights = []

        for tensor, bucket_size, dim_size in bucketized_inputs:
            if bucket_size > 1:
                tensor = torch.unsqueeze(tensor, dim=-1)
            distance = torch.abs(tensor - dim_keypoints[dim_size])
            weights = 1.0 - torch.minimum(
                distance, torch.tensor(1.0, dtype=distance.dtype)
            )
            if bucket_size == 1:
                one_d_interpolation_weights.append(weights)
            else:
                one_d_interpolation_weights.extend(torch.unbind(weights, dim=-2))

        return self._batch_outer_operation(one_d_interpolation_weights)

    @staticmethod
    def _batch_outer_operation(
        list_of_tensors: List[torch.Tensor],
        operation: Union[str, Callable] = "auto",
    ) -> torch.Tensor:
        """Computes the flattened outer product of a list of tensors.

        Args:
            list_of_tensors: List of tensors of same shape `(batch_size, ..., k[i])`
              where everything except `k_i` matches.

        Returns:
            `torch.Tensor` of shape `(batch_size, ..., k_i * k_j * ...)` containing a
            flattened version of the outer product.
        """
        if len(list_of_tensors) == 1:
            return list_of_tensors[0]

        result = torch.unsqueeze(list_of_tensors[0], dim=-1)

        for i, tensor in enumerate(list_of_tensors[1:]):
            # pylint: disable=invalid-name
            if operation == "auto":
                op = torch.mul if i < 6 else torch.matmul
            else:
                op = operation
            # pylint: enable=invalid-name

            result = op(result, torch.unsqueeze(tensor, dim=-2))
            shape = [-1] + [int(size) for size in result.shape[1:]]
            new_shape = shape[:-2] + [shape[-2] * shape[-1]]
            if i < len(list_of_tensors) - 2:
                new_shape.append(1)
            result = torch.reshape(result, new_shape)

        return result

    def _clip_onto_lattice_range(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """Clips inputs onto valid input range for given lattice_sizes.

        Args:
            inputs: `inputs` argument of `_compute_interpolation_weights()`.

        Returns:
            `torch.Tensor` of shape `inputs` with values within range
            `[0, dim_size - 1]`.
        """
        if not isinstance(inputs, list):
            upper_bounds = torch.tensor(
                [dim_size - 1.0 for dim_size in self.lattice_sizes]
            ).double()
            clipped_inputs = torch.clamp(
                inputs, min=torch.zeros_like(upper_bounds), max=upper_bounds
            )
        else:
            dim_upper_bounds = {}
            for dim_size in set(self.lattice_sizes):
                dim_upper_bounds[dim_size] = torch.tensor(
                    dim_size - 1.0, dtype=inputs[0].dtype
                )
            dim_lower_bound = torch.zeros(1, dtype=inputs[0].dtype)

            clipped_inputs = []
            for one_d_input, dim_size in zip(inputs, self.lattice_sizes):
                clipped_inputs.append(
                    torch.clamp(
                        one_d_input, min=dim_lower_bound, max=dim_upper_bounds[dim_size]
                    )
                )

        return clipped_inputs

    def _bucketize_consecutive_equal_dims(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Iterator[Tuple[torch.Tensor, int, int]]:
        """Creates buckets of equal sized dimensions for broadcasting ops.

        Args:
            inputs: `inputs` argument of `_compute_interpolation_weights()`.

        Returns:
            An `Iterable` containing `(torch.Tensor, int, int)` where the tensor
            contains individual values from "inputs" corresponding to its bucket, the
            first `int` is bucket size, and the second `int` is size of the dimension of
            the bucket.
        """
        if not isinstance(inputs, list):
            bucket_sizes = []
            bucket_dim_sizes = []
            current_size = 1
            for i in range(1, len(self.lattice_sizes)):
                if self.lattice_sizes[i] != self.lattice_sizes[i - 1]:
                    bucket_sizes.append(current_size)
                    bucket_dim_sizes.append(self.lattice_sizes[i - 1])
                    current_size = 1
                else:
                    current_size += 1
            bucket_sizes.append(current_size)
            bucket_dim_sizes.append(self.lattice_sizes[-1])
            inputs = torch.split(inputs, split_size_or_sections=bucket_sizes, dim=-1)
        else:
            bucket_sizes = [1] * len(self.lattice_sizes)
            bucket_dim_sizes = self.lattice_sizes

        return zip(inputs, bucket_sizes, bucket_dim_sizes)

    def _approximately_project_monotonicity(
        self,
        weights: torch.Tensor,
        lattice_sizes: List[int],
        monotonicities: List[Monotonicity],
    ) -> torch.Tensor:
        """Projects weights of lattice to meet monotonicity constraints.

        Note that this projection is an approximation which guarantees monotonicity
        constraints but is not an exact projection with respect to the L2 norm.

        Algorithm:
        1. `max_projection`: For each vertex V in the lattice, the weight is adjusted to
        be the maximum of all weights of vertices X such that X has all coordinates
        less than or equal to V in monotonic dimensions.

        2. `half_projection`: We adjust the weights to be the average of the original
        weights and the `max_projection` weights.

        3. `min_projection`: For each vertex V in the lattice, the weight is adjusted
        based on the `half_projection` to be the minimum of all weights of vertices X
        such that V has all coordinates less than or equal to X in monotonic dimensions.

        This algorithm ensures that weights conform to the monotonicity constraints
        while getting closer to a true projection by adjusting both up/downwards.

        Args:
            weights: `torch.Tensor` of kernel data reshaped into `(lattice_sizes)` if
              `units == 1` or `(lattice_sizes, units)` if `units > 1`.
            lattice_sizes: 'List' of size of each dimension of lattice, but for
              `units > 1`, `units` is appended to the end for computation purposes.
            monotonicities: `List` of `Monotonicity.NONE` or `Monotonicity.INCREASING`
              of length `len(lattice_sizes)` for `units == 1` or `len(lattice_sizes)+1`
              if `units > 1` specifying monotonicity of each feature of lattice.

        Returns:
            `torch.Tensor` of shape `self.kernel` with updated weights which meet
            monotonicity constraints.
        """
        max_projection = weights
        for dim in range(len(lattice_sizes)):
            if monotonicities[dim] == Monotonicity.NONE:
                continue
            layers = list(torch.unbind(max_projection, dim))
            for i in range(1, len(layers)):
                layers[i] = torch.max(layers[i], layers[i - 1])
            max_projection = torch.stack(layers, dim)

        half_projection = (weights + max_projection) / 2.0

        min_projection = half_projection
        for dim in range(len(lattice_sizes)):
            if monotonicities[dim] == Monotonicity.NONE:
                continue
            layers = list(torch.unbind(min_projection, dim))
            for i in range(len(layers) - 2, -1, -1):
                # Compute cumulative minimum in reverse order
                layers[i] = torch.min(layers[i], layers[i + 1])
            min_projection = torch.stack(layers, dim)

        return min_projection
