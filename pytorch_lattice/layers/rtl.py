"""A PyTorch module implementing a calibrated modeling layer for Random Tiny Lattices.

This module implements an ensemble of tiny lattices that each operate on a subset of the
inputs. It utilizes the multi-unit functionality of the Lattice module to better
optimize speed performance by putting feature subsets that have the same constraint
structure into the same Lattice module as multiple units.
"""
import logging
from typing import Optional, Union

import numpy as np
import torch

from ..enums import Interpolation, LatticeInit, Monotonicity
from .lattice import Lattice


class RTL(torch.nn.Module):
    """A module that efficiently implements Random Tiny Lattices.

    This module creates an ensemble of lattices where each lattice in the ensemble takes
    as input a subset of the input features. For further efficiency, input subsets with
    the same constraint structure all go through the same lattice as multiple units in
    parallel. When creating the ensemble structure, features are shuffled and uniformly
    repeated if there are more available slots in the ensemble structure than there are
    features.

    Attributes:
      - All `__init__` arguments.

    Example:
    ```python
    inputs=torch.tensor(...)  # shape: (batch_size, D)
    monotonicities = List[Monotonicity...]  # len: D
    random_tiny_lattices = RTL(
      monotonicities,
      num_lattices=5
      lattice_rank=3,  # num_lattices * lattice_rank must be greater than D
    )
    output1 = random_tiny_lattices(inputs)

    # You can stack RTL modules based on the previous RTL's output monotonicities.
    rtl2 = RTL(random_tiny_lattices.output_monotonicities(), ...)
    outputs2 = rtl2(outputs)
    ```
    """

    def __init__(
        self,
        monotonicities: list[Monotonicity],
        num_lattices: int,
        lattice_rank: int,
        lattice_size: int = 2,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        kernel_init: LatticeInit = LatticeInit.LINEAR,
        clip_inputs: bool = True,
        interpolation: Interpolation = Interpolation.HYPERCUBE,
        average_outputs: bool = False,
        random_seed: int = 42,
    ) -> None:
        """Initializes an instance of 'RTL'.

        Args:
            monotonicities: List of `Monotonicity.INCREASING` or `None`
              indicating monotonicities of input features, ordered respectively.
            num_lattices: number of lattices in RTL structure.
            lattice_rank: number of inputs for each lattice in RTL structure.
            output_min: Minimum output of each lattice in RTL.
            output_max: Maximum output of each lattice in RTL.
            kernel_init: Initialization scheme to use for lattices.
            clip_inputs: Whether input should be clipped to the range of each lattice.
            interpolation: Interpolation scheme for each lattice in RTL.
            average_outputs: Whether to average the outputs of every lattice RTL.
            random_seed: seed used for shuffling.

        Raises:
            ValueError: If size of RTL, determined by `num_lattices * lattice_rank`, is
             too small to support the number of input features.
        """
        super().__init__()

        if len(monotonicities) > num_lattices * lattice_rank:
            raise ValueError(
                f"RTL with {num_lattices}x{lattice_rank}D structure cannot support "
                + f"{len(monotonicities)} input features."
            )
        self.monotonicities = monotonicities
        self.num_lattices = num_lattices
        self.lattice_rank = lattice_rank
        self.lattice_size = lattice_size
        self.output_min = output_min
        self.output_max = output_max
        self.kernel_init = kernel_init
        self.clip_inputs = clip_inputs
        self.interpolation = interpolation
        self.average_outputs = average_outputs
        self.random_seed = random_seed

        rtl_indices = np.array(
            [i % len(self.monotonicities) for i in range(num_lattices * lattice_rank)]
        )
        np.random.seed(self.random_seed)
        np.random.shuffle(rtl_indices)
        split_rtl_indices = [list(arr) for arr in np.split(rtl_indices, num_lattices)]
        swapped_rtl_indices = self._ensure_unique_sublattices(split_rtl_indices)
        monotonicity_groupings = {}
        for lattice_indices in swapped_rtl_indices:
            monotonic_count = sum(
                1
                for idx in lattice_indices
                if self.monotonicities[idx] == Monotonicity.INCREASING
            )
            if monotonic_count not in monotonicity_groupings:
                monotonicity_groupings[monotonic_count] = [lattice_indices]
            else:
                monotonicity_groupings[monotonic_count].append(lattice_indices)
        for monotonic_count, groups in monotonicity_groupings.items():
            for i, lattice_indices in enumerate(groups):
                sorted_indices = sorted(
                    lattice_indices,
                    key=lambda x: (self.monotonicities[x] is None),
                    reverse=False,
                )
                groups[i] = sorted_indices

        self._lattice_layers = {}
        for monotonic_count, groups in monotonicity_groupings.items():
            self._lattice_layers[monotonic_count] = (
                Lattice(
                    lattice_sizes=[self.lattice_size] * self.lattice_rank,
                    output_min=self.output_min,
                    output_max=self.output_max,
                    kernel_init=self.kernel_init,
                    monotonicities=[Monotonicity.INCREASING] * monotonic_count
                    + [None] * (lattice_rank - monotonic_count),
                    clip_inputs=self.clip_inputs,
                    interpolation=self.interpolation,
                    units=len(groups),
                ),
                groups,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method computed by using forward methods of each lattice in ensemble.

        Args:
            x: input tensor of feature values with shape `(batch_size, num_features)`.

        Returns:
            `torch.Tensor` containing the outputs of each lattice within RTL structure.
            If `average_outputs == True`, then all outputs are averaged into a tensor of
            shape `(batch_size, 1)`. If `average_outputs == False`, shape of tensor is
            `(batch_size, num_lattices)`.
        """
        forward_results = []
        for _, (lattice, group) in sorted(self._lattice_layers.items()):
            if len(group) > 1:
                lattice_input = torch.stack([x[:, idx] for idx in group], dim=-2)
            else:
                lattice_input = x[:, group[0]]
            forward_results.append(lattice.forward(lattice_input))
        result = torch.cat(forward_results, dim=-1)
        if not self.average_outputs:
            return result
        result = torch.mean(result, dim=-1, keepdim=True)

        return result

    @torch.no_grad()
    def output_monotonicities(self) -> list[Union[Monotonicity, None]]:
        """Gives the monotonicities of the outputs of RTL.

        Returns:
            List of `Monotonicity` corresponding to each output of the RTL layer, in the
            same order as outputs.
        """
        monotonicities = []
        for monotonic_count, (lattice, _) in sorted(self._lattice_layers.items()):
            if monotonic_count:
                monotonicity = Monotonicity.INCREASING
            else:
                monotonicity = None
            for _ in range(lattice.units):
                monotonicities.append(monotonicity)

        return monotonicities

    @torch.no_grad()
    def apply_constraints(self) -> None:
        """Enforces constraints for each lattice in RTL."""
        for lattice, _ in self._lattice_layers.values():
            lattice.apply_constraints()

    @torch.no_grad()
    def assert_constraints(self, eps: float = 1e-6) -> list[list[str]]:
        """Asserts that each Lattice in RTL satisfies all constraints.

        Args:
          eps: allowed constraints violations.

        Returns:
          List of lists, each with constraints violations for an individual Lattice.
        """
        return list(
            lattice.assert_constraints(eps=eps)
            for lattice, _ in self._lattice_layers.values()
        )

    @staticmethod
    def _ensure_unique_sublattices(
        rtl_indices: list[list[int]],
        max_swaps: int = 10000,
    ) -> list[list[int]]:
        """Attempts to ensure every lattice in RTL structure contains unique features.

        Args:
            rtl_indices: list of lists where inner lists are groupings of
                indices of input features to RTL layer.
            max_swaps: maximum number of swaps to perform before giving up.

        Returns:
            List of lists where elements between inner lists have been swapped in
            an attempt to remove any duplicates from every grouping.
        """
        swaps = 0
        num_sublattices = len(rtl_indices)

        def find_swap_candidate(current_index, element):
            """Helper function to find the next sublattice not containing element."""
            for offset in range(1, num_sublattices):
                candidate_index = (current_index + offset) % num_sublattices
                if element not in rtl_indices[candidate_index]:
                    return candidate_index
            return None

        for i, sublattice in enumerate(rtl_indices):
            unique_elements = set()
            for element in sublattice:
                if element in unique_elements:
                    swap_with = find_swap_candidate(i, element)
                    if swap_with is not None:
                        for swap_element in rtl_indices[swap_with]:
                            if swap_element not in sublattice:
                                # Perform the swap
                                idx_to_swap = rtl_indices[swap_with].index(swap_element)
                                idx_duplicate = sublattice.index(element)
                                (
                                    rtl_indices[swap_with][idx_to_swap],
                                    sublattice[idx_duplicate],
                                ) = element, swap_element
                                swaps += 1
                                break
                    else:
                        logging.info(
                            "Some lattices in RTL may use the same feature multiple "
                            "times."
                        )
                        return rtl_indices
                else:
                    unique_elements.add(element)
                if swaps >= max_swaps:
                    logging.info(
                        "Some lattices in RTL may use the same feature multiple times."
                    )
                    return rtl_indices
        return rtl_indices
