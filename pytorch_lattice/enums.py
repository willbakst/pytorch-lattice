"""Enum Classes for PyTorch Lattice."""
from enum import Enum, EnumMeta
from typing import Any


class _Metaclass(EnumMeta):
    """Base `EnumMeta` subclass for accessing enum members directly."""

    def __getattribute__(cls, __name: str) -> Any:
        value = super().__getattribute__(__name)
        if isinstance(value, Enum):
            value = value.value
        return value


class _Enum(str, Enum, metaclass=_Metaclass):
    """Base Enum Class."""


class InputKeypointsInit(_Enum):
    """Type of initialization to use for NumericalCalibrator input keypoints.

    - QUANTILES: initialize the input keypoints such that each segment will see the same
        number of examples.
    - UNIFORM: initialize the input keypoints uniformly spaced in the feature range.
    """

    QUANTILES = "quantiles"
    UNIFORM = "uniform"


class InputKeypointsType(_Enum):
    """The type of input keypoints to use.

    - FIXED: the input keypoints will be fixed during initialization.
    """

    FIXED = "fixed"
    # TODO: add learned interior functionality
    # LEARNED = "learned_interior"


class NumericalCalibratorInit(_Enum):
    """Type of kernel initialization to use for NumericalCalibrator.

    - EQUAL_HEIGHTS: initialize the kernel such that all segments have the same height.
    - EQUAL_SLOPES: initialize the kernel such that all segments have the same slope.
    """

    EQUAL_HEIGHTS = "equal_heights"
    EQUAL_SLOPES = "equal_slopes"


class CategoricalCalibratorInit(_Enum):
    """Type of kernel initialization to use for CategoricalCalibrator.

    - UNIFORM: initialize the kernel with uniformly distributed values. The sample range
        will be [`output_min`, `output_max`] if both are provided.
    - CONSTANT: initialize the kernel with a constant value for all categories. This
        value will be `(output_min + output_max) / 2` if both are provided.
    """

    UNIFORM = "uniform"
    CONSTANT = "constant"


class Monotonicity(_Enum):
    """Type of monotonicity constraint.

    - NONE: no monotonicity constraint.
    - INCREASING: increasing monotonicity i.e. increasing input increases output.
    - DECREASING: decreasing monotonicity i.e. increasing input decreases output.
    """

    INCREASING = "increasing"
    DECREASING = "decreasing"


class Interpolation(_Enum):
    """Enum for interpolation method of lattice.

    - HYPERCUBE: n-dimensional hypercube surrounding input point(s).
    - SIMPLEX: uses only one of the n! simplices in the n-dim hypercube.
    """

    HYPERCUBE = "hypercube"
    SIMPLEX = "simplex"


class LatticeInit(_Enum):
    """Type of kernel initialization to use for CategoricalCalibrator.

    - LINEAR: initialize the kernel with weights represented by a linear function,
        conforming to monotonicity and unimodality constraints.
    - RANDOM_MONOTONIC: initialize the kernel with a uniformly random sampled
        lattice layer weight tensor, conforming to monotonicity and unimodality
        constraints.
    """

    LINEAR = "linear"
    RANDOM_MONOTONIC = "random_monotonic"
