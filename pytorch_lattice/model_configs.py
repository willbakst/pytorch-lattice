"""Model configurations classes for PyTorch Calibrated Models."""
from dataclasses import dataclass
from typing import Optional

from .enums import Interpolation, LatticeInit


@dataclass
class _BaseModelConfig:
    """Configuration for a calibrated model.

    Attributes:
        output_min: The minimum output value for the model. If None, then it will be
            assumed that there is no minimum output value.
        output_max: The maximum output value for the model. If None, then it will be
            assumed that there is no maximum output value.
        output_calibration_num_keypoints: The number of keypoints to use for the output
            calibrator. If `None`, no output calibration will be used.
    """

    output_min: Optional[float] = None
    output_max: Optional[float] = None
    output_calibration_num_keypoints: Optional[int] = None


@dataclass
class LinearConfig(_BaseModelConfig):
    """Configuration for a calibrated linear model.

    Attributes:
        - All `_BaseModelConfig` attributes.
        use_bias: Whether to use a bias term for the linear combination.
    """

    use_bias: bool = True


@dataclass
class LatticeConfig(_BaseModelConfig):
    """Configuration for a calibrated lattice model.

    Attributes:
        - All `_BaseModelConfig` attributes.
        kernel_init: The `LatticeInit` scheme to use to initialize the lattice kernel.
        interpolation: The `Interpolation` scheme to use in the lattice. Note that
            `HYPERCUBE` has exponential time complexity while `SIMPLEX` has
            log-linear time complexity.
    """

    kernel_init: LatticeInit = LatticeInit.LINEAR
    interpolation: Interpolation = Interpolation.HYPERCUBE
