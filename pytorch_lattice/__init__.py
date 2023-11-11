"""PyTorch Lattice"""

# This version must always be one version ahead of the current release, so it
# matches the current state of development, which will always be ahead of the
# current release. Use Semantic Versioning.
__version__ = "0.0.0"

from . import configs, datasets, utils
from .classifier import Classifier
from .enums import (
    CategoricalCalibratorInit,
    InputKeypointsInit,
    InputKeypointsType,
    Interpolation,
    LatticeInit,
    Monotonicity,
    NumericalCalibratorInit,
)
