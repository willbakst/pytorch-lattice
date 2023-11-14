"""PyTorch Calibrated Models to easily implement common calibrated model architectures.

PyTorch Calibrated Models make it easy to construct common calibrated model
architectures. To construct a PyTorch Calibrated Model, pass a calibrated modeling
config to the corresponding calibrated model.
"""
from .calibrated_lattice import CalibratedLattice
from .calibrated_linear import CalibratedLinear
