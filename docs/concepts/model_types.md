# Model Types

The PyTorch Lattice library currently supports two types of calibrated modeling:

- [`CalibratedLinear`](/pytorch-lattice/api/models/#pytorch_lattice.models.CalibratedLinear): a calibrated linear model combines calibrated features using a standard [linear](/pytorch-lattice/api/layers/#pytorch_lattice.layers.Linear) layer, optionally followed by an output calibrator.

- [`CalibratedLattice`](/pytorch-lattice/api/models/#pytorch_lattice.models.CalibratedLattice): a calibrated lattice model combines calibrated features using a [lattice](/pytorch-lattice/api/layers/#pytorch_lattice.layers.Lattice) layer, optionally followed by an output calibrator. The lattice layer can learn higher-order feature interactions, which can help increase model flexibility and thereby performance on more complex prediction tasks.
