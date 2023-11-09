"""Testing Utilities."""
from typing import Union

import torch

from pytorch_lattice.layers import Linear
from pytorch_lattice.models import CalibratedLinear


def _batch_data(examples: torch.Tensor, labels: torch.Tensor, batch_size: int):
    """A generator that yields batches of data."""
    num_examples = examples.size()[0]
    for i in range(0, num_examples, batch_size):
        yield (
            examples[i : i + batch_size],
            labels[i : i + batch_size],
        )


def train_calibrated_module(
    calibrated_module: Union[Linear, CalibratedLinear],
    examples: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
):
    """Trains a calibrated module for testing purposes."""
    for _ in range(epochs):
        for batched_inputs, batched_labels in _batch_data(examples, labels, batch_size):
            optimizer.zero_grad()
            outputs = calibrated_module(batched_inputs)
            loss = loss_fn(outputs, batched_labels)
            loss.backward()
            optimizer.step()
            calibrated_module.constrain()


class MockResponse:
    """Mock response class for testing."""

    def __init__(self, json_data, status_code=200):
        """Mock response for testing."""
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        """Return json data."""
        return self.json_data
