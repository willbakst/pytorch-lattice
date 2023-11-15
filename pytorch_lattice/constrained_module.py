"""A virtual base class for constrained modules."""
from abc import abstractmethod
from typing import Union

import torch


class ConstrainedModule(torch.nn.Module):
    """A base class for constrained implementations of a `torch.nn.Module`."""

    @torch.no_grad()
    @abstractmethod
    def apply_constraints(self) -> None:
        """Applies defined constraints to the module."""
        raise NotImplementedError()

    @torch.no_grad()
    @abstractmethod
    def assert_constraints(
        self, eps: float = 1e-6
    ) -> Union[list[str], dict[str, list[str]]]:
        """Asserts that the module satisfied specified constraints."""
        raise NotImplementedError()
