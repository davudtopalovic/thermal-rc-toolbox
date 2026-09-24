"""Abstract parameter-space interface for the identification pipeline.

Concrete subclasses build themselves from a ``building_config`` dict and may
carry additional physics context that their matching ``Problem`` subclass
needs for constraint evaluation.
"""
from abc import ABC, abstractmethod

import numpy as np


class BaseParameterSpace(ABC):
    free_param_names: list[str]
    free_param_bounds: dict[str, tuple[float, float]]

    @classmethod
    @abstractmethod
    def from_building_config(cls, building_config: dict) -> "BaseParameterSpace":
        """Build the parameter space from a building-config dict."""

    @property
    def xl(self) -> np.ndarray:
        return np.array(
            [self.free_param_bounds[k][0] for k in self.free_param_names],
            dtype=float,
        )

    @property
    def xu(self) -> np.ndarray:
        return np.array(
            [self.free_param_bounds[k][1] for k in self.free_param_names],
            dtype=float,
        )

    @property
    def n_free_params(self) -> int:
        return len(self.free_param_names)
