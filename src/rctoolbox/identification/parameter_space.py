"""Abstract parameter-space interface for the identification pipeline.

Concrete subclasses build themselves from a ``building_config`` dict (empty
when no building config is given; only physics-derived spaces read it) and may
carry additional physics context that their matching ``Problem`` subclass
needs for constraint evaluation. ``model_config["free_param_bounds"]``
overrides single bounds (``override_bounds``, applied by
``rctoolbox.pipeline.plugin.build_parameter_space``).
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

    def override_bounds(self, overrides: dict | None) -> None:
        """Replace the bounds of the parameters listed in ``overrides``
        ({name: [lower, upper]}, from ``model_config["free_param_bounds"]``);
        the other parameters keep their default/derived bounds."""
        if not overrides:
            return
        unknown = set(overrides) - set(self.free_param_bounds)
        if unknown:
            raise ValueError(
                f"Unknown parameters in free_param_bounds: {sorted(unknown)}. "
                f"Parameters: {self.free_param_names}"
            )
        for k, (lo, hi) in overrides.items():
            if not lo < hi:
                raise ValueError(f"free_param_bounds[{k!r}]: lower bound must be < upper bound, got {[lo, hi]}")
            self.free_param_bounds[k] = (float(lo), float(hi))
