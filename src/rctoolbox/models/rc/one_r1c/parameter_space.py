"""Parameter space for the 1R1C model.

Bounds are wide generic ranges (from a small cold room / freezer to a
building zone). They can be overridden per case through
``building_config["free_param_bounds"]``, e.g. for a freezer:

    "free_param_bounds": {"R_ia": [1e-3, 1.0], "C_i": [1e4, 1e8]}

Both R_ia and C_i must be given there.
"""
from rctoolbox.identification.parameter_space import BaseParameterSpace


class ParameterSpace_1R1C(BaseParameterSpace):
    """
    R_ia  [K/W] : indoor air <-> ambient (envelope + infiltration).
    C_i   [J/K] : lumped capacity of the indoor node (air + contents + envelope share).
    """

    DEFAULT_BOUNDS = {
        "R_ia":  (1e-4, 1.0),
        "C_i":   (1e1,  1e7),
    }
    PARAMS_LATEX = {
        "R_ia":  r"$R_{ia}$",
        "C_i":   r"$C_i$",
    }

    def __init__(self, building_config):
        self.building_config = building_config

        bounds = building_config.get("free_param_bounds")
        if bounds is None:
            bounds = self.DEFAULT_BOUNDS
        unknown = set(bounds) - set(self.DEFAULT_BOUNDS)
        if unknown:
            raise ValueError(
                f"Unknown 1R1C parameters in free_param_bounds: {sorted(unknown)}. "
                f"Valid: {sorted(self.DEFAULT_BOUNDS)}"
            )
        missing = {"R_ia", "C_i"} - set(bounds)
        if missing:
            raise ValueError(f"free_param_bounds must contain {sorted(missing)}")

        self.free_param_bounds = {k: tuple(v) for k, v in bounds.items()}
        self.free_param_names = list(self.free_param_bounds)
        self.params_latex = {k: self.PARAMS_LATEX[k] for k in self.free_param_names}

    @classmethod
    def from_building_config(cls, building_config: dict) -> "ParameterSpace_1R1C":
        return cls(building_config)
