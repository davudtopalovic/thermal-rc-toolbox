"""Parameter space for the 2R2C internal-mass model.

Bounds are wide generic ranges for a cold store / freezer. They can be
overridden per case through ``building_config["free_param_bounds"]``, e.g.

    "free_param_bounds": {"R_ia": [1e-3, 1.0], "R_im": [1e-4, 1.0],
                          "C_i": [1e3, 1e7], "C_m": [1e4, 1e8]}

The keys given there define the free parameters. ``alpha_m`` is free only if
it is listed there (e.g. ``"alpha_m": [0, 1]``); otherwise the model uses
alpha_m = 0 (all cooling power on the air node).
"""
from rctoolbox.identification.parameter_space import BaseParameterSpace


class ParameterSpace_2R2C_IM(BaseParameterSpace):
    """
    R_ia    [K/W] : air <-> ambient (envelope + infiltration).
    R_im    [K/W] : air <-> internal mass (convection to the stored products).
    C_i     [J/K] : capacity of the air node (air + light internal surfaces).
    C_m     [J/K] : capacity of the internal mass (stored products, shelves).
    alpha_m [-]   : share of the cooling power acting directly on the internal mass.
    """

    DEFAULT_BOUNDS = {
        "R_ia": (1e-3, 1.0),
        "R_im": (1e-4, 1.0),
        "C_i":  (1e3,  1e7),
        "C_m":  (1e4,  1e8),
    }
    OPTIONAL_BOUNDS = {
        "alpha_m": (0.0, 1.0),
    }
    PARAMS_LATEX = {
        "R_ia":    r"$R_{ia}$",
        "R_im":    r"$R_{im}$",
        "C_i":     r"$C_i$",
        "C_m":     r"$C_m$",
        "alpha_m": r"$\alpha_m$",
    }

    def __init__(self, building_config):
        self.building_config = building_config

        bounds = building_config.get("free_param_bounds")
        if bounds is None:
            bounds = self.DEFAULT_BOUNDS
        valid = set(self.DEFAULT_BOUNDS) | set(self.OPTIONAL_BOUNDS)
        unknown = set(bounds) - valid
        if unknown:
            raise ValueError(
                f"Unknown 2R2C_IM parameters in free_param_bounds: {sorted(unknown)}. "
                f"Valid: {sorted(valid)}"
            )
        missing = set(self.DEFAULT_BOUNDS) - set(bounds)
        if missing:
            raise ValueError(f"free_param_bounds must contain {sorted(missing)}")

        self.free_param_bounds = {k: tuple(v) for k, v in bounds.items()}
        self.free_param_names = list(self.free_param_bounds)
        self.params_latex = {k: self.PARAMS_LATEX[k] for k in self.free_param_names}

    @classmethod
    def from_building_config(cls, building_config: dict) -> "ParameterSpace_2R2C_IM":
        return cls(building_config)
