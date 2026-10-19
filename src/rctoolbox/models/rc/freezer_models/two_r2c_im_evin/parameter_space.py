"""Parameter space for the standalone 2R2C internal-mass model driven by the
measured evaporator temperature.

Bounds are wide generic ranges for a freezer. They can be overridden per case
through ``building_config["free_param_bounds"]``, e.g.

    "free_param_bounds": {"C_i": [1e3, 1e7], "R_ia": [1e-3, 1.0],
                          "R_im": [1e-4, 1.0], "C_m": [1e4, 1e8],
                          "R_ie_on": [1e-4, 1.0], "R_ie_off": [1e-3, 10.0]}

All six parameters are free; the keys given there must be exactly these.
"""
from rctoolbox.identification.parameter_space import BaseParameterSpace


class ParameterSpace_2R2C_IM_EVIN(BaseParameterSpace):
    """
    R_ia     [K/W] : air <-> neighbouring cold room (shared wall + leakage).
    C_i      [J/K] : capacity of the air node (air + light internal surfaces).
    R_im     [K/W] : air <-> internal mass (convection to the stored products).
    C_m      [J/K] : capacity of the internal mass (stored products, shelves).
    R_ie_on  [K/W] : air <-> evaporator coil with the fans on (forced convection).
    R_ie_off [K/W] : air <-> evaporator coil with the fans off (natural convection).
    """

    # same order as Freezer2R2C_IM_EVIN.DEFAULT_BOUNDS: the GA encodes the
    # parameters in this order, so a different order gives a different search
    DEFAULT_BOUNDS = {
        "C_i":      (1e3,  1e7),
        "R_ia":     (1e-3, 1.0),
        "R_im":     (1e-4, 1.0),
        "C_m":      (1e4,  1e8),
        "R_ie_on":  (1e-4, 1.0),
        "R_ie_off": (1e-3, 10.0),
    }
    PARAMS_LATEX = {
        "R_ia":     r"$R_{ia}$",
        "C_i":      r"$C_i$",
        "R_im":     r"$R_{im}$",
        "C_m":      r"$C_m$",
        "R_ie_on":  r"$R_{ie}^{on}$",
        "R_ie_off": r"$R_{ie}^{off}$",
    }

    def __init__(self, building_config):
        self.building_config = building_config

        bounds = building_config.get("free_param_bounds")
        if bounds is None:
            bounds = self.DEFAULT_BOUNDS
        unknown = set(bounds) - set(self.DEFAULT_BOUNDS)
        if unknown:
            raise ValueError(
                f"Unknown 2R2C_IM_EVIN parameters in free_param_bounds: {sorted(unknown)}. "
                f"Valid: {sorted(self.DEFAULT_BOUNDS)}"
            )
        missing = set(self.DEFAULT_BOUNDS) - set(bounds)
        if missing:
            raise ValueError(f"free_param_bounds must contain {sorted(missing)}")

        self.free_param_bounds = {k: tuple(v) for k, v in bounds.items()}
        self.free_param_names = list(self.free_param_bounds)
        self.params_latex = {k: self.PARAMS_LATEX[k] for k in self.free_param_names}

    @classmethod
    def from_building_config(cls, building_config: dict) -> "ParameterSpace_2R2C_IM_EVIN":
        return cls(building_config)
