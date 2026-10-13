"""Parameter space for the 5R4C envelope + internal-mass model.

Bounds are wide generic ranges for a cold store / freezer. They can be
overridden per case through ``building_config["free_param_bounds"]``; all
nine parameters must be listed there.
"""
from rctoolbox.identification.parameter_space import BaseParameterSpace


class ParameterSpace_5R4C_IM(BaseParameterSpace):
    """
    R_ie1   [K/W] : air <-> envelope part 1 (inner surface resistance).
    R_e1a   [K/W] : envelope part 1 <-> surrounding room T_a (insulation + outer surface).
    R_ie2   [K/W] : air <-> envelope part 2 (inner surface resistance).
    R_e2out [K/W] : envelope part 2 <-> outdoor air T_out.
    R_im    [K/W] : air <-> internal mass (convection to the stored products).
    C_i     [J/K] : capacity of the air node.
    C_e1    [J/K] : capacity of envelope part 1.
    C_e2    [J/K] : capacity of envelope part 2.
    C_m     [J/K] : capacity of the internal mass (stored products, shelves).
    """

    DEFAULT_BOUNDS = {
        "R_ie1":   (1e-4, 1.0),
        "R_e1a":   (1e-3, 10.0),
        "R_ie2":   (1e-4, 1.0),
        "R_e2out": (1e-3, 10.0),
        "R_im":    (1e-4, 1.0),
        "C_i":     (1e3,  1e7),
        "C_e1":    (1e3,  1e8),
        "C_e2":    (1e3,  1e8),
        "C_m":     (1e4,  1e8),
    }
    PARAMS_LATEX = {
        "R_ie1":   r"$R_{i,e1}$",
        "R_e1a":   r"$R_{e1,a}$",
        "R_ie2":   r"$R_{i,e2}$",
        "R_e2out": r"$R_{e2,out}$",
        "R_im":    r"$R_{im}$",
        "C_i":     r"$C_i$",
        "C_e1":    r"$C_{e1}$",
        "C_e2":    r"$C_{e2}$",
        "C_m":     r"$C_m$",
    }

    def __init__(self, building_config):
        self.building_config = building_config

        bounds = building_config.get("free_param_bounds")
        if bounds is None:
            bounds = self.DEFAULT_BOUNDS
        valid = set(self.DEFAULT_BOUNDS)
        unknown = set(bounds) - valid
        if unknown:
            raise ValueError(
                f"Unknown 5R4C_IM parameters in free_param_bounds: {sorted(unknown)}. "
                f"Valid: {sorted(valid)}"
            )
        missing = valid - set(bounds)
        if missing:
            raise ValueError(f"free_param_bounds must contain {sorted(missing)}")

        self.free_param_bounds = {k: tuple(v) for k, v in bounds.items()}
        self.free_param_names = list(self.free_param_bounds)
        self.params_latex = {k: self.PARAMS_LATEX[k] for k in self.free_param_names}

    @classmethod
    def from_building_config(cls, building_config: dict) -> "ParameterSpace_5R4C_IM":
        return cls(building_config)
