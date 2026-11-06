"""Parameter space for the generic 3R3C model.

Bounds are literature-style ranges that span typical residential values; tune
per building if the GA pegs against an edge.
"""
from rctoolbox.identification.parameter_space import BaseParameterSpace


class ParameterSpace_3R3C(BaseParameterSpace):
    """
    R_ih    [K/W] : heating system ↔ indoor air.
    R_ie    [K/W] : indoor air     ↔ envelope.
    R_ea    [K/W] : envelope       ↔ outdoor.
    R_ia    [K/W] : indoor air     ↔ outdoor (direct: ventilation, infiltration).
    C_i     [J/K] : indoor air + furnishings capacity.
    C_h     [J/K] : heating system (water content / radiator mass).
    C_e     [J/K] : envelope / thermal-mass capacity.
    k_sol_i [m²]  : solar aperture coupling indoor air to Phi_sol.
    k_sol_e [m²]  : solar aperture coupling envelope to Phi_sol.
    """

    def __init__(self, building_config):
        self.building_config = building_config
        self.free_param_bounds = {
            "R_ih":    (1e-4, 1e-2),
            "R_ie":    (1e-4, 1e-2),
            "R_ea":    (1e-4, 1e-1),
            "R_ia":    (1e-4, 1.0),
            "C_i":     (1e6,  1e9),
            "C_h":     (1e5,  1e9),
            "C_e":     (1e6,  1e10),
            "k_sol_i": (0.0,  20.0),
            "k_sol_e": (0.0,  50.0),
        }
        self.free_param_names = list(self.free_param_bounds)
        self.params_latex = {
            "R_ih":    r"$R_{ih}$",
            "R_ie":    r"$R_{ie}$",
            "R_ea":    r"$R_{ea}$",
            "R_ia":    r"$R_{ia}$",
            "C_i":     r"$C_i$",
            "C_h":     r"$C_h$",
            "C_e":     r"$C_e$",
            "k_sol_i": r"$k_{sol,i}$",
            "k_sol_e": r"$k_{sol,e}$",
        }

    @classmethod
    def from_building_config(cls, building_config: dict) -> "ParameterSpace_3R3C":
        return cls(building_config)
