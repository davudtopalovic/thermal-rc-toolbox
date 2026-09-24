"""Parameter space for the 6R3C model.

Bounds copied from the original VUBC-6R3C-ENISO13790-GA-Callibration.ipynb so a
re-run of that calibration through the toolbox stays comparable. Tweak per
building as needed.
"""
from rctoolbox.identification.parameter_space import BaseParameterSpace


class ParameterSpace_6R3C_ENISO13790(BaseParameterSpace):
    """
    A_f     [m²]       : conditioned floor area.
    H_em    [W/K]      : opaque-envelope conductance (T_m ↔ T_a).
    H_w     [W/K]      : window/glazing conductance.
    H_ve    [W/K]      : ventilation + infiltration conductance.
    c_f     [J/(K m²)] : thermal capacity per floor area (→ C_m = c_f · A_f).
    Phi_int [W]        : internal gains (baked into the E matrix).
    A_w     [m²]       : effective solar aperture of the glazed envelope.
    H_ih    [W/K]      : indoor-to-heating-system conductance.
    C_i   [J/K]      : indoor-air node capacity.
    C_h     [J/K]      : heating-system node capacity.
    """

    def __init__(self, building_config):
        self.building_config = building_config
        # Bounds reproduced from the VUBC-6R3C notebook.
        self.free_param_bounds = {
            "A_f":     (185.04, 354.66),
            "H_em":    (48.38,  1795.28),
            "H_w":     (4.19,   941.6),
            "H_ve":    (13.57,  1182.2),
            "c_f":     (80_000.0, 10_000_000.0),
            "Phi_int": (0, 3000),
            # "Phi_int": (0, 10),
            "A_w":     (2.0,    50.0),
            "H_ih":    (1e2,    1e3),
            "C_i":     (1e6,    1e9),
            "C_h":     (1e5,    1e8),
        }
        self.free_param_names = list(self.free_param_bounds)
        self.params_latex = {
            "A_f":     r"$A_f$",
            "H_em":    r"$H_{em}$",
            "H_w":     r"$H_w$",
            "H_ve":    r"$H_{ve}$",
            "c_f":     r"$c_f$",
            "Phi_int": r"$\Phi_{int}$",
            "A_w":     r"$A_w$",
            "H_ih":    r"$H_{ih}$",
            "C_i":     r"$C_i$",
            "C_h":     r"$C_h$",
        }

    @classmethod
    def from_building_config(cls, building_config: dict) -> "ParameterSpace_6R3C_ENISO13790":
        return cls(building_config)
