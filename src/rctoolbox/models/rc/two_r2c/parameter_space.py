"""Parameter space for the 2R2C model.

Bounds are literature-derived wide ranges (no per-building physics derivation
for this first version). Refine later if the GA gets stuck near the edges.
"""
from rctoolbox.identification.parameter_space import BaseParameterSpace


class ParameterSpace_2R2C(BaseParameterSpace):
    """Physics-derived parameter space for the 2R2C inverse heat-demand model.

    Bounds are derived from the same single-storey square-prism geometry
    approximation used for the 5R1C space (floor-area bounds from the building
    registry via ``JPD`` or ``exact_A_f_bounds``), so applying the model to a
    new building needs only the registered floor area, number of floors, and
    ``max_HP_power``. Unlike 5R1C, A_f is not a free parameter here, so all
    bounds collapse to a static box computed once at construction:

    R_ie  [K/W] : indoor-air <-> envelope coupling. 1/R_ie = h_ie * A_f with
                  h_ie in [1, 35] W/(m2 K) — widened around the ISO 13790
                  series coupling h_is–h_ms (~9–10 W/m2 K per floor area).
    R_ea  [K/W] : envelope <-> outdoor. 1/R_ea spans the total envelope
                  conductance H_em + H_w + H_ve built from the 5R1C ranges.
    C_i   [J/K] : air volume capacity rho_a*c_a*V times a furniture
                  multiplier in [1, 10] (Berthou et al. 2014 use x8).
    C_e   [J/K] : building mass capacity c_f * A_f, c_f per EN ISO 13790.
    k_sol [m2]  : effective solar aperture F_w * g_n * A_glazed (as in 5R1C).
    """

    def __init__(self, building_config):
        self.building_config = building_config

        if "max_HP_power" not in building_config:
            raise ValueError("building_config must contain 'max_HP_power'")
        self.max_HP_power = building_config["max_HP_power"]

        JPD = building_config.get("JPD")
        exact_A_f_bounds = building_config.get("exact_A_f_bounds")
        if JPD:
            A_f_min = 0.5 * JPD["n_floors"] * JPD["A_f"]
            A_f_max = (JPD["n_floors"] + 0.3) * JPD["A_f"]
        elif exact_A_f_bounds:
            A_f_min, A_f_max = exact_A_f_bounds
        else:
            raise ValueError(
                "Either JPD or exact_A_f_bounds must be provided in the building config."
            )

        # constants and base ranges (identical to the 5R1C space)
        rho_a_c_a = 1.2e3
        F_w = 0.9
        h_min, h_max = 2.2, 5
        c_f_min, c_f_max = 0.1e5, 5e5
        u_opaque_min, u_opaque_max = 0.1, 2
        u_windows_min, u_windows_max = 0.7, 5
        ach_tot_min, ach_tot_max = 0.1, 1
        ww_ratio_min, ww_ratio_max = 0.05, 0.5
        g_n_min, g_n_max = 0.5, 0.85
        # 2R2C-specific ranges
        h_ie_min, h_ie_max = 1.0, 35.0        # [W/(m2 K)] indoor coupling per floor area
        c_i_mult_min, c_i_mult_max = 1.0, 10.0  # furniture multiplier on the air capacity

        def geometry(A_f, h, ww_ratio):
            A_walls = 4 * (A_f ** 0.5) * h
            A_glazed = A_walls * ww_ratio
            A_opaque = (A_walls - A_glazed) + 2 * A_f
            room_vol = h * A_f
            return room_vol, A_opaque, A_glazed

        V_lo, A_op_lo, A_gl_lo = geometry(A_f_min, h_min, ww_ratio_min)
        V_hi, A_op_hi, A_gl_hi = geometry(A_f_max, h_max, ww_ratio_max)

        # total envelope conductance chain (opaque + glazed + ventilation)
        H_tot_lo = (u_opaque_min * A_op_lo
                    + u_windows_min * A_gl_lo
                    + rho_a_c_a * V_lo * ach_tot_min / 3600)
        H_tot_hi = (u_opaque_max * A_op_hi
                    + u_windows_max * A_gl_hi
                    + rho_a_c_a * V_hi * ach_tot_max / 3600)

        self.free_param_bounds = {
            "R_ie":  (1.0 / (h_ie_max * A_f_max), 1.0 / (h_ie_min * A_f_min)),
            "R_ea":  (1.0 / H_tot_hi,             1.0 / H_tot_lo),
            "C_i":   (rho_a_c_a * V_lo * c_i_mult_min, rho_a_c_a * V_hi * c_i_mult_max),
            "C_e":   (c_f_min * A_f_min,          c_f_max * A_f_max),
            "k_sol": (F_w * g_n_min * A_gl_lo,    F_w * g_n_max * A_gl_hi),
        }
        self.free_param_names = list(self.free_param_bounds)
        self.params_latex = {
            "R_ie":  r"$R_{ie}$",
            "R_ea":  r"$R_{ea}$",
            "C_i":   r"$C_i$",
            "C_e":   r"$C_e$",
            "k_sol": r"$k_{sol}$",
        }

    @classmethod
    def from_building_config(cls, building_config: dict) -> "ParameterSpace_2R2C":
        return cls(building_config)
