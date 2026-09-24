"""Physics-derived parameter space for the 3R2C inverse heat-demand model.

Same geometry-based derivation as the 2R2C space (single-storey square prism
from the building registry's floor area and number of floors) with three
additions:

- R_ia, the direct air<->ambient path. Its conductance 1/R_ia spans the
  ventilation/infiltration range up to ventilation + fully-glazed fast
  losses; correspondingly the envelope chain 1/R_ea keeps the full 2R2C
  range, so the GA is free to split the total loss between the massive (via
  T_e) and massless (direct) paths;
- the solar aperture k_sol split into envelope/air shares k_sol_e / k_sol_i
  (each allowed down to 0, jointly up to the full aperture);
- a constant internal-gains input Phi_int over EN ISO 13790's residential
  range of 3-9 W/m2.
"""
import numpy as np

from rctoolbox.identification.parameter_space import BaseParameterSpace


class RC_3R2CParameterSpace(BaseParameterSpace):
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

        # constants and base ranges (identical to the 2R2C space)
        rho_a_c_a = 1.2e3
        F_w = 0.9
        h_min, h_max = 2.2, 5
        c_f_min, c_f_max = 0.1e5, 5e5
        u_opaque_min, u_opaque_max = 0.1, 2
        u_windows_min, u_windows_max = 0.7, 5
        ach_tot_min, ach_tot_max = 0.1, 1
        ww_ratio_min, ww_ratio_max = 0.05, 0.5
        g_n_min, g_n_max = 0.5, 0.85
        h_ie_min, h_ie_max = 1.0, 35.0          # [W/(m2 K)] indoor coupling per floor area
        c_i_mult_min, c_i_mult_max = 1.0, 10.0  # furniture multiplier on the air capacity

        def geometry(A_f, h, ww_ratio):
            A_walls = 4 * (A_f ** 0.5) * h
            A_glazed = A_walls * ww_ratio
            A_opaque = (A_walls - A_glazed) + 2 * A_f
            room_vol = h * A_f
            return room_vol, A_opaque, A_glazed

        V_lo, A_op_lo, A_gl_lo = geometry(A_f_min, h_min, ww_ratio_min)
        V_hi, A_op_hi, A_gl_hi = geometry(A_f_max, h_max, ww_ratio_max)

        # envelope chain (opaque + glazed + ventilation) — as in the 2R2C space
        H_tot_lo = (u_opaque_min * A_op_lo
                    + u_windows_min * A_gl_lo
                    + rho_a_c_a * V_lo * ach_tot_min / 3600)
        H_tot_hi = (u_opaque_max * A_op_hi
                    + u_windows_max * A_gl_hi
                    + rho_a_c_a * V_hi * ach_tot_max / 3600)

        # direct air<->ambient path: ventilation/infiltration up to
        # ventilation + fully glazed fast losses
        H_ia_lo = rho_a_c_a * V_lo * ach_tot_min / 3600
        H_ia_hi = rho_a_c_a * V_hi * ach_tot_max / 3600 + u_windows_max * A_gl_hi

        # solar aperture, split into envelope/air shares (each down to 0)
        k_sol_hi = F_w * g_n_max * A_gl_hi

        self.free_param_bounds = {
            "R_ie":    (1.0 / (h_ie_max * A_f_max), 1.0 / (h_ie_min * A_f_min)),
            "R_ea":    (1.0 / H_tot_hi,             1.0 / H_tot_lo),
            "R_ia":    (1.0 / H_ia_hi,              1.0 / H_ia_lo),
            "C_i":     (rho_a_c_a * V_lo * c_i_mult_min, rho_a_c_a * V_hi * c_i_mult_max),
            "C_e":     (c_f_min * A_f_min,          c_f_max * A_f_max),
            "k_sol_e": (0.0,                        k_sol_hi),
            "k_sol_i": (0.0,                        k_sol_hi),
            "Phi_int": (3.0 * A_f_min,              9.0 * A_f_max),   # W, EN ISO 13790 range
        }
        self.free_param_names = list(self.free_param_bounds)
        self.params_latex = {
            "R_ie":    r"$R_{ie}$",
            "R_ea":    r"$R_{ea}$",
            "R_ia":    r"$R_{ia}$",
            "C_i":     r"$C_i$",
            "C_e":     r"$C_e$",
            "k_sol_e": r"$k_{sol,e}$",
            "k_sol_i": r"$k_{sol,i}$",
            "Phi_int": r"$\Phi_{int}$",
        }

    @classmethod
    def from_building_config(cls, building_config: dict) -> "RC_3R2CParameterSpace":
        return cls(building_config)


class RC_3R2CThermostatParameterSpace(RC_3R2CParameterSpace):
    """Adds the thermostat feed-through alpha = R_ti*f_t [K/W]: the thermostat
    overread per watt delivered. alpha = 0 recovers the 3R2C model; the upper
    bound allows the fast (5R1C-like) tracking regime to dominate the ZOH air
    slope."""

    def __init__(self, building_config):
        super().__init__(building_config)
        self.free_param_bounds["alpha"] = (0.0, 0.01)   # [K/W]
        self.free_param_names = list(self.free_param_bounds)
        self.params_latex["alpha"] = r"$\alpha$"
