"""6R2C multi-layer-wall extension of the EN ISO 13790 5R1C model.

State vector:  y = [T_m1, T_m2, T_s, T_i]
Dynamic states (with capacitance):  T_m1 (outer wall mass), T_m2 (inner wall mass)
Algebraic states (no capacitance):  T_s (indoor surface), T_i (indoor air)

Network:
    T_a --H_em--o--H_ww--o--H_ms--o--H_is--o T_i     (Φ_HC + 0.5·Φ_int)
              T_m1     T_m2      T_s
              C_m1     C_m2
    T_a --H_w------------------------------> (into surface/air node)
    T_a --H_ve-----------------------------> (into indoor air node)

The algebraic reduction for T_s and T_i is identical to EN ISO 13790 5R1C with
T_m replaced by T_m2. The two-mass dynamics are solved with Crank-Nicolson as a
2x2 linear system per time step.

Total thermal capacity is c_f * A_f, split 50/50 between C_m1 and C_m2 by default
(the split fraction is a class attribute `c_split` that can be adjusted).

Initialization: as for the 5R1C, the ``dynamic``/``regime_aware`` methods
replace the whole state with the lag-window simulation.
"""
import numpy as np

from rctoolbox.models.rc_base_model import RCBaseModel
from rctoolbox.models.rc.six_r2c_eniso13790.numba_core import forecast_heat_demand_numba


class SixR2C_ENISO13790(RCBaseModel):
    STATE_NAMES = ("T_m1", "T_m2", "T_s", "T_i")
    HIDDEN_STATE_IDX = None

    def __init__(self):
        super().__init__()
        self.c_split = 0.5   # fraction of total c_f going to C_m1; C_m2 gets (1 - c_split)

    def set_params(self, params_dict):
        self.params = params_dict
        self.process_input_params()

    def process_input_params(self):
        self.A_f  = self.params["A_f"]
        self.H_em = self.params["H_em"]
        self.H_ww = self.params["H_ww"]   # conductance between the two wall layers
        self.H_w  = self.params["H_w"]
        self.H_ve = self.params["H_ve"]
        self.phi_int_f = self.params["phi_int_f"]
        self.Phi_int = self.phi_int_f * self.A_f
        self.k_sol = self.params["k_sol"]

        # Total thermal capacity split into two layers.
        self.c_f = self.params["c_f"]
        C_total = self.c_f * self.A_f
        self.C_m1 = self.c_split * C_total
        self.C_m2 = (1.0 - self.c_split) * C_total

        # Effective mass area (same rule as 5R1C, so H_ms scales identically).
        self.eff_mass_coeff = self.calc_eff_mass_coeff(self.c_f)
        self.A_m = self.eff_mass_coeff * self.A_f
        self.h_ms_coeff = 9.1
        self.H_ms = self.h_ms_coeff * self.A_m

        self.a_t_coeff = 4.5
        self.A_t = self.a_t_coeff * self.A_f
        self.h_is_coeff = 3.45
        self.H_is = self.h_is_coeff * self.A_t

        # Auxiliary helper variables from EN ISO 13790 (unchanged; they only depend
        # on H_ve, H_is, H_w, H_ms — none of which changed).
        self.H_tr1 = 1.0 / (1.0 / self.H_ve + 1.0 / self.H_is)
        self.H_tr2 = self.H_tr1 + self.H_w
        self.H_tr3 = 1.0 / (1.0 / self.H_tr2 + 1.0 / self.H_ms)

        self.His_plus_Hve   = self.H_is + self.H_ve
        self.Hms_plus_Htr2  = self.H_ms + self.H_tr2
        self.Htr1_over_Hve  = self.H_tr1 / self.H_ve
        self.Htr3_over_Htr2 = self.H_tr3 / self.H_tr2

        self.Am_over_At = self.A_m / self.A_t

        # Alphas for the T_m dynamics after algebraic T_s substitution:
        #   C_m1 dT_m1/dt = H_em (T_a - T_m1) + H_ww (T_m2 - T_m1)
        #   C_m2 dT_m2/dt = H_ww (T_m1 - T_m2) + H_tr3 (T_a - T_m2)
        #                    + Htr3_over_Htr2 * (Htr1_over_Hve * Phi_ia + Phi_st) + Phi_m
        self.alpha1 = self.H_em + self.H_ww
        self.alpha2 = self.H_ww + self.H_tr3

    def calc_eff_mass_coeff(self, c_f):
        # Same tiered rule as 5R1C.
        if c_f >= 3.7e5:
            return 3.5
        if c_f >= 2.6e5:
            return 3.0
        return 2.5

    # ----------------------------------------------------- forward simulation
    def calc_heat_flows(self, T_a, Phi_sol, Phi_HC):
        """Same allocation of solar/internal gains as EN ISO 13790 5R1C."""
        Phi_sol_eff = self.k_sol * Phi_sol

        # Heat flow to the indoor air node
        Phi_ia_plus_Phi_HC = 0.5 * self.Phi_int + Phi_HC

        # Heat flow to the indoor surface node
        Phi_st = (1.0 - self.Am_over_At - self.H_w / (self.h_ms_coeff * self.A_t)) * \
                 (0.5 * self.Phi_int + Phi_sol_eff)

        # Heat flow to the (inner) mass node
        Phi_m = self.Am_over_At * (0.5 * self.Phi_int + Phi_sol_eff)

        return Phi_ia_plus_Phi_HC, Phi_st, Phi_m

    def calc_temperature(self, y_prev, dt, T_a, Phi_sol, Phi_HC):
        """One Crank-Nicolson step advancing (T_m1, T_m2), then algebraic T_s and T_i."""
        T_m1_prev = y_prev[:, 0]
        T_m2_prev = y_prev[:, 1]

        Phi_ia_plus_Phi_HC, Phi_st, Phi_m = self.calc_heat_flows(T_a, Phi_sol, Phi_HC)

        # T_m2 heat balance after algebraic T_s substitution — analogous to 5R1C.
        Phi_m2_tot = (Phi_m
                      + self.H_tr3 * T_a
                      + self.Htr3_over_Htr2 * (self.Htr1_over_Hve * Phi_ia_plus_Phi_HC + Phi_st))
        Phi_m1_tot = self.H_em * T_a

        # Crank-Nicolson: 2x2 implicit system
        #  [ a11  a12 ] [T_m1_next]   [ b1 ]
        #  [ a21  a22 ] [T_m2_next] = [ b2 ]
        a11 = self.C_m1 / dt + 0.5 * self.alpha1
        a12 = -0.5 * self.H_ww
        a21 = -0.5 * self.H_ww
        a22 = self.C_m2 / dt + 0.5 * self.alpha2

        b1 = (self.C_m1 / dt - 0.5 * self.alpha1) * T_m1_prev + 0.5 * self.H_ww * T_m2_prev + Phi_m1_tot
        b2 = 0.5 * self.H_ww * T_m1_prev + (self.C_m2 / dt - 0.5 * self.alpha2) * T_m2_prev + Phi_m2_tot

        det = a11 * a22 - a12 * a21
        T_m1_next = (a22 * b1 - a12 * b2) / det
        T_m2_next = (-a21 * b1 + a11 * b2) / det

        # Average for T_s / T_i evaluation (matches 5R1C's use of the average mass temperature)
        T_m2_avg = 0.5 * (T_m2_next + T_m2_prev)

        # Algebraic T_s and T_i (same closed form as 5R1C with T_m -> T_m2)
        T_s = (self.H_ms * T_m2_avg + Phi_st + self.H_tr2 * T_a
               + self.Htr1_over_Hve * Phi_ia_plus_Phi_HC) / self.Hms_plus_Htr2
        T_i = (self.H_is * T_s + self.H_ve * T_a + Phi_ia_plus_Phi_HC) / self.His_plus_Hve

        return np.stack([T_m1_next, T_m2_next, T_s, T_i], axis=1)

    # ------------------------------------------------ inverse (heat-demand) API
    def _probe_power(self):
        # 10 W/m2 heating-power probe of EN ISO 13790, Annex C.4
        return 10.0 * self.A_f

    def forecast_temperature(self, y_prev, t, T_a_BT, Phi_sol_BT, Phi_HC_BT):
        t, dt = self._dt_from_t(t)
        T = len(t)
        y_pred = [y_prev]
        for k in range(1, T):
            y_next = self.calc_temperature(y_prev, dt,
                                            T_a_BT[:, k], Phi_sol_BT[:, k], Phi_HC_BT[:, k])
            y_pred.append(y_next)
            y_prev = y_next
        return np.stack(y_pred, axis=1)

    def forecast_heat_demand(self, y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT):
        t, dt = self._dt_from_t(t)
        T = len(t)
        y_pred = [y_prev]
        Phi_HC_pred = [Phi_HC_prev]
        for k in range(1, T):
            y_k, Phi_HC_k = self.calc_energy_demand(y_prev, dt,
                                                     T_a_BT[:, k], Phi_sol_BT[:, k], T_i_set_BT[:, k])
            y_pred.append(y_k)
            Phi_HC_pred.append(Phi_HC_k)
            y_prev = y_k
        return np.stack(Phi_HC_pred, axis=1), np.stack(y_pred, axis=1)

    def forecast_heat_demand_fast(self, y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT):
        return forecast_heat_demand_numba(
            y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT,
            self.A_f,
            self.C_m1, self.C_m2,
            self.H_em, self.H_ww,
            self.alpha1, self.alpha2,
            self.H_tr2, self.H_tr3,
            self.Htr3_over_Htr2, self.Htr1_over_Hve,
            self.Hms_plus_Htr2, self.His_plus_Hve,
            self.H_ms, self.H_is, self.H_ve, self.H_w,
            self.A_m, self.A_t, self.h_ms_coeff,
            self.Phi_int, self.k_sol,
            self.heat_power_range[1],
        )

    # ------------------------------------------------- steady-state inversion
    def calc_Tm1_Tm2_Ts(self, T_i, T_a, Phi_sol, Phi_HC):
        """Given a measured T_i and inputs, back-solve T_s, T_m2, T_m1 assuming
        steady state (dT_m1/dt = 0). Mirrors 5R1C's calc_Ts_Tm and extends it to
        the two-mass topology.

            T_s  from  T_i balance:      (H_is + H_ve)*T_i = H_is*T_s + H_ve*T_a + Phi_ia
            T_m2 from  T_s balance:      (H_ms + H_tr2)*T_s = H_ms*T_m2 + H_tr2*T_a + Htr1_over_Hve*Phi_ia + Phi_st
            T_m1 from  dT_m1/dt = 0:     (H_em + H_ww)*T_m1 = H_em*T_a + H_ww*T_m2
        """
        Phi_ia_plus_Phi_HC, Phi_st, _ = self.calc_heat_flows(T_a, Phi_sol, Phi_HC)
        T_s  = (1 / self.H_is) * (self.His_plus_Hve * T_i - self.H_ve * T_a - Phi_ia_plus_Phi_HC)
        T_m2 = (1 / self.H_ms) * (self.Hms_plus_Htr2 * T_s - Phi_st - self.H_tr2 * T_a
                                   - self.Htr1_over_Hve * Phi_ia_plus_Phi_HC)
        T_m1 = (self.H_em * T_a + self.H_ww * T_m2) / self.alpha1
        return T_m1, T_m2, T_s

    def _init_static_inversion(self, data):
        y_0 = np.array(data["y_0"], dtype=np.float64)   # never mutate the caller's array

        T_i, T_a, Phi_sol, Phi_HC = self._get_static_inversion_inputs(data, self.static_inversion_method)
        T_m1_0, T_m2_0, T_s_0 = self.calc_Tm1_Tm2_Ts(T_i, T_a, Phi_sol, Phi_HC)

        y_0[:, 0] = T_m1_0
        y_0[:, 1] = T_m2_0
        y_0[:, 2] = T_s_0
        # y_0[:, 3] = T_i is already the measured value at the initial time
        return y_0
