"""5R1C simple hourly model of EN ISO 13790:2008 (Annex C).

State vector: y = [T_m, T_s, T_i]
    T_m — building thermal mass (hidden)
    T_s — internal surface (hidden, algebraic)
    T_i — indoor air (measured)

Network:
    T_a --H_em--o--H_ms--o--H_is--o T_i     (Phi_HC + 0.5*Phi_int)
              T_m      T_s
              C_m
    T_a --H_w------------> T_s
    T_a --H_ve-----------> T_i

The mass node is advanced with the Crank-Nicolson scheme of Annex C.3; T_s
and T_i follow algebraically from the interval-average T_m. The inverse
model uses the standard's two-probe interpolation (0 and 10 W/m2 * A_f).

Free parameters: A_f, H_em, H_w, H_ve, phi_int_f, k_sol, c_f.

Initialization note: the ``dynamic`` and ``regime_aware`` methods replace the
whole state, T_i included, with the lag-window simulation (legacy behaviour
kept for reproducibility of the published 5R1C results).
"""
import numpy as np

from rctoolbox.models.rc_base_model import RCBaseModel
from rctoolbox.models.rc.five_r1c_eniso13790.numba_core import forecast_heat_demand_numba


class FiveR1C_ENISO13790(RCBaseModel):
    STATE_NAMES = ("T_m", "T_s", "T_i")
    HIDDEN_STATE_IDX = None   # dynamic init overwrites every column (see module docstring)

    def set_params(self, params_dict):
        self.params = params_dict
        self.process_input_params()

    def process_input_params(self):
        self.A_f  = self.params["A_f"]  # [m2]  - Floor area of the conditioned zone
        self.H_em = self.params["H_em"] # [W/K] - Heat conductance through opaque elements of the envelope, between external environment (T_a) and thermal mass (T_m)
        self.H_w  = self.params["H_w"]  # [W/K] - Heat conductance through doors, windows, curtain walls and glazed walls, between external environment (T_a) and internal surface (T_s)
        self.H_ve = self.params["H_ve"] # [W/K] - Heat conductance by ventilation and infiltration, between external environment (T_a) and air (T_i)
        self.phi_int_f = self.params["phi_int_f"] # [W/m2] - Internal heat gains per floor area
        self.Phi_int = self.phi_int_f * self.A_f   # [W] - Total internal heat gains
        self.k_sol = self.params["k_sol"] # [W/m^2] - Solar heat gain factor of the glazed surface of the envelope

        self.c_f = self.params["c_f"]         # [J/(Km^2)] - Thermal capacity per floor area
        self.C_m = self.c_f * self.A_f        # [J/K] - Thermal capacitance of the building’s mass

        self.eff_mass_coeff = self.calc_eff_mass_coeff(self.c_f) # []
        self.A_m = self.eff_mass_coeff * self.A_f # [m^2] - Effective mass area, used for calculating the heat transfer between the thermal mass node and the surface node (T_m and T_s)
        self.h_ms_coeff = 9.1                     # [W/(Km^2] - Heat transfer coefficient between nodes m and s, with fixed value h_ms = 9.1
        self.H_ms = self.h_ms_coeff * self.A_m    # [W/K] - Heat conductance between internal surface (T_s) and building mass (T_m)

        self.a_t_coeff = 4.5                   # [] - Dimensionless ratio between the internal surfaces area and the floor area; can be assumed to be equal to 4.5.
        self.A_t = self.a_t_coeff * self.A_f   # [m^2] - Area of all surfaces facing the room (zone).
        self.h_is_coeff = 3.45                 # [W/(Km^2] - Heat transfer coefficient between the air node (T_i) and the internal surface node (T_s), with a fixed value h_is = 3.45
        self.H_is = self.h_is_coeff * self.A_t # [W/K] - Heat conductance between air (T_i) and internal surface (T_s)

        # Auxiliary helper variables from the EN ISO 13790
        self.H_tr1 =  1.0 / (1.0 / self.H_ve + 1.0 / self.H_is)
        self.H_tr2 =  self.H_tr1 + self.H_w
        self.H_tr3 =  1.0 / (1.0 / self.H_tr2 + 1.0 / self.H_ms)

        # Additional auxiliary variables
        self.His_plus_Hve   = self.H_is + self.H_ve
        self.Hms_plus_Htr2  = self.H_ms + self.H_tr2
        self.Htr1_over_Hve  = self.H_tr1 / self.H_ve
        self.Htr3_plus_Hem  = self.H_tr3 + self.H_em
        self.Htr3_over_Htr2 = self.H_tr3 / self.H_tr2

        self.Am_over_At  = self.A_m / self.A_t

    def calc_eff_mass_coeff(self, c_f):
        if c_f >= 3.7*1e5:
            coeff = 3.5
        elif 2.6*1e5 <= c_f < 3.7*1e5:
            coeff = 3
        elif c_f < 2.6*1e5:
            coeff = 2.5
        return coeff

    # ----------------------------------------------------- forward simulation
    def calc_heat_flows(self, T_a, Phi_sol, Phi_HC):
        """
        Calculates the heat flow from the solar gains, heating/cooling system, and internal gains into the building

        The input of the building is split into the air node (T_i), surface node (T_s), and thermal mass node (T_m) based on
        on the following equations: C.1 - C.3 in [Annex C - section C.2 in ISO 13790]
        """
        Phi_sol_eff = self.k_sol * Phi_sol

        # Heat flow to the air node (T_i)
        Phi_HC_to_ia = Phi_HC
        Phi_ia_plus_Phi_HC = 0.5 * self.Phi_int + Phi_HC_to_ia # (B,1)

        # Heat flow to the room surface node (Ts)
        Phi_st = (1 - (self.Am_over_At) - (self.H_w / (self.h_ms_coeff * self.A_t))) * (0.5 * self.Phi_int + Phi_sol_eff)  # (B,1)

        # Heatflow to the thermal mass node (T_m)
        Phi_m = (self.Am_over_At) * (0.5 * self.Phi_int + Phi_sol_eff)

        # Total heatflow  [(C.5) in [Annex C - section C.3 ISO 13790]]
        Phi_m_tot = Phi_m + self.Htr3_plus_Hem * T_a + self.Htr3_over_Htr2 * (self.Htr1_over_Hve * Phi_ia_plus_Phi_HC + Phi_st) # Shape: [B]

        return Phi_ia_plus_Phi_HC, Phi_st, Phi_m, Phi_m_tot

    def calc_temperature(self, y_prev, dt, T_a, Phi_sol, Phi_HC):

        T_m_prev = y_prev[:,0]

        # Calculate heat flows: Phi_ia, Phi_st, Phi_m, Phi_m_tot
        Phi_ia_plus_Phi_HC, Phi_st, _, Phi_m_tot = self.calc_heat_flows(T_a, Phi_sol, Phi_HC)

        # Calculate next T_m according to Crank–Nicolson discretizationn [(C.4) in [Annex C - section C.3 ISO 13790]]
        T_m_next = ((T_m_prev * ((self.C_m / dt) - 0.5 * self.Htr3_plus_Hem)) + Phi_m_tot) / ((self.C_m / dt) + 0.5 * self.Htr3_plus_Hem)

        # Then to calculate T_s and T_i, we need the average T_m across the timestep (C.9 in [Annex C - section C.3 ISO 13790])
        T_m = (T_m_next + T_m_prev) / 2.0

        # Calculate T_s according to (C.10) in [Annex C - section C.3 ISO 13790]
        T_s = (1 / self.Hms_plus_Htr2) * (self.H_ms * T_m + Phi_st + self.H_tr2 * T_a + self.Htr1_over_Hve * Phi_ia_plus_Phi_HC)

        # Calculate T_i according to (C.11) in [Annex C - section C.3 ISO 13790]
        T_i = (1 / self.His_plus_Hve) * (self.H_is * T_s + self.H_ve * T_a + Phi_ia_plus_Phi_HC)

        return np.stack([T_m_next, T_s, T_i], axis=1)

    # ------------------------------------------------ inverse (heat-demand) API
    def _probe_power(self):
        # 10 W/m2 heating-power probe of EN ISO 13790, Annex C.4
        return 10.0 * self.A_f

    def forecast_temperature(self, y_prev, t, T_a_BT, Phi_sol_BT, Phi_HC_BT):
        """
        Inputs:
        y_prev =  y_0
        t      = [t_0,     t_1,    t_2,    ..., t_N]
        d      = [d_0,     d_1,    d_2,    ..., d_N]   (d == T_a, Phi_sol -> disturbances)
        Φ_HC   = [Φ_HC_0,  Φ_HC_1, Φ_HC_2, ..., Φ_HC_N] (u ==  Φ_HC -> control signal)

        Output:
        y_pred =  [y_0, y_1, y_2, ..., y_N]
        #               [prediction-------]
        """
        t, dt = self._dt_from_t(t)
        T = len(t)

        y_pred = [y_prev] # adding initial value to the predictions

        # y_{k} = y_{k-1} + dt * f(d_{k}, u_{k}) - used approximation: for predicting state at k, use inputs at k.
        for k in range(1,T):

            T_a     = T_a_BT[:, k]
            Phi_sol = Phi_sol_BT[:, k]
            Phi_HC  = Phi_HC_BT[:, k]

            y_next = self.calc_temperature(y_prev, dt, T_a, Phi_sol, Phi_HC)

            y_pred.append(y_next)

            y_prev = y_next

        return np.stack(y_pred, axis=1)

    def forecast_heat_demand(self, y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT):
        """
        Inputs:
        y_prev =  y_0
        t      = [t_0,     t_1,    t_2,    ..., t_N]
        d      = [d_0,     d_1,    d_2,    ..., d_N]   (d == T_a, Phi_sol -> disturbances)
        T_set  = [Tset_0,  Tset_1, Tset_2, ..., Tset_N] (u ==  T_set -> control signal)

        Outputs:
        Φ_HC   =  [Φ_HC_0, Φ_HC_1, Φ_HC_2, ..., Φ_HC_N]
        y_pred =  [y_0,    y_1,     y_2,   ..., y_N]
        #                  [prediction--------------]
        """
        t, dt = self._dt_from_t(t)
        T = len(t)

        y_pred = [y_prev]
        Phi_HC_pred = [Phi_HC_prev]

        for k in range(1,T):
            # calculate Φ_HC demand for meeting the setpoint at time k
            T_a     = T_a_BT[:, k]
            Phi_sol = Phi_sol_BT[:, k]
            T_i_set = T_i_set_BT[:, k]

            y_k, Phi_HC_k = self.calc_energy_demand(y_prev, dt, T_a, Phi_sol, T_i_set)

            y_pred.append(y_k)
            Phi_HC_pred.append(Phi_HC_k)

            y_prev = y_k

        return np.stack(Phi_HC_pred, axis=1), np.stack(y_pred, axis=1)

    def forecast_heat_demand_fast(self, y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT):

        return forecast_heat_demand_numba(
            y_prev,
            Phi_HC_prev,
            t,
            T_a_BT,
            Phi_sol_BT,
            T_i_set_BT,

            self.A_f,
            self.C_m,
            self.H_tr1,
            self.H_tr2,
            self.H_tr3,
            self.Htr3_plus_Hem,
            self.Htr3_over_Htr2,
            self.Htr1_over_Hve,
            self.Hms_plus_Htr2,
            self.His_plus_Hve,
            self.H_ms,
            self.H_is,
            self.H_ve,
            self.H_w,
            self.A_m,
            self.A_t,
            self.h_ms_coeff,
            self.Phi_int,
            self.k_sol,
            self.heat_power_range[1]
        )

    # ------------------------------------------------- steady-state inversion
    def calc_Ts_Tm(self, T_i, T_a, Phi_sol, Phi_HC):
        Phi_ia_plus_Phi_HC, Phi_st, _, _ = self.calc_heat_flows(T_a, Phi_sol, Phi_HC)
        T_s = (1 / self.H_is) * (self.His_plus_Hve * T_i - self.H_ve * T_a - Phi_ia_plus_Phi_HC)
        T_m = (1 / self.H_ms) * (self.Hms_plus_Htr2 * T_s - Phi_st - self.H_tr2 * T_a - self.Htr1_over_Hve * Phi_ia_plus_Phi_HC)
        return T_s, T_m

    def _init_static_inversion(self, data):
        y_0 = np.array(data["y_0"], dtype=np.float64)   # never mutate the caller's array

        T_i, T_a, Phi_sol, Phi_HC = self._get_static_inversion_inputs(data, self.static_inversion_method)
        T_s0, T_m0 = self.calc_Ts_Tm(T_i, T_a, Phi_sol, Phi_HC)

        y_0[:, 0] = T_m0
        y_0[:, 1] = T_s0
        # y_0[:, 2] = T_i is already the measured value at the initial time
        return y_0
