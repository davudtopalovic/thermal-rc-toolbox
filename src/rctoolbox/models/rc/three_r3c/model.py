"""Three-resistance, three-capacitance grey-box model — generic topology.

States y = [T_i, T_h, T_e]:
    T_i — indoor air        (typically measured)
    T_h — heating system    (typically measured)
    T_e — envelope mass     (hidden)

Topology (each ── is a thermal resistance, dashed = solar coupling):

                
                  ┌───────–––R_ia─────────────┐
                  │                        │
    T_h ──R_ih── T_i ──R_ie── T_e ──R_ea── T_a
    C_h         C_i           C_e
     │           │             │
    Phi_HC  k_sol_i·Phi_sol   k_sol_e·Phi_sol

Continuous-time equations:
    C_i · dT_i/dt = (T_h - T_i)/R_ih + (T_e - T_i)/R_ie
                    + (T_a - T_i)/R_ia + k_sol_i · Phi_sol
    C_h · dT_h/dt = (T_i - T_h)/R_ih + Phi_HC
    C_e · dT_e/dt = (T_i - T_e)/R_ie + (T_a - T_e)/R_ea + k_sol_e · Phi_sol

Free parameters (9):
    R_ih, R_ie, R_ea, R_ia, C_i, C_h, C_e, k_sol_i, k_sol_e

Forward Euler discretization.

Initialization: ``default`` (y_0 from build_data_dict) or ``dynamic``
(simulate the lag window; T_i_0 and T_h_0 keep their measured boundary values,
T_e_0 is taken from the simulation) — both from ``RCBaseModel``.
"""
import numpy as np

from rctoolbox.models.rc_base_model import RCBaseModel


class ThreeR3C(RCBaseModel):
    STATE_NAMES = ("T_i", "T_h", "T_e")
    HIDDEN_STATE_IDX = (2,)   # dynamic init: T_i and T_h stay measured, T_e from the lag simulation
    VALID_INIT_METHODS = frozenset({"default", "dynamic"})

    def set_params(self, params: dict) -> None:
        self.params  = params
        self.R_ih    = params["R_ih"]
        self.R_ie    = params["R_ie"]
        self.R_ea    = params["R_ea"]
        self.R_ia    = params["R_ia"]
        self.C_i     = params["C_i"]
        self.C_h     = params["C_h"]
        self.C_e     = params["C_e"]
        self.k_sol_i = params["k_sol_i"]
        self.k_sol_e = params["k_sol_e"]

    # ----------------------------------------------------- forward simulation
    def calc_temperature(self, y_prev, dt, T_a, Phi_sol, Phi_HC):
        """One forward-Euler step.

        y_prev   : ndarray, shape (B, 3) — [T_i, T_h, T_e] at step k.
        dt       : float — time step [s].
        T_a, Phi_sol, Phi_HC : ndarray, shape (B,) — inputs at step k.

        Returns y_next : ndarray, shape (B, 3).
        """
        T_i = y_prev[:, 0]
        T_h = y_prev[:, 1]
        T_e = y_prev[:, 2]

        dT_i = ((T_h - T_i) / (self.R_ih * self.C_i)
                + (T_e - T_i) / (self.R_ie * self.C_i)
                + (T_a - T_i) / (self.R_ia * self.C_i)
                + self.k_sol_i * Phi_sol / self.C_i)
        dT_h = (T_i - T_h) / (self.R_ih * self.C_h) + Phi_HC / self.C_h
        dT_e = ((T_i - T_e) / (self.R_ie * self.C_e)
                + (T_a - T_e) / (self.R_ea * self.C_e)
                + self.k_sol_e * Phi_sol / self.C_e)

        # dt = 1
        T_i_next = T_i + dt * dT_i
        T_h_next = T_h + dt * dT_h
        T_e_next = T_e + dt * dT_e
        y_next = np.stack([T_i_next, T_h_next, T_e_next], axis=1)
        return np.clip(y_next, 0, 60)

    def forecast_temperature(self, y_prev, t, T_a_BT, Phi_sol_BT, Phi_HC_BT):
        """Multi-step forecast via forward Euler.

        y_prev   : ndarray, shape (B, 3) — initial [T_i_0, T_h_0, T_e_0].
        t        : ndarray, shape (T,) or (B, T) — time vector [s].
        T_a_BT, Phi_sol_BT, Phi_HC_BT : ndarray, shape (B, T) — inputs.

        Returns y_pred : ndarray, shape (B, T, 3) — state trajectory.
        """
        t, dt = self._dt_from_t(t)

        B, N = T_a_BT.shape
        y_pred = np.empty((B, N, 3))
        y_pred[:, 0, :] = y_prev

        y = y_prev
        for k in range(1,N):
            y = self.calc_temperature(
                y, dt,
                T_a_BT[:, k], Phi_sol_BT[:, k], Phi_HC_BT[:, k],
            )
            y_pred[:, k, :] = y

        return y_pred

    # ------------------------------------------------ inverse (heat-demand) API
    def forecast_heat_demand(self, y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT):
        """Multi-step heat-demand forecast via the two-probe inversion of the
        one-step map (``RCBaseModel.calc_energy_demand``).

        y_prev      : ndarray, shape (B, 3) — initial state.
        Phi_HC_prev : ndarray, shape (B,) — observed power at index 0.
        t           : ndarray, shape (T,) or (B, T) — time vector [s].
        T_a_BT, Phi_sol_BT, T_i_set_BT : ndarray, shape (B, T) — inputs.

        Returns (Phi_HC_pred, y_pred) of shapes (B, T) and (B, T, 3).
        """
        t, dt = self._dt_from_t(t)

        B, N = T_a_BT.shape
        y_pred = np.empty((B, N, 3))
        Phi_HC_pred = np.empty((B, N))
        y_pred[:, 0, :] = y_prev
        Phi_HC_pred[:, 0] = Phi_HC_prev

        y = y_prev
        for k in range(1, N):
            y, Phi_HC_k = self.calc_energy_demand(
                y, dt, T_a_BT[:, k], Phi_sol_BT[:, k], T_i_set_BT[:, k],
            )
            y_pred[:, k, :] = y
            Phi_HC_pred[:, k] = Phi_HC_k

        return Phi_HC_pred, y_pred

    def forecast_heat_demand_fast(self, y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT):
        """No compiled kernel for this model yet: runs the numpy loop."""
        return self.forecast_heat_demand(y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT)
