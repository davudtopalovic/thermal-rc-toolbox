"""Six-resistance, three-capacitance EN ISO 13790 grey-box model.

States y = [T_i, T_h, T_m]:
    T_i — indoor air (measured)
    T_h — heating system / supply circuit (measured)
    T_m — building thermal mass (hidden)

Continuous-time state-space (column vectors):
    C * dy/dt = A y + B u + E d
with u = [Phi_HC] and d = [T_a, Phi_sol, 1] (Phi_int is baked into E).

Free parameters (10):
    A_f, H_em, H_w, H_ve, c_f, C_i, A_w, H_ih, C_h, Phi_int

Forward Euler discretization.

Initialization: ``default`` (y_0 as provided) or ``dynamic`` (simulate the lag
window with measured inputs; T_i_0 and T_h_0 keep their measured values, T_m_0
is taken from the simulation) — both from ``RCBaseModel``.
"""
import numpy as np

from rctoolbox.models.rc_base_model import RCBaseModel


class SixR3C_ENISO13790(RCBaseModel):
    STATE_NAMES = ("T_i", "T_h", "T_m")
    HIDDEN_STATE_IDX = (2,)   # dynamic init: T_i and T_h stay measured, T_m from the lag simulation
    VALID_INIT_METHODS = frozenset({"default", "dynamic"})

    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.E = None
        self.inv_C = None

    def set_params(self, params: dict) -> None:
        self.params = params
        self._build_state_space()

    def _build_state_space(self) -> None:
        A_f     = self.params["A_f"]
        H_em    = self.params["H_em"]
        H_w     = self.params["H_w"]
        H_ve    = self.params["H_ve"]
        c_f     = self.params["c_f"]
        C_i     = self.params["C_i"]
        A_w     = self.params["A_w"]
        H_ih    = self.params["H_ih"]
        C_h     = self.params["C_h"]
        Phi_int = self.params["Phi_int"]

        # Derived constants (EN ISO 13790)
        A_t = 4.5 * A_f
        A_m = 3.5 * A_f
        H_is = A_t * 3.45
        H_ms = A_m * 9.1
        C_m = c_f * A_f

        X = H_is + H_w + H_ms
        Hms_over_X    = H_ms / X
        His_over_X    = H_is / X
        His2_over_X   = H_is * His_over_X
        Hms2_over_X   = H_ms * Hms_over_X
        HisHms_over_X = H_ms * His_over_X
        HisHw_over_X  = H_w  * His_over_X
        Am_over_At = A_m / A_t
        Y = 1 - Am_over_At - H_w / (9.1 * A_t)

        # A: state-coupling
        self.A = np.array([
            [His2_over_X - (H_ve + H_is + H_ih),         H_ih,  HisHms_over_X],
            [                              H_ih,        -H_ih,            0.0],
            [                    HisHms_over_X,          0.0,  Hms2_over_X - (H_em + H_ms)],
        ])

        # B: Phi_HC → T_h node
        self.B = np.array([[0.0], [1.0], [0.0]])

        # E: disturbance coupling, columns = [T_a, Phi_sol, 1]
        # (Phi_int baked into the third column coefficient)
        self.E = np.array([
            [H_ve + HisHw_over_X,           His_over_X * Y * A_w,           ((His_over_X * Y + 1) * 0.5) * Phi_int],
            [                0.0,                            0.0,                                              0.0],
            [H_em + (H_ms * H_w) / X,  (Hms_over_X * Y + Am_over_At) * A_w, ((Hms_over_X * Y + Am_over_At) * 0.5) * Phi_int],
        ])

        # Inverse heat capacities (state-wise)
        self.inv_C = np.array([1.0 / C_i, 1.0 / C_h, 1.0 / C_m])


    # ----------------------------------------------------- forward simulation
    def calc_temperature(self, y_prev, dt, T_a, Phi_sol, Phi_HC):
        """One forward-Euler step.

        y_prev   : ndarray, shape (B, 3) — [T_i, T_h, T_m] at step k.
        dt       : float — time step [s].
        T_a, Phi_sol, Phi_HC : ndarray, shape (B,) — inputs at step k.

        Returns y_next : ndarray, shape (B, 3).
        """
        B = y_prev.shape[0]
        # d = [T_a, Phi_sol, 1.0]: the '1.0' triggers the Phi_int-coefficient
        # column of E, which already contains Phi_int as a multiplier.
        d = np.stack([T_a, Phi_sol, np.ones(B)], axis=1)        # (B, 3)
        u = np.asarray(Phi_HC).reshape(B, 1)                    # (B, 1)

        A_term = y_prev @ self.A.T                              # (B, 3)
        B_term = u @ self.B.T                                   # (B, 3)
        E_term = d @ self.E.T                                   # (B, 3)

        dy = self.inv_C * (A_term + B_term + E_term)
        return y_prev + dt * dy

    def forecast_temperature(self, y_prev, t, T_a_BT, Phi_sol_BT, Phi_HC_BT):
        """Multi-step forecast via forward Euler.

        y_prev   : ndarray, shape (B, 3) — initial [T_i_0, T_h_0, T_m_0].
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
