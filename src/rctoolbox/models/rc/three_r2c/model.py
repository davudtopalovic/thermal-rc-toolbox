"""3R2C grey-box model: the 2R2C topology plus a direct air<->ambient
conductance R_ia — a fast loss path (ventilation/infiltration, optionally
glazing) bypassing the envelope mass — and two extra gain channels:

                                   Phi_HC + k_sol_i*Phi_sol + 0.5*Phi_int
                                               |
                T_a ---R_ea--- T_e ---R_ie--- T_i ---R_ia--- T_a
                                |              |
                               C_e            C_i
                                |
                   k_sol_e*Phi_sol + 0.5*Phi_int

- k_sol_e / k_sol_i: solar gains onto the envelope and directly onto the
  air/thermostat node;
- Phi_int [W]: constant internal gains, split 50/50 between the air and
  envelope nodes (EN ISO 13790-style fixed split).

Free parameters (8): R_ie, R_ea, R_ia, C_i, C_e, k_sol_e, k_sol_i, Phi_int.

Continuous model: dx/dt = A x + B u + E d + w, with state x = [T_i, T_e],
control u = [Phi_HC], disturbances d = [T_a, Phi_sol] and the constant
vector w = [0.5*Phi_int/C_i, 0.5*Phi_int/C_e]. Under ZOH the constant input
is discretized exactly like any other held input (extra column with input 1),
yielding a per-step offset; under Euler the offset is dt*w.

Discretization: selectable per model or per call (``integration``), with
the same options as the 2R2C:
- ``"euler"`` — explicit forward Euler (the fallback when nothing is set);
- ``"zoh"`` — exact zero-order-hold map from
  ``rctoolbox.models.discretization.zoh_discretize``, unconditionally stable.
Euler works with the continuous matrices (A, B, E, w); ZOH with the discrete
(A_d, B_d, off), where B_d multiplies the stacked input [Phi_HC, T_a, Phi_sol]
(control column B_d[:, 0]) and off is the discretized constant offset. The
numba fast path consumes the discrete affine map for either integration.

Inverse model: closed-form ideal-regulation inversion of the one-step map
(T_i is affine in the held power), clipped to the actuator range.

The initialization framework (default/dynamic/static_inversion/regime_aware)
comes from ``RCBaseModel``; only the static inversion of the hidden T_e is
model-specific. With R_ia -> inf, k_sol_i = 0 and Phi_int = 0 the model
reduces exactly to the 2R2C.
"""
import numpy as np

from rctoolbox.models.rc_base_model import RCBaseModel
from rctoolbox.models.discretization import zoh_discretize
from rctoolbox.models.rc.three_r2c.numba_core import forecast_heat_demand_offset_numba


class ThreeR2C(RCBaseModel):
    STATE_NAMES = ("T_i", "T_e")
    HIDDEN_STATE_IDX = (1,)   # dynamic/regime_aware init reconstructs T_e only; T_i stays measured

    VALID_INTEGRATIONS = {"euler", "zoh"}

    def __init__(self):
        super().__init__()
        self.integration = None   # unset -> resolves to "euler"
        self._disc_cache = {}

    def set_params(self, params: dict) -> None:
        self.params = params
        self.R_ie    = params["R_ie"]
        self.R_ea    = params["R_ea"]
        self.R_ia    = params["R_ia"]
        self.C_i     = params["C_i"]
        self.C_e     = params["C_e"]
        self.k_sol_e = params["k_sol_e"]
        self.k_sol_i = params["k_sol_i"]
        self.Phi_int = params["Phi_int"]
        self._disc_cache = {}   # discrete matrices depend on the parameters

    # --------------------------------------------------------- discretization
    def set_integration(self, integration) -> None:
        """Set the model-level default integration ('euler' or 'zoh').
        None clears it back to unset, which resolves to 'euler'."""
        if integration is not None and integration not in self.VALID_INTEGRATIONS:
            raise ValueError(
                f"Unknown integration '{integration}'. "
                f"Must be one of {sorted(self.VALID_INTEGRATIONS)} or None"
            )
        self.integration = integration
        self._disc_cache = {}

    def _resolve_integration(self, integration):
        """Resolve the integration for one call: the argument wins, otherwise
        the model-level setting; if neither is set (None), fall back to
        'euler' — 'zoh' is an explicit opt-in."""
        integration = integration if integration is not None else self.integration
        if integration is None:
            return "euler"
        if integration not in self.VALID_INTEGRATIONS:
            raise ValueError(
                f"Unknown integration '{integration}'. "
                f"Must be one of {sorted(self.VALID_INTEGRATIONS)}"
            )
        return integration

    def calc_matrices(self):
        """Continuous-time matrices (A, B, E, w) of the 3R2C:

            dx/dt = A x + B u + E d + w

        Compared to the 2R2C, the air row gains the direct ambient coupling
        1/R_ia (in A[0,0] and E[0,0]) and the solar-to-air gain k_sol_i
        (E[0,1]); w is the constant internal-gains vector."""
        A = np.array([
            [-(1.0 / self.R_ie + 1.0 / self.R_ia) / self.C_i,  1.0 / (self.R_ie * self.C_i)],
            [ 1.0 / (self.R_ie * self.C_e), -(1.0 / self.R_ie + 1.0 / self.R_ea) / self.C_e],
        ])
        B = np.array([
            [1.0 / self.C_i],
            [0.0],
        ])
        E = np.array([
            [1.0 / (self.R_ia * self.C_i), self.k_sol_i / self.C_i],
            [1.0 / (self.R_ea * self.C_e), self.k_sol_e / self.C_e],
        ])
        w = np.array([0.5 * self.Phi_int / self.C_i,
                      0.5 * self.Phi_int / self.C_e])
        return A, B, E, w

    def get_zoh_matrices(self, dt):
        """(A_d, B_d, off): B_d for the stacked input [Phi_HC, T_a, Phi_sol]
        plus the exactly discretized constant-gains offset (the w vector is
        treated as one more held input with value 1). Cached per (params, dt)."""
        if dt not in self._disc_cache:
            A, B, E, w = self.calc_matrices()
            Ad, Bd4 = zoh_discretize(A, np.hstack([B, E, w[:, None]]), dt)
            self._disc_cache[dt] = (Ad, Bd4[:, :3], Bd4[:, 3])
        return self._disc_cache[dt]

    # ----------------------------------------------------- forward simulation
    def calc_temperature(self, y_prev, dt, T_a, Phi_sol, Phi_HC, integration=None):
        """One discrete step of the 3R2C model.

        y_prev : ndarray, shape (B, 2) — state [T_i, T_e] at the interval start.
        dt     : float — time step [s].
        T_a, Phi_sol, Phi_HC : ndarray, shape (B,) — inputs held constant over
        the interval (the right-labeled hourly averages).
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Euler: y_next = y + dt * (A y + B u + E d + w);
        ZOH:   y_next = A_d y + B_d [u; d] + off.

        Returns y_next : ndarray, shape (B, 2) — state at the interval end.
        """
        integration = self._resolve_integration(integration)
        if integration == "euler":
            A, B, E, w = self.calc_matrices()
            d = np.stack([T_a, Phi_sol], axis=1)
            dydt = y_prev @ A.T + np.outer(Phi_HC, B[:, 0]) + d @ E.T + w
            return y_prev + dt * dydt
        Ad, Bd, off = self.get_zoh_matrices(dt)
        u = np.stack([Phi_HC, T_a, Phi_sol], axis=1)
        return y_prev @ Ad.T + u @ Bd.T + off

    def forecast_temperature(self, y_prev, t, T_a_BT, Phi_sol_BT, Phi_HC_BT, integration=None):
        """Multi-step forecast via the discrete one-step map.

        y_prev   : ndarray, shape (B, 2) — initial state [T_i_0, T_e_0].
        t        : ndarray, shape (T,) or (B, T) — time vector [s].
        T_a_BT, Phi_sol_BT, Phi_HC_BT : ndarray, shape (B, T) — inputs.
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Returns y_pred : ndarray, shape (B, T, 2) — state trajectory. The state
        at index k is computed with the inputs at index k (right-labeled).
        """
        t, dt = self._dt_from_t(t)

        B, N = T_a_BT.shape
        y_pred = np.empty((B, N, 2))
        y_pred[:, 0, :] = y_prev

        y_k = y_prev
        for k in range(1, N):
            y_k = self.calc_temperature(
                y_k, dt, T_a_BT[:, k], Phi_sol_BT[:, k], Phi_HC_BT[:, k], integration=integration
            )
            y_pred[:, k, :] = y_k

        return y_pred

    # ------------------------------------------------ inverse (heat-demand) API
    def calc_energy_demand(self, y_prev, dt, T_a, Phi_sol, T_i_set, integration=None):
        """One step of the inverse (setpoint-to-power) 3R2C model.

        Closed-form ideal-regulation inversion: T_i at the interval end is
        affine in the held power, T_i^k = free[0] + s * Phi, where ``free`` is
        the zero-power trajectory (including the internal-gains offset) and
        s the one-step power-to-air gain (dt/C_i under Euler, B_d[0, 0] under
        ZOH). The unrestricted demand (T_i_set - free[0]) / s is clipped to
        the actuator range and the state advanced with the realized power.

        Returns (y, Phi_HC_demand): state at the interval end and realized power.
        """
        integration = self._resolve_integration(integration)
        d = np.stack([T_a, Phi_sol], axis=1)
        if integration == "euler":
            A, B, E, w = self.calc_matrices()
            free = y_prev + dt * (y_prev @ A.T + d @ E.T + w)
            s = dt * B[0, 0]
            b_col = dt * B[:, 0]
        else:  # zoh
            Ad, Bd, off = self.get_zoh_matrices(dt)
            free = y_prev @ Ad.T + d @ Bd[:, 1:].T + off
            s = Bd[0, 0]
            b_col = Bd[:, 0]
        Phi_unres = (T_i_set - free[:, 0]) / s
        Phi_HC_demand = np.clip(Phi_unres, self.heat_power_range[0], self.heat_power_range[1])
        y = free + np.outer(Phi_HC_demand, b_col)
        return y, Phi_HC_demand

    def forecast_heat_demand(self, y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT, integration=None):
        """Multi-step heat-demand forecast.

        y_prev      : ndarray, shape (B, 2) — initial state [T_i_0, T_e_0].
        Phi_HC_prev : ndarray, shape (B,) — observed power at index 0.
        t           : ndarray, shape (T,) or (B, T) — time vector [s].
        T_a_BT, Phi_sol_BT, T_i_set_BT : ndarray, shape (B, T) — inputs.
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Returns (Phi_HC_pred, y_pred) of shapes (B, T) and (B, T, 2); the
        state and demand at index k are computed with the inputs at index k.
        """
        t, dt = self._dt_from_t(t)

        B, N = T_a_BT.shape
        y_pred = np.empty((B, N, 2))
        Phi_HC_pred = np.empty((B, N))
        y_pred[:, 0, :] = y_prev
        Phi_HC_pred[:, 0] = Phi_HC_prev

        y_k = y_prev
        for k in range(1, N):
            y_k, Phi_HC_k = self.calc_energy_demand(
                y_k, dt, T_a_BT[:, k], Phi_sol_BT[:, k], T_i_set_BT[:, k],
                integration=integration,
            )
            y_pred[:, k, :] = y_k
            Phi_HC_pred[:, k] = Phi_HC_k

        return Phi_HC_pred, y_pred

    def forecast_heat_demand_fast(self, y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT, integration=None):
        """Numba-JIT'd version of forecast_heat_demand (identical results);
        the kernel iterates the affine map with a per-step offset."""
        t, dt = self._dt_from_t(t)
        assert T_a_BT.shape[1] >= len(t) and Phi_sol_BT.shape[1] >= len(t) and T_i_set_BT.shape[1] >= len(t), \
            "input arrays have fewer time steps than t"
        assert np.allclose(np.diff(t), dt), "non-uniform time vector"

        integration = self._resolve_integration(integration)
        if integration == "euler":
            A, B, E, w = self.calc_matrices()
            Ad = np.eye(2) + dt * A
            Bd = dt * np.hstack([B, E])
            off = dt * w
        else:  # zoh
            Ad, Bd, off = self.get_zoh_matrices(dt)

        return forecast_heat_demand_offset_numba(
            y_prev, Phi_HC_prev, t, T_a_BT, Phi_sol_BT, T_i_set_BT,
            np.ascontiguousarray(Ad), np.ascontiguousarray(Bd), np.ascontiguousarray(off),
            self.heat_power_range[0], self.heat_power_range[1],
        )

    # ------------------------------------------------- steady-state inversion
    def calc_Te(self, T_i, Phi_HC, T_a, Phi_sol):
        """Static inversion of the indoor balance at quasi-steady state:
        0 = (T_e-T_i)/R_ie + (T_a-T_i)/R_ia + Phi_HC + k_sol_i*Phi_sol + 0.5*Phi_int."""
        return T_i - self.R_ie * (Phi_HC + (T_a - T_i) / self.R_ia
                                  + self.k_sol_i * Phi_sol + 0.5 * self.Phi_int)

    def _init_static_inversion(self, data) -> np.ndarray:
        y_0 = np.array(data["y_0"], dtype=np.float64)   # never mutate the caller's array
        T_i, T_a, Phi_sol, Phi_HC = self._get_static_inversion_inputs(data, self.static_inversion_method)
        # T_i at the window start is measured and already sits in y_0[:, 0];
        # only the hidden envelope state is reconstructed.
        y_0[:, 1] = self.calc_Te(T_i, Phi_HC, T_a, Phi_sol)
        return y_0
