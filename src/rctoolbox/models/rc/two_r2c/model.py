"""Two-state grey-box 2R2C model for indoor temperature prediction and
inverse heat-demand (setpoint-to-power) forecasting.

Topology:
                                            Phi_HC (known)
                                               |
                T_a ---R_ea--- T_e ---R_ie--- T_i
                                |              |
                               C_e            C_i
                                |
                          k_sol * Phi_sol

Free parameters: R_ie, R_ea, C_i, C_e, k_sol.

Discretization: selectable per model or per call (``integration``):
- ``"euler"`` — explicit forward Euler (the fallback when nothing is set).
  Conditionally stable: it diverges whenever the identified air-node time
  constant R_ie*C_i falls below dt/2, which is exactly the region the
  heat-demand identification prefers.
- ``"zoh"`` — exact zero-order-hold map computed by the shared
  ``rctoolbox.models.discretization.zoh_discretize`` (augmented-matrix
  exponential; no inverse of A required). Unconditionally stable for any
  parameter vector. The identification config's ``model_config["integration"]``
  passes through end-to-end: it is applied at training time, saved with the
  results, and reproduced exactly by the loader (absent -> euler).
The two integrations use different matrices:
- Euler works with the continuous matrices (A, B, E),
      y_k = y_{k-1} + dt * (A y_{k-1} + B u_k + E d_k);
- ZOH works with the textbook discrete matrices (A_d, B_d),
      y_k = A_d y_{k-1} + B_d [u_k; d_k],
  where B_d is the ZOH discretization of the lumped input matrix [B E] and
  its control column is B_d[:, 0]
(u = [Phi_HC], d = [T_a, Phi_sol]). The numba fast path always consumes a
discrete affine map, so for Euler it is handed the same update pre-multiplied
into matrix form (I + dt*A, dt*[B E]); it is integration-agnostic.

Selecting the integration: per call via the ``integration`` argument, or
model-wide via ``set_integration``. If neither is set, the model defaults to
'euler' — 'zoh' is an explicit opt-in, keeping the no-argument behavior
aligned with models that have no integration option (e.g. 5R1C).

Initialization (default/dynamic/static_inversion/regime_aware) comes from
``RCBaseModel``; only the static inversion of the hidden T_e is model-specific.
``get_initial_state(data, integration=...)`` forwards the per-call integration
to the lag-window simulation.
"""
import numpy as np

from rctoolbox.models.rc_base_model import RCBaseModel
from rctoolbox.models.discretization import zoh_discretize
from rctoolbox.models.rc.two_r2c.numba_core import forecast_heat_demand_numba


class TwoR2C(RCBaseModel):
    STATE_NAMES = ("T_i", "T_e")
    HIDDEN_STATE_IDX = (1,)   # dynamic/regime_aware init reconstructs T_e only; T_i stays measured

    VALID_INTEGRATIONS = {"euler", "zoh"}

    def __init__(self):
        super().__init__()
        self.integration = None   # unset -> resolves to "euler"
        self._disc_cache = {}

    def set_params(self, params: dict) -> None:
        self.params = params
        self.R_ie   = params["R_ie"]
        self.R_ea   = params["R_ea"]
        self.C_i    = params["C_i"]
        self.C_e    = params["C_e"]
        self.k_sol  = params["k_sol"]
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
        'euler' — so models without an explicit integration choice behave
        like the plain explicit-scheme models (e.g. 5R1C), and 'zoh' is an
        explicit opt-in."""
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
        """Continuous-time state-space matrices of the 2R2C:

            dx/dt = A x + B u + E d

        with state x = [T_i, T_e], control u = [Phi_HC], and disturbances
        d = [T_a, Phi_sol]. Returns (A, B, E) of shapes (2,2), (2,1), (2,2).
        """
        A = np.array([
            [-1.0 / (self.R_ie * self.C_i),  1.0 / (self.R_ie * self.C_i)],
            [ 1.0 / (self.R_ie * self.C_e), - (1 / self.C_e) * (1.0 / self.R_ie + 1.0 / self.R_ea)],
        ])
        B = np.array([
            [1.0 / self.C_i],
            [0.0],
        ])
        E = np.array([
            [0.0,                          0.0],
            [1.0 / (self.R_ea * self.C_e), self.k_sol / self.C_e],
        ])
        return A, B, E


    def get_zoh_matrices(self, dt):
        """Exact discrete (ZOH) matrices (A_d, B_d) for the current parameters
        and time step, in textbook form: the control and disturbance columns
        are lumped into one input matrix before discretization, so B_d
        multiplies the stacked input [Phi_HC, T_a, Phi_sol] and its control
        column is B_d[:, 0]. Cached per (parameters, dt)."""
        if dt not in self._disc_cache:
            A, B, E = self.calc_matrices()
            self._disc_cache[dt] = zoh_discretize(A, np.hstack([B, E]), dt)
        return self._disc_cache[dt]

    def calc_temperature(self, y_prev, dt, T_a, Phi_sol, Phi_HC, integration=None):
        """One discrete step of the 2R2C model.

        y_prev : ndarray, shape (B, 2) — state [T_i, T_e] at the interval start.
        dt     : float — time step [s].
        T_a, Phi_sol, Phi_HC : ndarray, shape (B,) — inputs held constant over
        the interval (the right-labeled hourly averages).
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Euler steps with the continuous matrices (A, B, E):
            y_next = y + dt * (A y + B u + E d);
        ZOH applies the exact discrete map (A_d, B_d):
            y_next = A_d y + B_d [u; d].

        Returns y_next : ndarray, shape (B, 2) — state at the interval end.
        """
        integration = self._resolve_integration(integration)

        if integration == "euler":
            A, B, E = self.calc_matrices()
            d = np.stack([T_a, Phi_sol], axis=1)
            dydt = y_prev @ A.T + np.outer(Phi_HC, B[:, 0]) + d @ E.T
            return y_prev + dt * dydt

        # zoh
        Ad, Bd = self.get_zoh_matrices(dt)
        u = np.stack([Phi_HC, T_a, Phi_sol], axis=1)  # stacked input [u; d]
        return y_prev @ Ad.T + u @ Bd.T

    def forecast_temperature(self, y_prev, t, T_a_BT, Phi_sol_BT, Phi_HC_BT, integration=None):
        """Multi-step forecast via the discrete one-step map.

        y_prev   : ndarray, shape (B, 2) — initial state [T_i_0, T_e_0].
        t        : ndarray, shape (T,) or (B, T) — time vector [s].
        T_a_BT, Phi_sol_BT, Phi_HC_BT : ndarray, shape (B, T) — inputs.
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Returns y_pred : ndarray, shape (B, T, 2) — state trajectory.

        Convention (same as the other models): the state at index k is computed
        with the inputs at index k, which under right-closed/right-labeled
        resampling hold the interval averages over (t_{k-1}, t_k].
        """
        t, dt = self._dt_from_t(t)

        B, N = T_a_BT.shape
        y_pred = np.empty((B, N, 2))
        y_pred[:, 0, :] = y_prev

        y_k = y_prev
        for k in range(1, N):
            T_k = T_a_BT[:, k]
            Phi_sol_k = Phi_sol_BT[:, k]
            Phi_HC_k = Phi_HC_BT[:, k]
            y_k = self.calc_temperature(y_k, dt, T_k, Phi_sol_k, Phi_HC_k, integration=integration)
            y_pred[:, k, :] = y_k

        return y_pred

    # ------------------------------------------------ inverse (heat-demand) API
    def calc_energy_demand(self, y_prev, dt, T_a, Phi_sol, T_i_set, integration=None):
        """One step of the inverse (setpoint-to-power) 2R2C model.

        Closed-form ideal-regulation inversion of the discrete one-step map:
        T_i at the interval end is affine in the held power,
        T_i^k = free[0] + s * Phi with s = B_d[0, 0], so the unrestricted
        demand is

            Phi_unres = (T_i_set - free[0]) / s,

        where ``free`` is the zero-power trajectory over the interval. It is
        then clipped to the actuator range and the state advanced with the
        realized power. This holds for either integration: under 'zoh',
        s is the exact one-hour power-to-air gain; under 'euler', s = dt/C_i
        and the formula reduces to the classical
        Phi_unres = C_i (T_set - T_i)/dt - (T_e - T_i)/R_ie. Because the map
        is affine in Phi_HC, both are algebraically identical to the
        EN ISO 13790 two-probe interpolation used by the 5R1C model.

        y_prev : ndarray, shape (B, 2) — state [T_i, T_e] at the interval start.
        dt     : float — time step [s].
        T_a, Phi_sol, T_i_set : ndarray, shape (B,) — inputs for the interval.
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Returns (y, Phi_HC_demand): state at the interval end and realized power.
        """
        integration = self._resolve_integration(integration)
        d = np.stack([T_a, Phi_sol], axis=1)

        if integration == "euler":
            # zero-power Euler step with the continuous matrices; the power
            # sensitivity of T_i over one step is s = dt * B[0, 0] = dt / C_i
            A, B, E = self.calc_matrices()
            free = y_prev + dt * (y_prev @ A.T + d @ E.T)
            s = dt * B[0, 0]
            b_col = dt * B[:, 0]
        else:  # zoh
            # B_d columns follow the stacked input [Phi_HC, T_a, Phi_sol]:
            # column 0 is the control, columns 1: are the disturbances
            Ad, Bd = self.get_zoh_matrices(dt)
            free = y_prev @ Ad.T + d @ Bd[:, 1:].T
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

        Returns (Phi_HC_pred, y_pred) of shapes (B, T) and (B, T, 2).

        Convention (same as the other models): the state and demand at index k
        are computed with the inputs at index k, which under right-closed/
        right-labeled resampling hold the interval averages over (t_{k-1}, t_k].
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
        """Numba-JIT'd version of forecast_heat_demand (identical results).

        The discrete matrices for the requested integration are precomputed
        here (cheap, once per parameter set) and handed to the kernel as plain
        arrays — the kernel itself is integration-agnostic: it only iterates
        the affine map and the closed-form inversion.
        """
        t, dt = self._dt_from_t(t)
        # numba does not bounds-check: a t longer than the input columns would
        # silently read out-of-bounds memory. Guard shapes and dt uniformity.
        assert T_a_BT.shape[1] >= len(t) and Phi_sol_BT.shape[1] >= len(t) and T_i_set_BT.shape[1] >= len(t), \
            "input arrays have fewer time steps than t"
        assert np.allclose(np.diff(t), dt), "non-uniform time vector"

        integration = self._resolve_integration(integration)
        if integration == "euler":
            # the kernel iterates a discrete affine map, so hand it the Euler
            # update pre-multiplied into matrix form: I + dt*A and dt*[B E]
            A, B, E = self.calc_matrices()
            Ad = np.eye(2) + dt * A
            Bd = dt * np.hstack([B, E])
        else:  # zoh
            Ad, Bd = self.get_zoh_matrices(dt)

        return forecast_heat_demand_numba(
            y_prev,
            Phi_HC_prev,
            t,
            T_a_BT,
            Phi_sol_BT,
            T_i_set_BT,

            np.ascontiguousarray(Ad),
            np.ascontiguousarray(Bd),  # columns: [Phi_HC, T_a, Phi_sol]

            self.heat_power_range[0],
            self.heat_power_range[1],
        )

    def calc_Te(self, T_i, Phi_HC):
        """Static inversion of the indoor-node balance at quasi-steady state:
        0 = (T_e - T_i)/R_ie + Phi_HC  =>  T_e = T_i - R_ie * Phi_HC."""
        return T_i - self.R_ie * Phi_HC

    # ------------------------------------------------- steady-state inversion
    def _init_static_inversion(self, data) -> np.ndarray:
        y_0 = np.array(data["y_0"], dtype=np.float64)   # never mutate the caller's array

        T_i, _, _, Phi_HC = self._get_static_inversion_inputs(data, self.static_inversion_method)
        # T_i at the window start is measured and already sits in y_0[:, 0];
        # only the hidden envelope state is reconstructed.
        y_0[:, 1] = self.calc_Te(T_i, Phi_HC)
        return y_0
