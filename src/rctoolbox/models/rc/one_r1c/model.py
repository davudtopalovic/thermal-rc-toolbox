"""Single-state grey-box 1R1C model for indoor temperature prediction.

Topology:
                       Phi_HC (known)
                          |
          T_a ---R_ia--- T_i
                          |
                         C_i

Continuous-time energy balance:
    C_i * dT_i/dt = (T_a - T_i) / R_ia + Phi_HC

Time constant tau = R_ia * C_i. Phi_HC > 0 heats, Phi_HC < 0 cools (e.g. a
freezer: Phi_HC = -P_el, with R_ia and C_i then being effective values that
absorb the COP).

Free parameters: R_ia, C_i. There is no solar input: unlike the other RC
models, ``calc_temperature`` and ``forecast_temperature`` take only T_a and
Phi_HC (no Phi_sol argument).

Discretization: selectable per model or per call (``integration``), with the
same semantics as ``TwoR2C``:
- ``"euler"`` — explicit forward Euler (the fallback when nothing is set),
      T_i,k = T_i,k-1 + dt * (A T_i,k-1 + B u_k + E d_k);
  stable only for dt < 2 * R_ia * C_i.
- ``"zoh"`` — exact zero-order-hold map from the shared
  ``rctoolbox.models.discretization.zoh_discretize``,
      T_i,k = A_d T_i,k-1 + B_d [u_k; d_k],
  which for one state reduces to A_d = exp(-dt / tau) and
  B_d = (1 - A_d) * [R_ia, 1]; exact for interval-constant inputs and
  unconditionally stable.
(u = [Phi_HC], d = [T_a].)

Initialization: the only state is the measured indoor air, so there is no
hidden state to reconstruct and only the ``default`` initialization (T_i_0 as
provided in ``y_0``) is supported.

The inverse (heat-demand) API is not implemented for this model.
"""
import numpy as np

from rctoolbox.models.rc_base_model import RCBaseModel
from rctoolbox.models.discretization import zoh_discretize


class OneR1C(RCBaseModel):
    STATE_NAMES = ("T_i",)
    HIDDEN_STATE_IDX = ()   # no hidden states
    VALID_INIT_METHODS = frozenset({"default"})

    VALID_INTEGRATIONS = {"euler", "zoh"}

    def __init__(self):
        super().__init__()
        self.integration = None   # unset -> resolves to "euler"
        self._disc_cache = {}

    def set_params(self, params: dict) -> None:
        self.params = params
        self.R_ia   = params["R_ia"]
        self.C_i    = params["C_i"]
        self._disc_cache = {}   # discrete matrices depend on the parameters

    @property
    def tau(self) -> float:
        """Time constant R_ia * C_i [s]."""
        return self.R_ia * self.C_i

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
        'euler'."""
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
        """Continuous-time state-space matrices of the 1R1C:

            dx/dt = A x + B u + E d

        with state x = [T_i], control u = [Phi_HC], and disturbance
        d = [T_a]. Returns (A, B, E) of shapes (1,1), (1,1), (1,1).
        """
        A = np.array([[-1.0 / (self.R_ia * self.C_i)]])
        B = np.array([[1.0 / self.C_i]])
        E = np.array([[1.0 / (self.R_ia * self.C_i)]])
        return A, B, E

    def get_zoh_matrices(self, dt):
        """Exact discrete (ZOH) matrices (A_d, B_d) for the current parameters
        and time step; B_d multiplies the stacked input [Phi_HC, T_a].
        Cached per (parameters, dt)."""
        if dt not in self._disc_cache:
            A, B, E = self.calc_matrices()
            self._disc_cache[dt] = zoh_discretize(A, np.hstack([B, E]), dt)
        return self._disc_cache[dt]

    # ----------------------------------------------------- forward simulation
    def calc_temperature(self, y_prev, dt, T_a, Phi_HC, integration=None):
        """One discrete step of the 1R1C model.

        y_prev : ndarray, shape (B, 1) — state [T_i] at the interval start.
        dt     : float — time step [s].
        T_a, Phi_HC : ndarray, shape (B,) — inputs held constant over the
        interval (the right-labeled interval averages).
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Returns y_next : ndarray, shape (B, 1) — state at the interval end.
        """
        integration = self._resolve_integration(integration)

        if integration == "euler":
            A, B, E = self.calc_matrices()
            dydt = y_prev @ A.T + np.outer(Phi_HC, B[:, 0]) + np.outer(T_a, E[:, 0])
            return y_prev + dt * dydt

        # zoh
        Ad, Bd = self.get_zoh_matrices(dt)
        u = np.stack([Phi_HC, T_a], axis=1)  # stacked input [u; d]
        return y_prev @ Ad.T + u @ Bd.T

    def forecast_temperature(self, y_prev, t, T_a_BT, Phi_HC_BT, integration=None):
        """Multi-step forecast via the discrete one-step map.

        y_prev   : ndarray, shape (B, 1) — initial state [T_i_0].
        t        : ndarray, shape (T,) or (B, T) — time vector [s].
        T_a_BT, Phi_HC_BT : ndarray, shape (B, T) — inputs.
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Returns y_pred : ndarray, shape (B, T, 1) — state trajectory.

        Convention (same as the other models): the state at index k is computed
        with the inputs at index k, which under right-closed/right-labeled
        resampling hold the interval averages over (t_{k-1}, t_k].
        """
        t, dt = self._dt_from_t(t)

        B, N = T_a_BT.shape
        y_pred = np.empty((B, N, 1))
        y_pred[:, 0, :] = y_prev

        y_k = y_prev
        for k in range(1, N):
            y_k = self.calc_temperature(
                y_k, dt, T_a_BT[:, k], Phi_HC_BT[:, k], integration=integration,
            )
            y_pred[:, k, :] = y_k

        return y_pred

    def calc_energy_demand(self, *args, **kwargs):
        """Not implemented: the inherited two-probe inversion assumes the
        (T_a, Phi_sol, Phi_HC) step signature, which this model does not have."""
        raise NotImplementedError("OneR1C does not implement the heat-demand (inverse) model.")
