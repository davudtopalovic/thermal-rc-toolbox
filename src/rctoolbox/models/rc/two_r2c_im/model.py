"""Two-state grey-box 2R2C model with an internal mass, for indoor temperature
prediction of a cold store / freezer.

Topology:
                       Phi_HC (known)
                          |
          T_a ---R_ia--- T_i ---R_im--- T_m
                          |              |
                         C_i            C_m

T_i is the (measured) air node, T_m the internal mass (stored products,
shelves) that exchanges heat only with the air. Continuous-time balances:

    C_i dT_i/dt = (T_a - T_i)/R_ia + (T_m - T_i)/R_im + (1 - alpha_m) Phi_HC
    C_m dT_m/dt = (T_i - T_m)/R_im + alpha_m Phi_HC

alpha_m is the share of the cooling power extracted directly from the internal
mass. For a forced-air evaporator all cooling acts on the air (alpha_m = 0, the
default when alpha_m is not a free parameter); alpha_m > 0 only makes sense for
plate / shelf evaporators in contact with the products. Phi_HC < 0 cools
(Phi_HC = -P_el gives effective R and C values that absorb the COP).

Free parameters: R_ia, R_im, C_i, C_m and optionally alpha_m. There is no solar
input: ``calc_temperature`` and ``forecast_temperature`` take only T_a and
Phi_HC (as in ``OneR1C``).

Discretization (``integration``), same semantics as ``OneR1C`` / ``TwoR2C``:
- ``"euler"`` — explicit forward Euler (the fallback when nothing is set),
      y_k = y_{k-1} + dt * (A y_{k-1} + B u_k + E d_k);
  stable only for dt below twice the fastest time constant.
- ``"zoh"`` — exact zero-order-hold map from
  ``rctoolbox.models.discretization.zoh_discretize``,
      y_k = A_d y_{k-1} + B_d [u_k; d_k];
  exact for interval-constant inputs and unconditionally stable.
(u = [Phi_HC], d = [T_a].)

Initialization: only ``default`` — the state y_0 = [T_i_0, T_m_0] is taken as
provided; ``build_data_dict`` sets T_m_0 = -17 °C (the middle of the freezer's
thermostat band).

The inverse (heat-demand) API is not implemented for this model.
"""
import numpy as np

from rctoolbox.models.rc_base_model import RCBaseModel
from rctoolbox.models.discretization import zoh_discretize


class TwoR2CInternalMass(RCBaseModel):
    STATE_NAMES = ("T_i", "T_m")
    HIDDEN_STATE_IDX = (1,)   # T_m is not measured
    VALID_INIT_METHODS = frozenset({"default"})

    VALID_INTEGRATIONS = {"euler", "zoh"}

    def __init__(self):
        super().__init__()
        self.integration = None   # unset -> resolves to "euler"
        self._disc_cache = {}

    def set_params(self, params: dict) -> None:
        self.params = params
        self.R_ia    = params["R_ia"]
        self.R_im    = params["R_im"]
        self.C_i     = params["C_i"]
        self.C_m     = params["C_m"]
        self.alpha_m = params.get("alpha_m", 0.0)
        self._disc_cache = {}   # discrete matrices depend on the parameters

    @property
    def time_constants(self) -> np.ndarray:
        """Time constants [s] of the two modes (-1 / eigenvalues of A), fast first."""
        A, _, _ = self.calc_matrices()
        return np.sort(-1.0 / np.linalg.eigvals(A).real)

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
        """Continuous-time state-space matrices:

            dx/dt = A x + B u + E d

        with state x = [T_i, T_m], control u = [Phi_HC], and disturbance
        d = [T_a]. Returns (A, B, E) of shapes (2,2), (2,1), (2,1).
        """
        A = np.array([
            [-(1.0 / self.R_ia + 1.0 / self.R_im) / self.C_i,  1.0 / (self.R_im * self.C_i)],
            [  1.0 / (self.R_im * self.C_m),                   -1.0 / (self.R_im * self.C_m)],
        ])
        B = np.array([
            [(1.0 - self.alpha_m) / self.C_i],
            [self.alpha_m / self.C_m],
        ])
        E = np.array([
            [1.0 / (self.R_ia * self.C_i)],
            [0.0],
        ])
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
        """One discrete step.

        y_prev : ndarray, shape (B, 2) — state [T_i, T_m] at the interval start.
        dt     : float — time step [s].
        T_a, Phi_HC : ndarray, shape (B,) — inputs held constant over the
        interval (the right-labeled interval averages).
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Returns y_next : ndarray, shape (B, 2) — state at the interval end.
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

        y_prev   : ndarray, shape (B, 2) — initial state [T_i_0, T_m_0].
        t        : ndarray, shape (T,) or (B, T) — time vector [s].
        T_a_BT, Phi_HC_BT : ndarray, shape (B, T) — inputs.
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
            y_k = self.calc_temperature(
                y_k, dt, T_a_BT[:, k], Phi_HC_BT[:, k], integration=integration,
            )
            y_pred[:, k, :] = y_k

        return y_pred

    def calc_energy_demand(self, *args, **kwargs):
        """Not implemented: the inherited two-probe inversion assumes the
        (T_a, Phi_sol, Phi_HC) step signature, which this model does not have."""
        raise NotImplementedError("TwoR2CInternalMass does not implement the heat-demand (inverse) model.")
