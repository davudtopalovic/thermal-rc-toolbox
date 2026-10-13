"""Four-state grey-box 5R4C model with two envelope nodes and an internal
mass, for indoor temperature prediction of a cold store / freezer.

Topology:
                                    Phi_HC (known)
                                       |
     T_a ---R_e1a--- T_e1 ---R_ie1---+ |
                      |              | |
                     C_e1            T_i ---R_im--- T_m
                                     | |             |
     T_out -R_e2out- T_e2 ---R_ie2---+ C_i          C_m
                      |
                     C_e2

The air T_i exchanges heat with two envelope parts: T_e1 (the part of the
envelope facing the surrounding room T_a) and T_e2 (the part facing the
outdoor air T_out). T_m is the internal mass (stored products, shelves), which
exchanges heat only with the air. All cooling power acts on the air.
Continuous-time balances:

    C_i  dT_i/dt  = (T_e1 - T_i)/R_ie1 + (T_e2 - T_i)/R_ie2
                    + (T_m - T_i)/R_im + Phi_HC
    C_e1 dT_e1/dt = (T_i - T_e1)/R_ie1 + (T_a - T_e1)/R_e1a
    C_e2 dT_e2/dt = (T_i - T_e2)/R_ie2 + (T_out - T_e2)/R_e2out
    C_m  dT_m/dt  = (T_i - T_m)/R_im

Phi_HC < 0 cools (Phi_HC = -P_el gives effective R and C values that absorb
the COP).

Free parameters: R_ie1, R_e1a, R_ie2, R_e2out, R_im, C_i, C_e1, C_e2, C_m.
Inputs: u = [Phi_HC], d = [T_a, T_out]; ``calc_temperature`` and
``forecast_temperature`` take T_a, Phi_HC and T_out (no solar input).

Discretization (``integration``), same semantics as ``OneR1C``:
- ``"euler"`` — explicit forward Euler (the fallback when nothing is set),
      y_k = y_{k-1} + dt * (A y_{k-1} + B u_k + E d_k);
  stable only for dt below twice the fastest time constant, which the air
  node can easily violate — prefer "zoh";
- ``"zoh"`` — exact zero-order-hold map,
      y_k = A_d y_{k-1} + B_d [u_k; d_k].

Initialization (hidden states T_e1, T_e2, T_m):
- ``default`` — y_0 as provided (``build_data_dict``: T_e1, T_e2 at the
  midpoints between T_i and T_a / T_out, T_m_0 = T_i_0 or a constant);
- ``static_inversion`` — T_e1 and T_e2 from the steady state of their series
  paths, T_e1 = (R_e1a T_i + R_ie1 T_a) / (R_ie1 + R_e1a) (likewise T_e2 with
  R_ie2, R_e2out, T_out), with T_i from y_0 and T_a, T_out the last
  ("last") or mean ("mean") lag-window values; T_m is kept from y_0;
- ``dynamic`` — the hidden states are warmed up by simulating the lag window.

The derivation of the balances and the state-space matrices is in
``derivation.ipynb`` next to this file. The inverse (heat-demand) API is not
implemented for this model.
"""
import numpy as np

from rctoolbox.models.rc_base_model import RCBaseModel
from rctoolbox.models.discretization import zoh_discretize


class FiveR4CInternalMass(RCBaseModel):
    STATE_NAMES = ("T_i", "T_e1", "T_e2", "T_m")
    HIDDEN_STATE_IDX = (1, 2, 3)   # only T_i is measured
    VALID_INIT_METHODS = frozenset({"default", "static_inversion", "dynamic"})

    VALID_INTEGRATIONS = {"euler", "zoh"}

    def __init__(self):
        super().__init__()
        self.integration = None   # unset -> resolves to "euler"
        self._disc_cache = {}

    def set_params(self, params: dict) -> None:
        self.params  = params
        self.R_ie1   = params["R_ie1"]
        self.R_e1a   = params["R_e1a"]
        self.R_ie2   = params["R_ie2"]
        self.R_e2out = params["R_e2out"]
        self.R_im    = params["R_im"]
        self.C_i     = params["C_i"]
        self.C_e1    = params["C_e1"]
        self.C_e2    = params["C_e2"]
        self.C_m     = params["C_m"]
        self._disc_cache = {}   # discrete matrices depend on the parameters

    @property
    def time_constants(self) -> np.ndarray:
        """Time constants [s] of the four modes (-1 / eigenvalues of A), fast first."""
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

        with state x = [T_i, T_e1, T_e2, T_m], control u = [Phi_HC], and
        disturbances d = [T_a, T_out]. Returns (A, B, E) of shapes (4,4),
        (4,1), (4,2).
        """
        g_ie1, g_e1a = 1.0 / self.R_ie1, 1.0 / self.R_e1a
        g_ie2, g_e2o = 1.0 / self.R_ie2, 1.0 / self.R_e2out
        g_im = 1.0 / self.R_im
        C_i, C_e1, C_e2, C_m = self.C_i, self.C_e1, self.C_e2, self.C_m

        A = np.array([
            [-(g_ie1 + g_ie2 + g_im) / C_i, g_ie1 / C_i,               g_ie2 / C_i,               g_im / C_i],
            [  g_ie1 / C_e1,               -(g_ie1 + g_e1a) / C_e1,    0.0,                       0.0],
            [  g_ie2 / C_e2,                0.0,                      -(g_ie2 + g_e2o) / C_e2,    0.0],
            [  g_im / C_m,                  0.0,                       0.0,                      -g_im / C_m],
        ])
        B = np.array([
            [1.0 / C_i],
            [0.0],
            [0.0],
            [0.0],
        ])
        E = np.array([
            [0.0,          0.0],
            [g_e1a / C_e1, 0.0],
            [0.0,          g_e2o / C_e2],
            [0.0,          0.0],
        ])
        return A, B, E

    def get_zoh_matrices(self, dt):
        """Exact discrete (ZOH) matrices (A_d, B_d) for the current parameters
        and time step; B_d multiplies the stacked input [Phi_HC, T_a, T_out].
        Cached per (parameters, dt)."""
        if dt not in self._disc_cache:
            A, B, E = self.calc_matrices()
            self._disc_cache[dt] = zoh_discretize(A, np.hstack([B, E]), dt)
        return self._disc_cache[dt]

    # ----------------------------------------------------- forward simulation
    def calc_temperature(self, y_prev, dt, T_a, Phi_HC, T_out, integration=None):
        """One discrete step.

        y_prev : ndarray, shape (B, 4) — state [T_i, T_e1, T_e2, T_m] at the
        interval start.
        dt     : float — time step [s].
        T_a, Phi_HC, T_out : ndarray, shape (B,) — inputs held constant over
        the interval (the right-labeled interval averages).
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Returns y_next : ndarray, shape (B, 4) — state at the interval end.
        """
        integration = self._resolve_integration(integration)
        u = np.stack([Phi_HC, T_a, T_out], axis=1)  # stacked input [u; d]

        if integration == "euler":
            A, B, E = self.calc_matrices()
            dydt = y_prev @ A.T + u @ np.hstack([B, E]).T
            return y_prev + dt * dydt

        # zoh
        Ad, Bd = self.get_zoh_matrices(dt)
        return y_prev @ Ad.T + u @ Bd.T

    def forecast_temperature(self, y_prev, t, T_a_BT, Phi_HC_BT, T_out_BT, integration=None):
        """Multi-step forecast via the discrete one-step map.

        y_prev   : ndarray, shape (B, 4) — initial state.
        t        : ndarray, shape (T,) or (B, T) — time vector [s].
        T_a_BT, Phi_HC_BT, T_out_BT : ndarray, shape (B, T) — inputs.
        integration : 'euler' | 'zoh' | None (model default; unset -> 'euler').

        Returns y_pred : ndarray, shape (B, T, 4) — state trajectory. The state
        at index k is computed with the inputs at index k (right-labeled).
        """
        t, dt = self._dt_from_t(t)

        B, N = T_a_BT.shape
        y_pred = np.empty((B, N, self.n_states))
        y_pred[:, 0, :] = y_prev

        y_k = y_prev
        for k in range(1, N):
            y_k = self.calc_temperature(
                y_k, dt, T_a_BT[:, k], Phi_HC_BT[:, k], T_out_BT[:, k],
                integration=integration,
            )
            y_pred[:, k, :] = y_k

        return y_pred

    # ------------------------------------------------------- initialization
    def _simulate_lag_window(self, data, mask=None, **step_kwargs) -> np.ndarray:
        """State at the end of the lag window (no solar input, T_out added)."""
        sel = slice(None) if mask is None else mask
        y_pred_ = self.forecast_temperature(
            np.array(data["y_0_"][sel], dtype=np.float64),
            data["t_"],
            data["T_a_"][sel],
            data["Phi_HC_"][sel],
            data["T_out_"][sel],
            **step_kwargs,
        )
        return y_pred_[:, -1, :]

    def _init_static_inversion(self, data) -> np.ndarray:
        """Envelope nodes from the steady state of their series paths
        T_i - R_ie - T_e - R_e - T_amb; T_i and T_m are kept from y_0."""
        y_0 = np.array(data["y_0"], dtype=np.float64)   # never mutate the caller's array
        if self.static_inversion_method == "mean":
            T_a, T_out = np.nanmean(data["T_a_"], axis=1), np.nanmean(data["T_out_"], axis=1)
        else:
            T_a, T_out = data["T_a_"][:, -1], data["T_out_"][:, -1]
        T_i = y_0[:, 0]
        y_0[:, 1] = (self.R_e1a * T_i + self.R_ie1 * T_a) / (self.R_ie1 + self.R_e1a)
        y_0[:, 2] = (self.R_e2out * T_i + self.R_ie2 * T_out) / (self.R_ie2 + self.R_e2out)
        return y_0

    def calc_energy_demand(self, *args, **kwargs):
        """Not implemented for this model."""
        raise NotImplementedError("FiveR4CInternalMass does not implement the heat-demand (inverse) model.")
