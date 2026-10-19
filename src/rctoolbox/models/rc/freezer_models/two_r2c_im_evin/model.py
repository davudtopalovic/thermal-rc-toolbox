"""Two-state grey-box 2R2C model of a freezer with an internal mass, driven by
the measured evaporator (coil) temperature instead of the cooling power.

Standalone implementation: the matrices are written out explicitly here and
nothing is taken from the configurable ``freezer_models/network.py``. It is the
same model as ``fz_2r2c_im_evin.Freezer2R2C_IM_EVIN``.

Topology:

          T_a ---R_ia--- T_i ---g(s)--- T_ev (measured)
                          |  \\
                         C_i  R_im
                                \\
                                 T_m --- C_m

T_i is the (measured) freezer air, T_m the internal mass (stored products,
shelves, inner surfaces) that exchanges heat only with the air. The coil is
not a state: its measured temperature T_ev is an input. Continuous-time
balances:

    C_i dT_i/dt = (T_a - T_i)/R_ia + (T_m - T_i)/R_im + g(s) (T_ev - T_i)
    C_m dT_m/dt = (T_i - T_m)/R_im

    g(s) = s / R_ie_on + (1 - s) / R_ie_off

s in [0, 1] is the fraction of the interval the compressor (and the
evaporator fans) ran (input ``on``). With the fans on the coil-air coupling is
forced convection (small R_ie_on); with the fans off only natural convection
remains (R_ie_off >> R_ie_on). g(s) is the time average of the two
conductances over the interval. The cooling power and the COP do not appear:
the model is COP-free.

Free parameters: R_ia, C_i, R_im, C_m, R_ie_on, R_ie_off.
Inputs: T_a, T_ev, on (all (B, T), right-labeled interval averages).

Discretization (``integration``):
- ``"zoh"`` (the fallback when nothing is set) — exact zero-order-hold map
  from ``rctoolbox.models.discretization.zoh_discretize``,
      y_k = A_d(s_k) y_{k-1} + B_d(s_k) d_k,   d = [T_a, T_ev];
  the matrices depend on s, which is rounded to ``S_LEVELS`` levels so only a
  few matrix exponentials are needed (cached per parameters and dt).
- ``"euler"`` — explicit forward Euler, a cross-check only: with the fans on
  the fast time constant is a few minutes, so Euler is unstable for coarse dt.

Initialization (``set_initialization``):
- ``default``: y_0 as provided; ``build_data_dict`` sets T_m_0 = -17 °C.
- ``dynamic``: simulate the lag window from ``y_0_`` and take T_m at its end.
- ``kalman``: a Kalman filter over the lag window with the measured T_i
  (``T_i_``); T_m at the lag-window end is the filtered estimate. Options
  ``q`` (process-noise std per step [K]), ``r`` (measurement-noise std [K])
  and ``p0`` (prior std of T_m [K]). NaN samples at the start of the lag
  window are skipped.

The inverse (heat-demand) API is not implemented for this model.
"""
import numpy as np

from rctoolbox.models.rc_base_model import RCBaseModel
from rctoolbox.models.discretization import zoh_discretize


class TwoR2CInternalMassEvIn(RCBaseModel):
    STATE_NAMES = ("T_i", "T_m")
    HIDDEN_STATE_IDX = (1,)   # T_m is not measured
    INPUT_NAMES = ("T_a", "T_ev", "on")
    VALID_INIT_METHODS = frozenset({"default", "dynamic", "kalman"})

    VALID_INTEGRATIONS = {"euler", "zoh"}
    S_LEVELS = 11             # quantization of the on-fraction s

    def __init__(self):
        super().__init__()
        self.integration = None   # unset -> resolves to "zoh"
        self.kalman_q = 0.05
        self.kalman_r = 0.05
        self.kalman_p0 = 2.0
        self._disc_cache = {}

    def set_params(self, params: dict) -> None:
        self.params = params
        self.R_ia     = params["R_ia"]
        self.C_i      = params["C_i"]
        self.R_im     = params["R_im"]
        self.C_m      = params["C_m"]
        self.R_ie_on  = params["R_ie_on"]
        self.R_ie_off = params["R_ie_off"]
        self._disc_cache = {}   # discrete matrices depend on the parameters

    def g_ie(self, s):
        """Air-coil conductance [W/K] at on-fraction s."""
        return s / self.R_ie_on + (1.0 - s) / self.R_ie_off

    @property
    def time_constants(self) -> np.ndarray:
        """Time constants [s] with the compressor off (-1 / eigenvalues of A(0)), fast first."""
        A, _ = self.calc_matrices(0.0)
        return np.sort(-1.0 / np.linalg.eigvals(A).real)

    def time_constants_on(self) -> np.ndarray:
        """Time constants [s] with the compressor on (A(1)), fast first."""
        A, _ = self.calc_matrices(1.0)
        return np.sort(-1.0 / np.linalg.eigvals(A).real)

    # --------------------------------------------------------- discretization
    def set_integration(self, integration) -> None:
        """Set the model-level default integration ('euler' or 'zoh').
        None clears it back to unset, which resolves to 'zoh'."""
        if integration is not None and integration not in self.VALID_INTEGRATIONS:
            raise ValueError(
                f"Unknown integration '{integration}'. "
                f"Must be one of {sorted(self.VALID_INTEGRATIONS)} or None"
            )
        self.integration = integration
        self._disc_cache = {}

    def _resolve_integration(self, integration):
        """The argument wins, otherwise the model-level setting; if neither is
        set (None), fall back to 'zoh'."""
        integration = integration if integration is not None else self.integration
        if integration is None:
            return "zoh"
        if integration not in self.VALID_INTEGRATIONS:
            raise ValueError(
                f"Unknown integration '{integration}'. "
                f"Must be one of {sorted(self.VALID_INTEGRATIONS)}"
            )
        return integration

    def calc_matrices(self, s=0.0):
        """Continuous-time state-space matrices at on-fraction s:

            dx/dt = A(s) x + E(s) d

        with state x = [T_i, T_m] and inputs d = [T_a, T_ev].
        Returns (A, E) of shapes (2, 2), (2, 2).
        """
        g = self.g_ie(s)
        A = np.array([
            [-(1.0 / self.R_ia + 1.0 / self.R_im + g) / self.C_i,  1.0 / (self.R_im * self.C_i)],
            [  1.0 / (self.R_im * self.C_m),                       -1.0 / (self.R_im * self.C_m)],
        ])
        E = np.array([
            [1.0 / (self.R_ia * self.C_i), g / self.C_i],
            [0.0,                          0.0],
        ])
        return A, E

    def _quantize(self, s):
        """On-fraction s -> level index 0..S_LEVELS-1."""
        n = self.S_LEVELS - 1
        return np.clip(np.rint(np.asarray(s, dtype=np.float64) * n), 0, n).astype(int)

    def get_discrete_matrices(self, dt, integration=None):
        """Discrete matrices for every s level: (A_d, B_d) of shapes
        (S_LEVELS, 2, 2) each; B_d multiplies d = [T_a, T_ev].
        Cached per (parameters, dt, integration)."""
        integration = self._resolve_integration(integration)
        key = (dt, integration)
        if key not in self._disc_cache:
            Ad, Bd = [], []
            for level in range(self.S_LEVELS):
                A, E = self.calc_matrices(level / (self.S_LEVELS - 1))
                if integration == "zoh":
                    a, b = zoh_discretize(A, E, dt)
                else:
                    a, b = np.eye(2) + dt * A, dt * E
                Ad.append(a)
                Bd.append(b)
            self._disc_cache[key] = (np.stack(Ad), np.stack(Bd))
        return self._disc_cache[key]

    # ----------------------------------------------------- forward simulation
    def calc_temperature(self, y_prev, dt, T_a, T_ev, on, integration=None):
        """One discrete step.

        y_prev : ndarray, shape (B, 2) — state [T_i, T_m] at the interval start.
        dt     : float — time step [s].
        T_a, T_ev, on : ndarray, shape (B,) — inputs held constant over the
        interval (the right-labeled interval averages).
        integration : 'euler' | 'zoh' | None (model default; unset -> 'zoh').

        Returns y_next : ndarray, shape (B, 2) — state at the interval end.
        """
        Ad, Bd = self.get_discrete_matrices(dt, integration)
        lev = self._quantize(on)
        d = np.stack([T_a, T_ev], axis=1)
        return np.einsum("bij,bj->bi", Ad[lev], y_prev) + np.einsum("bij,bj->bi", Bd[lev], d)

    def forecast_temperature(self, y_prev, t, T_a_BT, T_ev_BT, on_BT, integration=None):
        """Multi-step forecast via the discrete one-step map.

        y_prev  : ndarray, shape (B, 2) — initial state [T_i_0, T_m_0].
        t       : ndarray, shape (T,) or (B, T) — time vector [s].
        T_a_BT, T_ev_BT, on_BT : ndarray, shape (B, T) — inputs.
        integration : 'euler' | 'zoh' | None (model default; unset -> 'zoh').

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
                y_k, dt, T_a_BT[:, k], T_ev_BT[:, k], on_BT[:, k], integration=integration,
            )
            y_pred[:, k, :] = y_k

        return y_pred

    # ------------------------------------------------------- initialization
    def set_initialization(self, init_config: dict) -> None:
        """
        init_config examples:
            {"method": "default"}                            # y_0 taken as provided
            {"method": "dynamic"}                            # simulate the lag window
            {"method": "kalman", "q": 0.01, "r": 0.03, "p0": 1.0}
        """
        if "method" not in init_config:
            raise ValueError("Initialization config must contain key 'method'")
        method = init_config["method"]
        if method not in self.VALID_INIT_METHODS:
            raise ValueError(
                f"Unknown {type(self).__name__} initialization method '{method}'. "
                f"Must be one of {sorted(self.VALID_INIT_METHODS)}"
            )
        self.init_type = method
        self.kalman_q = float(init_config.get("q", self.kalman_q))
        self.kalman_r = float(init_config.get("r", self.kalman_r))
        self.kalman_p0 = float(init_config.get("p0", self.kalman_p0))

    def get_initial_state(self, data: dict, **step_kwargs) -> np.ndarray:
        if self.init_type == "default":
            return data["y_0"]
        if self.init_type == "dynamic":
            return self._init_dynamic(data, **step_kwargs)
        if self.init_type == "kalman":
            return self._init_kalman(data, **step_kwargs)
        raise ValueError(
            f"Unknown initialization method '{self.init_type}'. "
            f"Use one of {sorted(self.VALID_INIT_METHODS)}."
        )

    def _simulate_lag_window(self, data, mask=None, **step_kwargs) -> np.ndarray:
        """State at the end of the lag window, simulated from the lag-window
        start with the measured inputs. ``mask`` restricts the batch."""
        sel = slice(None) if mask is None else mask
        y_pred_ = self.forecast_temperature(
            np.array(data["y_0_"][sel], dtype=np.float64),
            data["t_"],
            data["T_a_"][sel],
            data["T_ev_"][sel],
            data["on_"][sel],
            **step_kwargs,
        )
        return y_pred_[:, -1, :]

    def _init_kalman(self, data, integration=None) -> np.ndarray:
        """Kalman filter over the lag window with the measured T_i.

        State [T_i, T_m], measurement T_i, P_0 = diag(0, p0^2),
        Q = q^2 I, R = r^2. A row starts at its first sample where T_i and the
        inputs are all finite (T_i set to the measurement, T_m to y_0_, or to
        the measured T_i if y_0_ has no finite T_m); later NaN samples are
        skipped (no prediction, no update). Returns y_0 with
        T_m replaced by the filtered estimate at the lag-window end.
        """
        T_i_ = np.asarray(data["T_i_"], dtype=np.float64)
        _, dt = self._dt_from_t(data["t_"])
        B, N = T_i_.shape
        d = np.stack([np.asarray(data["T_a_"], dtype=np.float64),
                      np.asarray(data["T_ev_"], dtype=np.float64)], axis=-1)   # (B, N, 2)
        on = np.asarray(data["on_"], dtype=np.float64)
        Ad, Bd = self.get_discrete_matrices(dt, integration)
        lev = self._quantize(np.nan_to_num(on))
        valid = np.isfinite(T_i_) & np.all(np.isfinite(d), axis=-1) & np.isfinite(on)

        x = np.array(data["y_0_"], dtype=np.float64)
        P0 = np.diag([0.0, self.kalman_p0 ** 2])
        P = np.repeat(P0[None], B, axis=0)
        Q = np.eye(2) * self.kalman_q ** 2
        R = self.kalman_r ** 2

        started = valid[:, 0].copy()
        x[started, 0] = T_i_[started, 0]
        for k in range(1, N):
            go = started & valid[:, k]
            if go.any():
                A = Ad[lev[go, k]]
                xp = np.einsum("bij,bj->bi", A, x[go]) + np.einsum("bij,bj->bi", Bd[lev[go, k]], d[go, k])
                Pp = A @ P[go] @ np.transpose(A, (0, 2, 1)) + Q
                K = Pp[:, :, 0] / (Pp[:, 0, 0] + R)[:, None]          # gain (b, 2)
                innov = T_i_[go, k] - xp[:, 0]
                x[go] = xp + K * innov[:, None]
                P[go] = Pp - K[:, :, None] * Pp[:, 0, :][:, None, :]
            new = ~started & valid[:, k]
            if new.any():
                x[new, 0] = T_i_[new, k]
                x[new, 1] = np.where(np.isfinite(x[new, 1]), x[new, 1], T_i_[new, k])   # no T_m_0 given: start at T_i
                P[new] = P0
                started |= new

        y_0 = np.array(data["y_0"], dtype=np.float64)   # never mutate the caller's array
        y_0[:, 1] = x[:, 1]
        return y_0

    def calc_energy_demand(self, *args, **kwargs):
        """Not implemented: the model has no cooling-power input to invert."""
        raise NotImplementedError("TwoR2CInternalMassEvIn does not implement the heat-demand (inverse) model.")
