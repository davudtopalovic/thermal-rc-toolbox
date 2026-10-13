"""Shared machinery of the freezer RC models (``rctoolbox.models.rc.freezer_models``).

Every freezer model is a linear (or fan-switched linear) RC network

    dx/dt = A(s) x + B(s) u,

with state ``x`` (node temperatures, ``STATE_NAMES``; ``"T_i"`` is the measured
freezer air) and input vector ``u`` whose columns are named by ``INPUT_NAMES``
(e.g. ``("Phi_HC", "T_a")``). A concrete model only declares its structure
and writes ``calc_matrices``; discretization, simulation, initialization and
the toolbox glue (parameter space, problem, data dict) are shared here.

Switched models (``SWITCHED = True``) take the compressor/fan state
``s in [0, 1]`` as an extra input named ``"on"`` (the fraction of the interval
the compressor ran). ``calc_matrices(s)`` must be affine in ``s``; within an
interval the matrices are taken at the interval-mean ``s`` (quantized to
``S_LEVELS`` levels, so only a few matrix exponentials are needed).

Conventions (as in the rest of the toolbox)
-------------------------------------------
- Inputs are right-labeled interval averages: the input at index k drives the
  step k-1 -> k.
- Discretization is the exact zero-order hold (``zoh_discretize``); ``"euler"``
  is available as a cross-check.
- ``Phi_HC < 0`` cools. With ``Phi_HC = -P_el`` all R and C are *effective*
  values (they absorb the COP).

Initialization of the hidden states (``set_initialization``)
-------------------------------------------------------------
- ``default``: ``y_0`` as provided by ``build_data_dict`` (hidden nodes at the
  measured air temperature).
- ``dynamic``: simulate the lag window from ``y_0_`` (toolbox semantics).
- ``kalman``: a Kalman filter over the lag window with the measured T_i
  (``T_i_``); the filtered state at the lag-window end is the initial state.
  This is the proper forecast initialization: it only uses past data.
  Options ``q`` (process-noise std per step and state [K]) and ``r``
  (measurement-noise std [K]).
"""
import numpy as np

from rctoolbox.identification.parameter_space import BaseParameterSpace
from rctoolbox.models.rc_base_model import RCBaseModel
from rctoolbox.models.discretization import zoh_discretize


class FreezerRCModel(RCBaseModel):
    STATE_NAMES: tuple = ("T_i",)
    INPUT_NAMES: tuple = ("Phi_HC", "T_a")
    # parameter name -> (lower, upper) default bounds of the free parameters
    DEFAULT_BOUNDS: dict = {}
    # optional parameters: name -> (value when fixed, (lower, upper) when free)
    OPTIONAL_PARAMS: dict = {}
    PARAMS_LATEX: dict = {}
    SWITCHED: bool = False
    S_LEVELS: int = 11

    VALID_INIT_METHODS = frozenset({"default", "dynamic", "kalman"})
    VALID_INTEGRATIONS = {"euler", "zoh"}

    def __init__(self):
        super().__init__()
        self.integration = None
        self.kalman_q = 0.05
        self.kalman_r = 0.05
        self.kalman_p0 = 2.0          # K, prior std of hidden temperature states
        self.kalman_rho = 0.0         # prior correlation between hidden temperature states
        self.kalman_q_dist = 2.0      # W per step, random walk of disturbance states
        self.kalman_p0_dist = 300.0   # W, prior std of disturbance states
        self._disc_cache = {}

    # ------------------------------------------------------------ structure
    @property
    def all_inputs(self) -> tuple:
        """Data inputs the model needs (``unit`` is synthesized internally)."""
        return tuple(k for k in self.INPUT_NAMES if k != "unit") + (("on",) if self.SWITCHED else ())

    DIST_STATES = ("Q_d", "K_d")

    @property
    def dist_idx(self) -> list:
        return [i for i, n in enumerate(self.STATE_NAMES) if n in self.DIST_STATES]

    def _zero_dist(self, y0):
        y0 = np.array(y0, dtype=float)
        if self.dist_idx:
            y0[:, self.dist_idx] = 0.0
        return y0

    @property
    def hidden_idx(self) -> list:
        return [i for i, n in enumerate(self.STATE_NAMES) if n != "T_i"]

    @property
    def HIDDEN_STATE_IDX(self):   # noqa: N802 (toolbox attribute name)
        return tuple(self.hidden_idx)

    # ------------------------------------------------------------ parameters
    def set_params(self, params: dict) -> None:
        p = {k: v[0] for k, v in self.OPTIONAL_PARAMS.items()}
        p.update({k: float(v) for k, v in params.items()})
        missing = set(self.DEFAULT_BOUNDS) - set(p)
        if missing:
            raise KeyError(f"{type(self).__name__}: missing parameters {sorted(missing)}")
        self.params = p
        for k, v in p.items():
            setattr(self, k, v)
        self._disc_cache = {}

    def calc_matrices(self, s=0.0):
        """Continuous (A, B): dx/dt = A x + B u, u ordered as INPUT_NAMES."""
        raise NotImplementedError

    @property
    def time_constants(self) -> np.ndarray:
        """Time constants [s] (-1/eigenvalues of A), fast first. For switched
        models the compressor-off matrices are used."""
        A, _ = self.calc_matrices(0.0)
        return np.sort(-1.0 / np.linalg.eigvals(A).real)

    def time_constants_on(self) -> np.ndarray:
        A, _ = self.calc_matrices(1.0)
        return np.sort(-1.0 / np.linalg.eigvals(A).real)

    # --------------------------------------------------------- discretization
    def set_integration(self, integration) -> None:
        if integration is not None and integration not in self.VALID_INTEGRATIONS:
            raise ValueError(f"Unknown integration {integration!r}")
        self.integration = integration
        self._disc_cache = {}

    def _resolve_integration(self, integration):
        integration = integration if integration is not None else self.integration
        return "zoh" if integration is None else integration

    def _quantize(self, s):
        n = self.S_LEVELS - 1
        return np.clip(np.rint(np.asarray(s, dtype=float) * n), 0, n).astype(int)

    def get_discrete(self, dt, level=0, integration=None):
        """Discrete (A_d, B_d) for switch level ``level`` (0..S_LEVELS-1)."""
        integration = self._resolve_integration(integration)
        key = (dt, level, integration)
        if key not in self._disc_cache:
            s = level / (self.S_LEVELS - 1)
            A, B = self.calc_matrices(s)
            if integration == "zoh":
                self._disc_cache[key] = zoh_discretize(A, B, dt)
            else:
                n = A.shape[0]
                self._disc_cache[key] = (np.eye(n) + dt * A, dt * B)
        return self._disc_cache[key]

    def _discrete_stack(self, dt, integration=None):
        levels = range(self.S_LEVELS) if self.SWITCHED else range(1)
        mats = [self.get_discrete(dt, l, integration) for l in levels]
        return np.stack([m[0] for m in mats]), np.stack([m[1] for m in mats])

    # ----------------------------------------------------- forward simulation
    def _stack_inputs(self, inputs, B, N):
        cols = []
        for name in self.INPUT_NAMES:
            if name == "unit":                              # constant input of affine terms
                cols.append(np.ones((B, N)))
                continue
            if name not in inputs:
                raise KeyError(f"{type(self).__name__} needs input {name!r}")
            cols.append(np.broadcast_to(np.asarray(inputs[name], dtype=float), (B, N)))
        return np.stack(cols, axis=-1)                     # (B, N, m)

    def forecast_temperature(self, y_prev, t, integration=None, **inputs):
        """Multi-step simulation.

        y_prev : (B, n) initial state; t : (T,) or (B, T) time vector [s];
        inputs : (B, T) arrays named as ``all_inputs`` (``"on"`` for switched
        models). Returns (B, T, n); ``y[:, 0] = y_prev``, ``y[:, k]`` is driven
        by the inputs at index k.
        """
        t, dt = self._dt_from_t(t)
        y_prev = np.asarray(y_prev, dtype=float)
        Bsz = y_prev.shape[0]
        N = np.asarray(next(iter(inputs.values()))).shape[-1]
        U = self._stack_inputs(inputs, Bsz, N)
        Ad, Bd = self._discrete_stack(dt, integration)
        n = y_prev.shape[1]
        y = np.empty((Bsz, N, n))
        y[:, 0] = y_prev
        yk = y_prev
        if not self.SWITCHED:
            A0, B0 = Ad[0], Bd[0]
            UB = U @ B0.T                                   # (B, N, n)
            for k in range(1, N):
                yk = yk @ A0.T + UB[:, k]
                y[:, k] = yk
            return y
        lev = self._quantize(np.broadcast_to(np.asarray(inputs["on"], dtype=float), (Bsz, N)))
        for k in range(1, N):
            l = lev[:, k]
            yk = np.einsum("bij,bj->bi", Ad[l], yk) + np.einsum("bij,bj->bi", Bd[l], U[:, k])
            y[:, k] = yk
        return y

    def calc_temperature(self, y_prev, dt, integration=None, **inputs):
        """One discrete step; inputs are (B,) arrays named as ``all_inputs``."""
        t = np.array([0.0, dt])
        two = {k: np.stack([np.asarray(v, float), np.asarray(v, float)], axis=1) for k, v in inputs.items()}
        return self.forecast_temperature(y_prev, t, integration=integration, **two)[:, 1]

    # ------------------------------------------------------- initialization
    def set_initialization(self, init_config: dict) -> None:
        method = init_config.get("method", "default")
        if method not in self.VALID_INIT_METHODS:
            raise ValueError(f"Unknown init method {method!r}; use {sorted(self.VALID_INIT_METHODS)}")
        self.init_type = method
        self.kalman_q = float(init_config.get("q", self.kalman_q))
        self.kalman_r = float(init_config.get("r", self.kalman_r))
        self.kalman_q_dist = float(init_config.get("q_dist", self.kalman_q_dist))
        self.kalman_p0 = float(init_config.get("p0", self.kalman_p0))
        self.kalman_rho = float(init_config.get("rho", self.kalman_rho))
        self.kalman_p0_dist = float(init_config.get("p0_dist", self.kalman_p0_dist))

    def get_initial_state(self, data: dict, **step_kwargs) -> np.ndarray:
        if self.init_type == "default" or not self.hidden_idx:
            return self._zero_dist(data["y_0"])
        if self.init_type == "dynamic":
            data = dict(data, y_0=self._zero_dist(data["y_0"]), y_0_=self._zero_dist(data["y_0_"]))
            return self._init_dynamic(data, **step_kwargs)
        if self.init_type == "kalman":
            return self._init_kalman(data, **step_kwargs)
        raise ValueError(self.init_type)

    def _lag_inputs(self, data):
        return {k: data[f"{k}_"] for k in self.all_inputs}

    def _simulate_lag_window(self, data, mask=None, **step_kwargs):
        y_ = self.forecast_temperature(np.asarray(data["y_0_"], float), data["t_"],
                                       **self._lag_inputs(data), **step_kwargs)
        return y_[:, -1, :]

    def _init_kalman(self, data, integration=None):
        """Kalman filter over the lag window. NaN measurements (or inputs) at
        the start of the lag window are skipped: the filter starts at the
        first valid sample of each row with P_0 = diag(0 for T_i, 2^2 hidden)."""
        y0_ = self._zero_dist(data["y_0_"])
        Ti_ = np.asarray(data["T_i_"], float)
        t_, dt = self._dt_from_t(data["t_"])
        Bsz, N = Ti_.shape
        n = self.n_states
        U = self._stack_inputs(self._lag_inputs(data), Bsz, N)
        Ad, Bd = self._discrete_stack(dt, integration)
        lev = (self._quantize(np.broadcast_to(np.nan_to_num(np.asarray(data["on_"], float)), (Bsz, N)))
               if self.SWITCHED else np.zeros((Bsz, N), dtype=int))
        valid = np.isfinite(Ti_) & np.all(np.isfinite(U), axis=-1)
        if self.SWITCHED:
            valid &= np.isfinite(np.asarray(data["on_"], float))
        iT = self.T_i_idx
        x = y0_.copy()
        hid = self.hidden_idx
        dist = self.dist_idx
        p0 = np.zeros(n)
        p0[hid] = self.kalman_p0 ** 2
        p0[dist] = self.kalman_p0_dist ** 2
        P = np.repeat(np.diag(p0)[None], Bsz, axis=0)
        qd = np.full(n, self.kalman_q ** 2)
        qd[dist] = self.kalman_q_dist ** 2
        Q = np.diag(qd)
        R = self.kalman_r ** 2
        if self.kalman_rho:
            # correlated prior of the temperature hidden states
            th = [i for i in hid if i not in dist]
            for a in th:
                for c in th:
                    if a != c:
                        P[:, a, c] = self.kalman_rho * self.kalman_p0 ** 2
        p0full = P[0].copy()
        started = valid[:, 0].copy()
        x[started, iT] = Ti_[started, 0]
        for k in range(1, N):
            l = lev[:, k]
            go = started & valid[:, k]
            if go.any():
                A = Ad[l[go]]
                Bm = Bd[l[go]]
                xp = np.einsum("bij,bj->bi", A, x[go]) + np.einsum("bij,bj->bi", Bm, np.nan_to_num(U[go, k]))
                Pp = A @ P[go] @ np.transpose(A, (0, 2, 1)) + Q
                S = Pp[:, iT, iT] + R
                K = Pp[:, :, iT] / S[:, None]
                innov = Ti_[go, k] - xp[:, iT]
                x[go] = xp + K * innov[:, None]
                P[go] = Pp - K[:, :, None] * Pp[:, iT, :][:, None, :]
            new = (~started) & valid[:, k]
            if new.any():
                x[new] = Ti_[new, k][:, None]
                if dist:
                    x[np.ix_(np.flatnonzero(new), dist)] = 0.0
                P[new] = p0full
                started |= new
        y0 = self._zero_dist(data["y_0"])
        y0[:, hid] = x[:, hid]
        return y0

    def calc_energy_demand(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} has no inverse (heat-demand) model.")


# ---------------------------------------------------------------- toolbox glue
def make_parameter_space(model_cls):
    """ParameterSpace class for a FreezerRCModel subclass. Free parameters are
    the keys of ``building_config["free_param_bounds"]`` (defaults:
    ``DEFAULT_BOUNDS``); optional parameters are free only if listed."""

    class _PS(BaseParameterSpace):
        MODEL = model_cls

        def __init__(self, building_config):
            bounds = building_config.get("free_param_bounds") or model_cls.DEFAULT_BOUNDS
            valid = set(model_cls.DEFAULT_BOUNDS) | set(model_cls.OPTIONAL_PARAMS)
            unknown = set(bounds) - valid
            if unknown:
                raise ValueError(f"Unknown {model_cls.__name__} parameters {sorted(unknown)}; valid {sorted(valid)}")
            missing = set(model_cls.DEFAULT_BOUNDS) - set(bounds)
            if missing:
                raise ValueError(f"free_param_bounds must contain {sorted(missing)}")
            self.building_config = building_config
            self.free_param_bounds = {k: tuple(v) for k, v in bounds.items()}
            self.free_param_names = list(self.free_param_bounds)
            self.params_latex = {k: model_cls.PARAMS_LATEX.get(k, k) for k in self.free_param_names}

        @classmethod
        def from_building_config(cls, building_config):
            return cls(building_config)

    _PS.__name__ = f"ParameterSpace_{model_cls.__name__}"
    return _PS


def initial_state(model, T_i):
    """y_0 (B, n): T_i measured at the window start, hidden nodes = T_i_0."""
    T_i0 = np.asarray(T_i, dtype=float)[:, 0]
    return np.repeat(T_i0[:, None], model.n_states, axis=1)


def build_data_dict(model, dataloader, split_name, column_map=None):
    """Data dict for ``FreezerTemperatureProblem`` from a toolbox DataLoader
    batch. ``column_map`` maps model input names to dataloader features
    (default: identical names, measured air ``"y1"``)."""
    cmap = {"T_i": "y1"} | (column_map or {})
    b = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]
    col = lambda k: cmap.get(k, k)
    d = {"t": b["t_array"], "T_i": di[col("T_i")]}
    init = {"y_0": initial_state(model, di[col("T_i")]),
            "y_0_": initial_state(model, di[f"{col('T_i')}_1Dlag"]),
            "t_": di["t_array_1Dlag"], "T_i_": di[f"{col('T_i')}_1Dlag"]}
    for k in model.all_inputs:
        d[k] = di[col(k)]
        init[f"{k}_"] = di[f"{col(k)}_1Dlag"]
    d["data_for_init_state"] = init
    return d


def predict_for_batch(model, batch, raw_dfs, column_map=None):
    """Simulate every window of a toolbox DataLoader batch; adds ``<state>_pred``
    columns (``y1_pred`` for the air) to copies of the raw DataFrames."""
    import pandas as pd
    cmap = {"T_i": "y1"} | (column_map or {})
    di = batch["data_inputs"]
    col = lambda k: cmap.get(k, k)
    init = {"y_0": initial_state(model, di[col("T_i")]),
            "y_0_": initial_state(model, di[f"{col('T_i')}_1Dlag"]),
            "t_": di["t_array_1Dlag"], "T_i_": di[f"{col('T_i')}_1Dlag"]}
    for k in model.all_inputs:
        init[f"{k}_"] = di[f"{col(k)}_1Dlag"]
    y0 = model.get_initial_state(init)
    Y = model.forecast_temperature(y0, batch["t_array"], **{k: di[col(k)] for k in model.all_inputs})
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, bdf in enumerate(batch_dfs.values()):
        bdf["y1_pred"] = Y[i, :, model.T_i_idx]
        for j, name in enumerate(model.STATE_NAMES):
            if name != "T_i":
                bdf[f"{name}_pred"] = Y[i, :, j]
    dfbs = pd.concat(list(batch_dfs.values()), axis=0).dropna()
    return dfbs, batch_dfs
