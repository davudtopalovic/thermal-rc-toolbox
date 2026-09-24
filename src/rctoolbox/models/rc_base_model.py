"""Common base class for every reduced-order RC building model.

An RC model is a discrete-time state-space model of a building zone,

    y_k = f(y_{k-1}, dt, T_a_k, Phi_sol_k, Phi_HC_k),

whose state vector ``y`` collects the node temperatures (some measured, some
hidden) and whose inputs are the ambient temperature ``T_a``, the solar
irradiance ``Phi_sol`` and the heating/cooling power ``Phi_HC``. Everything
that only depends on this abstraction lives here; a concrete model supplies
the network itself:

Required in a subclass
----------------------
``STATE_NAMES``
    Tuple of node names in state-vector order, e.g. ``("T_m", "T_s", "T_i")``.
    It must contain ``"T_i"``: the indoor-air (thermostat) node is what the
    inverse heat-demand model regulates.
``set_params(params)``
    Store the parameter dict and derive whatever the step needs.
``calc_temperature(y_prev, dt, T_a, Phi_sol, Phi_HC, **step_kwargs)``
    The one-step forward map. Batched: ``y_prev`` is ``(B, n_states)``, the
    inputs are ``(B,)``, the result is ``(B, n_states)``.
``forecast_temperature(y_prev, t, T_a_BT, Phi_sol_BT, Phi_HC_BT, **step_kwargs)``
    The multi-step forward simulation, returning ``(B, T, n_states)``. The
    forecast loops (this one, ``forecast_heat_demand`` and its compiled
    ``forecast_heat_demand_fast`` variant) are deliberately written out on
    each model rather than inherited: they are the model's public interface
    and the identification problems call them directly. ``_dt_from_t`` is
    provided for the shared time-vector handling.

Optional hooks
--------------
``_init_static_inversion(data)``
    Reconstruct the hidden states from a quasi-steady-state balance. Needed
    for the ``static_inversion`` and ``regime_aware`` initializations.
``calc_energy_demand(...)``
    Defaults to the EN ISO 13790 two-probe interpolation, which is exact for
    any model whose one-step map is affine in ``Phi_HC``. Override when a
    closed form is cheaper (see ``TwoR2C``).
``HIDDEN_STATE_IDX``
    Which state columns the ``dynamic`` / ``regime_aware`` initializations
    take from the lag-window simulation. ``None`` (the default) replaces the
    whole state; a tuple of indices keeps the measured columns of ``y_0`` and
    replaces only the listed ones.

Data conventions
----------------
Inputs are right-labeled interval averages: the state at index ``k`` is
produced by the inputs at index ``k``, which hold the averages over
``(t_{k-1}, t_k]``. Forecast arrays are ``(B, T)`` for inputs and ``(B, T,
n_states)`` for states; ``t`` is ``(T,)`` or ``(B, T)`` in seconds and must be
uniformly spaced.

Initialization ``data`` dict
----------------------------
``y_0`` (B, n_states)
    State at the forecast start; the measured columns are already filled.
``y_0_``, ``t_``, ``T_a_``, ``Phi_sol_``, ``Phi_HC_``, ``T_i_``, ``T_i_set_``
    The same quantities over the lag window preceding the forecast start
    (trailing underscore), used to warm up or invert the hidden states.
"""
from abc import ABC, abstractmethod

import numpy as np


class RCBaseModel(ABC):
    STATE_NAMES: tuple[str, ...] = ()
    HIDDEN_STATE_IDX: tuple[int, ...] | None = None

    VALID_INIT_METHODS: frozenset[str] = frozenset(
        {"default", "dynamic", "static_inversion", "regime_aware"}
    )
    VALID_STATIC_VARIANTS: frozenset[str] = frozenset({"last", "mean"})

    # A setpoint at or below the setback threshold, or above the boost
    # threshold, marks an interval where the heat pump was intentionally
    # controlled for demand response rather than left to regulate freely.
    REGIME_SETBACK_T: float = 20.0
    REGIME_BOOST_T: float = 23.5

    # Relative step used by the default two-probe inversion when a model has
    # no more natural reference power (see ``_probe_power``).
    PROBE_EPS: float = 1e-9

    def __init__(self):
        self.params = None
        self.init_type = "default"
        self.static_inversion_method = "last"
        self.heat_power_range = None

    # ------------------------------------------------------------ structure
    @property
    def n_states(self) -> int:
        return len(self.STATE_NAMES)

    @property
    def T_i_idx(self) -> int:
        """Column of the indoor-air node in the state vector."""
        return self.STATE_NAMES.index("T_i")

    # ------------------------------------------------------------ parameters
    @abstractmethod
    def set_params(self, params: dict) -> None: ...

    def set_heat_power_range(self, heat_power_range: list) -> None:
        """Actuator limits ``[Phi_min, Phi_max]`` [W] used to clip the demand."""
        self.heat_power_range = list(heat_power_range)

    # ---------------------------------------------------------- forward map
    @abstractmethod
    def calc_temperature(self, y_prev, dt, T_a, Phi_sol, Phi_HC, **step_kwargs):
        """One discrete step: state at the interval end given the state at the
        interval start and the inputs held over the interval."""

    @staticmethod
    def _dt_from_t(t) -> tuple[np.ndarray, float]:
        """Return ``(t_1d, dt)`` for a ``(T,)`` or ``(B, T)`` time vector [s]."""
        t = np.asarray(t, dtype=np.float64)
        if t.ndim == 2:
            t = t[0]
        return t, float(t[1] - t[0])

    @abstractmethod
    def forecast_temperature(self, y_prev, t, T_a_BT, Phi_sol_BT, Phi_HC_BT, **step_kwargs):
        """Multi-step forward simulation with known heating power.

        y_prev : ndarray, shape (B, n_states) — initial state.
        t      : ndarray, shape (T,) or (B, T) — time vector [s].
        T_a_BT, Phi_sol_BT, Phi_HC_BT : ndarray, shape (B, T) — inputs.

        Returns y_pred : ndarray, shape (B, T, n_states); ``y_pred[:, 0]`` is
        ``y_prev`` and ``y_pred[:, k]`` is driven by the inputs at index k.
        Implemented on each model (see the module docstring).
        """

    # ---------------------------------------------------- inverse (demand) map
    def _probe_power(self):
        """Reference power [W] for the second probe of the two-probe inversion.
        Any nonzero value works for an affine map; the actuator maximum is the
        natural scale. EN ISO 13790 models override this with 10 W/m2 * A_f."""
        return self.heat_power_range[1]

    def calc_energy_demand(self, y_prev, dt, T_a, Phi_sol, T_i_set, **step_kwargs):
        """One step of the inverse (setpoint-to-power) model.

        EN ISO 13790 ideal-regulation inversion: the interval is simulated
        twice, with zero power and with the probe power, the required power is
        interpolated linearly from the two indoor-air responses, clipped to the
        actuator range, and the state advanced with the realized power. Exact
        whenever ``calc_temperature`` is affine in ``Phi_HC``.

        Returns ``(y, Phi_HC_demand)``: state at the interval end, shape
        (B, n_states), and realized power, shape (B,).
        """
        if self.heat_power_range is None:
            raise RuntimeError("call set_heat_power_range() before the inverse model")
        B = y_prev.shape[0]
        i = self.T_i_idx

        Phi_0 = np.zeros(B)
        T_i_0 = self.calc_temperature(y_prev, dt, T_a, Phi_sol, Phi_0, **step_kwargs)[:, i]

        Phi_p = np.full(B, float(self._probe_power()))
        T_i_p = self.calc_temperature(y_prev, dt, T_a, Phi_sol, Phi_p, **step_kwargs)[:, i]

        Phi_unres = Phi_p * (T_i_set - T_i_0) / ((T_i_p - T_i_0) + self.PROBE_EPS)
        Phi_HC_demand = np.clip(Phi_unres, self.heat_power_range[0], self.heat_power_range[1])

        y = self.calc_temperature(y_prev, dt, T_a, Phi_sol, Phi_HC_demand, **step_kwargs)
        return y, Phi_HC_demand

    # ------------------------------------------------------- initialization
    def set_initialization(self, init_config: dict) -> None:
        """
        init_config examples:
            {"method": "default"}                            # y_0 taken as provided
            {"method": "dynamic"}                            # simulate the lag window
            {"method": "static_inversion", "variant": "mean"}
            {"method": "static_inversion"}                   # variant defaults to "last"
            {"method": "regime_aware"}                       # static, dynamic under DR
        """
        if "method" not in init_config:
            raise ValueError("Initialization config must contain key 'method'")
        method = init_config["method"]
        variant = init_config.get("variant", None)

        if method not in self.VALID_INIT_METHODS:
            raise ValueError(
                f"Unknown {type(self).__name__} initialization method '{method}'. "
                f"Must be one of {sorted(self.VALID_INIT_METHODS)}"
            )
        self.init_type = method

        if method in ("static_inversion", "regime_aware"):
            if variant is None:
                variant = "last"
            if variant not in self.VALID_STATIC_VARIANTS:
                raise ValueError(
                    f"Unknown steady-state variant '{variant}'. "
                    f"Must be one of {sorted(self.VALID_STATIC_VARIANTS)}"
                )
            self.static_inversion_method = variant
        else:
            self.static_inversion_method = None

    def get_initial_state(self, data: dict, **step_kwargs) -> np.ndarray:
        """Initial state (B, n_states) for the configured method.

        ``step_kwargs`` are forwarded to the lag-window simulation of the
        dynamic/regime_aware methods so that initialization and forecasting
        can be kept under the same per-call options (e.g. ``integration``).
        """
        if self.init_type == "default":
            return data["y_0"]
        if self.init_type == "dynamic":
            return self._init_dynamic(data, **step_kwargs)
        if self.init_type == "static_inversion":
            return self._init_static_inversion(data)
        if self.init_type == "regime_aware":
            return self._init_regime_aware(data, **step_kwargs)
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
            data["Phi_sol_"][sel],
            data["Phi_HC_"][sel],
            **step_kwargs,
        )
        return y_pred_[:, -1, :]

    def _hidden_cols(self):
        return slice(None) if self.HIDDEN_STATE_IDX is None else list(self.HIDDEN_STATE_IDX)

    def _init_dynamic(self, data, **step_kwargs) -> np.ndarray:
        """Warm up the hidden states by simulating the lag window; the
        measured columns keep the values given in ``y_0`` (unless
        ``HIDDEN_STATE_IDX`` is None, in which case the whole simulated state
        is taken)."""
        y_0 = np.array(data["y_0"], dtype=np.float64)   # never mutate the caller's array
        cols = self._hidden_cols()
        y_0[:, cols] = self._simulate_lag_window(data, **step_kwargs)[:, cols]
        return y_0

    def _init_static_inversion(self, data) -> np.ndarray:
        """Reconstruct the hidden states from a quasi-steady-state balance at
        the lag-window end. Model-specific: the subclass knows its network."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement static inversion; "
            "use init method 'default' or 'dynamic'."
        )

    def _get_static_inversion_inputs(self, data, variant):
        """Operating point ``(T_i, T_a, Phi_sol, Phi_HC)`` for the static
        inversion: the last lag-window sample or the lag-window mean."""
        keys = ("T_i_", "T_a_", "Phi_sol_", "Phi_HC_")
        if variant == "last":
            return tuple(data[k][:, -1] for k in keys)
        if variant == "mean":
            return tuple(np.nanmean(data[k], axis=1) for k in keys)
        raise ValueError(
            f"Unknown steady-state variant '{variant}'. "
            f"Use one of {sorted(self.VALID_STATIC_VARIANTS)}."
        )

    def _init_regime_aware(self, data, **step_kwargs) -> np.ndarray:
        """Static inversion as the baseline; where the last lag-window
        setpoint indicates intentionally controlled (setback/boost) operation
        the steady-state assumption fails, so those batch rows take the
        hidden states from the lag-window simulation instead."""
        T_set_last = data["T_i_set_"][:, -1]
        controlled = (T_set_last <= self.REGIME_SETBACK_T) | (T_set_last > self.REGIME_BOOST_T)

        base_data = dict(data)
        base_data["y_0"] = np.array(data["y_0"], dtype=np.float64)
        y_0 = np.array(self._init_static_inversion(base_data), dtype=np.float64)

        if np.any(controlled):
            cols = self._hidden_cols()
            y_end = self._simulate_lag_window(data, mask=controlled, **step_kwargs)
            y_0[np.ix_(np.flatnonzero(controlled), np.arange(y_0.shape[1])[cols])] = y_end[:, cols]
        return y_0
