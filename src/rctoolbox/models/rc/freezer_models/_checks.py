"""Numerical checks used by the derivation notebooks of the freezer models."""
import numpy as np

PLAUSIBLE = {
    "R_ia": (0.02, 0.1), "C_i": (1e4, 5e5), "R_im": (1e-3, 0.05), "C_m": (1e5, 5e6),
    "R_ie_on": (1e-3, 0.02), "R_ie_off": (0.01, 1.0), "C_e": (1e3, 1e5), "R_ea": (0.1, 10.0),
    "R_iw": (1e-3, 0.05), "R_wa": (0.01, 0.2), "C_w": (1e5, 1e7), "R_iout": (0.5, 20.0),
    "R_isoil": (0.5, 20.0), "Q_0": (-100.0, 200.0), "beta": (-0.03, 0.01), "alpha_m": (0.0, 0.5),
}


def random_params(model, rng):
    p = {}
    for k in model.DEFAULT_BOUNDS:
        a, b = PLAUSIBLE[k]
        p[k] = float(np.exp(rng.uniform(np.log(a), np.log(b)))) if a > 0 else float(rng.uniform(a, b))
    return p


def _inputs(model, rng, B, N, on=None):
    on = (rng.random((B, N)) > 0.6).astype(float) if on is None else on
    T_out = 10 + 5 * rng.normal(size=(B, N))
    d = {"T_a": 5 + rng.normal(size=(B, N)), "Phi_HC": -2350.0 * on, "on": on, "T_out": T_out,
         "T_ev": np.where(on > 0, -26.0, -15.0) + rng.normal(size=(B, N)), "T_soil": np.full((B, N), 12.0)}
    d["Phi_HC_Tout"] = d["Phi_HC"] * (T_out - 10.0)
    return {k: d[k] for k in model.all_inputs}


def check_zoh_vs_euler(model, params, dt=120.0, N=61, fine=1000, seed=1):
    """Max |ZOH - forward Euler with dt/fine| over N steps (K)."""
    rng = np.random.default_rng(seed)
    model.set_params(params)
    inp = _inputs(model, rng, 3, N)
    y0 = np.full((3, model.n_states), -18.0)
    t = np.arange(N) * dt
    model.set_integration("zoh")
    z = model.forecast_temperature(y0, t, **inp)
    rep = lambda a: np.concatenate([a[:, :1], np.repeat(a[:, 1:], fine, axis=1)], axis=1)
    model.set_integration("euler")
    e = model.forecast_temperature(y0, np.arange((N - 1) * fine + 1) * dt / fine,
                                   **{k: rep(v) for k, v in inp.items()})[:, ::fine]
    model.set_integration("zoh")
    return float(np.abs(z - e).max())


def check_steady_state(model, params, s=0.0, seed=2):
    """Max |x(t -> oo) - (-A^-1 B u)| for constant inputs (compressor state s)."""
    rng = np.random.default_rng(seed)
    model.set_params(params)
    N = 2000
    on = np.full((1, N), s)
    inp = {k: np.repeat(v[:, :1], N, axis=1) for k, v in _inputs(model, rng, 1, N, on=on).items()}
    A, B = model.calc_matrices(s)
    u = np.array([1.0 if k == "unit" else inp[k][0, 0] for k in model.INPUT_NAMES])
    x_ss = -np.linalg.solve(A, B @ u)
    dt = 20 * max(model.time_constants.max(), 3600.0) / N
    y = model.forecast_temperature(np.full((1, model.n_states), -18.0), np.arange(N) * dt, **inp)
    return float(np.abs(y[0, -1] - x_ss).max())


def check_reduction(model, parent, extra, rng, N=61, dt=120.0):
    """Max |T_i(model with ``extra`` params) - T_i(parent)|: the extra element
    removed (e.g. R_im -> oo) must reproduce the parent model."""
    p = random_params(parent, rng)
    parent.set_params(p)
    model.set_params(p | extra | {k: v for k, v in random_params(model, rng).items()
                                  if k not in p and k not in extra})
    inp = _inputs(model, rng, 3, N)
    pin = {k: v for k, v in inp.items() if k in parent.all_inputs}
    t = np.arange(N) * dt
    a = model.forecast_temperature(np.full((3, model.n_states), -18.0), t, **inp)[:, :, model.T_i_idx]
    b = parent.forecast_temperature(np.full((3, parent.n_states), -18.0), t, **pin)[:, :, parent.T_i_idx]
    return float(np.abs(a - b).max())
