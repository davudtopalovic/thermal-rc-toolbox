"""Numba-JIT'd cores for the 2R2C model.

State vector y = [T_i, T_e]; stacked input vector u = [Phi_HC, T_a, Phi_sol].

Continuous-time energy balance:
    C_i * dT_i/dt = (T_e - T_i) / R_ie + Phi_HC
    C_e * dT_e/dt = (T_i - T_e) / R_ie + (T_a - T_e) / R_ea + k_sol * Phi_sol

Two kernels:
- ``forecast_heat_demand_numba`` — inverse (setpoint-to-power) forecast using
  the exact ZOH one-step map (discrete matrices precomputed by the model),
  matching ``TwoR2C.forecast_heat_demand``
  (right-labeled convention: inputs at index k produce state/demand at k).
- ``simulate_2r2c`` — Crank-Nicolson forward simulation,
  (I - dt/2 A) y_{k+1} = (I + dt/2 A) y_k + dt/2 (B u_k + B u_{k+1}),
  solved per timestep as a 2x2 linear system in closed form.
  NOTE: currently unused and still on the trapezoidal-input (left-labeled)
  pairing — align with the right-labeled convention before first use.
"""
import numpy as np
from numba import njit


@njit
def forecast_heat_demand_numba(
    y_prev,
    Phi_HC_prev,
    t,
    T_a_BT,
    Phi_sol_BT,
    T_i_set_BT,

    # discrete one-step matrices (euler or exact ZOH), precomputed per
    # parameter set by the model:  y_k = Ad @ y_{k-1} + Bd @ [Phi_HC, T_a, Phi_sol]
    Ad,   # (2, 2)
    Bd,   # (2, 3) — control column stacked with the disturbance columns
    heat_power_min,
    heat_power_max,
):
    """Numba twin of ``TwoR2C.forecast_heat_demand`` (integration-agnostic).

    The kernel only iterates the discrete affine map and its closed-form
    ideal-regulation inversion: T_i^k = free_i + s * Phi with s = Bd[0, 0], so
        Phi_unres = (T_i_set - free_i) / s
    clipped to [heat_power_min, heat_power_max], then the state advance with
    the realized power. Which integration this realizes (forward Euler or the
    unconditionally stable exact ZOH) is decided by the matrices passed in.

    Returns (Phi_HC_pred, y_pred) of shapes (B, T) and (B, T, 2).
    """
    B = y_prev.shape[0]
    T = len(t)

    y_pred = np.zeros((B, T, 2))
    Phi_HC_pred = np.zeros((B, T))

    y_pred[:, 0, :] = y_prev
    Phi_HC_pred[:, 0] = Phi_HC_prev

    s = Bd[0, 0]

    for k in range(1, T):

        T_a = T_a_BT[:, k]
        Phi_sol = Phi_sol_BT[:, k]
        T_i_set = T_i_set_BT[:, k]

        T_i = y_prev[:, 0]
        T_e = y_prev[:, 1]

        # ============================
        # EXACT ZERO-POWER TRAJECTORY
        # ============================
        free_i = (Ad[0, 0] * T_i + Ad[0, 1] * T_e) + Bd[0, 1] * T_a + Bd[0, 2] * Phi_sol
        free_e = (Ad[1, 0] * T_i + Ad[1, 1] * T_e) + Bd[1, 1] * T_a + Bd[1, 2] * Phi_sol

        # ============================
        # CLOSED-FORM INVERSION
        # ============================
        Phi_unres = (T_i_set - free_i) / s

        # clipping to the actuator range
        Phi_HC_k = np.clip(Phi_unres, heat_power_min, heat_power_max)

        # ============================
        # STATE UPDATE (exact ZOH)
        # ============================
        y_prev = np.stack((free_i + Bd[0, 0] * Phi_HC_k,
                           free_e + Bd[1, 0] * Phi_HC_k), axis=1)

        y_pred[:, k, :] = y_prev
        Phi_HC_pred[:, k] = Phi_HC_k

    return Phi_HC_pred, y_pred
