"""Numba kernel for the gains-augmented 3R2C (constant-input offset variant).

Same integration-agnostic affine map as the two_r2c kernel, extended with a
constant per-step offset vector (the discretized contribution of the constant
internal-gains input):

    y_k = Ad @ y_{k-1} + Bd @ [Phi_HC, T_a, Phi_sol] + off
"""
import numpy as np
from numba import njit


@njit
def forecast_heat_demand_offset_numba(
    y_prev,
    Phi_HC_prev,
    t,
    T_a_BT,
    Phi_sol_BT,
    T_i_set_BT,

    Ad,    # (2, 2)
    Bd,    # (2, 3) columns: [Phi_HC, T_a, Phi_sol]
    off,   # (2,)   discretized constant-input offset per step

    heat_power_min,
    heat_power_max,
):
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

        free_i = (Ad[0, 0] * T_i + Ad[0, 1] * T_e) + Bd[0, 1] * T_a + Bd[0, 2] * Phi_sol + off[0]
        free_e = (Ad[1, 0] * T_i + Ad[1, 1] * T_e) + Bd[1, 1] * T_a + Bd[1, 2] * Phi_sol + off[1]

        Phi_unres = (T_i_set - free_i) / s
        Phi_HC_k = np.clip(Phi_unres, heat_power_min, heat_power_max)

        y_prev = np.stack((free_i + Bd[0, 0] * Phi_HC_k,
                           free_e + Bd[1, 0] * Phi_HC_k), axis=1)

        y_pred[:, k, :] = y_prev
        Phi_HC_pred[:, k] = Phi_HC_k

    return Phi_HC_pred, y_pred


@njit
def forecast_heat_demand_thermostat_numba(
    y_prev,
    Phi_HC_prev,
    t,
    T_a_BT,
    Phi_sol_BT,
    T_i_set_BT,

    Ad,     # (2, 2)
    Bd,     # (2, 3) columns: [Phi_HC, T_a, Phi_sol]
    off,    # (2,)   discretized constant-input offset per step
    alpha,  # [K/W]  thermostat feed-through: T_t = T_i + alpha * Phi_HC

    heat_power_min,
    heat_power_max,
):
    """Thermostat-node variant: the controller pins the algebraic thermostat
    signal T_t = T_i + alpha*Phi (not T_i itself) to the setpoint, so the
    inversion slope becomes s + alpha:

        Phi_unres = (T_i_set - free_i) / (Bd[0, 0] + alpha)

    alpha = 0 recovers forecast_heat_demand_offset_numba exactly.
    """
    B = y_prev.shape[0]
    T = len(t)

    y_pred = np.zeros((B, T, 2))
    Phi_HC_pred = np.zeros((B, T))

    y_pred[:, 0, :] = y_prev
    Phi_HC_pred[:, 0] = Phi_HC_prev

    s_eff = Bd[0, 0] + alpha

    for k in range(1, T):
        T_a = T_a_BT[:, k]
        Phi_sol = Phi_sol_BT[:, k]
        T_i_set = T_i_set_BT[:, k]

        T_i = y_prev[:, 0]
        T_e = y_prev[:, 1]

        free_i = (Ad[0, 0] * T_i + Ad[0, 1] * T_e) + Bd[0, 1] * T_a + Bd[0, 2] * Phi_sol + off[0]
        free_e = (Ad[1, 0] * T_i + Ad[1, 1] * T_e) + Bd[1, 1] * T_a + Bd[1, 2] * Phi_sol + off[1]

        Phi_unres = (T_i_set - free_i) / s_eff
        Phi_HC_k = np.clip(Phi_unres, heat_power_min, heat_power_max)

        y_prev = np.stack((free_i + Bd[0, 0] * Phi_HC_k,
                           free_e + Bd[1, 0] * Phi_HC_k), axis=1)

        y_pred[:, k, :] = y_prev
        Phi_HC_pred[:, k] = Phi_HC_k

    return Phi_HC_pred, y_pred
