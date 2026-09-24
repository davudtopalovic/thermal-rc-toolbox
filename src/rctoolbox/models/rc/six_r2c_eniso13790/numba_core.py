"""Numba-JIT'd inner loop for 6R2C heat-demand forecasting.

Same three-part step as the 5R1C core (probe at 0 W, probe at 10 W/m², interpolate),
but the state is (T_m1, T_m2) instead of just T_m and each probe advances the pair
via a 2x2 Crank-Nicolson linear solve.

State ordering in `y_pred`: [T_m1, T_m2, T_s, T_i] (4 columns).
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

    # scalar parameters (order must match model.forecast_heat_demand_fast)
    A_f,
    C_m1, C_m2,
    H_em, H_ww,
    alpha1, alpha2,
    H_tr2, H_tr3,
    Htr3_over_Htr2, Htr1_over_Hve,
    Hms_plus_Htr2, His_plus_Hve,
    H_ms, H_is, H_ve, H_w,
    A_m, A_t, h_ms_coeff,
    Phi_int, k_sol,
    heat_power_max,
):
    B = y_prev.shape[0]
    T = len(t)

    y_pred = np.zeros((B, T, 4))
    Phi_HC_pred = np.zeros((B, T))

    y_pred[:, 0, :] = y_prev
    Phi_HC_pred[:, 0] = Phi_HC_prev

    dt = t[1] - t[0]

    Phi_st_coeff = 1.0 - (A_m / A_t) - (H_w / (h_ms_coeff * A_t))
    Phi_m_coeff = A_m / A_t

    # 2x2 system coefficients that don't depend on inputs — compute once.
    a11 = C_m1 / dt + 0.5 * alpha1
    a12 = -0.5 * H_ww
    a21 = -0.5 * H_ww
    a22 = C_m2 / dt + 0.5 * alpha2
    det = a11 * a22 - a12 * a21
    inv_det = 1.0 / det

    c11 = C_m1 / dt - 0.5 * alpha1     # coefficient of T_m1_prev on RHS of eq. 1
    c22 = C_m2 / dt - 0.5 * alpha2     # coefficient of T_m2_prev on RHS of eq. 2
    c12 = 0.5 * H_ww                    # coefficient of T_m2_prev on RHS of eq. 1
    c21 = 0.5 * H_ww                    # coefficient of T_m1_prev on RHS of eq. 2

    for k in range(1, T):
        T_a = T_a_BT[:, k]
        Phi_sol = Phi_sol_BT[:, k]
        T_i_set = T_i_set_BT[:, k]

        T_m1_prev = y_prev[:, 0]
        T_m2_prev = y_prev[:, 1]

        # Time-invariant-per-step heat allocations
        Phi_sol_eff = k_sol * Phi_sol
        Phi_st = Phi_st_coeff * (0.5 * Phi_int + Phi_sol_eff)
        Phi_m = Phi_m_coeff * (0.5 * Phi_int + Phi_sol_eff)
        Phi_m1_tot = H_em * T_a

        # ============================
        # PROBE WITH 0 POWER
        # ============================
        Phi_HC0 = np.zeros(B)
        Phi_ia0 = 0.5 * Phi_int + Phi_HC0
        Phi_m2_tot0 = Phi_m + H_tr3 * T_a + Htr3_over_Htr2 * (Htr1_over_Hve * Phi_ia0 + Phi_st)

        b1_0 = c11 * T_m1_prev + c12 * T_m2_prev + Phi_m1_tot
        b2_0 = c21 * T_m1_prev + c22 * T_m2_prev + Phi_m2_tot0

        T_m1_0 = (a22 * b1_0 - a12 * b2_0) * inv_det
        T_m2_0 = (-a21 * b1_0 + a11 * b2_0) * inv_det

        T_m2_avg0 = 0.5 * (T_m2_0 + T_m2_prev)
        T_s0 = (H_ms * T_m2_avg0 + Phi_st + H_tr2 * T_a + Htr1_over_Hve * Phi_ia0) / Hms_plus_Htr2
        T_i0 = (H_is * T_s0 + H_ve * T_a + Phi_ia0) / His_plus_Hve

        # ============================
        # PROBE WITH 10 W/m²
        # ============================
        Phi_HC10 = np.full(B, 10.0 * A_f)
        Phi_ia10 = 0.5 * Phi_int + Phi_HC10
        Phi_m2_tot10 = Phi_m + H_tr3 * T_a + Htr3_over_Htr2 * (Htr1_over_Hve * Phi_ia10 + Phi_st)

        b1_10 = c11 * T_m1_prev + c12 * T_m2_prev + Phi_m1_tot
        b2_10 = c21 * T_m1_prev + c22 * T_m2_prev + Phi_m2_tot10

        T_m1_10 = (a22 * b1_10 - a12 * b2_10) * inv_det
        T_m2_10 = (-a21 * b1_10 + a11 * b2_10) * inv_det

        T_m2_avg10 = 0.5 * (T_m2_10 + T_m2_prev)
        T_s10 = (H_ms * T_m2_avg10 + Phi_st + H_tr2 * T_a + Htr1_over_Hve * Phi_ia10) / Hms_plus_Htr2
        T_i10 = (H_is * T_s10 + H_ve * T_a + Phi_ia10) / His_plus_Hve

        # ============================
        # INTERPOLATION
        # ============================
        denom = T_i10 - T_i0
        eps = 1e-9
        Phi_unres = Phi_HC10 * (T_i_set - T_i0) / (denom + eps)
        Phi_HC_k = np.clip(Phi_unres, 0.0, heat_power_max)

        # ============================
        # FINAL STATE UPDATE at Phi_HC_k
        # ============================
        Phi_ia = 0.5 * Phi_int + Phi_HC_k
        Phi_m2_tot = Phi_m + H_tr3 * T_a + Htr3_over_Htr2 * (Htr1_over_Hve * Phi_ia + Phi_st)

        b1 = c11 * T_m1_prev + c12 * T_m2_prev + Phi_m1_tot
        b2 = c21 * T_m1_prev + c22 * T_m2_prev + Phi_m2_tot

        T_m1_next = (a22 * b1 - a12 * b2) * inv_det
        T_m2_next = (-a21 * b1 + a11 * b2) * inv_det

        T_m2_avg = 0.5 * (T_m2_next + T_m2_prev)
        T_s = (H_ms * T_m2_avg + Phi_st + H_tr2 * T_a + Htr1_over_Hve * Phi_ia) / Hms_plus_Htr2
        T_i = (H_is * T_s + H_ve * T_a + Phi_ia) / His_plus_Hve

        y_prev = np.stack((T_m1_next, T_m2_next, T_s, T_i), axis=1)

        y_pred[:, k, :] = y_prev
        Phi_HC_pred[:, k] = Phi_HC_k
        Phi_HC_prev = Phi_HC_k

    return Phi_HC_pred, y_pred
