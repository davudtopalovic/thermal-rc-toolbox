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

    # parameters
    A_f,
    C_m,
    H_tr1,
    H_tr2,
    H_tr3,
    H_tr3_plus_Hem,
    Htr3_over_Htr2,
    Htr1_over_Hve,
    Hms_plus_Htr2,
    His_plus_Hve,
    H_ms,
    H_is,
    H_ve,
    H_w,
    A_m,
    A_t,
    h_ms_coeff,
    Phi_int,
    k_sol,

    heat_power_max
):

    B = y_prev.shape[0]
    T = len(t)

    y_pred = np.zeros((B, T, 3))
    Phi_HC_pred = np.zeros((B, T))

    y_pred[:, 0, :] = y_prev
    Phi_HC_pred[:, 0] = Phi_HC_prev

    dt = t[1] - t[0]

    for k in range(1, T):

        T_a = T_a_BT[:, k]
        Phi_sol = Phi_sol_BT[:, k]
        T_i_set = T_i_set_BT[:, k]

        T_m_prev = y_prev[:, 0]

        # ============================
        # PROBE WITH 0 POWER
        # ============================
        Phi_HC0 = np.zeros(B)

        # heat flows
        Phi_sol_eff = k_sol * Phi_sol
        Phi_ia0 = 0.5 * Phi_int + Phi_HC0
        Phi_st = (1 - (A_m / A_t) - (H_w / (h_ms_coeff * A_t))) * (0.5 * Phi_int + Phi_sol_eff)
        Phi_m = (A_m / A_t) * (0.5 * Phi_int + Phi_sol_eff)

        Phi_m_tot0 = Phi_m + H_tr3_plus_Hem * T_a + Htr3_over_Htr2 * (Htr1_over_Hve * Phi_ia0 + Phi_st)

        T_m0 = ((T_m_prev * ((C_m / dt) - 0.5 * H_tr3_plus_Hem)) + Phi_m_tot0) / ((C_m / dt) + 0.5 * H_tr3_plus_Hem)
        T_m_avg0 = (T_m0 + T_m_prev) / 2.0

        T_s0 = (H_ms * T_m_avg0 + Phi_st + H_tr2 * T_a + Htr1_over_Hve * Phi_ia0) / Hms_plus_Htr2
        T_i0 = (H_is * T_s0 + H_ve * T_a + Phi_ia0) / His_plus_Hve

        # ============================
        # PROBE WITH 10 W/m²
        # ============================
        Phi_HC10 = np.full(B, 10.0 * A_f)

        Phi_ia10 = 0.5 * Phi_int + Phi_HC10

        Phi_m_tot10 = Phi_m + H_tr3_plus_Hem * T_a + Htr3_over_Htr2 * (Htr1_over_Hve * Phi_ia10 + Phi_st)

        T_m10 = ((T_m_prev * ((C_m / dt) - 0.5 * H_tr3_plus_Hem)) + Phi_m_tot10) / ((C_m / dt) + 0.5 * H_tr3_plus_Hem)
        T_m_avg10 = (T_m10 + T_m_prev) / 2.0

        T_s10 = (H_ms * T_m_avg10 + Phi_st + H_tr2 * T_a + Htr1_over_Hve * Phi_ia10) / Hms_plus_Htr2
        T_i10 = (H_is * T_s10 + H_ve * T_a + Phi_ia10) / His_plus_Hve

        # ============================
        # INTERPOLATION
        # ============================
        denom = T_i10 - T_i0
        eps = 1e-9

        Phi_unres = Phi_HC10 * (T_i_set - T_i0) / (denom + eps)

        # max clipping
        Phi_HC_k = np.clip(Phi_unres, 0.0, heat_power_max)

        # ============================
        # FINAL STATE UPDATE
        # ============================
        Phi_ia = 0.5 * Phi_int + Phi_HC_k

        Phi_m_tot = Phi_m + H_tr3_plus_Hem * T_a + Htr3_over_Htr2 * (Htr1_over_Hve * Phi_ia + Phi_st)

        T_m_next = ((T_m_prev * ((C_m / dt) - 0.5 * H_tr3_plus_Hem)) + Phi_m_tot) / ((C_m / dt) + 0.5 * H_tr3_plus_Hem)
        T_m_avg = (T_m_next + T_m_prev) / 2.0

        T_s = (H_ms * T_m_avg + Phi_st + H_tr2 * T_a + Htr1_over_Hve * Phi_ia) / Hms_plus_Htr2
        T_i = (H_is * T_s + H_ve * T_a + Phi_ia) / His_plus_Hve

        y_prev = np.stack((T_m_next, T_s, T_i), axis=1)

        y_pred[:, k, :] = y_prev
        Phi_HC_pred[:, k] = Phi_HC_k

        Phi_HC_prev = Phi_HC_k

    return Phi_HC_pred, y_pred