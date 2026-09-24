"""Build the per-batch data dict consumed by TemperatureProblem when
calibrating a 3R3C model.

State vector is [T_i, T_h, T_e]; ``y_0`` and ``y_0_`` (lag) are pre-computed
with measured ``T_i`` and ``T_h`` plus the ad-hoc envelope guess
``T_e = T_i - 2``. ``T_h`` is read from ``T_h_col`` (default ``"T_h"``) and
its 1D-lag companion ``f"{T_h_col}_1Dlag"``.
"""
import numpy as np


def build_data_dict(dataloader, split_name, Phi_HC_col, T_h_col="T_h"):
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]

    T_i  = di["y1"]
    T_h  = di[T_h_col]
    T_i0 = T_i[:, 0]
    T_h0 = T_h[:, 0]
    T_e0 = T_i0 - 2
    y_0  = np.stack([T_i0, T_h0, T_e0], axis=1)

    T_i_ = di["y1_1Dlag"]
    T_h_ = di[f"{T_h_col}_1Dlag"]
    T_i0_ = T_i_[:, 0]
    T_h0_ = T_h_[:, 0]
    T_e0_ = T_i0_ - 2
    y_0_ = np.stack([T_i0_, T_h0_, T_e0_], axis=1)

    return {
        "t":       b["t_array"],
        "T_i":     T_i,
        "T_h":     T_h,
        "T_a":     di["T_a"],
        "Phi_sol": di["Phi_sol"],
        "Phi_HC":  di[Phi_HC_col],
        "data_for_init_state": {
            "y_0":      y_0,      "y_0_":     y_0_,
            "t_":       di["t_array_1Dlag"],
            "T_i_":     T_i_,
            "T_h_":     T_h_,
            "T_a_":     di["T_a_1Dlag"],
            "Phi_sol_": di["Phi_sol_1Dlag"],
            "Phi_HC_":  di[f"{Phi_HC_col}_1Dlag"],
        },
    }
