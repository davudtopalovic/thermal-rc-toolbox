"""Build the per-batch data dict consumed by HeatDemandProblem for 6R2C.

State vector for 6R2C is [T_m1, T_m2, T_s, T_i]; ``y_0`` and ``y_0_`` (lag) use
approximate offsets from T_i: T_m1 = T_i - 2, T_m2 = T_i - 1.5, T_s = T_i - 1.
"""
import numpy as np


def _make_initial_state(T_i_col):
    T_m1_0 = T_i_col - 2.0
    T_m2_0 = T_i_col - 1.5
    T_s_0  = T_i_col - 1.0
    T_i_0  = T_i_col
    return np.stack([T_m1_0, T_m2_0, T_s_0, T_i_0], axis=1)


def build_data_dict(dataloader, split_name, Phi_HC_col="Phi_HC", y1_set_col="y1_set"):
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]

    T_i  = di["y1"]
    y_0  = _make_initial_state(T_i[:, 0])

    T_i_ = di["y1_1Dlag"]
    y_0_ = _make_initial_state(T_i_[:, 0])

    return {
        "t":        b["t_array"],
        "T_i":      T_i,
        "T_a":      di["T_a"],
        "Phi_sol":  di["Phi_sol"],
        "Phi_HC":   di[Phi_HC_col],
        "Phi_HC_0": di[Phi_HC_col][:, 0],
        "T_i_set":  di[y1_set_col],
        "DR":       di.get("DR"),
        "controlled_flag": di.get("controlled_flag"),
        "data_for_init_state": {
            "y_0":      y_0,       "y_0_":     y_0_,
            "t_":       di["t_array_1Dlag"],
            "T_i_":     T_i_,
            "T_a_":     di["T_a_1Dlag"],
            "Phi_sol_": di["Phi_sol_1Dlag"],
            "Phi_HC_":  di[f"{Phi_HC_col}_1Dlag"],
            "T_i_set_": di[f"{y1_set_col}_1Dlag"],
        },
    }
