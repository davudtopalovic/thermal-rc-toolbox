"""Build the per-batch data dict consumed by HeatDemandProblem.

State vector for 5R1C is [T_m, T_s, T_i]; ``y_0`` and ``y_0_`` (lag) are
pre-computed here with the conventional ad-hoc offsets ``T_m = T_i - 2`` and
``T_s = T_i - 1``.
"""
import numpy as np


def build_data_dict(dataloader, split_name, Phi_HC_col="Phi_HC", y1_set_col="y1_set"):
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]

    T_i  = di["y1"]
    T_i0 = T_i[:, 0]
    T_m0 = T_i0 - 2
    T_s0 = T_i0 - 1
    y_0  = np.stack([T_m0, T_s0, T_i0], axis=1)

    T_i_ = di["y1_1Dlag"]
    T_i0_ = T_i_[:, 0]
    T_m0_ = T_i0_ - 2
    T_s0_ = T_i0_ - 1
    y_0_ = np.stack([T_m0_, T_s0_, T_i0_], axis=1)

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
