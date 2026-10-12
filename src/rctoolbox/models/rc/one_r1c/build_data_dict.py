"""Build the per-batch data dict consumed by TemperatureProblem when
calibrating a 1R1C model.

State vector for 1R1C is [T_i] (measured), so ``y_0`` and ``y_0_`` (lag) are
simply the first measured indoor temperatures of the window and lag window.
There is no solar input, so the dataloader needs only the features y1, T_a and
the Phi_HC column.
"""
import numpy as np


def build_data_dict(dataloader, split_name, Phi_HC_col):
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]

    T_i  = di["y1"]
    y_0  = T_i[:, [0]]

    T_i_ = di["y1_1Dlag"]
    y_0_ = T_i_[:, [0]]

    return {
        "t":       b["t_array"],
        "T_i":     T_i,
        "T_a":     di["T_a"],
        "Phi_HC":  di[Phi_HC_col],
        "data_for_init_state": {
            "y_0":      np.asarray(y_0, dtype=np.float64),
            "y_0_":     np.asarray(y_0_, dtype=np.float64),
            "t_":       di["t_array_1Dlag"],
            "T_i_":     T_i_,
            "T_a_":     di["T_a_1Dlag"],
            "Phi_HC_":  di[f"{Phi_HC_col}_1Dlag"],
        },
    }
