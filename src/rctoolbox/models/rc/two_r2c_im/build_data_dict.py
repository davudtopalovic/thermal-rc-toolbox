"""Build the per-batch data dict consumed by TemperatureProblem when
calibrating the 2R2C internal-mass model.

State vector is [T_i, T_m]: T_i is measured, the internal mass T_m is not and
starts at a constant T_m_0 = -17 °C, the middle of the freezer's thermostat
band (for ``y_0`` and the lag-window ``y_0_``). There is no solar input, so the dataloader needs only the
features y1, T_a and the Phi_HC column.
"""
import numpy as np


def initial_state(T_i):
    """y_0 (B, 2) = [T_i_0, T_m_0] with T_m_0 = -17 °C, from measured T_i (B, T)."""
    T_i0 = np.asarray(T_i, dtype=np.float64)[:, 0]
    T_m0 = np.full_like(T_i0, fill_value=-17)
    return np.stack([T_i0, T_m0], axis=1)


def build_data_dict(dataloader, split_name, Phi_HC_col):
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]

    T_i  = di["y1"]
    T_i_ = di["y1_1Dlag"]

    return {
        "t":       b["t_array"],
        "T_i":     T_i,
        "T_a":     di["T_a"],
        "Phi_HC":  di[Phi_HC_col],
        "data_for_init_state": {
            "y_0":      initial_state(T_i),
            "y_0_":     initial_state(T_i_),
            "t_":       di["t_array_1Dlag"],
            "T_i_":     T_i_,
            "T_a_":     di["T_a_1Dlag"],
            "Phi_HC_":  di[f"{Phi_HC_col}_1Dlag"],
        },
    }
