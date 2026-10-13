"""Build the per-batch data dict consumed by TemperatureProblem when
calibrating the 3R2C internal-mass + outdoor-link model.

State vector is [T_i, T_m]: T_i is measured, the internal mass T_m is not. It
starts at the constant ``T_m0`` (default -17 °C, the middle of the freezer's
thermostat band); pass ``T_m0=None`` to start it at the measured air
temperature, T_m_0 = T_i_0, instead. The dataloader needs the features
y1, T_a, T_out and the Phi_HC column.
"""
import numpy as np


T_M0_DEFAULT = -17.0


def initial_state(T_i, T_m0=T_M0_DEFAULT):
    """y_0 (B, 2) = [T_i_0, T_m_0] from measured T_i (B, T); T_m_0 = T_m0
    (constant), or T_i_0 if ``T_m0`` is None."""
    T_i0 = np.asarray(T_i, dtype=np.float64)[:, 0]
    T_m0 = T_i0.copy() if T_m0 is None else np.full_like(T_i0, float(T_m0))
    return np.stack([T_i0, T_m0], axis=1)


def data_for_init_state(di, Phi_HC_col, T_m0=T_M0_DEFAULT):
    """Initialization dict (y_0 and the lag-window inputs) from the data inputs."""
    return {
        "y_0":      initial_state(di["y1"], T_m0),
        "y_0_":     initial_state(di["y1_1Dlag"], T_m0),
        "t_":       di["t_array_1Dlag"],
        "T_i_":     di["y1_1Dlag"],
        "T_a_":     di["T_a_1Dlag"],
        "T_out_":   di["T_out_1Dlag"],
        "Phi_HC_":  di[f"{Phi_HC_col}_1Dlag"],
    }


def build_data_dict(dataloader, split_name, Phi_HC_col, T_m0=T_M0_DEFAULT):
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]

    return {
        "t":       b["t_array"],
        "T_i":     di["y1"],
        "T_a":     di["T_a"],
        "T_out":   di["T_out"],
        "Phi_HC":  di[Phi_HC_col],
        "data_for_init_state": data_for_init_state(di, Phi_HC_col, T_m0),
    }
