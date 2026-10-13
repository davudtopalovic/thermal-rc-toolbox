"""Build the per-batch data dict consumed by TemperatureProblem when
calibrating the 5R4C envelope + internal-mass model.

State vector is [T_i, T_e1, T_e2, T_m]; only T_i is measured. ``y_0`` (and
the lag-window ``y_0_``) start the envelope nodes at the midpoints between
the air and their ambient, T_e1_0 = (T_i_0 + T_a_0)/2 and
T_e2_0 = (T_i_0 + T_out_0)/2, and the internal mass at T_m_0 = T_i_0 (or a
constant ``T_m0``). With the ``static_inversion`` / ``dynamic`` initializations
the model refines the envelope nodes at run time. The dataloader needs the
features y1, T_a, T_out and the Phi_HC column.
"""
import numpy as np


def initial_state(T_i, T_a, T_out, T_m0=None):
    """y_0 (B, 4) = [T_i_0, T_e1_0, T_e2_0, T_m_0] from the measured T_i,
    T_a and T_out (B, T)."""
    T_i0   = np.asarray(T_i,   dtype=np.float64)[:, 0]
    T_a0   = np.asarray(T_a,   dtype=np.float64)[:, 0]
    T_out0 = np.asarray(T_out, dtype=np.float64)[:, 0]
    T_e10 = 0.5 * (T_i0 + T_a0)
    T_e20 = 0.5 * (T_i0 + T_out0)
    T_m0 = T_i0.copy() if T_m0 is None else np.full_like(T_i0, float(T_m0))
    return np.stack([T_i0, T_e10, T_e20, T_m0], axis=1)


def data_for_init_state(di, Phi_HC_col, T_m0=None):
    """Initialization dict (y_0 and the lag-window inputs) from the data inputs."""
    return {
        "y_0":      initial_state(di["y1"], di["T_a"], di["T_out"], T_m0),
        "y_0_":     initial_state(di["y1_1Dlag"], di["T_a_1Dlag"], di["T_out_1Dlag"], T_m0),
        "t_":       di["t_array_1Dlag"],
        "T_i_":     di["y1_1Dlag"],
        "T_a_":     di["T_a_1Dlag"],
        "T_out_":   di["T_out_1Dlag"],
        "Phi_HC_":  di[f"{Phi_HC_col}_1Dlag"],
    }


def build_data_dict(dataloader, split_name, Phi_HC_col, T_m0=None):
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
