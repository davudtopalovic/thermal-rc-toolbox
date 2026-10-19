"""Build the per-batch data dict consumed by TemperatureProblem when
calibrating the standalone 2R2C internal-mass model driven by the measured
evaporator temperature.

State vector is [T_i, T_m]: T_i is measured, the internal mass T_m is not and
starts at a constant T_m_0 = -17 °C, the middle of the freezer's thermostat
band (for ``y_0`` and the lag-window ``y_0_``); ``T_m0=None`` starts it at the
measured T_i instead (as ``fz_2r2c_im_evin``, which then gives identical
results). With the ``kalman`` or
``dynamic`` initialization T_m_0 is only the starting point at the beginning
of the lag window. The dataloader needs the features y1, T_a, T_ev and on.
"""
import numpy as np


def initial_state(T_i, T_m0=-17.0):
    """y_0 (B, 2) = [T_i_0, T_m_0] from measured T_i (B, T); T_m0=None -> T_m_0 = T_i_0."""
    T_i0 = np.asarray(T_i, dtype=np.float64)[:, 0]
    T_m0 = T_i0.copy() if T_m0 is None else np.full_like(T_i0, fill_value=T_m0)
    return np.stack([T_i0, T_m0], axis=1)


def build_data_dict(dataloader, split_name, T_m0=-17.0):
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]

    T_i  = di["y1"]
    T_i_ = di["y1_1Dlag"]

    return {
        "t":     b["t_array"],
        "T_i":   T_i,
        "T_a":   di["T_a"],
        "T_ev":  di["T_ev"],
        "on":    di["on"],
        "data_for_init_state": {
            "y_0":    initial_state(T_i, T_m0),
            "y_0_":   initial_state(T_i_, T_m0),
            "t_":     di["t_array_1Dlag"],
            "T_i_":   T_i_,
            "T_a_":   di["T_a_1Dlag"],
            "T_ev_":  di["T_ev_1Dlag"],
            "on_":    di["on_1Dlag"],
        },
    }
