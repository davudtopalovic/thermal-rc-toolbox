"""Build the per-batch data dict consumed by TemperatureProblem when
calibrating the 5R4C envelope + internal-mass model.

The state is [T_i, T_e1, T_e2, T_m]; only T_i is measured. The dict holds
measurements only; the model builds its starting state itself
(``FiveR4C_TiTe1Te2Tm.seed_state``: the envelope nodes at the midpoints
between the air and their ambient, T_e1 = (T_i + T_a)/2 and
T_e2 = (T_i + T_out)/2, the internal mass at T_m = T_i or the constant
``init_config["T_m0"]``), from T_i, T_a and T_out at the forecast start and at
the lag-window start. The dataloader needs the features y1, T_a, T_out and
the Phi_HC column. ``columns`` maps the model signals to data columns
(``DEFAULT_COLUMNS``); ``Phi_HC_col`` is the older shorthand. Passing ``T_m0``
here is deprecated (it writes explicit seeds into the data).
"""
import numpy as np

from rctoolbox.models.init_data import UNSET, init_data as _init_data, legacy_seeds, resolve_columns as _resolve
from rctoolbox.models.rc.FiveR4C_TiTe1Te2Tm.model import FiveR4C_TiTe1Te2Tm

DEFAULT_COLUMNS = {"T_i": "y1", "T_a": "T_a", "T_out": "T_out", "Phi_HC": "Phi_HC"}
LAG_SIGNALS = ("T_i", "T_a", "T_out", "Phi_HC")


def initial_state(T_i, T_a, T_out, T_m0=None):
    """y_0 (B, 4) = [T_i_0, T_e1_0, T_e2_0, T_m_0] from the measured T_i,
    T_a and T_out (B, T). Kept for older code; the model builds this itself
    (``seed_state``)."""
    T_i0   = np.asarray(T_i,   dtype=np.float64)[:, 0]
    T_a0   = np.asarray(T_a,   dtype=np.float64)[:, 0]
    T_out0 = np.asarray(T_out, dtype=np.float64)[:, 0]
    T_e10 = 0.5 * (T_i0 + T_a0)
    T_e20 = 0.5 * (T_i0 + T_out0)
    T_m0 = T_i0.copy() if T_m0 is None else np.full_like(T_i0, float(T_m0))
    return np.stack([T_i0, T_e10, T_e20, T_m0], axis=1)


def resolve_columns(columns=None, Phi_HC_col=None) -> dict:
    return _resolve(DEFAULT_COLUMNS, columns, "5R4C_IM", Phi_HC=Phi_HC_col)


def data_for_init_state(di, Phi_HC_col=None, T_m0=UNSET, columns=None):
    """Initialization data of a batch (``di`` = its ``data_inputs``)."""
    c = resolve_columns(columns, Phi_HC_col)
    init = _init_data(di, c, FiveR4C_TiTe1Te2Tm.SEED_SIGNALS, LAG_SIGNALS)
    if T_m0 is not UNSET:
        init = legacy_seeds(FiveR4C_TiTe1Te2Tm(), init, T_m0=None if T_m0 is None else float(T_m0))
    return init


def build_data_dict(dataloader, split_name, Phi_HC_col=None, T_m0=UNSET, columns=None):
    c  = resolve_columns(columns, Phi_HC_col)
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]
    return {
        "t":       b["t_array"],
        "T_i":     di[c["T_i"]],
        "T_a":     di[c["T_a"]],
        "T_out":   di[c["T_out"]],
        "Phi_HC":  di[c["Phi_HC"]],
        "data_for_init_state": data_for_init_state(di, T_m0=T_m0, columns=c),
    }
