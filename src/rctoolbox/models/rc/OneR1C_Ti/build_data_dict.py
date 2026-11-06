"""Build the per-batch data dict consumed by TemperatureProblem when
calibrating a 1R1C model.

The state is [T_i] (measured). The dict holds measurements only; the model
builds its starting state itself (``OneR1C_Ti.seed_state``) from the measured T_i
at the forecast start (``T_i_0``) and at the lag-window start. There is no
solar input, so the dataloader needs only the features y1, T_a and the Phi_HC
column. ``columns`` maps the model signals to data columns
(``DEFAULT_COLUMNS``); ``Phi_HC_col`` is the older shorthand for
``columns["Phi_HC"]``.
"""
from rctoolbox.models.init_data import init_data as _init_data, resolve_columns as _resolve
from rctoolbox.models.rc.OneR1C_Ti.model import OneR1C_Ti

DEFAULT_COLUMNS = {"T_i": "y1", "T_a": "T_a", "Phi_HC": "Phi_HC"}
LAG_SIGNALS = ("T_i", "T_a", "Phi_HC")


def resolve_columns(columns=None, Phi_HC_col=None) -> dict:
    return _resolve(DEFAULT_COLUMNS, columns, "1R1C", Phi_HC=Phi_HC_col)


def init_data(di, columns) -> dict:
    """``data_for_init_state`` of a batch (``di`` = its ``data_inputs``)."""
    return _init_data(di, columns, OneR1C_Ti.SEED_SIGNALS, LAG_SIGNALS)


def build_data_dict(dataloader, split_name, Phi_HC_col=None, columns=None):
    c  = resolve_columns(columns, Phi_HC_col)
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]
    return {
        "t":       b["t_array"],
        "T_i":     di[c["T_i"]],
        "T_a":     di[c["T_a"]],
        "Phi_HC":  di[c["Phi_HC"]],
        "data_for_init_state": init_data(di, c),
    }
