"""Build the per-batch data dict consumed by TemperatureProblem when
calibrating a 3R3C model.

The state is [T_i, T_h, T_e]; T_i and T_h are measured. The dict holds
measurements only; the model builds its starting state itself
(``ThreeR3C_TiThTe.seed_state``: T_e = T_i - 2) from the measured T_i and T_h at the
forecast start (``T_i_0``, ``T_h_0``) and at the lag-window start.
``columns`` maps the model signals to data columns (``DEFAULT_COLUMNS``; T_h
defaults to ``"y2"``, the heating-circuit column, as for the 6R3C);
``Phi_HC_col`` and ``T_h_col`` are the older shorthands.
"""
from rctoolbox.models.init_data import init_data as _init_data, resolve_columns as _resolve
from rctoolbox.models.rc.ThreeR3C_TiThTe.model import ThreeR3C_TiThTe

DEFAULT_COLUMNS = {"T_i": "y1", "T_h": "y2", "T_a": "T_a", "Phi_sol": "Phi_sol", "Phi_HC": "Phi_HC"}
LAG_SIGNALS = ("T_i", "T_h", "T_a", "Phi_sol", "Phi_HC")


def resolve_columns(columns=None, Phi_HC_col=None, T_h_col=None) -> dict:
    return _resolve(DEFAULT_COLUMNS, columns, "3R3C", Phi_HC=Phi_HC_col, T_h=T_h_col)


def init_data(di, columns) -> dict:
    """``data_for_init_state`` of a batch (``di`` = its ``data_inputs``)."""
    return _init_data(di, columns, ThreeR3C_TiThTe.SEED_SIGNALS, LAG_SIGNALS)


def build_data_dict(dataloader, split_name, Phi_HC_col=None, T_h_col=None, columns=None):
    c  = resolve_columns(columns, Phi_HC_col, T_h_col)
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]
    return {
        "t":       b["t_array"],
        "T_i":     di[c["T_i"]],
        "T_h":     di[c["T_h"]],
        "T_a":     di[c["T_a"]],
        "Phi_sol": di[c["Phi_sol"]],
        "Phi_HC":  di[c["Phi_HC"]],
        "data_for_init_state": init_data(di, c),
    }
