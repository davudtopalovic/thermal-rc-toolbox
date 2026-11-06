"""Build the per-batch data dicts for calibrating a 2R2C model.

``build_data_dict``             -> TemperatureProblem (temperature task)
``build_data_dict_heat_demand`` -> HeatDemandProblem (setpoint-to-power task)

The state is [T_i, T_e]. The dicts hold measurements only; the model builds its
starting state itself (``TwoR2C_TiTe.seed_state``: T_e = T_i - 2, refined by the
static-inversion / regime-aware initializers) from the measured T_i at the
forecast start (``T_i_0``) and at the lag-window start. ``columns`` maps the
model signals to data columns (``DEFAULT_COLUMNS``); ``Phi_HC_col`` and
``y1_set_col`` are the older shorthands for ``columns["Phi_HC"]`` and
``columns["T_i_set"]``. The 3R2C uses the same helpers.
"""
from rctoolbox.models.init_data import init_data as _init_data, resolve_columns as _resolve
from rctoolbox.models.rc.TwoR2C_TiTe.model import TwoR2C_TiTe

DEFAULT_COLUMNS = {
    "T_i":     "y1",
    "T_a":     "T_a",
    "Phi_sol": "Phi_sol",
    "Phi_HC":  "Phi_HC",
    "T_i_set": "y1_set",
    "DR":      "DR",
    "controlled_flag": "controlled_flag",
}
LAG_SIGNALS = ("T_i", "T_a", "Phi_sol", "Phi_HC")                     # temperature task
LAG_SIGNALS_HEAT_DEMAND = LAG_SIGNALS + ("T_i_set",)                  # heat-demand task


def resolve_columns(columns=None, Phi_HC_col=None, y1_set_col=None) -> dict:
    return _resolve(DEFAULT_COLUMNS, columns, "2R2C", Phi_HC=Phi_HC_col, T_i_set=y1_set_col)


def init_data(di, columns, heat_demand=False) -> dict:
    """``data_for_init_state`` of a batch (``di`` = its ``data_inputs``)."""
    lag = LAG_SIGNALS_HEAT_DEMAND if heat_demand else LAG_SIGNALS
    return _init_data(di, columns, TwoR2C_TiTe.SEED_SIGNALS, lag)


def build_data_dict(dataloader, split_name, Phi_HC_col=None, columns=None):
    c  = resolve_columns(columns, Phi_HC_col)
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]
    return {
        "t":       b["t_array"],
        "T_i":     di[c["T_i"]],
        "T_a":     di[c["T_a"]],
        "Phi_sol": di[c["Phi_sol"]],
        "Phi_HC":  di[c["Phi_HC"]],
        "data_for_init_state": init_data(di, c),
    }


def build_data_dict_heat_demand(dataloader, split_name, Phi_HC_col="Phi_HC", y1_set_col="y1_set", columns=None):
    """Data dict for the inverse (setpoint-to-power) 2R2C identification.

    Mirrors five_r1c_eniso13790.build_data_dict — same keys, 2-state model.
    """
    c  = resolve_columns(columns, Phi_HC_col, y1_set_col)
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]
    return {
        "t":        b["t_array"],
        "T_i":      di[c["T_i"]],
        "T_a":      di[c["T_a"]],
        "Phi_sol":  di[c["Phi_sol"]],
        "Phi_HC":   di[c["Phi_HC"]],
        "Phi_HC_0": di[c["Phi_HC"]][:, 0],
        "T_i_set":  di[c["T_i_set"]],
        "DR":       di.get(c["DR"]),
        "controlled_flag": di.get(c["controlled_flag"]),
        "data_for_init_state": init_data(di, c, heat_demand=True),
    }
