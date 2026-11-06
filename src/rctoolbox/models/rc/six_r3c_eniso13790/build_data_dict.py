"""Build the per-batch data dict consumed by TemperatureProblem when
calibrating a 6R3C model.

The dict holds measurements only. The model builds its starting state itself
(``SixR3C_ENISO13790.seed_state``: [T_i, T_h, T_m] = [T_i, T_h, T_i - 1]) from
the measured T_i and T_h at the forecast start (``T_i_0``, ``T_h_0``) and at
the lag-window start, and ``init_config["lag_h"]`` can shorten the lag window.
Data dicts written earlier, which carry ``y_0`` / ``y_0_``, still work.

The model's signals are read from the dataframe columns given by ``columns``
(signal -> column); signals left out keep their default column
(``DEFAULT_COLUMNS``). ``Phi_HC_col`` and ``T_h_col`` are the older shorthands
for ``columns["Phi_HC"]`` and ``columns["T_h"]``.
"""
DEFAULT_COLUMNS = {
    "T_i":     "y1",       # indoor air temperature [degC]
    "T_h":     "y2",       # heating circuit temperature [degC]
    "T_a":     "T_a",      # outdoor air temperature [degC]
    "Phi_sol": "Phi_sol",  # solar irradiance [W/m2]
    "Phi_HC":  "Phi_HC",   # heat power [W]
}
# signals the initialization reads over the lag window (``<signal>_``)
LAG_SIGNALS = ("T_i", "T_h", "T_a", "Phi_sol", "Phi_HC")


def resolve_columns(columns=None, Phi_HC_col=None, T_h_col=None) -> dict:
    """Signal -> data column for every signal (defaults, then the shorthands,
    then ``columns``)."""
    legacy = {"Phi_HC": Phi_HC_col, "T_h": T_h_col}
    columns = {**DEFAULT_COLUMNS, **{k: v for k, v in legacy.items() if v is not None}, **(columns or {})}
    unknown = set(columns) - set(DEFAULT_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown 6R3C signals {sorted(unknown)}; expected a subset of {list(DEFAULT_COLUMNS)}")
    return columns


def init_data(di, columns) -> dict:
    """Initialization data of a batch (``di`` = its ``data_inputs``): T_i and
    T_h at the forecast start and the lag-window measurements; ``columns`` as
    returned by ``resolve_columns``."""
    return {
        "T_i_0": di[columns["T_i"]][:, 0],
        "T_h_0": di[columns["T_h"]][:, 0],
        "t_":    di["t_array_1Dlag"],
        **{f"{s}_": di[f"{columns[s]}_1Dlag"] for s in LAG_SIGNALS},
    }


def build_data_dict(dataloader, split_name, Phi_HC_col=None, T_h_col=None, columns=None):
    columns = resolve_columns(columns, Phi_HC_col, T_h_col)

    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]
    x  = lambda signal: di[columns[signal]]

    return {
        "t":       b["t_array"],
        "T_i":     x("T_i"),
        "T_h":     x("T_h"),
        "T_a":     x("T_a"),
        "Phi_sol": x("Phi_sol"),
        "Phi_HC":  x("Phi_HC"),
        "data_for_init_state": init_data(di, columns),
    }
