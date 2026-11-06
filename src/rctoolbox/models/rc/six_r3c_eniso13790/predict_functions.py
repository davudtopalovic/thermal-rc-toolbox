import pandas as pd

from rctoolbox.models.rc.six_r3c_eniso13790.build_data_dict import init_data, resolve_columns


def predict_for_batch(model, batch, raw_dfs, columns=None):
    """Simulate every window of the batch. ``columns`` maps the model signals to
    data columns (``build_data_dict.resolve_columns``, same defaults as the
    training data dict). Adds ``<state>_pred`` for every state (T_i_pred,
    T_h_pred, T_m_pred) to copies of the raw DataFrames."""
    c  = resolve_columns(columns)
    di = batch["data_inputs"]
    t  = batch["t_array"]

    T_a      = di[c["T_a"]]
    Phi_sol  = di[c["Phi_sol"]]
    Phi_HC   = di[c["Phi_HC"]]

    # Initial state: built by the model from the measurements (seed + init method)
    y_0 = model.get_initial_state(data=init_data(di, c))
    Y_pred = model.forecast_temperature(y_0, t, T_a, Phi_sol, Phi_HC)

    # Deep-copy inner DataFrames so repeated calls with the same `raw_dfs`
    # don't mutate the caller's data in place. See sibling 5R1C fix.
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (sampled_idx, bdf) in enumerate(batch_dfs.items()):
        for j, name in enumerate(model.STATE_NAMES):   # T_i, T_h, T_m
            bdf[f"{name}_pred"] = Y_pred[i, :, j]

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)
    
    return dfbs, batch_dfs