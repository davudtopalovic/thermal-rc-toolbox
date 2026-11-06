"""Batch prediction helper for the 1R1C temperature model."""
import pandas as pd

from rctoolbox.models.rc.OneR1C_Ti.build_data_dict import init_data, resolve_columns


def predict_for_batch(model, batch, raw_dfs, Phi_HC_col="Phi_HC", columns=None):
    c  = resolve_columns(columns, Phi_HC_col)
    di = batch["data_inputs"]
    t  = batch["t_array"]

    # Initial state: built by the model from the measurements (the single state T_i is measured)
    y_0 = model.get_initial_state(data=init_data(di, c))
    Y_pred = model.forecast_temperature(y_0, t, di[c["T_a"]], di[c["Phi_HC"]])
    T_i_pred = Y_pred[:, :, 0]

    # Deep-copy inner DataFrames so repeated calls with the same `raw_dfs`
    # don't mutate the caller's data in place.
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (sampled_idx, bdf) in enumerate(batch_dfs.items()):
        bdf["y1_pred"] = T_i_pred[i]

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)

    return dfbs, batch_dfs
