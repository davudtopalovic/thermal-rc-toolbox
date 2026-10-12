"""Batch prediction helper for the 1R1C temperature model."""
import numpy as np
import pandas as pd


def predict_for_batch(model, batch, raw_dfs, Phi_HC_col="Phi_HC"):
    di = batch["data_inputs"]
    t  = batch["t_array"]

    T_a     = di["T_a"]
    Phi_HC  = di[Phi_HC_col]

    # State initialization: the single state T_i is measured
    data_for_init_state = {
        "y_0":      np.asarray(di["y1"][:, [0]], dtype=np.float64),
        "y_0_":     np.asarray(di["y1_1Dlag"][:, [0]], dtype=np.float64),
        "t_":       di["t_array_1Dlag"],
        "T_i_":     di["y1_1Dlag"],
        "T_a_":     di["T_a_1Dlag"],
        "Phi_HC_":  di[f"{Phi_HC_col}_1Dlag"],
    }

    y_0 = model.get_initial_state(data=data_for_init_state)
    Y_pred = model.forecast_temperature(y_0, t, T_a, Phi_HC)
    T_i_pred = Y_pred[:, :, 0]

    # Deep-copy inner DataFrames so repeated calls with the same `raw_dfs`
    # don't mutate the caller's data in place.
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (sampled_idx, bdf) in enumerate(batch_dfs.items()):
        bdf["y1_pred"] = T_i_pred[i]

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)

    return dfbs, batch_dfs
