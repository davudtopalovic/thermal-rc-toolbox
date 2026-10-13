"""Batch prediction helper for the 2R2C internal-mass model."""
import numpy as np
import pandas as pd

from rctoolbox.models.rc.two_r2c_im.build_data_dict import initial_state


def predict_for_batch(model, batch, raw_dfs, Phi_HC_col="Phi_HC"):
    """Simulate every window of the batch. Adds the columns y1_pred (air, T_i)
    and T_m_pred (internal mass) to copies of the raw DataFrames."""
    di = batch["data_inputs"]
    t  = batch["t_array"]

    T_a     = di["T_a"]
    Phi_HC  = di[Phi_HC_col]

    # State initialization: T_i measured, T_m_0 = -17 °C
    data_for_init_state = {
        "y_0":      initial_state(di["y1"]),
        "y_0_":     initial_state(di["y1_1Dlag"]),
        "t_":       di["t_array_1Dlag"],
        "T_i_":     di["y1_1Dlag"],
        "T_a_":     di["T_a_1Dlag"],
        "Phi_HC_":  di[f"{Phi_HC_col}_1Dlag"],
    }

    y_0 = model.get_initial_state(data=data_for_init_state)
    Y_pred = model.forecast_temperature(y_0, t, T_a, Phi_HC)

    # Deep-copy inner DataFrames so repeated calls with the same `raw_dfs`
    # don't mutate the caller's data in place.
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (sampled_idx, bdf) in enumerate(batch_dfs.items()):
        bdf["y1_pred"] = Y_pred[i, :, 0]
        bdf["T_m_pred"] = Y_pred[i, :, 1]

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)

    return dfbs, batch_dfs
