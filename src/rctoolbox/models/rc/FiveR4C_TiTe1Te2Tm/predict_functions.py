"""Batch prediction helper for the 5R4C envelope + internal-mass model."""
import pandas as pd

from rctoolbox.models.init_data import UNSET
from rctoolbox.models.rc.FiveR4C_TiTe1Te2Tm.build_data_dict import data_for_init_state, resolve_columns


def predict_for_batch(model, batch, raw_dfs, Phi_HC_col="Phi_HC", T_m0=UNSET, columns=None):
    """Simulate every window of the batch. Adds the columns y1_pred (air, T_i),
    T_e1_pred, T_e2_pred (envelope) and T_m_pred (internal mass) to copies of
    the raw DataFrames. The start value of T_m is the model's ``T_m0``
    (``init_config["T_m0"]``); passing ``T_m0`` here is deprecated."""
    c  = resolve_columns(columns, Phi_HC_col)
    di = batch["data_inputs"]
    t  = batch["t_array"]

    y_0 = model.get_initial_state(data=data_for_init_state(di, T_m0=T_m0, columns=c))
    Y_pred = model.forecast_temperature(y_0, t, di[c["T_a"]], di[c["Phi_HC"]], di[c["T_out"]])

    # Deep-copy inner DataFrames so repeated calls with the same `raw_dfs`
    # don't mutate the caller's data in place.
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (sampled_idx, bdf) in enumerate(batch_dfs.items()):
        bdf["y1_pred"]   = Y_pred[i, :, 0]
        bdf["T_e1_pred"] = Y_pred[i, :, 1]
        bdf["T_e2_pred"] = Y_pred[i, :, 2]
        bdf["T_m_pred"]  = Y_pred[i, :, 3]

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)

    return dfbs, batch_dfs
