"""Batch prediction helper for the 3R2C internal-mass + outdoor-link model."""
import pandas as pd

from rctoolbox.models.rc.three_r2c_im_out.build_data_dict import data_for_init_state, T_M0_DEFAULT


def predict_for_batch(model, batch, raw_dfs, Phi_HC_col="Phi_HC", T_m0=T_M0_DEFAULT):
    """Simulate every window of the batch. Adds the columns y1_pred (air, T_i)
    and T_m_pred (internal mass) to copies of the raw DataFrames."""
    di = batch["data_inputs"]
    t  = batch["t_array"]

    y_0 = model.get_initial_state(data=data_for_init_state(di, Phi_HC_col, T_m0))
    Y_pred = model.forecast_temperature(y_0, t, di["T_a"], di[Phi_HC_col], di["T_out"])

    # Deep-copy inner DataFrames so repeated calls with the same `raw_dfs`
    # don't mutate the caller's data in place.
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (sampled_idx, bdf) in enumerate(batch_dfs.items()):
        bdf["y1_pred"] = Y_pred[i, :, 0]
        bdf["T_m_pred"] = Y_pred[i, :, 1]

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)

    return dfbs, batch_dfs
