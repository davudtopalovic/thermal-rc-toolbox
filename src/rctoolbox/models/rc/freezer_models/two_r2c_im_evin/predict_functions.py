"""Batch prediction helper for the standalone 2R2C internal-mass model driven
by the measured evaporator temperature."""
import pandas as pd

from rctoolbox.models.rc.freezer_models.two_r2c_im_evin.build_data_dict import initial_state


def predict_for_batch(model, batch, raw_dfs, T_m0=-17.0):
    """Simulate every window of the batch. Adds the columns y1_pred (air, T_i)
    and T_m_pred (internal mass) to copies of the raw DataFrames."""
    di = batch["data_inputs"]
    t  = batch["t_array"]

    # State initialization: T_i measured, T_m_0 (then per the model's init method)
    data_for_init_state = {
        "y_0":    initial_state(di["y1"], T_m0),
        "y_0_":   initial_state(di["y1_1Dlag"], T_m0),
        "t_":     di["t_array_1Dlag"],
        "T_i_":   di["y1_1Dlag"],
        "T_a_":   di["T_a_1Dlag"],
        "T_ev_":  di["T_ev_1Dlag"],
        "on_":    di["on_1Dlag"],
    }

    y_0 = model.get_initial_state(data=data_for_init_state)
    Y_pred = model.forecast_temperature(y_0, t, di["T_a"], di["T_ev"], di["on"])

    # Deep-copy inner DataFrames so repeated calls with the same `raw_dfs`
    # don't mutate the caller's data in place.
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (sampled_idx, bdf) in enumerate(batch_dfs.items()):
        bdf["y1_pred"] = Y_pred[i, :, 0]
        bdf["T_m_pred"] = Y_pred[i, :, 1]

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)

    return dfbs, batch_dfs
