import numpy as np
import pandas as pd

def predict_for_batch(model, batch, raw_dfs):
    
    di = batch["data_inputs"]
    t  = batch["t_array"]

    T_i = di["y1"]
    T_h = di["y2"]
    T_a      = di["T_a"]
    Phi_sol  = di["Phi_sol"]
    Phi_HC   = di["Phi_HC"]

    # State initialization
    T_i0 = T_i[:, 0]
    T_h0 = T_h[:, 0]
    T_m0 = T_i0 -  1
    y_0  = np.stack([T_i0, T_h0, T_m0], axis=1)

    T_i_ = di["y1_1Dlag"]
    T_h_ = di["y2_1Dlag"]
    T_i0_ = T_i_[:, 0]
    T_h0_ = T_h_[:, 0]
    T_m0_ = T_i0_ - 1
    y_0_ = np.stack([T_i0_, T_h0_, T_m0_], axis=1)

    data_for_init_state =  {
            "y_0":      y_0,      "y_0_":     y_0_,
            "t_":       di["t_array_1Dlag"],
            "T_i_":     T_i_,
            "T_h_":     T_h_,
            "T_a_":     di["T_a_1Dlag"],
            "Phi_sol_": di["Phi_sol_1Dlag"],
            "Phi_HC_":  di["Phi_HC_1Dlag"],
        }

    y_0 = model.get_initial_state(data=data_for_init_state)
    Y_pred = model.forecast_temperature(y_0, t, T_a, Phi_sol, Phi_HC)
    T_i_pred = Y_pred[:, :, 0]
    T_h_pred = Y_pred[:, :, 1]
    T_e_pred = Y_pred[:, :, 2]

    # Deep-copy inner DataFrames so repeated calls with the same `raw_dfs`
    # don't mutate the caller's data in place. See sibling 5R1C fix.
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (sampled_idx, bdf) in enumerate(batch_dfs.items()):
        bdf["T_i_pred"] = T_i_pred[i]
        bdf["T_h_pred"] = T_h_pred[i]
        bdf["T_e_pred"] = T_e_pred[i]

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)
    
    return dfbs, batch_dfs