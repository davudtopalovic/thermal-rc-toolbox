import numpy as np
import pandas as pd
from rctoolbox.evaluation.forecast_metrics import evaluate_forecast
from rctoolbox.evaluation.error_threshold import err_thr


def predict_for_batch_and_evaluate(model, batch, y1_set_col, err_thr=err_thr, use_control_day_mask: bool = False):
    t        = batch["t_array"]
    t_datetime  = batch["data_inputs"]["t_datetime"]
    T_i      = batch["data_inputs"]["y1"]
    T_a      = batch["data_inputs"]["T_a"]
    Phi_sol  = batch["data_inputs"]["Phi_sol"]
    Phi_HC   = batch["data_inputs"]["Phi_HC"]
    Phi_HC_0 = Phi_HC[:,0]
    T_i_set  = batch["data_inputs"][y1_set_col]

    T_i_0 = T_i[:,0]; T_s_0 = T_i_0 - 1; T_m_0 = T_i_0 - 2
    y_0 = np.stack([T_m_0, T_s_0, T_i_0], axis=1)

    # State initialization batches
    t_       = batch["data_inputs"]["t_array_1Dlag"]
    T_i_     = batch["data_inputs"]["y1_1Dlag"]
    T_a_     = batch["data_inputs"]["T_a_1Dlag"]
    Phi_sol_ = batch["data_inputs"]["Phi_sol_1Dlag"]
    Phi_HC_  = batch["data_inputs"]["Phi_HC_1Dlag"]
    T_i_set_ = batch["data_inputs"][f"{y1_set_col}_1Dlag"]

    T_i_0_ = T_i_[:,0]; T_s_0_ = T_i_0_ - 1; T_m_0_ = T_i_0_ - 2
    y_0_ = np.stack([T_m_0_, T_s_0_, T_i_0_], axis=1)

    data_for_init_state = {"y_0": y_0, # default approximated initial state
                                # pre-window lag inputs for getting initial state of the model
                                "y_0_": y_0_, "t_": t_, "T_i_": T_i_, "T_a_": T_a_, "Phi_sol_": Phi_sol_, "Phi_HC_": Phi_HC_, "T_i_set_": T_i_set_}
    
    y_0 = model.get_initial_state(data=data_for_init_state)
    Phi_HC_pred, _ = model.forecast_heat_demand(y_0, Phi_HC_0, t, T_a, Phi_sol, T_i_set)

    Y_true = Phi_HC.copy() / 1e3       # convert to kW
    Y_pred = Phi_HC_pred.copy() / 1e3  # convert to kW

    missing_flag = batch["data_inputs"]["missing_flag"] == 1
    Y_true[missing_flag] = np.nan
    Y_pred[missing_flag] = np.nan

    if use_control_day_mask:
        control_day_mask = batch["data_inputs"]["control_day"] == 1
        Y_true[control_day_mask] = np.nan
        Y_pred[control_day_mask] = np.nan
    
    Y_true = Y_true[:,1:] # exclude first time step (t=0) from evaluation as it is used for initialization
    Y_pred = Y_pred[:,1:] # exclude first time step (t=0) from evaluation as it is used for initialization

    p = evaluate_forecast(Y_true, Y_pred, error_thresholds=err_thr)
    p["all_errors"]["t_start"] = pd.DatetimeIndex([row[0] for row in t_datetime])
    p["all_errors"]["t_end"]   = pd.DatetimeIndex([row[-1] for row in t_datetime])
    
    return p

def predict_for_batch(model, batch,  y1_set_col, raw_dfs, use_control_day_mask: bool = False):
    t        = batch["t_array"]
    T_i      = batch["data_inputs"]["y1"]
    T_a      = batch["data_inputs"]["T_a"]
    Phi_sol  = batch["data_inputs"]["Phi_sol"]
    Phi_HC   = batch["data_inputs"]["Phi_HC"]
    Phi_HC_0 = Phi_HC[:,0]
    T_i_set  = batch["data_inputs"][y1_set_col]

    T_i_0 = T_i[:,0]; T_s_0 = T_i_0 - 1; T_m_0 = T_i_0 - 2
    y_0 = np.stack([T_m_0, T_s_0, T_i_0], axis=1)

    # State initialization batches
    t_       = batch["data_inputs"]["t_array_1Dlag"]
    T_i_     = batch["data_inputs"]["y1_1Dlag"]
    T_a_     = batch["data_inputs"]["T_a_1Dlag"]
    Phi_sol_ = batch["data_inputs"]["Phi_sol_1Dlag"]
    Phi_HC_  = batch["data_inputs"]["Phi_HC_1Dlag"]
    T_i_set_ = batch["data_inputs"][f"{y1_set_col}_1Dlag"]

    T_i_0_ = T_i_[:,0]; T_s_0_ = T_i_0_ - 1; T_m_0_ = T_i_0_ - 2
    y_0_ = np.stack([T_m_0_, T_s_0_, T_i_0_], axis=1)

    data_for_init_state = {"y_0": y_0, # default approximated initial state
                                # pre-window lag inputs for getting initial state of the model
                                "y_0_": y_0_, "t_": t_, "T_i_": T_i_, "T_a_": T_a_, "Phi_sol_": Phi_sol_, "Phi_HC_": Phi_HC_, "T_i_set_": T_i_set_}
    
    y_0 = model.get_initial_state(data=data_for_init_state)
    Phi_HC_pred, Y_pred = model.forecast_heat_demand(y_0, Phi_HC_0, t, T_a, Phi_sol, T_i_set)

    if "missing_flag" in batch["data_inputs"]:
        missing_flag = batch["data_inputs"]["missing_flag"] == 1
        Phi_HC_pred[missing_flag] = np.nan
        Y_pred[missing_flag] = np.nan

    if use_control_day_mask:
        if "control_day" in batch["data_inputs"]:
            control_day_mask = batch["data_inputs"]["control_day"] == 1
            Phi_HC_pred[control_day_mask] = np.nan
            Y_pred[control_day_mask] = np.nan

    # Integrate predictions into dataframes.
    # NOTE: deep-copy each sample DataFrame so repeated calls with the same
    # `raw_dfs` (e.g. evaluating multiple models against the same test set)
    # don't mutate the caller's data in place. The previous shallow `.copy()`
    # aliased the inner DataFrames and let the last model's predictions
    # overwrite all earlier ones.
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (sampled_idx, bdf) in enumerate(batch_dfs.items()):
        bdf["Phi_HC_pred"] = Phi_HC_pred[i]

        Y_pred_b = Y_pred[i];
        bdf["y3_pred"] = Y_pred_b[:,0] # T_m
        bdf["y2_pred"] = Y_pred_b[:,1] # T_s
        bdf["y1_pred"] = Y_pred_b[:,2] # T_i

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)

    return dfbs, batch_dfs
