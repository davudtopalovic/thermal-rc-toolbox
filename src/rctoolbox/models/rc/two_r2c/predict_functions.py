"""Batch prediction helpers for the inverse (setpoint-to-power) 2R2C model.

Mirrors five_r1c_eniso13790.predict_functions with the 2R2C state vector
[T_i, T_e]; the default hidden-state guess is T_e = T_i - 2 (refined by the
model's static-inversion / regime-aware initializers at run time).
"""
import numpy as np
import pandas as pd

from rctoolbox.evaluation.forecast_metrics import evaluate_forecast
from rctoolbox.evaluation.error_threshold import err_thr


def _build_init_data(batch, y1_set_col):
    di = batch["data_inputs"]

    T_i_0 = di["y1"][:, 0]
    y_0 = np.stack([T_i_0, T_i_0 - 2], axis=1)

    T_i_ = di["y1_1Dlag"]
    T_i_0_ = T_i_[:, 0]
    y_0_ = np.stack([T_i_0_, T_i_0_ - 2], axis=1)

    return {
        "y_0": y_0,  # default approximated initial state
        # pre-window lag inputs for getting the initial state of the model
        "y_0_":     y_0_,
        "t_":       di["t_array_1Dlag"],
        "T_i_":     T_i_,
        "T_a_":     di["T_a_1Dlag"],
        "Phi_sol_": di["Phi_sol_1Dlag"],
        "Phi_HC_":  di["Phi_HC_1Dlag"],
        "T_i_set_": di[f"{y1_set_col}_1Dlag"],
    }


def predict_for_batch_and_evaluate(model, batch, y1_set_col, err_thr=err_thr, use_control_day_mask: bool = False):
    t          = batch["t_array"]
    t_datetime = batch["data_inputs"]["t_datetime"]
    T_a        = batch["data_inputs"]["T_a"]
    Phi_sol    = batch["data_inputs"]["Phi_sol"]
    Phi_HC     = batch["data_inputs"]["Phi_HC"]
    Phi_HC_0   = Phi_HC[:, 0]
    T_i_set    = batch["data_inputs"][y1_set_col]

    y_0 = model.get_initial_state(data=_build_init_data(batch, y1_set_col))
    Phi_HC_pred, _ = model.forecast_heat_demand_fast(y_0, Phi_HC_0, t, T_a, Phi_sol, T_i_set)

    Y_true = Phi_HC.copy() / 1e3       # convert to kW
    Y_pred = Phi_HC_pred.copy() / 1e3  # convert to kW

    missing_flag = batch["data_inputs"]["missing_flag"] == 1
    Y_true[missing_flag] = np.nan
    Y_pred[missing_flag] = np.nan

    if use_control_day_mask:
        control_day_mask = batch["data_inputs"]["control_day"] == 1
        Y_true[control_day_mask] = np.nan
        Y_pred[control_day_mask] = np.nan

    Y_true = Y_true[:, 1:]  # exclude first time step (t=0) from evaluation as it is used for initialization
    Y_pred = Y_pred[:, 1:]  # exclude first time step (t=0) from evaluation as it is used for initialization

    p = evaluate_forecast(Y_true, Y_pred, error_thresholds=err_thr)
    p["all_errors"]["t_start"] = pd.DatetimeIndex([row[0] for row in t_datetime])
    p["all_errors"]["t_end"]   = pd.DatetimeIndex([row[-1] for row in t_datetime])

    return p


def predict_for_batch(model, batch, y1_set_col, raw_dfs, use_control_day_mask: bool = False):
    t        = batch["t_array"]
    T_a      = batch["data_inputs"]["T_a"]
    Phi_sol  = batch["data_inputs"]["Phi_sol"]
    Phi_HC   = batch["data_inputs"]["Phi_HC"]
    Phi_HC_0 = Phi_HC[:, 0]
    T_i_set  = batch["data_inputs"][y1_set_col]

    y_0 = model.get_initial_state(data=_build_init_data(batch, y1_set_col))
    Phi_HC_pred, Y_pred = model.forecast_heat_demand_fast(y_0, Phi_HC_0, t, T_a, Phi_sol, T_i_set)

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
    # `raw_dfs` don't mutate the caller's data in place (see the 5R1C twin).
    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (sampled_idx, bdf) in enumerate(batch_dfs.items()):
        bdf["Phi_HC_pred"] = Phi_HC_pred[i]

        Y_pred_b = Y_pred[i]
        bdf["y1_pred"] = Y_pred_b[:, 0]  # T_i
        bdf["Te_pred"] = Y_pred_b[:, 1]  # T_e (hidden envelope state)

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)

    return dfbs, batch_dfs
