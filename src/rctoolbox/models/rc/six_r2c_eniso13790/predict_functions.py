"""Batch prediction helpers for the 6R2C model.

State vector is [T_m1, T_m2, T_s, T_i] — 4 columns instead of 3.
Approximate initial state uses T_i as anchor with fixed offsets, same convention
as 5R1C: T_m1 = T_i - 2, T_m2 = T_i - 1.5, T_s = T_i - 1.
"""
import numpy as np
import pandas as pd
from rctoolbox.evaluation.forecast_metrics import evaluate_forecast


def _make_initial_state(T_i_col):
    """T_i_col shape (B,) -> y_0 shape (B, 4) with approximate offsets."""
    T_m1_0 = T_i_col - 2.0
    T_m2_0 = T_i_col - 1.5
    T_s_0  = T_i_col - 1.0
    T_i_0  = T_i_col
    return np.stack([T_m1_0, T_m2_0, T_s_0, T_i_0], axis=1)


def predict_for_batch_and_evaluate(model, batch, y1_set_col, err_thr, use_control_day_mask: bool = False):
    t         = batch["t_array"]
    t_datetime = batch["data_inputs"]["t_datetime"]
    T_i       = batch["data_inputs"]["y1"]
    T_a       = batch["data_inputs"]["T_a"]
    Phi_sol   = batch["data_inputs"]["Phi_sol"]
    Phi_HC    = batch["data_inputs"]["Phi_HC"]
    Phi_HC_0  = Phi_HC[:, 0]
    T_i_set   = batch["data_inputs"][y1_set_col]

    y_0 = _make_initial_state(T_i[:, 0])

    T_i_   = batch["data_inputs"]["y1_1Dlag"]
    T_a_   = batch["data_inputs"]["T_a_1Dlag"]
    Phi_sol_ = batch["data_inputs"]["Phi_sol_1Dlag"]
    Phi_HC_  = batch["data_inputs"]["Phi_HC_1Dlag"]
    t_       = batch["data_inputs"]["t_array_1Dlag"]
    T_i_set_ = batch["data_inputs"][f"{y1_set_col}_1Dlag"]

    y_0_ = _make_initial_state(T_i_[:, 0])

    data_for_init_state = {
        "y_0": y_0, "y_0_": y_0_, "t_": t_,
        "T_i_": T_i_, "T_a_": T_a_, "Phi_sol_": Phi_sol_,
        "Phi_HC_": Phi_HC_, "T_i_set_": T_i_set_,
    }

    y_0 = model.get_initial_state(data=data_for_init_state)
    Phi_HC_pred, _ = model.forecast_heat_demand(y_0, Phi_HC_0, t, T_a, Phi_sol, T_i_set)

    Y_true = Phi_HC.copy() / 1e3
    Y_pred = Phi_HC_pred.copy() / 1e3

    missing_flag = batch["data_inputs"]["missing_flag"] == 1
    Y_true[missing_flag] = np.nan
    Y_pred[missing_flag] = np.nan

    if use_control_day_mask:
        control_day_mask = batch["data_inputs"]["control_day"] == 1
        Y_true[control_day_mask] = np.nan
        Y_pred[control_day_mask] = np.nan

    Y_true = Y_true[:, 1:]
    Y_pred = Y_pred[:, 1:]

    p = evaluate_forecast(Y_true, Y_pred, error_thresholds=err_thr)
    p["all_errors"]["t_start"] = pd.DatetimeIndex([row[0]  for row in t_datetime])
    p["all_errors"]["t_end"]   = pd.DatetimeIndex([row[-1] for row in t_datetime])
    return p


def predict_for_batch(model, batch, y1_set_col, raw_dfs, use_control_day_mask: bool = False):
    t         = batch["t_array"]
    T_i       = batch["data_inputs"]["y1"]
    T_a       = batch["data_inputs"]["T_a"]
    Phi_sol   = batch["data_inputs"]["Phi_sol"]
    Phi_HC    = batch["data_inputs"]["Phi_HC"]
    Phi_HC_0  = Phi_HC[:, 0]
    T_i_set   = batch["data_inputs"][y1_set_col]

    y_0 = _make_initial_state(T_i[:, 0])

    T_i_   = batch["data_inputs"]["y1_1Dlag"]
    T_a_   = batch["data_inputs"]["T_a_1Dlag"]
    Phi_sol_ = batch["data_inputs"]["Phi_sol_1Dlag"]
    Phi_HC_  = batch["data_inputs"]["Phi_HC_1Dlag"]
    t_       = batch["data_inputs"]["t_array_1Dlag"]
    T_i_set_ = batch["data_inputs"][f"{y1_set_col}_1Dlag"]

    y_0_ = _make_initial_state(T_i_[:, 0])

    data_for_init_state = {
        "y_0": y_0, "y_0_": y_0_, "t_": t_,
        "T_i_": T_i_, "T_a_": T_a_, "Phi_sol_": Phi_sol_,
        "Phi_HC_": Phi_HC_, "T_i_set_": T_i_set_,
    }

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

    batch_dfs = {k: v.copy() for k, v in raw_dfs.items()}
    for i, (_, bdf) in enumerate(batch_dfs.items()):
        bdf["Phi_HC_pred"] = Phi_HC_pred[i]
        y_p = Y_pred[i]
        bdf["y4_pred"] = y_p[:, 0]   # T_m1
        bdf["y3_pred"] = y_p[:, 1]   # T_m2
        bdf["y2_pred"] = y_p[:, 2]   # T_s
        bdf["y1_pred"] = y_p[:, 3]   # T_i

    dfbs = pd.concat([df for df in batch_dfs.values()], axis=0)
    dfbs.dropna(inplace=True)
    return dfbs, batch_dfs
