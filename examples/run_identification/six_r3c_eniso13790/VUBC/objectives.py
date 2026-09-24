"""Custom objective for the VUBC 6R3C temperature identification.

Referenced from identification_config.json as
``"objective_function": "objectives.py:weighted_temp_objective"`` and hence usable
from the command line:

    rct-train --building-config VUBC \\
              --data-path examples/run_identification/six_r3c_eniso13790/VUBC/data_train.joblib \\
              --identification-config examples/run_identification/six_r3c_eniso13790/VUBC/identification_config.json \\
              --results-directory-path <out> --seed 23 --save-figures --save-dfRtopN
"""
import numpy as np

W_TI = 0.95   # weight on the indoor-air error; the rest goes to the heating circuit


def weighted_temp_objective(y_true, y_pred):
    """Weighted RMSE over the two supervised states.

    y_true, y_pred : ndarray, shape (B, T, 2) — states [T_i, T_h] as stacked by
    ``TemperatureProblem`` with ``output_states=[[0, "T_i"], [1, "T_h"]]``.
    """
    rmse_Ti = float(np.sqrt(np.nanmean((y_true[..., 0] - y_pred[..., 0]) ** 2)))
    rmse_Th = float(np.sqrt(np.nanmean((y_true[..., 1] - y_pred[..., 1]) ** 2)))
    return W_TI * rmse_Ti + (1.0 - W_TI) * rmse_Th
