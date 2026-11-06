"""Batch prediction helper for the 3R3C temperature model.

The 3R3C state vector [T_i, T_h, T_e] has the same layout as the 6R3C's
[T_i, T_h, T_m] (two measured nodes followed by one hidden node) and the same
initialization data (T_i_0, T_h_0 and the lag window), so the 6R3C helper is
re-exported here; the starting state comes from the model passed in (3R3C:
T_e = T_i - 2), and the columns are named after its states (T_e_pred).
"""
from rctoolbox.models.rc.six_r3c_eniso13790.predict_functions import predict_for_batch  # noqa: F401
