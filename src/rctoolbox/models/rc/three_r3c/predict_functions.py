"""Batch prediction helper for the 3R3C temperature model.

The 3R3C state vector [T_i, T_h, T_e] has the same layout as the 6R3C's
[T_i, T_h, T_m] (two measured nodes followed by one hidden node), and the
init-data schema is identical, so the 6R3C helper is re-exported here. With
the ``dynamic`` initialization the hidden node's starting guess is replaced
by the lag-window simulation, so the two models can share the same batches.
"""
from rctoolbox.models.rc.six_r3c_eniso13790.predict_functions import predict_for_batch  # noqa: F401
