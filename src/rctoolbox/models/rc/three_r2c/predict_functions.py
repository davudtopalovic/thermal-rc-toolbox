"""Batch prediction helpers for the 3R2C inverse model.

The state vector [T_i, T_e] and the init-data schema are identical to the
2R2C, so the helpers are shared with two_r2c — re-exported here so the
3R2C notebooks can import from their own package.
"""
from rctoolbox.models.rc.two_r2c.predict_functions import (  # noqa: F401
    _build_init_data,
    predict_for_batch,
    predict_for_batch_and_evaluate,
)
