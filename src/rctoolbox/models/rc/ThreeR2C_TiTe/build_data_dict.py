"""Build the per-batch data dicts for calibrating a 3R2C model.

The state vector [T_i, T_e] and the data schema are identical to the 2R2C, so
the helpers are shared with TwoR2C_TiTe — re-exported here so the 3R2C
notebooks can import from their own package.
"""
from rctoolbox.models.rc.TwoR2C_TiTe.build_data_dict import (  # noqa: F401
    DEFAULT_COLUMNS,
    build_data_dict,
    build_data_dict_heat_demand,
    init_data,
    resolve_columns,
)
