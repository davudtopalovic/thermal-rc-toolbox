"""Data dict for ``TemperatureProblem`` from a toolbox DataLoader batch.

The dataloader needs the features ``y1`` (measured T_i) and the model inputs
``Freezer2R2C_IM().all_inputs``; ``column_map`` renames them if needed. Hidden
states start at the measured air temperature (``default`` initialization) or
are estimated by the Kalman filter over the lag window (``kalman``).
"""
from rctoolbox.models.rc.freezer_models._base import build_data_dict as _build, initial_state as _init
from rctoolbox.models.rc.freezer_models.fz_2r2c_im.model import Freezer2R2C_IM


def initial_state(T_i):
    return _init(Freezer2R2C_IM(), T_i)


def build_data_dict(dataloader, split_name, column_map=None):
    return _build(Freezer2R2C_IM(), dataloader, split_name, column_map)
