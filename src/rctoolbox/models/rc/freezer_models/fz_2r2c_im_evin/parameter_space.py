"""Parameter space of ``Freezer2R2C_IM_EVIN``: default bounds are
``Freezer2R2C_IM_EVIN.DEFAULT_BOUNDS``; override with ``building_config["free_param_bounds"]``."""
from rctoolbox.models.rc.freezer_models._base import make_parameter_space
from rctoolbox.models.rc.freezer_models.fz_2r2c_im_evin.model import Freezer2R2C_IM_EVIN

ParameterSpace_Freezer2R2C_IM_EVIN = make_parameter_space(Freezer2R2C_IM_EVIN)
