"""Parameter space of ``Freezer3R3C_IM_EV_COP``: default bounds are
``Freezer3R3C_IM_EV_COP.DEFAULT_BOUNDS``; override with ``building_config["free_param_bounds"]``."""
from rctoolbox.models.rc.freezer_models._base import make_parameter_space
from rctoolbox.models.rc.freezer_models.fz_3r3c_im_ev_cop.model import Freezer3R3C_IM_EV_COP

ParameterSpace_Freezer3R3C_IM_EV_COP = make_parameter_space(Freezer3R3C_IM_EV_COP)
