"""Parameter space of ``Freezer3R3C_IM_EV``: default bounds are
``Freezer3R3C_IM_EV.DEFAULT_BOUNDS``; override with ``building_config["free_param_bounds"]``."""
from rctoolbox.models.rc.freezer_models._base import make_parameter_space
from rctoolbox.models.rc.freezer_models.fz_3r3c_im_ev.model import Freezer3R3C_IM_EV

ParameterSpace_Freezer3R3C_IM_EV = make_parameter_space(Freezer3R3C_IM_EV)
