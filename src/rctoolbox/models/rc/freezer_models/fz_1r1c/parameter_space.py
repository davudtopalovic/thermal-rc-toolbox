"""Parameter space of ``Freezer1R1C``: default bounds are
``Freezer1R1C.DEFAULT_BOUNDS``; override with ``building_config["free_param_bounds"]``."""
from rctoolbox.models.rc.freezer_models._base import make_parameter_space
from rctoolbox.models.rc.freezer_models.fz_1r1c.model import Freezer1R1C

ParameterSpace_Freezer1R1C = make_parameter_space(Freezer1R1C)
