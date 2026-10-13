"""Identification problem of ``Freezer3R3C_IM_EV_OUT`` (shared freezer problem: inputs passed by name,
supervised states in ``output_states``, default the measured air T_i)."""
from rctoolbox.models.rc.freezer_models._problem import FreezerTemperatureProblem as TemperatureProblem

__all__ = ["TemperatureProblem"]
