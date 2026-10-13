"""Identification problem of ``Freezer1R1C`` (shared freezer problem: inputs passed by name,
supervised states in ``output_states``, default the measured air T_i)."""
from rctoolbox.models.rc.freezer_models._problem import FreezerTemperatureProblem as TemperatureProblem

__all__ = ["TemperatureProblem"]
