"""Identification problem for the 2R2C internal-mass model.

The 1R1C ``TemperatureProblem`` is state-count agnostic (it simulates with
``forecast_temperature(y_0, t, T_a, Phi_HC)`` and supervises the states listed
in ``output_states``, default ``[(T_i_idx, "T_i")]``), so it is reused here:
only the measured air temperature T_i is supervised, T_m is hidden.

The inverse (heat-demand) problem is not implemented for this model.
"""
from rctoolbox.models.rc.one_r1c.problem import TemperatureProblem

__all__ = ["TemperatureProblem"]
