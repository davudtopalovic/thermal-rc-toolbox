"""Identification problem for the 5R4C envelope + internal-mass model.

Reuses the T_out-aware ``TemperatureProblem`` of the 3R2C internal-mass +
outdoor-link model: only the measured air temperature T_i is supervised
(default ``output_states``), T_e1, T_e2 and T_m are hidden.

The inverse (heat-demand) problem is not implemented for this model.
"""
from rctoolbox.models.rc.three_r2c_im_out.problem import TemperatureProblem

__all__ = ["TemperatureProblem"]
