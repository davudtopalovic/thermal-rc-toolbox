r"""Freezer model: 2R2C: air + internal mass.

                     Phi_HC
                       |
       T_a ---R_ia--- T_i ---R_im--- T_m
                       |              |
                      C_i            C_m

Continuous-time heat balances (derivation: ``derivation.ipynb``):

    C_i dT_i/dt = (T_a - T_i) / R_ia + (T_m - T_i) / R_im + Phi_HC
    C_m dT_m/dt = (T_i - T_m) / R_im

States ('T_i', 'T_m'); inputs ('Phi_HC', 'T_a').
Free parameters: C_i, R_ia, R_im, C_m.

Configuration of :class:`~rctoolbox.models.rc.freezer_models.network.FreezerNetwork`
with ``SPEC = {'mass': True}``; discretization (exact ZOH), simulation and the
Kalman initialization of the hidden states are shared
(``freezer_models/_base.py``).
"""
from rctoolbox.models.rc.freezer_models.network import FreezerNetwork


class Freezer2R2C_IM(FreezerNetwork):
    SPEC = {'mass': True}
