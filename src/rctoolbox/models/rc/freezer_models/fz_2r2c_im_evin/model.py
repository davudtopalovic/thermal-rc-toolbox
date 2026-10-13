r"""Freezer model: 2R2C: air + internal mass, driven by the measured evaporator temperature (COP-free).

       T_a ---R_ia--- T_i ---g(s)--- T_ev (measured)
                       |  \
                      C_i  R_im
                             \
                              T_m --- C_m

Continuous-time heat balances (derivation: ``derivation.ipynb``):

    C_i dT_i/dt = (T_a - T_i) / R_ia + (T_m - T_i) / R_im + (s / R_ie_on + (1 - s) / R_ie_off) * (T_ev - T_i)
    C_m dT_m/dt = (T_i - T_m) / R_im

States ('T_i', 'T_m'); inputs ('T_a', 'T_ev') + 's' (compressor on-fraction, fan-gated coupling).
Free parameters: C_i, R_ia, R_im, C_m, R_ie_on, R_ie_off.

Configuration of :class:`~rctoolbox.models.rc.freezer_models.network.FreezerNetwork`
with ``SPEC = {'mass': True, 'ev_input': True}``; discretization (exact ZOH), simulation and the
Kalman initialization of the hidden states are shared
(``freezer_models/_base.py``).
"""
from rctoolbox.models.rc.freezer_models.network import FreezerNetwork


class Freezer2R2C_IM_EVIN(FreezerNetwork):
    SPEC = {'mass': True, 'ev_input': True}
