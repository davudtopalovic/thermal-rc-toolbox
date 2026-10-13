r"""Freezer model: 1R1C driven by the measured evaporator temperature (COP-free).

       T_a ---R_ia--- T_i ---g(s)--- T_ev (measured)     g(s) = s/R_ie_on + (1-s)/R_ie_off
                       |
                      C_i

Continuous-time heat balances (derivation: ``derivation.ipynb``):

    C_i dT_i/dt = (T_a - T_i) / R_ia + (s / R_ie_on + (1 - s) / R_ie_off) * (T_ev - T_i)

States ('T_i',); inputs ('T_a', 'T_ev') + 's' (compressor on-fraction, fan-gated coupling).
Free parameters: C_i, R_ia, R_ie_on, R_ie_off.

Configuration of :class:`~rctoolbox.models.rc.freezer_models.network.FreezerNetwork`
with ``SPEC = {'ev_input': True}``; discretization (exact ZOH), simulation and the
Kalman initialization of the hidden states are shared
(``freezer_models/_base.py``).
"""
from rctoolbox.models.rc.freezer_models.network import FreezerNetwork


class Freezer1R1C_EVIN(FreezerNetwork):
    SPEC = {'ev_input': True}
