r"""Freezer model: 3R3C: air + internal mass + evaporator coil, coil linked to the cold room.

                                      Phi_HC
                                        |
       T_a ---R_ia--- T_i ---g(s)------ T_e ---R_ea--- T_a
                       |  \              |
                      C_i  R_im          C_e
                             \
                              T_m --- C_m

Continuous-time heat balances (derivation: ``derivation.ipynb``):

    C_i dT_i/dt = (T_a - T_i) / R_ia + (T_m - T_i) / R_im + (s / R_ie_on + (1 - s) / R_ie_off) * (T_e - T_i)
    C_m dT_m/dt = (T_i - T_m) / R_im
    C_e dT_e/dt = (s / R_ie_on + (1 - s) / R_ie_off) * (T_i - T_e) + (T_a - T_e) / R_ea + Phi_HC

States ('T_i', 'T_m', 'T_e'); inputs ('Phi_HC', 'T_a') + 's' (compressor on-fraction, fan-gated coupling).
Free parameters: C_i, R_ia, R_im, C_m, R_ie_on, R_ie_off, C_e, R_ea.

Configuration of :class:`~rctoolbox.models.rc.freezer_models.network.FreezerNetwork`
with ``SPEC = {'mass': True, 'coil': True, 'coil_to_ta': True}``; discretization (exact ZOH), simulation and the
Kalman initialization of the hidden states are shared
(``freezer_models/_base.py``).
"""
from rctoolbox.models.rc.freezer_models.network import FreezerNetwork


class Freezer3R3C_IM_EV_TA(FreezerNetwork):
    SPEC = {'mass': True, 'coil': True, 'coil_to_ta': True}
