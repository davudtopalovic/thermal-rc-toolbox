r"""Freezer model: 3R3C: air + internal mass + evaporator coil, COP(T_out) and outdoor link.

                         Phi = Phi_HC (1 + beta (T_out - 10))
                                        |
       T_a ---R_ia--- T_i ---g(s)------ T_e
       T_out-R_iout--/ |  \              |
                      C_i  R_im          C_e
                             \
                              T_m --- C_m

Continuous-time heat balances (derivation: ``derivation.ipynb``):

    C_i dT_i/dt = (T_a - T_i) / R_ia + (T_m - T_i) / R_im + (s / R_ie_on + (1 - s) / R_ie_off) * (T_e - T_i) + (T_out - T_i) / R_iout
    C_m dT_m/dt = (T_i - T_m) / R_im
    C_e dT_e/dt = (s / R_ie_on + (1 - s) / R_ie_off) * (T_i - T_e) + Phi_HC * (1 + beta * (T_out - 10))

States ('T_i', 'T_m', 'T_e'); inputs ('Phi_HC', 'T_a', 'Phi_HC_Tout', 'T_out') + 's' (compressor on-fraction, fan-gated coupling).
Free parameters: C_i, R_ia, R_im, C_m, R_ie_on, R_ie_off, C_e, R_iout, beta.

Configuration of :class:`~rctoolbox.models.rc.freezer_models.network.FreezerNetwork`
with ``SPEC = {'mass': True, 'coil': True, 'cop_tout': True, 'link_out': True}``; discretization (exact ZOH), simulation and the
Kalman initialization of the hidden states are shared
(``freezer_models/_base.py``).
"""
from rctoolbox.models.rc.freezer_models.network import FreezerNetwork


class Freezer3R3C_IM_EV_COP(FreezerNetwork):
    SPEC = {'mass': True, 'coil': True, 'cop_tout': True, 'link_out': True}
