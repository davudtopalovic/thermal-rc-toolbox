r"""Freezer model: 1R1C (baseline).

                     Phi_HC
                       |
       T_a ---R_ia--- T_i
                       |
                      C_i

Continuous-time heat balances (derivation: ``derivation.ipynb``):

    C_i dT_i/dt = (T_a - T_i) / R_ia + Phi_HC

States ('T_i',); inputs ('Phi_HC', 'T_a').
Free parameters: C_i, R_ia.

Configuration of :class:`~rctoolbox.models.rc.freezer_models.network.FreezerNetwork`
with ``SPEC = {}``; discretization (exact ZOH), simulation and the
Kalman initialization of the hidden states are shared
(``freezer_models/_base.py``).
"""
from rctoolbox.models.rc.freezer_models.network import FreezerNetwork


class Freezer1R1C(FreezerNetwork):
    SPEC = {}
