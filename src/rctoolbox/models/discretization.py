"""Exact zero-order-hold (ZOH) discretization for linear RC state-space models.

Textbook form. Continuous-time model:

    dx/dt = A x + B u

where u collects ALL inputs held constant over each step (zero-order hold) —
controls and disturbances alike; the model class decides the column layout of
B (e.g. TwoR2C stacks its control column and disturbance columns before
calling this). The exact discrete one-step map is

    x_k = A_d x_{k-1} + B_d u_k,

    A_d = expm(A*dt),
    B_d = integral_0^dt expm(A s) ds * B = A^{-1} (expm(A*dt) - I) B,

evaluated here with the augmented-matrix exponential

    expm([[A, B], [0, 0]] * dt) = [[A_d, B_d], [0, I]],

which requires no inverse of A (singular A, e.g. a network without an ambient
loss path, is fine) and works for any state/input dimensions. Exact for
interval-constant inputs and unconditionally stable for stable A. The
k-labeled inputs are the interval averages over (t_{k-1}, t_k] (the
project-wide right-labeled convention).

Forward Euler needs no discretizer: it steps directly with the continuous
matrices, x_k = x_{k-1} + dt * (A x_{k-1} + B u_k). It is only conditionally
stable — a mode with time constant tau diverges for dt > 2*tau.
"""
import numpy as np
from scipy.linalg import expm


def zoh_discretize(A, B, dt):
    """Exact ZOH discrete matrices (A_d, B_d) for any shapes of A (n,n), B (n,m)."""
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]
