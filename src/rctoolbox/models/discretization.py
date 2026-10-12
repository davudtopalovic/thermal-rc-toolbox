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


import numpy as np
from scipy.linalg import expm, solve

def zoh_discretize_explicit(A, B, dt):
    """
    ZOH discretization using the explicit formula:

        Ad = expm(A * dt)
        Bd = A^{-1} (Ad - I) B

    This requires A to be invertible or at least well-conditioned.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Continuous-time state matrix.
    B : array_like, shape (n, m)
        Continuous-time input matrix.
    dt : float
        Sampling time.

    Returns
    -------
    Ad : ndarray, shape (n, n)
        Discrete-time state matrix.
    Bd : ndarray, shape (n, m)
        Discrete-time input matrix.
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    n = A.shape[0]

    Ad = expm(A * dt)

    I = np.eye(n)

    # Avoid np.linalg.inv(A) explicitly.
    # This solves A X = (Ad - I) B for X.
    Bd = solve(A, (Ad - I) @ B)

    return Ad, Bd
