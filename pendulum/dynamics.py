"""
Pendulum dynamics for the two-rung Schrodinger-Bridge validation ladder.

State: x = (theta, theta_dot) in R^2, theta measured from UPRIGHT.
  theta = 0  <=>  pole upright   (desired equilibrium x* = 0, UNSTABLE open-loop)
  theta = pi <=>  pole hanging down (stable open-loop)

Single torque input u in R (g = B, the velocity channel), matched noise:
  prior      dx = f0(x) dt + sigma * B dW,        W in R^1
  controlled dx = (f0(x) + B u) dt + sigma * B dW

Two rungs share this file:
  Rung 1 (linear, ph_drift_lin): f0(x) = A x, the upright linearization.
          (A, B) controllable with one unstable eigenvalue +sqrt(g/l-ish).
          The linear-Gaussian SB has a CLOSED FORM (linear_sb.py), so the lg
          bridge target is EXACT, not a surrogate -- the decisive correctness
          check for the DSBM machinery.
  Rung 2 (nonlinear, ph_drift): f0(x) with gravity (g/l) sin(theta).  Swing-up
          with torque saturation |u| <= U_MAX makes it effectively
          underactuated (cannot drive straight up; must pump energy).

This is the cart-pole minus the cart minus (rung 1) the nonlinearity.  If DSBM
cannot solve these, cart-pole is hopeless; the point of the ladder is to
localize where the difficulty actually lives.
"""

import jax
import jax.numpy as jnp

# Physical parameters (m l^2 = 1 so B = e_2; clean torque channel).
M      = 1.0     # pole mass (kg)
L      = 1.0     # pole length to COM (m)
G_GRAV = 9.81    # gravitational acceleration (m/s^2)
B_DAMP = 0.2     # viscous damping on theta_dot
U_MAX  = 5.0     # torque saturation for rung 2 (effectively underactuated)

# Desired equilibrium (upright, at rest)
X_STAR = jnp.zeros(2)

# Input matrix g = B (torque enters the theta_dot channel).
B_VEC = jnp.array([0.0, 1.0 / (M * L**2)])
G_VEC = B_VEC            # alias matching the paper's g


def ph_drift(x):
    """
    Nonlinear open-loop drift f0(x) for the inverted pendulum (rung 2).

      theta_ddot = (g/l) sin(theta) - b theta_dot

    Gravity destabilizes upright: d/dtheta[(g/l) sin theta]|_0 = g/l > 0.
    """
    theta, theta_dot = x
    return jnp.array([
        theta_dot,
        (G_GRAV / L) * jnp.sin(theta) - B_DAMP * theta_dot,
    ])


def ph_drift_lin(x):
    """Linearized drift f0(x) = A x about upright (rung 1)."""
    return A_MAT @ x


def linearize():
    """Return (A, B) of the upright linearization x_dot = A x + B u."""
    A = jax.jacobian(ph_drift)(X_STAR)
    B = B_VEC.reshape(2, 1)
    return A, B


# A computed once at import (constant; ph_drift_lin uses it).
A_MAT = jax.jacobian(ph_drift)(X_STAR)


def controlled_drift(x, u, nonlinear=True):
    """Closed-loop drift f0(x) + B u."""
    f0 = ph_drift(x) if nonlinear else ph_drift_lin(x)
    return f0 + B_VEC * u


def saturate(u):
    """Torque saturation for rung 2."""
    return jnp.clip(u, -U_MAX, U_MAX)


if __name__ == "__main__":
    import numpy as np

    A, B = linearize()
    print("A =\n", np.array(A))
    print("B =", np.array(B).ravel())
    w = np.linalg.eigvals(np.array(A))
    print("eig(A) =", w, " (expect one positive real -> unstable upright)")

    # Controllability
    C = np.hstack([np.array(B), np.array(A) @ np.array(B)])
    print("rank[B AB] =", np.linalg.matrix_rank(C), "(expect 2, controllable)")
