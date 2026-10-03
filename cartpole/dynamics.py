"""
Cart-pole port-Hamiltonian dynamics.

Parameters match dm_control cartpole.xml (arXiv:2006.12983):
  m_c = 1.0 kg  (cart), m_p = 0.1 kg (pole), l = 0.5 m (half-pole length),
  d_s = 5e-4 (slider damping), d_h = 2e-6 (hinge damping), g_grav = 9.81 m/s^2.

Coordinate convention: theta is measured from the UPRIGHT position.
  theta = 0  <=>  pole upright  (desired equilibrium x* = 0 in R^4)
  theta = pi <=>  pole hanging down

This places x* = (0, 0, 0, 0) at the origin, which is the natural choice for
IDA-PBC (nabla H_d(x*) = 0 at the origin) and for the Schrödinger Bridge
(target distribution centred at 0).

State: x = (theta, s, theta_dot, s_dot) in R^4
  theta     : pole angle from upright (rad)
  s         : cart position (m)
  theta_dot : pole angular velocity (rad/s)
  s_dot     : cart velocity (m/s)

Port-Hamiltonian form:
  [q_dot]   = (J - R) nabla_x H + g(x) u
  [p_dot]

with H = (1/2) p^T M(q)^{-1} p + V(q),  p = M(q) q_dot.
Input matrix g = (0, 0, 0, 1)^T (constant), u in R (force on cart, post-gear).
Matched-noise SDE: dx = f0(x) dt + sigma * g dW, W in R^1.
"""

import jax
import jax.numpy as jnp

# ── Physical parameters ────────────────────────────────────────────────────────
M_C   = 1.0      # cart mass (kg)
M_P   = 0.1      # pole mass (kg)
L     = 0.5      # pole half-length (m)  — capsule from 0 to 1 m, COM at 0.5 m
G_GRAV = 9.81    # gravitational acceleration (m/s^2)
D_S   = 5e-4     # slider (cart) damping coefficient
D_H   = 2e-6     # hinge (pole) damping coefficient
GEAR  = 10.0     # actuator gear ratio: physical cart force = GEAR * u_cmd

# Desired equilibrium at origin in shifted coordinates
X_STAR = jnp.zeros(4)

# Input matrix g (now explicitly includes GEAR so the PDE knows the actuator strength)
G_VEC = jnp.array([0.0, 0.0, 0.0, GEAR])


def mass_matrix(theta):
    """
    2x2 configuration-space mass matrix M(theta).

    Lagrangian mass matrix for (theta, s) with point-mass pole approximation
    (I_total = m_p l^2 about the pivot):

      M = [[ m_p l^2,          m_p l cos(theta) ],
           [ m_p l cos(theta),  m_c + m_p        ]]

    With theta measured from upright, cos(theta) = 1 at theta=0.
    """
    m11 = M_P * L**2
    m12 = M_P * L * jnp.cos(theta)
    m22 = M_C + M_P
    return jnp.array([[m11, m12],
                      [m12, m22]])


def potential_energy(theta):
    """
    V(theta) = -m_p g l cos(theta).

    With theta measured from upright (theta=0 is upright):
      V(0)  = -m_p g l   (local maximum of -cos: cos(0)=1 is the maximum of cos,
                           so -cos(0) is the minimum of -cos)

    Wait — we need upright to be an UNSTABLE equilibrium of the open-loop system,
    meaning it is a saddle point of H.  For an inverted pendulum:
      - At theta=0 (upright), gravity pulls the pole away -> unstable.
      - At theta=pi (hanging), gravity restores the pole -> stable.

    The gravitational torque on the pole is  tau = -m_p g l sin(theta)  (from upright).
    In the Hamiltonian framework this comes from  -dV/dtheta = m_p g l sin(theta),
    so dV/dtheta = -m_p g l sin(theta), giving V = m_p g l cos(theta) + const.

    Choosing V = m_p g l cos(theta):
      dV/dtheta = -m_p g l sin(theta) -> zero at theta=0 and theta=pi. Good.
      d²V/dtheta²|_{theta=0}  = -m_p g l < 0  -> local MAXIMUM -> unstable. Correct.
      d²V/dtheta²|_{theta=pi} = +m_p g l > 0  -> local MINIMUM -> stable.   Correct.

    With this choice nabla_x H(x*=0) = 0 still holds (dV/dtheta|_0 = 0, velocities=0).
    """
    return M_P * G_GRAV * L * jnp.cos(theta)


def hamiltonian(x):
    """H(x) = (1/2) p^T M^{-1} p + V(q).  Minimum at x* = 0."""
    theta, s, p_theta, p_s = x
    p = jnp.array([p_theta, p_s])
    M = mass_matrix(theta)
    KE = 0.5 * p @ jnp.linalg.solve(M, p)
    PE = potential_energy(theta)
    return KE + PE


def ph_drift(x):
    """
    Open-loop port-Hamiltonian drift f0(x) = (J - R) nabla_x H.

    J = block-skew:  J nabla_H = (dH/dp, -dH/dq) = (q_dot, -dH/dq)
    R = diag(0, 0, d_h, d_s)   (damping on velocity components only)

    Returns f0(x) in R^4.
    """
    nabla_H = jax.grad(hamiltonian)(x)
    J_nabla = jnp.array([ nabla_H[2],
                           nabla_H[3],
                          -nabla_H[0],
                          -nabla_H[1]])
    R_nabla = jnp.array([0.0, 0.0, D_H * nabla_H[2], D_S * nabla_H[3]])
    return J_nabla - R_nabla


def controlled_drift(x, u):
    """Closed-loop drift f0(x) + g * u."""
    return ph_drift(x) + G_VEC * u


def diffusion_vec(sigma):
    """Diffusion vector sigma * g (constant in x)."""
    return sigma * G_VEC


# ── Linearization at x* = 0 ───────────────────────────────────────────────────

def linearize():
    """
    Return A = df0/dx|_{x*=0} and B (4x1).

    The linear model is  x_dot = A x + B u  where u is the pre-gear command.
    """
    A = jax.jacobian(ph_drift)(X_STAR)
    B = G_VEC.reshape(4, 1)
    return A, B


def lqr_gains(Q=None, R=None):
    """
    LQR value-function matrix P and gain K for the linearization at x* = 0.

    Solves the continuous-time ARE for the linear model x_dot = A x + B u (u the
    pre-gear command, B carries GEAR).  Returned P is the catch-region value
    function (½ xᵀP x); K is the stabilizing gain.  Shared by the trainer (local
    anchor for V) and the evaluator (LQR catch controller) so both agree.
    """
    import numpy as np
    import scipy.linalg
    A, B = linearize()
    A = np.array(A); B = np.array(B)
    if Q is None:
        Q = np.diag([10.0, 1.0, 1.0, 1.0])
    if R is None:
        R = np.array([[0.1]])
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = (np.linalg.inv(R) @ B.T @ P)[0]
    return P, K


if __name__ == "__main__":
    import numpy as np

    A, B = linearize()
    print("A =")
    print(np.array(A))
    print("\nB =", np.array(B).ravel())

    eigvals = np.linalg.eigvals(np.array(A))
    print("\nEigenvalues of A:", eigvals)
    print("(expect one real positive eigenvalue for the unstable pole mode)")

    print("\nH(x*) =", hamiltonian(X_STAR))
    print("nabla H(x*) =", jax.grad(hamiltonian)(X_STAR),
          "  (should be zero at the upright equilibrium)")
