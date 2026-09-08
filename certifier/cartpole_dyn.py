"""
Cart-pole lifted polynomial dynamics for the SOS certifier, in port-Hamiltonian
MOMENTUM coordinates (so the input g=(0,0,0,GEAR) is constant and the
eigenfunction control u=-GEAR dV/dps stays polynomial).

State lift z = (si, co, xc, pth, ps) = (sin th, cos th, cart pos, p_theta, p_s),
with si^2+co^2 = 1.  Open-loop drift matches cartpole/dynamics.py exactly
(validated numerically against the jax ph_drift below).

The accelerations carry 1/detM (velocities) and 1/detM^2 (Coriolis in p_theta),
detM = a*dd - bb^2 co^2 > 0; the SOS programs clear it by multiplying through.
"""
from __future__ import annotations
import sympy as sp

# parameters identical to cartpole/dynamics.py, as EXACT rationals so that the
# symbolic detM^2 / detM cancellation is exact (float coeffs break it).
M_C = sp.Rational(1)
M_P = sp.Rational(1, 10)
L = sp.Rational(1, 2)
G_GRAV = sp.Rational(981, 100)
D_S = sp.Rational(1, 2000)
D_H = sp.Rational(1, 500000)
GEAR = sp.Integer(10)
a = M_P * L**2            # M11
bb = M_P * L              # M12 = bb*cos
dd = M_C + M_P            # M22

si, co, xc, pth, ps = sp.symbols("si co xc pth ps", real=True)
u = sp.symbols("u", real=True)
VARS = (si, co, xc, pth, ps)

detM = a * dd - bb**2 * co**2                 # > 0
# velocities theta_dot, s_dot = M^{-1} p   (carry 1/detM)
thd = (dd * pth - bb * co * ps) / detM
sdt = (-bb * co * pth + a * ps) / detM
# dH/dtheta = (kinetic) + dV/dtheta = bb*si*thd*sdt - bb*g*si
dHdth = bb * si * thd * sdt - bb * G_GRAV * si

# port-Hamiltonian drift f0 + g u  (g=(0,0,0,GEAR) constant)
f = {
    si:  co * thd,                       # d/dt sin th = cos th * th_dot
    co: -si * thd,                       # d/dt cos th = -sin th * th_dot
    xc:  sdt,                            # cart velocity
    pth: -dHdth - D_H * thd,             # p_theta_dot
    ps:  -D_S * sdt + GEAR * u,          # p_s_dot (+ control)
}


def drift_num(state, uval=0.0):
    """Numeric (si,co,xc,pth,ps)->dz/dt for validation (u given)."""
    subs = dict(zip(VARS, state)); subs[u] = uval
    return [float(sp.N(f[v].subs(subs))) for v in VARS]


def _validate(n=200, tol=1e-7):
    """Check the lifted drift equals the jax ph_drift at random states (u=0)."""
    import jax; jax.config.update("jax_enable_x64", True)   # match sympy float64
    import numpy as np, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cartpole"))
    from dynamics import ph_drift          # jax, momentum state (th,s,pth,ps)
    import jax.numpy as jnp
    rng = np.random.default_rng(0); worst = 0.0
    for _ in range(n):
        th = rng.uniform(-np.pi, np.pi)
        x = np.array([th, rng.uniform(-1, 1), rng.uniform(-3, 3),
                      rng.uniform(-3, 3)])  # (th, s, pth, ps)
        dj = np.array(ph_drift(jnp.array(x, dtype=jnp.float64)))   # thd,sd,pthd,psd
        z = [np.sin(th), np.cos(th), x[1], x[2], x[3]]
        dz = np.array(drift_num(z, 0.0))                   # si',co',xc',pth',ps'
        ref = np.array([np.cos(th) * dj[0], -np.sin(th) * dj[0], dj[1], dj[2], dj[3]])
        worst = max(worst, np.abs(dz - ref).max())
    return worst


if __name__ == "__main__":
    err = _validate()
    print(f"max |lifted drift - jax ph_drift| over 200 random states = {err:.2e}")
    print("OK" if err < 1e-6 else "MISMATCH -- do not use for SOS")
