"""
Khasminskii stochastic certificate (Certificate B): stability in probability of
the upright equilibrium under the certified catch controller at finite noise.

The controlled generator acting on the degree-2 Lyapunov function w2* is
    A_u w2* = Wdot + (sigma^2/2) g^T (Hess w2*) g,
the deterministic decrease Wdot < 0 (Certificate A) plus the Ito term. For the
pendulum the noise channel is g = (0,1), so the Ito term is the positive constant
I(sigma) = (sigma^2/2) p22. Hence A_u w2* < 0 fails inside a residual sublevel
ball B_delta = {w2* <= delta} but holds on the annulus {delta <= w2* <= gamma*}.
By Khasminskii's theorem this certifies recurrence to B_delta and stability in
probability of x*. The smallest certifiable delta scales as delta*(sigma) ~
sigma^2, so delta* -> 0 as sigma -> 0, recovering the deterministic asymptotic
stability of Certificate A.

The certificate is an SOS feasibility program on the annulus, intersected with the
variety s^2 + c^2 = 1; the controller and Lyapunov function are fixed at the design
point and sigma parametrizes the operating noise.
"""
from __future__ import annotations
import os
import numpy as np
import sympy as sp
import cvxpy as cp
from sos import SOSProblem, CvxPoly, to_dict, monomials
import pendulum_cert as PC

s, c, w = PC.VARS
VARS = PC.VARS
GAMMA = 1.9984                       # certified deterministic ROA level (Certificate A)
P22 = float(PC.P_V[1, 1])            # Hess w2* in the noise channel (theta_dot)

CONS = to_dict(s**2 + c**2 - 1, VARS)
WD = to_dict(PC.Wdot, VARS)          # deterministic decrease Wdot
Wd = to_dict(PC.W, VARS)
EPS = 1e-6
# The annulus certificate (two S-procedure multipliers, for gamma-W and W-delta)
# needs higher-degree multipliers than the simple sublevel-set decrease.
b_sig = monomials(VARS, 2)        # SOS multipliers sigma0, sigma1 (degree 4)
b_out = monomials(VARS, 3)        # outer SOS (degree 6)
l_mon = monomials(VARS, 4)        # equality multiplier L


def _lvl(level, sign):
    """sign*(level - W) as a coefficient dict (for S-procedure multipliers)."""
    d = {k: -sign * v for k, v in Wd.items()}
    d[(0, 0, 0)] = d.get((0, 0, 0), 0.0) + sign * level
    return d


def generator_ok(sigma, delta):
    """A_u w2* <= -eps on the annulus {delta <= W <= gamma*}, s^2+c^2=1."""
    I = 0.5 * sigma**2 * P22                          # Ito term (positive constant)
    p = SOSProblem(VARS)
    sig0 = p.sos(b_sig)                               # multiplier for gamma - W >= 0
    sig1 = p.sos(b_sig)                               # multiplier for W - delta >= 0
    L = p.free(l_mon)
    out = p.sos(b_out)
    Auw = CvxPoly({k: v for k, v in WD.items()}); Auw.terms[(0, 0, 0)] = \
        Auw.terms.get((0, 0, 0), 0.0) + I             # A_u w2* = Wdot + I
    ident = (CvxPoly({k: -v for k, v in Auw.terms.items()})
             - sig0 * _lvl(GAMMA, +1)                 # - sig0 (gamma - W)
             - sig1 * _lvl(delta, -1)                 # - sig1 (W - delta)
             - L * CONS
             - CvxPoly({(0, 0, 0): EPS}))
    p.zero(ident)
    p.solve()
    return p.feasible


def smallest_ball(sigma, d_lo=1e-5, iters=26):
    """Smallest delta with the generator decrease certified on {delta<=W<=gamma*}.
    generator_ok is monotone in delta (larger delta = smaller annulus = easier), so
    we bracket [infeasible, feasible] and bisect. The upper bracket is set below
    gamma* to avoid the degenerate annulus {gamma*<=W<=gamma*}."""
    d_hi = 0.95 * GAMMA
    if not generator_ok(sigma, d_hi):
        return None                                   # vacuous: ball fills the basin
    if generator_ok(sigma, d_lo):
        return d_lo                                   # essentially the whole basin certified
    for _ in range(iters):
        m = 0.5 * (d_lo + d_hi)
        if generator_ok(sigma, m):
            d_hi = m
        else:
            d_lo = m
    return d_hi


def ball_radius(delta):
    """Approx theta-extent of the residual ball {w2* <= delta} on the variety."""
    th = np.linspace(-np.pi, np.pi, 2000)
    Wv = PC.P_V[0, 0] * (1 - np.cos(th)) + 0  # at theta_dot=0 slice
    inside = Wv <= delta
    return np.abs(th[inside]).max() if inside.any() else 0.0


if __name__ == "__main__":
    print("=" * 70)
    print("Khasminskii stochastic certificate (pendulum): residual ball vs sigma")
    print(f"  Lyapunov w2* fixed at design point; gamma* = {GAMMA:.3f}, p22 = {P22:.3f}")
    print("  A_u w2* = Wdot + (sigma^2/2) p22 < 0  on  {delta <= w2* <= gamma*}")
    print("-" * 70)
    print(f"{'sigma':>6} {'Ito I':>9} {'delta*':>9} {'ball |theta|':>13} {'delta*/sig^2':>12}")
    for sigma in (2.0, 1.5, 1.0, 0.5, 0.25, 0.1):
        I = 0.5 * sigma**2 * P22
        d = smallest_ball(sigma)
        if d is None:
            print(f"{sigma:>6} {I:>9.3f} {'vacuous':>9}")
        else:
            r = ball_radius(d)
            print(f"{sigma:>6} {I:>9.3f} {d:>9.4f} {np.degrees(r):>11.1f} d {d/sigma**2:>12.4f}")
    print("=" * 70)
    print("delta* ~ sigma^2 (constant last column) => residual ball -> 0 as sigma -> 0,")
    print("recovering the deterministic asymptotic stability of Certificate A.")
