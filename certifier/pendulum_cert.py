"""
Certified region of attraction for the inverted pendulum under the eigenfunction
feedback u = -g^T grad w, on the EXACT lifted (sin, cos) dynamics.

State lift: z = (s, c, w) = (sin theta, cos theta, theta_dot), with s^2+c^2=1.
Dynamics (m l^2 = 1, no mass-matrix denominator):
    s' = c w,   c' = -s w,   w' = (g/l) s - b w + u.
Value function w2*(z): the degree-2 eigenfunction value function, whose Hessian
P_V solves the local Riccati of the ergodic HJB (cost q = 3(1-cos)+0.15 w^2,
control penalty 1/(2 sigma^2)). Feedback u = -dw2*/dw.

Certificate (deterministic, Certificate A): the LQR-type quadratic gives a
strictly Hurwitz closed loop, so we certify
    w2*'(z) <= -eps (s^2 + w^2)   on   Omega_gamma = {w2* <= gamma}, s^2+c^2=1,
via SOS, and maximize gamma by bisection.
"""
from __future__ import annotations
import os
import numpy as np
import sympy as sp
import scipy.linalg
from sos import SOSProblem, monomials

# ---- pendulum model -----------------------------------------------------------
G_OVER_L = 9.81
B_DAMP = 0.2
U_MAX = 5.0
SIGMA = 2.0

s, c, w = sp.symbols("s c w", real=True)
VARS = (s, c, w)

# ---- degree-2 eigenfunction value function from the local Riccati -------------
A = np.array([[0.0, 1.0], [G_OVER_L, -B_DAMP]])
B = np.array([[0.0], [1.0]])
Q = np.diag([3.0, 0.3])              # Hess q at x*: q = 3(1-cos th) + 0.15 th_dot^2
R = np.array([[1.0 / SIGMA**2]])     # control penalty 1/(2 sigma^2) ||u||^2
P_W = scipy.linalg.solve_continuous_are(A, B, Q, R)
P_V = SIGMA**2 * P_W                 # Hessian of V* = sigma^2 W
p11, p12, p22 = float(P_V[0, 0]), float(P_V[0, 1]), float(P_V[1, 1])

K = B.T @ P_V                        # eigenfunction feedback gain u = -K x
Acl = A - B @ K
print(f"P_V = [[{p11:.3f},{p12:.3f}],[{p12:.3f},{p22:.3f}]]")
print(f"feedback gain K = {K.ravel()}  (need K[0]>{G_OVER_L} to beat gravity)")
print(f"closed-loop eigs = {np.linalg.eigvals(Acl)}  (must be Hurwitz)")

# lifted quadratic value function: PD near upright, =2 p11 at hanging
W = p11 * (1 - c) + p12 * s * w + sp.Rational(1, 2) * p22 * w**2
W = sp.expand(W)

# control and closed-loop decrease in lifted coordinates
u_ctrl = -sp.diff(W, w)                              # u = -dW/dw  (= -(p12 s + p22 w))
sdot, cdot, wdot = c * w, -s * w, G_OVER_L * s - B_DAMP * w + u_ctrl
Wdot = sp.expand(sp.diff(W, s) * sdot + sp.diff(W, c) * cdot + sp.diff(W, w) * wdot)

# ---- SOS certificate at a fixed gamma (dict-based sos.py API) -----------------
from sos import CvxPoly, to_dict
EPS = 1e-6                                            # strict-decrease margin
b_half = monomials(VARS, 2)        # outer-SOS half basis (deg<=2)
b_sig = monomials(VARS, 1)         # sigma0 half basis  (deg<=1) -> sigma0 deg 2
l_mon = monomials(VARS, 2)         # equality multiplier L (deg<=2)
CONS = to_dict(s**2 + c**2 - 1, VARS)
WD = to_dict(Wdot, VARS)
Wd = to_dict(W, VARS)
MARG = to_dict(EPS * (s**2 + w**2), VARS)
SAT = to_dict(U_MAX**2 - u_ctrl**2, VARS)


def _gW(gamma):
    d = {k: -v for k, v in Wd.items()}; d[(0, 0, 0)] = d.get((0, 0, 0), 0.0) + gamma
    return d


def decrease_ok(gamma):
    """Wdot <= -eps (s^2+w^2) on {W<=gamma}, s^2+c^2=1."""
    prob = SOSProblem(VARS)
    sigma0 = prob.sos(b_sig); L = prob.free(l_mon); outer = prob.sos(b_half)
    prob.zero(CvxPoly({k: -v for k, v in WD.items()}) - sigma0 * _gW(gamma)
              - L * CONS - CvxPoly(MARG) - outer)
    prob.solve(); return prob.feasible


def saturation_ok(gamma):
    """U_max^2 - u^2 >= 0 on {W<=gamma}, s^2+c^2=1  (control unsaturated in basin)."""
    prob = SOSProblem(VARS)
    sigma1 = prob.sos(b_sig); L = prob.free(l_mon); outer = prob.sos(b_half)
    prob.zero(CvxPoly(SAT) - sigma1 * _gW(gamma) - L * CONS - outer)
    prob.solve(); return prob.feasible


def certified(gamma):
    return decrease_ok(gamma) and saturation_ok(gamma)


# disturbance enters the control channel: u -> u + d, |d| <= eps. The disturbance
# gain is g^T grad V = dW/dw (= dW/d theta_dot). The ISS / robustness margin eps*
# is the largest disturbance for which Omega_{gamma*} stays forward-invariant, i.e.
# Wdot +/- eps*(dW/dw) <= 0 on the boundary {W = gamma*} (both disturbance signs).
import cvxpy as cp
DISTG = to_dict(sp.diff(W, w), VARS)


def iss_margin(gamma):
    prob = SOSProblem(VARS)
    eps = cp.Variable(nonneg=True)
    Wm = {k: v for k, v in Wd.items()}; Wm[(0, 0, 0)] = Wm.get((0, 0, 0), 0.0) - gamma  # W - gamma
    cons = []
    for sgn in (1.0, -1.0):
        mu = prob.free(monomials(VARS, 2)); L = prob.free(l_mon); outer = prob.sos(b_half)
        ident = (CvxPoly({k: -v for k, v in WD.items()})
                 + CvxPoly({k: sgn * eps * v for k, v in DISTG.items()})
                 - mu * Wm - L * CONS - outer)
        for c in ident.terms.values():
            if isinstance(c, (int, float)):
                if abs(c) > 1e-9:
                    return 0.0
            else:
                cons.append(c == 0)
    cons += [Q >> 0 for Q in prob._psd]
    try:
        cp.Problem(cp.Maximize(eps), cons).solve(solver=cp.MOSEK)
        return float(eps.value) if eps.value is not None else 0.0
    except cp.error.SolverError:
        return 0.0


def bisect_gamma(g_lo=1e-4, g_hi=0.1, iters=34):
    cap = 2.0 * p11 * 0.98
    while certified(g_hi) and g_hi < cap:
        g_lo, g_hi = g_hi, min(g_hi * 2, cap)
    if not certified(g_lo):
        print("  base gamma infeasible.")
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (g_lo + g_hi)
        if certified(mid):
            g_lo = mid
        else:
            g_hi = mid
    return g_lo


def report_region(gamma, n=400):
    """Honest extent of Omega_gamma by sampling the variety (theta in [-pi,pi])."""
    th = np.linspace(-np.pi, np.pi, n)
    wd = np.linspace(-40, 40, n)
    TH, WD = np.meshgrid(th, wd)
    S, C = np.sin(TH), np.cos(TH)
    Wval = p11 * (1 - C) + p12 * S * WD + 0.5 * p22 * WD**2
    Uval = np.abs(p12 * S + p22 * WD)
    inside = Wval <= gamma
    return (np.abs(TH[inside]).max(), np.abs(WD[inside]).max(), Uval[inside].max())


if __name__ == "__main__":
    print("\nCertifying ROA (deterministic, degree-2 w2*) ...")
    g_star = bisect_gamma()
    print(f"\ncertified gamma* = {g_star:.4f}  (decrease AND |u|<=U_max)")
    th_m, thd_m, u_m = report_region(g_star)
    print(f"basin extent on variety: |theta| <= {th_m:.3f} rad ({np.degrees(th_m):.1f} deg), "
          f"|theta_dot| <= {thd_m:.3f} rad/s,  max|u| = {u_m:.3f} (U_max={U_MAX})")
