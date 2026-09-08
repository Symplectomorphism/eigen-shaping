"""
Certified ROA for the cart-pole upright, velocity coords z=(si,co,xc,vth,vx).

Lyapunov function w2* = 1/2 y^T P_V y (y=(si,xc,vth,vx)) is the degree-2
eigenfunction value function (P_V from the ergodic Riccati). The CATCH controller
is the constant-gain feedback u = -K x (K the r=2 linearization of the
eigenfunction feedback = the deployed LQR catch). A constant gain is polynomial,
so the closed loop needs only SINGLE detM clearing (the rational eigenfunction
feedback u*=-g^T grad V would force detM^2 -> degree 7, intractable here).

Normalize by detM0 = detM|_{co=1} = 0.025.  Certificates: V_dot < 0 on
Omega_gamma={w2*<=gamma}, and |u| <= U_max containment; gamma by bisection.
Damping set to zero (negligible; conservative).
"""
from __future__ import annotations
import os, time, random
import numpy as np
import sympy as sp
import scipy.linalg
import cvxpy as cp
from sos import SOSProblem, CvxPoly, to_dict, monomials
import cartpole_dyn as D

a, bb, dd, g = D.a, D.bb, D.dd, D.G_GRAV
GEAR = D.GEAR
SIGMA, U_MAX = 1.0, 3.0

si, co, xc, vth, vx = sp.symbols("si co xc vth vx", real=True)
VARS = (si, co, xc, vth, vx)
detM = a*dd - bb**2*co**2
detM0 = float(a*dd - bb**2)

# ---- ergodic Riccati (velocity coords) ----------------------------------------
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
              [float(dd*bb*g/detM0), 0, 0, 0],
              [float(-bb**2*g/detM0), 0, 0, 0]])
B = np.array([[0.], [0.], [float(-bb*GEAR/detM0)], [float(a*GEAR/detM0)]])
Q = np.diag([2.0, 0.4, 0.04, 0.04]); R = np.array([[1.0/SIGMA**2]])
P_W = scipy.linalg.solve_continuous_are(A, B, Q, R)
P_V = SIGMA**2 * P_W
Klin = (np.linalg.inv(R) @ B.T @ P_W).flatten()        # constant catch gain
print("P_V diag:", np.round(np.diag(P_V), 3), " cond=%.1e" % np.linalg.cond(P_V))
print("catch gain K =", np.round(Klin, 3),
      " closed-loop eigs:", np.round(np.linalg.eigvals(A - B @ Klin.reshape(1, -1)), 2))

# Lyapunov function: 1/2 x^T P_V x with the angle lifted as theta^2 -> 2(1-cos),
# theta -> si in cross terms. The (1-co) term is ESSENTIAL: using si alone fails to
# separate upright (theta=0) from hanging (theta=pi) (both have si=0), so {W<=gamma}
# on the variety would wrongly include the un-stabilized hanging equilibrium.
P = P_V
W = sp.expand(
    P[0, 0]*(1 - co)
    + P[1, 1]/2*xc**2 + P[2, 2]/2*vth**2 + P[3, 3]/2*vx**2
    + P[0, 1]*si*xc + P[0, 2]*si*vth + P[0, 3]*si*vx
    + P[1, 2]*xc*vth + P[1, 3]*xc*vx + P[2, 3]*vth*vx)
Wv = {v: sp.diff(W, v) for v in VARS}
u_lin = -(Klin[0]*si + Klin[1]*xc + Klin[2]*vth + Klin[3]*vx)   # polynomial, constant gain


def Wdot_times_detM(damp):
    dh, ds = (D.D_H, D.D_S) if damp else (sp.Integer(0), sp.Integer(0))
    tau_th = bb*g*si - dh*vth
    tau_x = bb*si*vth**2 - ds*vx + GEAR*u_lin                    # control force on cart
    thdd_dM = dd*tau_th - bb*co*tau_x                            # = th_ddot * detM
    xdd_dM = -bb*co*tau_th + a*tau_x                             # = x_ddot * detM
    wd = (detM*(Wv[si]*co*vth - Wv[co]*si*vth + Wv[xc]*vx)
          + Wv[vth]*thdd_dM + Wv[vx]*xdd_dM)
    return sp.expand(wd), thdd_dM, xdd_dM


WdM_d, thdd_d, xdd_d = Wdot_times_detM(damp=True)               # for validation
WdM_u, _, _ = Wdot_times_detM(damp=False)                       # for SOS
Wdot = WdM_u / detM0                                            # normalize by detM0

# validate damped accelerations against jax-validated momentum drift D.f
for _ in range(40):
    pt = {si: random.uniform(-.9, .9), xc: random.uniform(-1, 1),
          vth: random.uniform(-3, 3), vx: random.uniform(-3, 3)}
    pt[co] = (1 - pt[si]**2)**0.5
    Mloc = np.array([[float(a), float(bb)*pt[co]], [float(bb)*pt[co], float(dd)]])
    p = Mloc @ np.array([pt[vth], pt[vx]])
    mom = {D.si: pt[si], D.co: pt[co], D.xc: pt[xc], D.pth: p[0], D.ps: p[1]}
    uval = float(u_lin.subs(pt))
    pd = np.array([float(D.f[D.pth].subs(D.u, uval).subs(mom)),
                   float(D.f[D.ps].subs(D.u, uval).subs(mom))])
    Mdot = np.array([[0, -float(bb)*pt[si]*pt[vth]], [-float(bb)*pt[si]*pt[vth], 0]])
    acc = np.linalg.solve(Mloc, pd - Mdot @ np.array([pt[vth], pt[vx]]))
    dM = float(detM.subs(pt))
    assert abs(float(thdd_d.subs(pt))/dM - acc[0]) < 1e-6*(1+abs(acc[0]))
    assert abs(float(xdd_d.subs(pt))/dM - acc[1]) < 1e-6*(1+abs(acc[1]))

deg = sp.Poly(Wdot, *VARS).total_degree()
cs = [abs(float(t)) for t in sp.Poly(Wdot, *VARS).coeffs() if abs(float(t)) > 1e-13]
print(f"deg(Wdot)={deg}  coeff range [{min(cs):.2e},{max(cs):.2e}] ratio {max(cs)/min(cs):.1e}")

# ---- SOS ----------------------------------------------------------------------
CONS = to_dict(si**2 + co**2 - 1, VARS)
WD = to_dict(Wdot, VARS); Wd = to_dict(W, VARS)
MARG1 = to_dict(si**2 + xc**2 + vth**2 + vx**2, VARS)
SAT = to_dict(U_MAX**2 - u_lin**2, VARS)
ZERO = (0, 0, 0, 0, 0)
HD = (deg + 1) // 2                          # outer SOS half-degree (deg 5 -> HD 3)
print(f"bases: outer(deg{2*HD})={len(monomials(VARS,HD))} L(deg{deg})={len(monomials(VARS,deg))}")


def gW(gamma):
    d = {e: -c for e, c in Wd.items()}; d[ZERO] = d.get(ZERO, 0.) + gamma; return d


def _solve_eps(ident_builder, sig_hd, l_deg, out_hd):
    p = SOSProblem(VARS)
    s0 = p.sos(monomials(VARS, sig_hd)); L = p.free(monomials(VARS, l_deg)); out = p.sos(monomials(VARS, out_hd))
    eps = cp.Variable(nonneg=True)
    ident = ident_builder(s0, L, out, eps)
    cons = [Q >> 0 for Q in p._psd]
    for c in ident.terms.values():
        if isinstance(c, (int, float)):
            if abs(c) > 1e-9:
                return -1.0
        else:
            cons.append(c == 0)
    try:
        cp.Problem(cp.Maximize(eps), cons).solve(solver=cp.MOSEK)
        return float(eps.value) if eps.value is not None else -1.0
    except cp.error.SolverError:
        return -1.0


def decrease_ok(gamma):
    e = _solve_eps(
        lambda s0, L, out, eps: (CvxPoly({k: -v for k, v in WD.items()})
                                 - s0*gW(gamma) - L*CONS
                                 - CvxPoly({k: eps*v for k, v in MARG1.items()}) - out),
        sig_hd=HD-1, l_deg=2*HD-2, out_hd=HD)
    return e > 1e-7


def saturation_ok(gamma):
    e = _solve_eps(
        lambda s0, L, out, eps: (CvxPoly(SAT) - s0*gW(gamma) - L*CONS
                                 - CvxPoly({k: eps*v for k, v in MARG1.items()}) - out),
        sig_hd=2, l_deg=4, out_hd=3)
    return e > 1e-9


def certified(gamma):
    return decrease_ok(gamma) and saturation_ok(gamma)


# ISS / robustness margin: disturbance d on the cart-force channel (u -> u + d).
# Its (normalized) gain in Wdot is utilde/detM0. eps* is the largest |d| for which
# Omega_{gamma*} stays forward-invariant, i.e. Wdot +/- eps*(utilde/detM0) <= 0 on
# the boundary {W = gamma*} (both signs).
DISTG = to_dict(GEAR*(-bb*co*Wv[vth] + a*Wv[vx]) / detM0, VARS)


def iss_margin(gamma):
    p = SOSProblem(VARS); eps = cp.Variable(nonneg=True); cons = []
    Wm = {k: v for k, v in Wd.items()}; Wm[ZERO] = Wm.get(ZERO, 0.0) - gamma
    for sgn in (1.0, -1.0):
        mu = p.free(monomials(VARS, 4)); L = p.free(monomials(VARS, 2*HD-2)); out = p.sos(monomials(VARS, HD))
        ident = (CvxPoly({k: -v for k, v in WD.items()})
                 + CvxPoly({k: sgn*eps*v for k, v in DISTG.items()})
                 - mu*CvxPoly(Wm) - L*CONS - out)
        for c in ident.terms.values():
            if isinstance(c, (int, float)):
                if abs(c) > 1e-9:
                    return 0.0
            else:
                cons.append(c == 0)
    cons += [Q >> 0 for Q in p._psd]
    try:
        cp.Problem(cp.Maximize(eps), cons).solve(solver=cp.MOSEK)
        return float(eps.value) if eps.value is not None else 0.0
    except cp.error.SolverError:
        return 0.0


def bisect(g_lo=1e-3, g_hi=0.1, iters=26):
    while certified(g_hi) and g_hi < 100:
        g_lo, g_hi = g_hi, g_hi*2
    if not certified(g_lo):
        return 0.0
    for _ in range(iters):
        m = 0.5*(g_lo+g_hi)
        g_lo, g_hi = (m, g_hi) if certified(m) else (g_lo, m)
    return g_lo


def report(gamma, n=100):
    th = np.linspace(-np.pi, np.pi, n); v = np.linspace(-6, 6, n)
    b = dict(th=0., vth=0., vx=0., u=0.)
    P = P_V
    for t in th:
        st = np.sin(t)
        for a2 in v:
            for a3 in v[::3]:
                Wval = (P[0,0]*(1-np.cos(t)) + P[2,2]/2*a2**2 + P[3,3]/2*a3**2
                        + P[0,2]*st*a2 + P[0,3]*st*a3 + P[2,3]*a2*a3)
                if Wval <= gamma:
                    yv = np.array([st, 0., a2, a3])
                    b["th"] = max(b["th"], abs(t)); b["vth"] = max(b["vth"], abs(a2))
                    b["vx"] = max(b["vx"], abs(a3)); b["u"] = max(b["u"], abs(Klin@yv))
    return b


if __name__ == "__main__":
    t0 = time.time()
    print("\nCertifying cart-pole ROA (velocity coords, constant-gain catch) ...")
    gs = bisect()
    print(f"\ncertified gamma* = {gs:.4f}   (elapsed {time.time()-t0:.1f}s)")
    b = report(gs)
    print(f"basin (xc=0): |theta|<={b['th']:.3f} rad ({np.degrees(b['th']):.1f} deg), "
          f"|theta_dot|<={b['vth']:.2f}, |x_dot|<={b['vx']:.2f}; max|u|~{b['u']:.2f} (U_max={U_MAX})")
